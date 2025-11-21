import os
import sys
import tempfile
import whisperx
from whisperx.diarize import DiarizationPipeline
import plotly.express as px
import pandas as pd
try:
    import librosa
    import soundfile as sf
except ImportError:
    print("FATAL ERROR: Please install librosa and soundfile: pip install librosa soundfile")
    sys.exit(1)
try:
    import noisereduce as nr
    HAVE_NOISEREDUCE = True
except ImportError:
    HAVE_NOISEREDUCE = False
pyannote_available = True
try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
except ImportError:
    pyannote_available = False
    Annotation = None
    Segment = None
    DiarizationErrorRate = None
    print("WARNING: Pyannote dependencies not installed. DER calculation will be skipped.")
device = "cpu"
token = os.environ.get("HF_TOKEN")
perform_diarization = bool(token)
if not perform_diarization:
    print("WARNING: HF_TOKEN not set. Skipping speaker diarization and DER calculation.")
def rms_normalize(y, target_rms=0.1, eps=1e-6):
    rms = (y**2).mean() ** 0.5
    if rms < eps:
        return y
    gain = target_rms / (rms + eps)
    return y * gain
def apply_filter(y, sr, highpass=None, lowpass=None):
    if highpass and highpass > 0:
        y = librosa.effects.preemphasis(y, coef=0.97)
    if lowpass and lowpass < sr / 2:
        target_sr = int(min(sr, max(8000, int(lowpass * 2))))
        if target_sr < sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            y = librosa.resample(y, orig_sr=target_sr, target_sr=sr)
    return y
def preprocess_audio(input_path,
                     target_sr=16000,
                     normalize_rms=True,
                     target_rms=0.08,
                     denoise=False,
                     highpass=None,
                     lowpass=None,
                     verbose=True):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio not found: {input_path}")
    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    if verbose:
        print(f"\n--- Preprocessing {os.path.basename(input_path)} ---")
    y, sr = librosa.load(input_path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if highpass or lowpass:
        y = apply_filter(y, sr, highpass=highpass, lowpass=lowpass)
    if denoise:
        if not HAVE_NOISEREDUCE:
            if verbose:
                print("noisereduce not installed; skipping denoise.")
        else:
            noise_len = int(min(len(y), int(0.5 * sr)))
            noise_clip = y[:noise_len]
            try:
                y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip, prop_decrease=0.9, verbose=False)
                if verbose:
                    print("Applied noise reduction.")
            except Exception as e:
                if verbose:
                    print(f"Noise reduction failed: {e}; continuing without denoise.")
    if normalize_rms:
        y = rms_normalize(y, target_rms=target_rms)
    sf.write(output_path, y, sr, subtype="PCM_16")
    if verbose:
        print(f"Wrote preprocessed WAV to temporary file: {output_path} (sr={sr})")
    return output_path
def load_rttm(path):
    if not pyannote_available or not os.path.exists(path):
        return None
    ann = Annotation()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                try:
                    start = float(parts[3])
                    dur = float(parts[4])
                    spk = parts[7]
                    ann[Segment(start, start + dur)] = spk
                except ValueError:
                    continue
    return ann
def diarization_to_annotation(diarize_output):
    """Convert diarization output (list-of-dicts or pyannote-like) to pyannote.Annotation."""
    if not pyannote_available:
        return None
    ann = Annotation()
    try:
        if hasattr(diarize_output, "itertracks"):
            for segment, track, label in diarize_output.itertracks(yield_label=True):
                ann[Segment(segment.start, segment.end)] = label
            return ann
    except Exception:
        pass
    try:
        for seg in diarize_output:
            if all(k in seg for k in ("start", "end", "speaker")):
                s = float(seg["start"])
                e = float(seg["end"])
                lbl = str(seg["speaker"])
                ann[Segment(s, e)] = lbl
        return ann
    except Exception:
        return None
def analyze_audio(audio_file,
                  reference_rttm_file=None,
                  preprocess=True,
                  preprocess_params=None):
    if not os.path.exists(audio_file):
        print(f"Error: Input audio file '{audio_file}' not found.")
        return
    audio_for_model = audio_file
    temp_preproc = None
    if preprocess:
        params = {
            "target_sr": 16000,
            "normalize_rms": True,
            "target_rms": 0.08,
            "denoise": False,
            "highpass": None,
            "lowpass": None,
            "verbose": True
        }
        if isinstance(preprocess_params, dict):
            params.update(preprocess_params)
        try:
            temp_preproc = preprocess_audio(audio_file, **params)
            audio_for_model = temp_preproc
        except Exception as e:
            print(f"Preprocessing failed: {e}. Falling back to original audio.")
            audio_for_model = audio_file
            temp_preproc = None
    print("\n1. Loading WhisperX model (small)...")
    model = whisperx.load_model("small", device)
    print(f"2. Loading audio for transcription: {os.path.basename(audio_for_model)}...")
    audio = whisperx.load_audio(audio_for_model)
    print("3. Transcribing audio...")
    result = model.transcribe(audio, batch_size=4)
    language_code = result.get("language", "en")
    print(f"4. Loading alignment model for language: {language_code}...")
    align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    print("5. Aligning segments for word-level timestamps...")
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)
    diarize_output = None
    if perform_diarization:
        print("6. Performing speaker diarization (Requires HF_TOKEN)...")
        try:
            diarize_model = DiarizationPipeline(use_auth_token=token, device=device)
            diarize_output = diarize_model(audio_for_model)
        except Exception as e:
            print(f"Error during diarization (likely token/model failure): {e}")
            diarize_output = None

    if diarize_output is None:
        print("Diarization skipped/failed. Assigning 'Speaker_1' to all words.")
        last = 0.0
        if isinstance(aligned, dict) and "segments" in aligned:
            last = max((seg.get("end", 0.0) for seg in aligned["segments"]), default=0.0)
        else:
            last = max((seg.get("end", 0.0) for seg in result.get("segments", [])), default=0.0)
        diarize_output = [{"start": 0.0, "end": last, "speaker": "Speaker_1"}]
    print("7. Merging diarization with word-aligned transcription...")
    final = whisperx.assign_word_speakers(diarize_output, aligned)
    rows = []
    for seg in final.get("segments", []):
        if "words" in seg:
            for w in seg["words"]:
                speaker_id = w.get("speaker") or seg.get("speaker", "Speaker_1")
                rows.append({
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "text": w["text"].strip(),
                    "speaker": str(speaker_id),
                })
        else:
            speaker_id = seg.get("speaker", "Speaker_1")
            rows.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", "").strip(),
                "speaker": str(speaker_id),
            })
    timeline_df = pd.DataFrame(rows)
    if timeline_df.empty:
        print("No transcription data available to build timeline.")
        if temp_preproc and os.path.exists(temp_preproc):
            try:
                os.remove(temp_preproc)
            except Exception:
                pass
        return
    timeline_df["wrapped_text"] = timeline_df["text"].str.wrap(30)
    print("\n8. Generating interactive timeline (Plotly)...")
    fig = px.timeline(
        timeline_df,
        x_start="start",
        x_end="end",
        y="speaker",
        hover_data={"text": True, "start": ":.2f", "end": ":.2f", "speaker": True},
        text="wrapped_text",
        color="speaker",
        title=f"Interactive Audio Timeline for {os.path.basename(audio_file)} (Language: {language_code})"
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(textposition="outside")
    fig.show()
    if pyannote_available and reference_rttm_file and perform_diarization:
        print("\n9. Computing Diarization Error Rate (DER)...")
        hypothesis = diarization_to_annotation(diarize_output)
        reference = load_rttm(reference_rttm_file)
        if reference is None:
            print(f"WARNING: Reference RTTM not found at {reference_rttm_file}; skipping DER.")
        elif hypothesis is None:
            print("WARNING: Diarization output conversion failed; skipping DER.")
        else:
            try:
                metric = DiarizationErrorRate()
                der = metric(reference, hypothesis)
                print("\n--- Diarization Quality Metrics ---")
                print(f"Overall Diarization Error Rate (DER): {der:.4f}")
            except Exception as e:
                print(f"Error computing DER: {e}")
    else:
        if reference_rttm_file:
            print("DER not computed because pyannote is not available, HF token missing, or diarization skipped.")
    if temp_preproc and os.path.exists(temp_preproc):
        try:
            os.remove(temp_preproc)
            print(f"Cleaned up temporary file: {temp_preproc}")
        except Exception:
            pass
if __name__ == "__main__":
    audio_file_path = "audio_sample.wav"
    rttm_file_path = None  
    preprocess_params = {
        "target_sr": 16000,
        "normalize_rms": True,
        "target_rms": 0.08,
        "denoise": False,
        "highpass": 80,
        "lowpass": None,
        "verbose": True
    }
    print("--- Starting Advanced Audio Analysis Pipeline ---")
    analyze_audio(
        audio_file_path,
        reference_rttm_file=rttm_file_path,
        preprocess=True,
        preprocess_params=preprocess_params
    )


