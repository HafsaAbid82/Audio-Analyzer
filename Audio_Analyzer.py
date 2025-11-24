from __future__ import annotations
import os
import tempfile
import whisperx
from whisperx.diarize import DiarizationPipeline
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Optional, Dict, List, Any
import soundfile as sf_
import torch 
from dataclasses import dataclass, field
try:
    import noisereduce as nr
    HAVE_NOISEREDUCE = True
except ImportError:
    HAVE_NOISEREDUCE = False
Annotation = None
Segment = None
DiarizationErrorRate = None
pyannote_available = False
try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    pyannote_available = True
except ImportError:
    pass
device = "cuda" if torch.cuda.is_available() else "cpu"
token = os.environ.get("HF_TOKEN")
perform_diarization = bool(token)
model_name = "small"
@dataclass
class AnalysisResults:
    """Structured container for the output of the audio analysis pipeline."""
    timelineData: List[Dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0
    languageCode: str = "unknown"
    diarizationErrorRate: Optional[float] = None
    speakerError: Optional[float] = None
    missedSpeech: Optional[float] = None
    falseAlarm: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    success: bool = False
    message: str = "Analysis initiated."
def warn(results: AnalysisResults, code: str, detail: str) -> None:
    msg = f"{code}: {detail}"
    if msg not in results.warnings:
        results.warnings.append(msg)
def set_message(results: AnalysisResults, msg: str) -> None:
    initial_message = "Analysis initiated." 
    if results.message and results.message != initial_message:
        results.message += f" | {msg}"
    else:
        results.message = msg
def normalize_speaker(lbl: str) -> str:
    lbl_str = str(lbl)
    return lbl_str.replace("SPEAKER_", "Speaker_").replace("speaker_", "Speaker_")
def temp_wav_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        return f.name
def butter_filter(y, sr, lowpass=None, highpass=None, order=4):
    nyq = 0.5 * sr
    if highpass and highpass > 0 and highpass < nyq:
        b, a = butter(order, highpass / nyq, btype="highpass", analog=False)
        y = filtfilt(b, a, y)
    if lowpass and lowpass > 0 and lowpass < nyq:
        b, a = butter(order, lowpass / nyq, btype="lowpass", analog=False)
        y = filtfilt(b, a, y)
    return y
def rms_normalize(y, target_rms=0.1, eps=1e-6):
    rms = (y**2).mean() ** 0.5
    if rms < eps:
        return y
    gain = target_rms / (rms + eps)
    return y * gain
def preprocess_audio(input_path,
                     target_sr=16000,
                     normalize_rms=True,
                     target_rms=0.08,
                     denoise=False,
                     highpass=None,
                     lowpass=None,
                     output_subtype="PCM_16",
                     verbose=False):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio not found: {input_path}")
    output_path = temp_wav_path()
    y_stereo, sr = sf_.read(input_path, dtype='float64')
    if y_stereo.ndim > 1:
        y = librosa.to_mono(y_stereo.T)
    else:
        y = y_stereo
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if highpass or lowpass:
        y = butter_filter(y, sr, highpass=highpass, lowpass=lowpass)
    if denoise and HAVE_NOISEREDUCE:
        try:
            noise_len = int(min(len(y), int(0.5 * sr)))
            noise_clip = y[:noise_len]
            y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip, prop_decrease=0.9, verbose=False)
        except Exception:
            pass 
    if normalize_rms:
        y = rms_normalize(y, target_rms=target_rms)
    sf.write(output_path, y, sr, subtype=output_subtype)
    return output_path
def load_rttm(path: str) -> Optional[None["Annotation"]]:
    if not pyannote_available or not os.path.exists(path):
        return None
    ann = Annotation()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(";;"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start = float(parts[3])
                    dur = float(parts[4])
                    spk = normalize_speaker(parts[7])
                    ann[Segment(start, start + dur)] = spk
        return ann
    except Exception as e:
        print(f"Error loading RTTM: {e}")
        return None
def diarization_to_annotation(diarize_output: Any) -> Optional[None["Annotation"]]:
    if not pyannote_available:
        return None
    ann = Annotation()
    try:
        if hasattr(diarize_output, "itertracks"):
            for segment, _, label in diarize_output.itertracks(yield_label=True):
                ann[Segment(segment.start, segment.end)] = normalize_speaker(label)
            return ann
    except Exception:
        pass
    
    try:
        for seg in diarize_output:
            if all(k in seg for k in ("start", "end", "speaker")):
                s = float(seg["start"])
                e = float(seg["end"])
                lbl = normalize_speaker(seg["speaker"])
                ann[Segment(s, e)] = lbl
        return ann
    except Exception:
        return None
def analyze_audio(audio_file: str,
                  reference_rttm_file: Optional[str] = None,
                  preprocess: bool = True,
                  preprocess_params: Optional[Dict[str, Any]] = None) -> AnalysisResults:
    results = AnalysisResults()   
    if not os.path.exists(audio_file):
        results.message = f"Error: Input audio file '{audio_file}' not found."
        return results    
    audio_for_model = audio_file
    temp_preproc = None
    if preprocess:
        params = {
            "target_sr": 16000, "normalize_rms": True, "target_rms": 0.08,
            "denoise": False, "highpass": None, "lowpass": None, 
            "output_subtype": "PCM_16",
            "verbose": False
        }
        if isinstance(preprocess_params, dict):
            params.update(preprocess_params)       
        if params.get("denoise") and not HAVE_NOISEREDUCE:
            warn(results, "DENOISE_SKIP", "Denoise requested but noisereduce not installed; skipping denoise.")
            params["denoise"] = False        
        try:
            temp_preproc = preprocess_audio(audio_file, **params)
            audio_for_model = temp_preproc
        except Exception as e:
            warn(results, "PREP_FAIL", f"Preprocessing failed: {e}. Falling back to original audio.")
            audio_for_model = audio_file
            temp_preproc = None
    try:
        print(f"Loading Whisper model '{model_name}' on {device}...")
        model = whisperx.load_model(model_name, device)
        audio_loaded = whisperx.load_audio(audio_for_model)        
        print("Transcribing audio...")
        result = model.transcribe(audio_loaded, batch_size=4)
        language_code = result.get("language") or result.get("detected_language") or "en"
        results.languageCode = language_code       
        print(f"Detected language: {language_code}. Aligning transcription...")
        try:
            align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
            aligned = whisperx.align(result["segments"], align_model, metadata, audio_loaded, device)
        except Exception:
            aligned = {"segments": result["segments"]}
            warn(results, "ALIGN_SKIP", "Alignment unavailable; using raw Whisper segments.")        
        diarize_output = None
        if perform_diarization:
            print("Performing speaker diarization (Requires HF_TOKEN)...")
            try:
                diarize_model = DiarizationPipeline(use_auth_token=token, device=device)
                diarize_output = diarize_model(audio_for_model)
            except Exception as e:
                warn(results, "DIAR_SKIP", f"Error during diarization (likely token/model failure): {e}. Skipping diarization.")
                diarize_output = None
        else:
            warn(results, "DIAR_SKIP", "HF_TOKEN not set. Skipping speaker diarization.")
        if diarize_output is None:
            last = max((seg.get("end", 0.0) for seg in aligned["segments"]), default=0.0)
            diarize_output = [{"start": 0.0, "end": last, "speaker": normalize_speaker("Speaker_1")}]
            print("No diarization performed. Assigning all speech to Speaker_1.")
        print("Assigning speakers to words...")
        final = whisperx.assign_word_speakers(diarize_output, aligned)
        rows = []
        for seg in final.get("segments", []):
            if "words" in seg and seg["words"]:
                for w in seg["words"]:
                    speaker_id = normalize_speaker(w.get("speaker") or seg.get("speaker", "Speaker_1"))
                    rows.append({
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        "text": w["text"].strip(),
                        "speaker": str(speaker_id),
                    })
            else:
                speaker_id = normalize_speaker(seg.get("speaker", "Speaker_1"))
                rows.append({
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": seg.get("text", "").strip(),
                    "speaker": str(speaker_id),
                })        
        results.timelineData = rows
        ends = [w.get("end", 0.0) for w in rows if isinstance(w.get("end"), (int, float))]
        results.duration = max(ends) if ends else 0.0
        if pyannote_available and reference_rttm_file and perform_diarization:
            print(f"Computing DER using reference RTTM: {reference_rttm_file}...")
            reference = load_rttm(reference_rttm_file)
            hypothesis = diarization_to_annotation(diarize_output)
            
            if reference is None:
                warn(results, "DER_SKIP", f"Reference RTTM not loaded correctly from {os.path.basename(reference_rttm_file)}; skipping DER.")
            elif hypothesis is None:
                warn(results, "DER_SKIP", "Diarization output conversion failed; skipping DER.")
            else:
                try:
                    # pyannote.metrics computation
                    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
                    der_report = metric.compute_components(reference, hypothesis)
                    results.diarizationErrorRate = der_report['diarization error rate']
                    results.speakerError = der_report['speaker error']
                    results.missedSpeech = der_report['missed speech']
                    results.falseAlarm = der_report['false alarm']
                    print(f"DER computed: {results.diarizationErrorRate:.4f}")
                except Exception as e:
                    warn(results, "DER_ERROR", f"Error computing DER: {e}")
        elif reference_rttm_file:
            warn(results, "DER_SKIP", "DER not computed because pyannote is unavailable or diarization was skipped (no HF_TOKEN).")
            
        results.success = True
        set_message(results, "Analysis complete.")
        if reference_rttm_file and results.diarizationErrorRate is None:
            warn(results, "DER_MISSING", "Reference RTTM provided but DER was not computed. Check for missing pyannote dependencies or diarization failure (HF_TOKEN).")
            
    except Exception as e:
        results.message = f"Core analysis failed: {type(e).__name__}: {e}"
        results.success = False
        warn(results, "FATAL_ERROR", f"Fatal analysis error: {type(e).__name__}: {e}")         
    finally:
        if temp_preproc and os.path.exists(temp_preproc):
            try:
                os.remove(temp_preproc)
            except Exception:
                pass                
    return results
if __name__ == "__main__":
    AUDIO_FILE = "Sample.mp3"
    RTTM_FILE = "segments.rttm"
    if not os.path.exists(AUDIO_FILE):
        print(f"\n--- ATTENTION ---")
        print(f"ERROR: Audio file '{AUDIO_FILE}' not found.")
        print("Please place your audio file in the same directory and name it 'Sample.mp3'.")
        print("-----------------\n")   
    if not perform_diarization:
        print(f"\n--- WARNING ---")
        print(f"HF_TOKEN environment variable is not set.")
        print("Diarization (speaker assignment) and DER calculation will be skipped.")
        print("-----------------\n")
    print(f"Starting analysis for audio: {AUDIO_FILE}")
    print(f"Reference RTTM: {RTTM_FILE}")
    preprocessing_config = {"denoise": False} 
    analysis_result = analyze_audio(
        audio_file=AUDIO_FILE,
        reference_rttm_file=RTTM_FILE,
        preprocess_params=preprocessing_config
    )
    print("\n" + "="*50)
    print("ANALYSIS REPORT")
    print("="*50)
    print(f"Success: {analysis_result.success}")
    print(f"Message: {analysis_result.message}")
    print(f"Duration: {analysis_result.duration:.2f} seconds")
    print(f"Language: {analysis_result.languageCode}") 
    if analysis_result.diarizationErrorRate is not None:
        print("\n--- Diarization Error Rate (DER) Metrics ---")
        print(f"DER: {analysis_result.diarizationErrorRate:.4f}")
        print(f"Speaker Error: {analysis_result.speakerError:.4f}")
        print(f"Missed Speech: {analysis_result.missedSpeech:.4f}")
        print(f"False Alarm: {analysis_result.falseAlarm:.4f}")      
    if analysis_result.warnings:
        print("\n--- Warnings ---")
        for w in analysis_result.warnings:
            print(f"- {w}")
    print("\n--- Timeline Data (First 5 entries) ---")
    if analysis_result.timelineData:
        for entry in analysis_result.timelineData[:5]:
            print(f"[{entry['start']:.2f}-{entry['end']:.2f}] ({entry['speaker']}): {entry['text']}")
    else:
        print("No transcription data generated.")
    print("="*50)
