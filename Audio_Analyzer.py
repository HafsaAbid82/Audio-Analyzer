import os
import tempfile
import whisperx
from pyannote.audio import Pipeline
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Optional, Dict, List, Any
import torch
from dataclasses import dataclass, field
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import shutil
try:
    import noisereduce as nr
    HAVE_NOISEREDUCE = True
except ImportError:
    HAVE_NOISEREDUCE = False
Annotation: Any = None
Segment: Any = None

device = "cuda" if torch.cuda.is_available() else "cpu"
token = os.environ.get("HF_TOKEN")
try:
    if token:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=token
        )
        diarization_pipeline.to(torch.device(device))
    else:
        diarization_pipeline = None
except Exception as e:
    print(f"Error loading pyannote pipeline: {type(e).__name__}: {e}. Diarization will be skipped.")
    diarization_pipeline = None
global_diarizer = diarization_pipeline
model_name = "medium"
ALIGN_MODEL_MAP = {
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu"}
global_align_model_cache = {}
class TimelineItem(BaseModel):
    start: float
    end: float
    speaker: str | None = None
    text: str
class AnalysisResult(BaseModel):
    duration: float
    language: str
    der: float | None = None
    speaker_error: float | None = None
    missed_speech: float | None = None
    false_alarm: float | None = None
    timeline_data: list[TimelineItem]
    raw_transcription: str

app = FastAPI(title="Audio Analyzer Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend-audio-analyzer.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@dataclass
class AnalysisResults:
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
def force_float(value: Optional[Any]) -> Optional[float]:
    """Ensures value is a native Python float or None. Returns None for NaN/Inf."""
    if value is None:
        return None
    try:
        f_val = float(value)
        if np.isnan(f_val) or np.isinf(f_val):
            return None
        return f_val
    except (TypeError, ValueError, AttributeError):
        return None

def butter_filter(y, sr, lowpass=None, highpass=None, order=4):
    nyq = 0.5 * sr
    if highpass and highpass > 0 and highpass < nyq:
        b, a = butter(order, highpass / nyq, btype="highpass", analog=False)
        y = filtfilt(b, a, y)
    if lowpass and lowpass > 0 and lowpass < nyq:
        b, a = butter(order, lowpass / nyq, btype="lowpass", analog=False)
        y = filtfilt(b, a, y)
    return y

def rms_normalize(y, target_rms=0.8, eps=1e-6):
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
                     verbose=False) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio not found: {input_path}")
    output_path = temp_wav_path()
    y_stereo, sr = sf.read(input_path, dtype='float64')
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
def analyze_audio(audio_file: str, 
                  reference_rttm_file: Optional[str] = None, 
                  preprocess: bool = True, 
                  preprocess_params: Optional[Dict[str, Any]] = None) -> AnalysisResults:
    global global_align_model_cache, ALIGN_MODEL_MAP
    results = AnalysisResults()
    ends: List[float] = []      
    rows: List[Dict[str, Any]] = []
    rawTranscriptionText: str = ""
    if not os.path.exists(audio_file):
        results.message = f"Error: Input audio file '{audio_file}' not found."
        return results
    audio_for_model = audio_file
    temp_preproc = None
    if preprocess:
        params = {
            "target_sr": 16000, "normalize_rms": True, "target_rms": 0.08,
            "denoise": False, "highpass": None, "lowpass": None,
            "output_subtype": "PCM_16", "verbose": False
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
            
    start_ml_time = time.time()
    try:
        print(f"Loading Whisper model '{model_name}' on {device}...")
        model = whisperx.load_model(model_name, device, compute_type="float32")
        audio_loaded = whisperx.load_audio(audio_for_model) 
        lang_result = model.transcribe(audio_loaded)
        language_code_detected = lang_result.get("language") or lang_result.get("detected_language")
        languageCode = language_code_detected
        results.languageCode = languageCode
        print("Transcribing audio...")
        transcribed_language = "ur"
        result = model.transcribe(audio_loaded, batch_size=4, language= transcribed_language 
          )
        full_text = " ".join([seg['text'] for seg in result.get("segments", [])]).strip()
        results.rawTranscriptionText = full_text
        aligned = {"segments": result["segments"]} 
        print(f"Detected language: {languageCode}. Aligning transcription...")
        aligner_lookup_language = transcribed_language
        align_model = None
        metadata = None
        if aligner_lookup_language not in global_align_model_cache:
            align_model_name = ALIGN_MODEL_MAP.get(aligner_lookup_language)
            try:
                align_model, metadata = whisperx.load_align_model(
                                                  language_code=aligner_lookup_language, 
                                                  model_name=align_model_name, 
                                                  device=device
                )
                global_align_model_cache[aligner_lookup_language] = (align_model, metadata)
                print(f"Alignment model successfully loaded/cached for language: {aligner_lookup_language}")
            except Exception as e:
                warn(results, "ALIGN_LOAD_FAIL", f"Failed to load alignment model for {aligner_lookup_language}: {type(e).__name__}: {e}. Alignment skipped.")
                global_align_model_cache[aligner_lookup_language] = (None, None)
        else:
            align_model, metadata = global_align_model_cache[aligner_lookup_language]
            if align_model:
                print(f"Alignment model loaded from cache for language: {aligner_lookup_language}")
        if align_model:
            try:
                print("Performing word-level alignment...")
                aligned = whisperx.align(
                    result["segments"], 
                    align_model, 
                    metadata, 
                    audio_loaded, 
                    device
                )
            except Exception as e:
                warn(results, "ALIGN_RUN_FAIL", f"Alignment execution failed: {type(e).__name__}: {e}. Using raw segments.")
        else:
            warn(results, "ALIGN_SKIP", "Alignment model unavailable; using raw Whisper segments.")
        diarize_output = None
        if global_diarizer is not None:
            print("Performing speaker diarization (Requires HF_TOKEN)...")
            try:
                diarize_output = global_diarizer(audio_for_model)
                for segment, _, label in diarize_output.itertracks(yield_label=True):
                    print(f"start={segment.start:.1f}s stop={segment.end:.1f}s {label}")
            except Exception as e:
                warn(results, "DIAR_SKIP", f"Error during diarization (likely token/model failure): {type(e).__name__}: {e}. Skipping diarization.")
                diarize_output = None
        else:
            warn(results, "DIAR_SKIP", "HF_TOKEN not set or Diarization Pipeline failed to load globally. Skipping speaker diarization.")
        print("Assigning speakers to words...")                           
        try:
            diarize_segments_for_assignment = []
            if diarize_output is not None and hasattr(diarize_output, "itertracks"):
                for segment, _, label in diarize_output.itertracks(yield_label=True):
                    diarize_segments_for_assignment.append({
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "speaker": normalize_speaker(label)
                    })
                print(f"DEBUG: Converted {len(diarize_segments_for_assignment)} diarization segments.")
            if diarize_segments_for_assignment:
                diarize_df = pd.DataFrame(diarize_segments_for_assignment)
                final = whisperx.assign_word_speakers(diarize_df, aligned)
            else:
                warn(results, "ASSIGN_FAIL", "Diarization segments were empty or unavailable. Defaulting all to Speaker_1.")
                final = aligned
                for seg in final.get("segments", []):
                    seg["speaker"] = "Speaker_1"
        except Exception as e:
            warn(results, "ASSIGN_SPEAKERS_ERROR", f"Error assigning speakers: {type(e).__name__}: {e}. Falling back to unassigned segments.")
            final = aligned
            for seg in final.get("segments", []):
                seg["speaker"] = "Speaker_1"
        def _get_time_field(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
            """Try multiple possible keys and coerce to native float, returning None if not possible."""
            for k in keys:
                if k in d:
                    try:
                        v = d[k]
                        if v is None:
                            continue
                        f = float(v)
                        if np.isnan(f) or np.isinf(f):
                            return None
                        return f
                    except (TypeError, ValueError):
                        continue
            return None
        for seg in final.get("segments", []):
            seg_speaker = normalize_speaker(seg.get("speaker") or seg.get("speaker_label") or "Speaker_1")
            word_list = seg.get("words") or seg.get("tokens") or seg.get("items") or []
            if not word_list:
                word_start = _get_time_field(seg, ["start", "s", "timestamp", "t0"])
                word_end = _get_time_field(seg, ["end", "e", "t1"])
                if word_start is None:
                    continue
                if word_end is None:
                    word_end = word_start
                rows.append({
                    "start": float(word_start),
                    "end": float(word_end),
                    "text": str(seg.get("text", "")).strip(),
                    "speaker": str(seg_speaker),
                })
                continue
            for w in word_list:
                if not isinstance(w, dict):
                    continue
                word_start = _get_time_field(w, ["start", "s", "timestamp", "t0"])
                word_end = _get_time_field(w, ["end", "e", "t1"])
                if word_start is None:
                    word_start = _get_time_field(seg, ["start", "s"])
                if word_end is None:
                    word_end = _get_time_field(seg, ["end", "e"])          
                if word_start is None:
                    continue
                if word_end is None:
                    word_end = word_start
                word_speaker = normalize_speaker(w.get("speaker") or seg_speaker)
                word_text = (w.get("text") or w.get("word") or w.get("label") or "").strip()
                rows.append({
                    "start": float(word_start),
                    "end": float(word_end),
                    "text": str(word_text),
                    "speaker": str(word_speaker),
                })
        rows = sorted(rows, key=lambda r: r.get("start", 0.0))
        results.timelineData = rows
        for w in rows:
            e = w.get("end")
            f_e = force_float(e)
            if f_e is not None:
                ends.append(f_e)  
    except Exception as e:
        results.message = f"Error during ML processing: {type(e).__name__}: {e}"
        return results
    finally:
        if temp_preproc and os.path.exists(temp_preproc):
            os.remove(temp_preproc)
        results.duration = force_float(max(ends) if ends else 0.0) or 0.0
        end_ml_time = time.time()
        print(f"ML Processing finished in {end_ml_time - start_ml_time:.2f} seconds.")
    results.success = True
    return results
@app.post("/upload", response_model=AnalysisResult)
async def upload_file(audio_file: UploadFile = File(...)):
    start_time = time.time()
    audio_path: Optional[str] = None
    try:
        print("Incoming upload:", getattr(audio_file, "filename", None))

        suffix = audio_file.filename.split(".")[-1] if audio_file.filename else "tmp"
        with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as tmp_audio:
            shutil.copyfileobj(audio_file.file, tmp_audio)
            audio_path = tmp_audio.name
        print(f"Received audio file: {audio_file.filename} (saved to {audio_path}), size: {os.path.getsize(audio_path)} bytes")
        preprocessing_config = {"denoise": False}
        print(f"Starting ML processing with audio: {audio_path}, preprocess_params: {preprocessing_config}")
        analysis_result = analyze_audio(
            audio_file=audio_path,
            preprocess_params=preprocessing_config
        )
        print("MESSAGE:", analysis_result.message)
        if not analysis_result.success:
            raise HTTPException(status_code=500, detail=analysis_result.message)
        print("DURATION BEFORE RETURN:", analysis_result.duration)
        if analysis_result.duration is None:
            analysis_result.duration = 0.0
        return AnalysisResult(
            duration=force_float(analysis_result.duration) or 0.0,
            language=analysis_result.languageCode,
            timeline_data=[
                TimelineItem(
                    start=force_float(item.get('start')) or 0.0,
                    end=force_float(item.get('end')) or 0.0,
                    speaker=str(item.get('speaker')) if item.get('speaker') else None,
                    text=str(item.get('text', ""))
                ) for item in analysis_result.timelineData
            ],
            raw_transcription=analysis_result.rawTranscriptionText
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during upload process: {type(e).__name__}: {e}")
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        end_time = time.time()
        print(f"API Request processed in {end_time - start_time:.2f} seconds.")
@app.get("/")
def root():
    return {"message": "Audio Analyzer Backend is running."}
