# backend.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

import subprocess
import librosa
import os

# -------------------- APP INIT --------------------
app = FastAPI()

# Load faster-whisper model ONCE
fw_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="float32"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- UTILITIES --------------------

def extract_audio(video_path: str) -> str:
    """
    Extract mono 16kHz WAV audio from video using ffmpeg.
    """
    audio_path = os.path.splitext(video_path)[0] + ".wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        audio_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    return audio_path


def analyze_audio(audio_path: str):
    """
    Analyze speech duration and silence ratio.
    """
    y, sr = librosa.load(audio_path, sr=16000)

    intervals = librosa.effects.split(y, top_db=30)
    total_speech_duration = sum((end - start) / sr for start, end in intervals)
    total_duration = len(y) / sr

    silence_ratio = (
        (total_duration - total_speech_duration) / total_duration
        if total_duration > 0 else 0
    )

    return total_speech_duration, silence_ratio


def transcribe_audio(audio_path: str):
    """
    Transcribe audio and return segments + transcript.
    """
    segments, info = fw_model.transcribe(audio_path)
    segments = list(segments)

    transcript = " ".join(segment.text for segment in segments)

    return segments, transcript


def compute_feedback(
    transcript: str,
    speech_duration: float,
    silence_ratio: float,
    segments
):
    """
    Generate public speaking feedback.
    """
    words = transcript.split()
    pace_wpm = len(words) / (speech_duration / 60) if speech_duration > 0 else 0

    filler_words = sum(
        transcript.lower().count(w) for w in ["um", "uh", "like"]
    )

    # Pronunciation confidence (segment-level)
    unclear_segments = [
        s.text.strip()
        for s in segments
        if s.avg_logprob < -0.7
    ]

    return {
        "pace_wpm": round(pace_wpm, 2),
        "silence_ratio": round(silence_ratio, 2),
        "filler_word_count": filler_words,
        "unclear_segments": unclear_segments,
        "transcript": transcript
    }

# -------------------- ROUTES --------------------

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save uploaded video
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # 1️⃣ Extract audio
    audio_path = extract_audio(video_path)

    # 2️⃣ Analyze audio
    speech_duration, silence_ratio = analyze_audio(audio_path)

    # 3️⃣ Transcribe audio
    segments, transcript = transcribe_audio(audio_path)

    # 4️⃣ Generate feedback
    feedback = compute_feedback(
        transcript,
        speech_duration,
        silence_ratio,
        segments
    )

    # 5️⃣ Cleanup
    os.remove(video_path)
    os.remove(audio_path)

    return JSONResponse(content=feedback)


@app.get("/")
async def root():
    return {"message": "Public Speaking Analyzer Backend Running!"}
