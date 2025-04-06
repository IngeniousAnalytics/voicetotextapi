from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import subprocess
import torch
import os

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)

@app.post("/stream/")
async def stream_transcription(file: UploadFile = File(...)):
    # Save the uploaded .webm file to a temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_webm:
        tmp_webm.write(await file.read())
        webm_path = tmp_webm.name

    # Create another temp file for the .wav output
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)  # We just need the path

    try:
        # Convert .webm to .wav using ffmpeg
        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Transcribe using Whisper
        result = model.transcribe(wav_path)

        return {"text": result["text"]}

    except subprocess.CalledProcessError as e:
        return {"text": "[Error converting audio file]"}

    finally:
        # Clean up temporary files
        if os.path.exists(webm_path):
            os.remove(webm_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
