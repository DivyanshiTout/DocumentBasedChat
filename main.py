from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import List
import os
import tempfile
import uuid
import asyncio
import ollama
import edge_tts
from dotenv import load_dotenv


from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings

from utils import (
    extract_model_names,
    handle_document_upload,  # Update this too to use Qdrant
    process_question
)

app = FastAPI(title="Ollama Document QA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv()



QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "doc-qa")  

# === GLOBAL VARS ===
vector_db = None
selected_model = "llama3.2:latest"
OUTPUT_DIR = "/tmp/wav2lip_output"
WAV2LIP_DIR = os.path.join(os.path.dirname(__file__), "Wav2Lip")
CHECKPOINT_PATH = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
STATIC_IMG = os.path.join(WAV2LIP_DIR, "samples", "aigirl.jpeg")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.on_event("startup")
def connect_qdrant():
    global vector_db
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        vector_db = Qdrant(
            client=client,
            collection_name=QDRANT_COLLECTION,
            Embeddings=embeddings,
        )
        print("✅ Connected to Qdrant and collection ready.")
    except Exception as e:
        print("❌ Failed to initialize Qdrant:", e)

@app.get("/models")
def list_models():
    return {"models": extract_model_names(ollama.list())}

@app.post("/upload-documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    categories: List[str] = Form(...),
    display_names: List[str] = Form(...)
):
    if len(files) != len(categories) or len(files) != len(display_names):
        raise HTTPException(status_code=400, detail="Mismatched input lengths.")
    
    global vector_db
    temp_files = []

    try:
        for file, category, display_name in zip(files, categories, display_names):
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as tmp:
                tmp.write(await file.read())
                temp_files.append((tmp.name, category, display_name))

        # Update handle_document_upload() to push to Qdrant
        vector_db = handle_document_upload(temp_files)

        return {"status": "success", "message": "Documents processed."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        for f, _, _ in temp_files:
            os.remove(f)

@app.post("/ask_question")
async def ask_question(question: str = Form(...), user_role: str = Form(...)):
    if not vector_db:
        return JSONResponse(status_code=400, content={"error": "No vector DB loaded."})
    try:
        response = process_question(question, vector_db, selected_model, user_role)
        return {"response": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/delete-vectors")
def delete_vectors():
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        return {"status": "success", "message": "Collection deleted from Qdrant."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/uploaded-documents")
def get_uploaded_documents():
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        response = client.scroll(collection_name=QDRANT_COLLECTION, with_payload=True, limit=100)
        filenames = set()
        for point in response[0]:
            metadata = point.payload
            if "display_name" in metadata and "category" in metadata:
                filenames.add((metadata["display_name"], metadata["category"]))
        return {"status": "success", "documents": sorted(filenames)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

async def generate_edge_tts(text, audio_path):
    communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural")
    await communicate.save(audio_path)

async def enhance_video_quality(input_path, output_path):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "scale=2160:2160:flags=lanczos,unsharp=5:5:1.5:5:5:0.0,eq=contrast=1.1:saturation=1.2",
        "-c:v", "libx264", "-b:v", "5M", "-preset", "ultrafast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", output_path
    ]
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {stderr.decode()}")

async def run_wav2lip(audio_path, video_path):
    process = await asyncio.create_subprocess_exec(
        "python3", "inference.py",
        "--checkpoint_path", CHECKPOINT_PATH,
        "--face", STATIC_IMG,
        "--audio", audio_path,
        "--outfile", video_path,
        cwd=WAV2LIP_DIR,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Wav2Lip error: {stderr.decode()}")

@app.post("/video_call")
async def video_call(question: str = Form(...), user_role: str = Form(...), request: Request = None):
    try:
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith(".wav") or file.endswith(".mp4"):
                os.remove(os.path.join(OUTPUT_DIR, file))

        ai_response = process_question(question, vector_db, selected_model, user_role)
        unique_id = str(uuid.uuid4())
        audio_path = f"{OUTPUT_DIR}/{unique_id}.wav"
        raw_video_path = f"{OUTPUT_DIR}/{unique_id}_raw.mp4"
        enhanced_video_path = f"{OUTPUT_DIR}/{unique_id}.mp4"

        await generate_edge_tts(ai_response, audio_path)
        await run_wav2lip(audio_path, raw_video_path)
        await enhance_video_quality(raw_video_path, enhanced_video_path)

        base_url = str(request.base_url).rstrip("/")
        return {
            "reply": ai_response,
            "video_url": f"{base_url}/stream?path={enhanced_video_path}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/stream")
def stream_video(path: str):
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Video not found"})
    return FileResponse(path, media_type="video/mp4")
