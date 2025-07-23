from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.embeddings import HuggingFaceEmbeddings


import uuid
import os
import tempfile
import asyncio
from typing import List
from dotenv import load_dotenv

from utils import (
    extract_model_names, 
    handle_document_upload, 
    process_question
)

import ollama
import edge_tts

load_dotenv()

app = FastAPI(title="Ollama Document QA")

# Paths
WAV2LIP_DIR = os.path.join(os.path.dirname(__file__), "Wav2Lip")
CHECKPOINT_PATH = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
STATIC_IMG = os.path.join(WAV2LIP_DIR, "samples", "aigirl.jpeg")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "temp")

OUTPUT_DIR = "/tmp"
print("OUTPUT_DIR",OUTPUT_DIR)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_db = None
selected_model = "llama3.2:latest"
VECTOR_COLLECTION_NAME = "doc_qa"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


@app.on_event("startup")
def load_persisted_vector_db():
    global vector_db
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        existing = [c.name for c in client.get_collections().collections]
        if VECTOR_COLLECTION_NAME not in existing:
            client.recreate_collection(
                collection_name=VECTOR_COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        vector_db = Qdrant(
            client=client,
            collection_name=VECTOR_COLLECTION_NAME,
            embeddings=embeddings,
        )
        print("✅ Qdrant Cloud vector DB loaded with HuggingFace embeddings.")
    except Exception as e:
        print("❌ Failed to load Qdrant DB:", e)


@app.get("/models")
def list_models():
    models_info = ollama.list()
    model_names = extract_model_names(models_info)
    return {"models": model_names}

@app.post("/upload-documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    categories: List[str] = Form(...),  
    display_names: List[str] = Form(...),
):
    if len(files) != len(categories) or len(files) != len(display_names):
        raise HTTPException(status_code=400, detail="Files and categories count mismatch.")
    global vector_db
    temp_files = []
    try:
        file_data_list = []
        for file, category, display_name in zip(files, categories, display_names):
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                temp_files.append((tmp.name, category, display_name))

        vector_db = handle_document_upload(temp_files)
        return {"status": "success", "message": "Documents processed."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        for f, _, _ in temp_files:
            os.remove(f)

@app.post("/ask_question")
async def ask_question(
    question: str = Form(...),
    user_role: str = Form(...),
):
    if not vector_db:
        return JSONResponse(status_code=400, content={"error": "No documents uploaded."})
    try:
        response = process_question(question, vector_db, user_role)
        return {"response": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/delete-vectors")
def delete_vectors():
    global vector_db
    try:
        if vector_db:
            vector_db.client.delete_collection(collection_name=VECTOR_COLLECTION_NAME)
            vector_db = None
            return {"status": "success", "message": "Vector DB deleted."}
        return {"status": "info", "message": "No vector DB to delete."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/uploaded-documents")
def get_uploaded_documents():
    if not vector_db:
        return {"status": "error", "message": "No vector DB loaded."}
    try:
        
        collection = vector_db.client.scroll(collection_name=VECTOR_COLLECTION_NAME, with_payload=True)
        # print("collection=====",collection)
        filenames = set()
        for record in collection[0]:
            meta = record.payload
            # Safely extract nested metadata
            metadata = meta.get("metadata", {})
            display_name = metadata.get("display_name")
            category = metadata.get("category")
            print(display_name,category)
            if display_name and category:
                filenames.add((display_name, category))
        return {"status": "success", "documents": sorted(filenames)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

async def generate_edge_tts(text, audio_path):
    communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural")
    await communicate.save(audio_path)

async def enhance_video_quality(input_path: str, output_path: str):
    try:
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf",
            "scale=2160:2160:flags=lanczos,"
            "unsharp=5:5:1.5:5:5:0.0,"
            "eq=contrast=1.1:saturation=1.2",
            "-c:v", "libx264",
            "-b:v", "5M",
            "-preset", "ultrafast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path
        ]
        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {stderr.decode()}")
    except Exception as e:
        print("Video enhancement failed:", str(e))
        raise

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
        print("Wav2Lip Error:", stderr.decode())

@app.post("/video_call")
async def ask(question: str = Form(...), user_role: str = Form(...)):
    try:
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith(".wav") or file.endswith(".mp4"):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, file))
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")

        ai_response = process_question(question, vector_db, user_role)
        unique_id = str(uuid.uuid4())
        audio_path = f"{OUTPUT_DIR}/{unique_id}.wav"
        raw_video_path = f"{OUTPUT_DIR}/{unique_id}_raw.mp4"
        enhanced_video_path = f"{OUTPUT_DIR}/{unique_id}.mp4"

        await generate_edge_tts(ai_response, audio_path)
        await run_wav2lip(audio_path, raw_video_path)
        await enhance_video_quality(raw_video_path, enhanced_video_path)

        return {
            "reply": ai_response,
            "video_url": f"http://localhost:8000/stream?path={enhanced_video_path}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/stream")
def stream_video(path: str):
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Video not found"})
    return FileResponse(path, media_type="video/mp4")
