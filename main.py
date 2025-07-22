from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from fastapi.responses import JSONResponse,FileResponse
import uuid
import os
import tempfile
from typing import List
from utils import (
    extract_model_names, 
    handle_document_upload, 
    process_question
)
import ollama
import edge_tts
import asyncio

app = FastAPI()

# Paths (make sure Wav2Lip repo is cloned and checkpoints exist)
WAV2LIP_DIR = os.path.join(os.path.dirname(__file__), "Wav2Lip")
CHECKPOINT_PATH = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
STATIC_IMG = os.path.join(WAV2LIP_DIR, "samples", "aigirl.jpeg")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "temp")

app = FastAPI(title="Ollama Document QA")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory state (use Redis or DB for persistence in production)
vector_db = None
selected_model = "llama3.2:latest"
PERSIST_DIRECTORY = os.path.join("data", "vectors")
VECTOR_COLLECTION_NAME = "multi_file_rag"


@app.on_event("startup")
def load_persisted_vector_db():
    global vector_db
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_db = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
                collection_name=VECTOR_COLLECTION_NAME
            )
            collection = vector_db._collection
            metadatas = collection.get(include=["metadatas"])["metadatas"]

            print("✅ Vector DB loaded successfully from disk.")
        except Exception as e:
            print("❌ Failed to load persisted vector DB:", e)
    else:
        print("ℹ️ No persisted vector DB found. Upload documents first.")

@app.get("/models")
def list_models():
    models_info = ollama.list()
    model_names = extract_model_names(models_info)
    return {"models": model_names}

# @app.post("/upload-documents")
# async def upload_documents(
#     files: List[UploadFile] = File(...),
#     user_role: str = Form(...),  # Accepting role from the frontend
# ):
#     global vector_db
#     temp_files = []
#     try:
#         # Check the role and validate if it's allowed to upload documents
#         if user_role not in ['employee', 'guest']:  # Add any other validation if needed
#             raise HTTPException(status_code=400, detail="Invalid user role.")

#         # Process files and assign role-based metadata
#         for file in files:
#             suffix = os.path.splitext(file.filename)[1]
#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                 tmp.write(await file.read())
#                 temp_files.append(tmp.name)

#         # Call the modified handle_document_upload function with the role
#         vector_db = handle_document_upload(temp_files, user_role)

#         return {"status": "success", "message": "Documents processed."}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
#     finally:
#         for f in temp_files:
#             os.remove(f)



@app.post("/upload-documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    categories: List[str] = Form(...),  
    display_names: List[str] = Form(...),

):
    if len(files) != len(categories)or len(files) != len(display_names):
        raise HTTPException(status_code=400, detail="Files and categories count mismatch.")
    global vector_db
    temp_files = []
    try:
        file_data_list = []
        for file, category,display_name in zip(files, categories,display_names):
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                temp_files.append((tmp.name, category,display_name))

        vector_db = handle_document_upload(temp_files)

        return {"status": "success", "message": "Documents processed."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        for f, _, _ in temp_files:
            os.remove(f)


# @app.post("/ask_question")
# async def ask_question(question: str = Form(...)):
#     if not vector_db:
#         return JSONResponse(status_code=400, content={"error": "No documents uploaded."})

#     try:
#         response = process_question(question, vector_db, selected_model)
#         return {"response": response}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask_question")
async def ask_question(
    question: str = Form(...),
    user_role: str = Form(...),  # new field: either "authorized" or "unauthorized"
):
    if not vector_db:
        return JSONResponse(status_code=400, content={"error": "No documents uploaded."})

    try:
        print(user_role)
        response = process_question(question, vector_db, selected_model, user_role)
        return {"response": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/delete-vectors")
def delete_vectors():
    global vector_db
    try:
        if vector_db:
            vector_db.delete_collection()
            vector_db = None
            return {"status": "success", "message": "Vector DB deleted."}
        return {"status": "info", "message": "No vector DB to delete."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/uploaded-documents")
def get_uploaded_documents():
    print(vector_db)
    if not vector_db:
        return {"status": "error", "message": "No vector DB loaded."}
    
    try:
        collection = vector_db._collection.get(include=["metadatas"])
        filenames = set()

        for meta in collection["metadatas"]:
            if meta and "display_name" in meta and "category" in meta:
                filenames.add( (meta["display_name"], meta["category"]) )


        return {"status": "success", "documents": sorted(filenames)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

async def generate_edge_tts(text, audio_path):
    communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural")  # Pick voice
    await communicate.save(audio_path)


import asyncio

# async def enhance_video_quality(input_path, output_path):
#     cmd = [
#         "ffmpeg", "-y",
#         "-i", input_path,
#         "-vf", "scale=2160:2160:flags=lanczos",  # Change scale as needed
#         "-c:v", "libx264",
#         "-crf", "18",  # Lower means better quality (default is 23)
#         "-preset", "slow",
#         "-c:a", "aac",
#         "-b:a", "256k",
#         output_path
#     ]
#     process = await asyncio.create_subprocess_exec(*cmd)
#     await process.communicate()


async def enhance_video_quality(input_path: str, output_path: str):
    try:
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf",
            "scale=2160:2160:flags=lanczos,"        # upscale to square 2K
            "unsharp=5:5:1.5:5:5:0.0,"              # gentle sharpening
            "eq=contrast=1.1:saturation=1.2",       # realistic tone
            "-c:v", "libx264",
            "-b:v", "5M",                           # solid bitrate
            "-preset", "ultrafast",
            "-crf", "18",                           # good quality
            "-c:a", "aac",
            "-b:a", "192k",
            output_path
        ]


        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE
        )
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
        stdout=asyncio.subprocess.DEVNULL,  # 👈 discard logs to save time,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        print("Wav2Lip Error:", stderr.decode())

@app.post("/video_call")
async def ask(question: str = Form(...),
    user_role: str = Form(...)
,  ):
    try:
        # Clean up previous output files before generating new ones
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith(".wav") or file.endswith(".mp4"):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, file))
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")

        ai_response = process_question(question, vector_db, selected_model,user_role)
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


# GET /stream endpoint to serve video
@app.get("/stream")
def stream_video(path: str):
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Video not found"})
    return FileResponse(path, media_type="video/mp4")

