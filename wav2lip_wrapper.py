import subprocess
import uuid
import os

def generate_lip_video(audio_path: str, face_img: str) -> str:
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, f"lip_{uuid.uuid4().hex}.mp4")

    command = [
        "python3", "inference.py",
        "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
        "--face", face_img,
        "--audio", audio_path,
        "--outfile", output_path
    ]
    subprocess.run(command, cwd="Wav2Lip", check=True)
    return output_path
