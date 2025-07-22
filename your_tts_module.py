from gtts import gTTS
import io

def synthesize_speech(text: str) -> bytes:
    tts = gTTS(text)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()
