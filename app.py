"""
Bazarr-compatible Whisper API wrapper.

Translates whisper-asr-webservice API format (what Bazarr speaks)
into whisper.cpp server API format (/inference endpoint).
"""

import io
import os
import struct
import logging

import httpx
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger("uvicorn.error")

WHISPER_SERVER_URL = os.environ.get("WHISPER_SERVER_URL", "http://whisper-server:8081")

# ISO 639-1 code -> language name (subset that Whisper supports)
WHISPER_LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
    "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
    "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}

app = FastAPI(title="Bazarr Whisper Wrapper")
client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0))


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Wrap raw PCM s16le data in a WAV header."""
    data_size = len(pcm_data)
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,       # ChunkSize
        b"WAVE",
        b"fmt ",
        16,                   # Subchunk1Size (PCM)
        1,                    # AudioFormat (PCM = 1)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_data


def map_output_format(output: str) -> str:
    """Map whisper-asr-webservice output names to whisper.cpp response_format names."""
    mapping = {
        "srt": "srt",
        "vtt": "vtt",
        "txt": "text",
        "json": "verbose_json",
        "tsv": "text",  # no direct equivalent, fall back to text
    }
    return mapping.get(output, "srt")


@app.get("/health")
async def health():
    """Proxy health check to whisper-server."""
    try:
        r = await client.get(f"{WHISPER_SERVER_URL}/health")
        return Response(content=r.content, status_code=r.status_code)
    except httpx.ConnectError:
        return JSONResponse({"status": "whisper-server unavailable"}, status_code=503)


@app.post("/asr")
async def asr(
    audio_file: UploadFile = File(...),
    task: str = Query(default="transcribe"),
    language: str = Query(default=None),
    output: str = Query(default="srt"),
    encode: str = Query(default="true"),
    initial_prompt: str = Query(default=None),
    video_file: str = Query(default=None),
    vad_filter: bool = Query(default=False),
    word_timestamps: bool = Query(default=False),
):
    """
    Bazarr-compatible /asr endpoint.
    Translates requests to whisper.cpp /inference format.
    """
    audio_data = await audio_file.read()

    # Bazarr sends raw PCM s16le mono 16kHz when encode=false
    if encode == "false":
        audio_data = pcm_to_wav(audio_data)
        filename = "audio.wav"
    else:
        filename = audio_file.filename or "audio.wav"

    # Build multipart form for whisper.cpp
    form_data = {
        "temperature": "0.0",
        "temperature_inc": "0.2",
        "response_format": map_output_format(output),
    }

    if language:
        form_data["language"] = language

    if task == "translate":
        form_data["translate"] = "true"

    if initial_prompt:
        form_data["prompt"] = initial_prompt

    files = {"file": (filename, audio_data, "audio/wav")}

    logger.info(f"Forwarding ASR request: lang={language}, task={task}, format={output}")

    r = await client.post(
        f"{WHISPER_SERVER_URL}/inference",
        data=form_data,
        files=files,
    )

    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type="text/plain",
    )


@app.post("/detect-language")
async def detect_language(
    audio_file: UploadFile = File(...),
    encode: str = Query(default="true"),
    video_file: str = Query(default=None),
):
    """
    Bazarr-compatible /detect-language endpoint.
    Calls whisper.cpp /inference with verbose_json to extract the detected language.
    """
    audio_data = await audio_file.read()

    if encode == "false":
        audio_data = pcm_to_wav(audio_data)
        filename = "audio.wav"
    else:
        filename = audio_file.filename or "audio.wav"

    # Use a short duration to speed up language detection
    form_data = {
        "temperature": "0.0",
        "response_format": "verbose_json",
        "language": "auto",
    }

    files = {"file": (filename, audio_data, "audio/wav")}

    logger.info("Forwarding language detection request")

    r = await client.post(
        f"{WHISPER_SERVER_URL}/inference",
        data=form_data,
        files=files,
    )

    if r.status_code != 200:
        return JSONResponse(
            {"detected_language": "", "language_code": "", "confidence": 0.0},
            status_code=r.status_code,
        )

    try:
        result = r.json()
        lang_code = result.get("language", "")
        lang_name = WHISPER_LANGUAGES.get(lang_code, lang_code)

        return JSONResponse({
            "detected_language": lang_name,
            "language_code": lang_code,
            "confidence": 1.0,  # whisper.cpp verbose_json doesn't expose per-language confidence
        })
    except Exception as e:
        logger.error(f"Failed to parse language detection response: {e}")
        return JSONResponse(
            {"detected_language": "", "language_code": "", "confidence": 0.0},
            status_code=500,
        )
