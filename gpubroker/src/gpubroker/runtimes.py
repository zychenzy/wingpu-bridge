"""Recommended broker runtime ids.

``acquire`` treats a second call with the same runtime id as a reassert: it
refreshes the pidfile and keeps the lease. Two processes that share an id
therefore both stay resident. Whisper and Qwen ASR must use different ids.
"""

LLAMA = "llama"
LOCATE = "locate"
OCR = "ocr"
MINERU = "mineru"
ASR_WHISPER = "live-asr"
ASR_QWEN = "live-asr-qwen"

__all__ = [
    "ASR_QWEN",
    "ASR_WHISPER",
    "LLAMA",
    "LOCATE",
    "MINERU",
    "OCR",
]
