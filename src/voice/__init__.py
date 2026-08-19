"""Public exports for the Whisper voice transcription subpackage."""
from .transcriber import TranscriptionError, WhisperTranscriber

__all__ = ["TranscriptionError", "WhisperTranscriber"]
