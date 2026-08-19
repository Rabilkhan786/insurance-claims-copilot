"""Whisper-based speech-to-text service for uploaded audio."""
from pathlib import Path

from config import settings


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be converted to usable text."""


class WhisperTranscriber:
    """Load the configured Whisper model and transcribe audio files."""

    def __init__(self) -> None:
        import whisper

        self.model = whisper.load_model(settings.voice_model)

    def transcribe(self, audio_path: Path) -> str:
        """Return a stable English transcription or raise a clear error."""
        try:
            result = self.model.transcribe(
                str(audio_path),
                language=settings.voice_language,
                fp16=False,
                temperature=0.0,
                condition_on_previous_text=True,
            )
        except Exception as error:
            raise TranscriptionError("Audio transcription failed") from error

        text = str(result.get("text", "")).strip()
        if not text:
            raise TranscriptionError("No speech was detected in the audio")

        return text
