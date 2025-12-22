"""
Whisper Voice Integration

Provides voice-to-text transcription using OpenAI's Whisper API.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


class WhisperClient:
    """
    Client for OpenAI Whisper transcription API.

    Supports:
    - Audio file transcription
    - Multiple audio formats (mp3, mp4, wav, m4a, webm)
    - Language detection or forced language
    """

    SUPPORTED_FORMATS = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, 'OPENAI_API_KEY', None) or os.getenv("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("OpenAI API key not configured for voice input")

            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)

        return self._client

    def is_available(self) -> bool:
        """Check if Whisper is available."""
        return bool(self.api_key)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: ISO language code (e.g., 'en', 'es'). None for auto-detect.
            prompt: Optional prompt to guide transcription style

        Returns:
            Transcribed text
        """
        if not self.is_available():
            raise RuntimeError("Whisper API not available - OPENAI_API_KEY not set")

        path = Path(audio_path)

        # Validate file
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}. Supported: {self.SUPPORTED_FORMATS}")

        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB. Max: 25MB")

        try:
            client = self._get_client()

            with open(audio_path, "rb") as audio_file:
                kwargs = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "text",
                }

                if language:
                    kwargs["language"] = language

                if prompt:
                    kwargs["prompt"] = prompt

                transcript = client.audio.transcriptions.create(**kwargs)

            logger.info(f"Transcribed {path.name}: {len(transcript)} chars")
            return transcript.strip()

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def transcribe_bytes(
        self,
        audio_data: bytes,
        file_extension: str = ".wav",
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> str:
        """
        Transcribe audio from bytes.

        Args:
            audio_data: Raw audio bytes
            file_extension: File extension (e.g., '.wav', '.mp3')
            language: ISO language code
            prompt: Optional prompt

        Returns:
            Transcribed text
        """
        if not file_extension.startswith('.'):
            file_extension = '.' + file_extension

        if file_extension.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {file_extension}")

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            return self.transcribe(tmp_path, language, prompt)
        finally:
            # Cleanup temp file
            Path(tmp_path).unlink(missing_ok=True)


def save_audio_to_temp(audio_elements: list) -> str:
    """
    Save Chainlit audio elements to a temporary file.

    Args:
        audio_elements: List of Chainlit audio elements

    Returns:
        Path to saved audio file
    """
    if not audio_elements:
        raise ValueError("No audio elements provided")

    # Find audio element
    audio_element = None
    for element in audio_elements:
        if hasattr(element, 'content') or hasattr(element, 'path'):
            audio_element = element
            break

    if not audio_element:
        raise ValueError("No valid audio element found")

    # Determine format and get content
    if hasattr(audio_element, 'path') and audio_element.path:
        return audio_element.path

    if hasattr(audio_element, 'content') and audio_element.content:
        # Save content to temp file
        mime = getattr(audio_element, 'mime', 'audio/wav')
        ext = '.wav'
        if 'mp3' in mime:
            ext = '.mp3'
        elif 'webm' in mime:
            ext = '.webm'
        elif 'm4a' in mime:
            ext = '.m4a'

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_element.content)
            return tmp.name

    raise ValueError("Audio element has no content or path")


# Convenience function
def transcribe_audio(audio_path: str) -> str:
    """Quick transcription helper."""
    client = WhisperClient()
    if not client.is_available():
        return "[Voice input requires OPENAI_API_KEY]"
    return client.transcribe(audio_path)
