"""
Portable audio duration detection for preprocessing and dataset building.

Uses torchcodec (fast, supports all ffmpeg formats) with a soundfile
fallback.  Returns integer seconds to avoid float-precision issues
with ffprobe-style outputs.

Vendored from upstream ``audio_io.get_audio_duration`` so Side-Step
stays fully standalone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: str) -> int:
    """Return the duration of an audio file in whole seconds.

    Resolution chain:
        1. ``torchcodec.decoders.AudioDecoder`` (fast, all ffmpeg formats).
        2. ``soundfile.info`` (fast for wav/flac/ogg, all platforms).
        3. Returns ``0`` if both fail.

    The result is truncated to ``int`` so callers never deal with
    sub-second float precision.
    """
    # Primary: torchcodec (ships with torchaudio >=2.9)
    try:
        from torchcodec.decoders import AudioDecoder
        decoder = AudioDecoder(audio_path)
        return int(decoder.metadata.duration_seconds)
    except ImportError:
        logger.debug("torchcodec not available, trying soundfile")
    except Exception as exc:
        logger.debug("torchcodec failed for %s: %s", audio_path, exc)

    # Fallback: soundfile
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return int(info.duration)
    except Exception as exc:
        logger.warning("Failed to get duration for %s: %s", audio_path, exc)
        return 0


def detect_max_duration(audio_files: List[Path]) -> int:
    """Scan *audio_files* and return the longest duration in seconds.

    Returns ``0`` when the list is empty or every probe fails.
    """
    if not audio_files:
        return 0

    longest = 0
    for af in audio_files:
        dur = get_audio_duration(str(af))
        logger.debug("[Side-Step] %s: %ds", af.name, dur)
        if dur > longest:
            longest = dur

    logger.info(
        "[Side-Step] Detected longest clip: %ds (across %d files)",
        longest,
        len(audio_files),
    )
    return longest
