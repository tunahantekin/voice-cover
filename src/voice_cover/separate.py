"""Demucs (htdemucs, two-stems vocals) ile şarkıyı vokal ve enstrümantal olarak ikiye ayırır."""

from __future__ import annotations

from pathlib import Path


def separate_vocals(song_path: Path, out_dir: Path, model: str = "htdemucs") -> tuple[Path, Path]:
    """(vokal, enstrümantal) dosya yollarını döndürür."""
    raise NotImplementedError
