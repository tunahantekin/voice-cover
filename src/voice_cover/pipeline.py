"""Doğrulama, ayırma, dönüştürme ve miks adımlarını sırayla çalıştırır, iş bitince yüklenen sesleri siler."""

from __future__ import annotations

from pathlib import Path


def run_pipeline(
    reference: Path,
    song_file: Path | None = None,
    song_url: str | None = None,
    semitones: int = 0,
    work_dir: Path | None = None,
) -> Path:
    raise NotImplementedError


def cleanup(paths: list[Path]) -> None:
    raise NotImplementedError
