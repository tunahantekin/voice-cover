"""Ses okuma, yazma, referans kaydı doğrulama, ses yüksekliği normalizasyonu ve perde kaydırma yardımcıları."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    raise NotImplementedError


def save_audio(path: Path, audio: np.ndarray, sr: int) -> Path:
    raise NotImplementedError


def validate_reference(path: Path) -> None:
    raise NotImplementedError


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    raise NotImplementedError


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    raise NotImplementedError
