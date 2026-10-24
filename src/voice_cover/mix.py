"""Dönüştürülmüş vokali, aynı yarım ton kadar kaydırılmış enstrümantalle birleştirip son miksi üretir."""

from __future__ import annotations

from pathlib import Path


def mix_tracks(
    vocals: Path,
    instrumental: Path,
    out_path: Path,
    semitones: int = 0,
    vocal_gain_db: float = 0.0,
) -> Path:
    raise NotImplementedError
