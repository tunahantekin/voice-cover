"""third_party/seed-vc altındaki Seed-VC'nin F0 koşullu şarkı modelini, kendi venv'i ile subprocess üzerinden çağırıp vokali referans sese dönüştürür."""

from __future__ import annotations

from pathlib import Path

from voice_cover.seedvc_env import SEEDVC_DIR, SEEDVC_PYTHON


def convert_vocals(
    source_vocals: Path,
    reference: Path,
    out_dir: Path,
    semitones: int = 0,
    diffusion_steps: int = 30,
    seedvc_dir: Path = SEEDVC_DIR,
    seedvc_python: Path = SEEDVC_PYTHON,
) -> Path:
    raise NotImplementedError
