"""third_party/seed-vc altındaki Seed-VC'nin F0 koşullu şarkı modelini subprocess ile çağırıp vokali referans sese dönüştürür."""

from __future__ import annotations

from pathlib import Path

SEEDVC_DIR = Path(__file__).resolve().parents[2] / "third_party" / "seed-vc"


def convert_vocals(
    source_vocals: Path,
    reference: Path,
    out_dir: Path,
    semitones: int = 0,
    diffusion_steps: int = 30,
    seedvc_dir: Path = SEEDVC_DIR,
) -> Path:
    raise NotImplementedError
