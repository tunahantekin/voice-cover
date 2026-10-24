"""Demucs (htdemucs, two-stems vocals) ile şarkıyı vokal ve enstrümantal olarak ikiye ayırır."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODEL = "htdemucs"
STDERR_TAIL = 800


class SeparationError(RuntimeError):
    """Demucs ayırma işlemi başarısız olduğunda fırlatılır."""


def separate(audio_path: Path, workdir: Path) -> tuple[Path, Path]:
    """Şarkıyı ayırır ve (vocals.wav, no_vocals.wav) yollarını döndürür."""
    audio_path = Path(audio_path)
    out_root = Path(workdir) / "separated"

    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", MODEL,
        "--two-stems", "vocals",
        "-o", str(out_root),
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise SeparationError(
            f"Demucs hata verdi (çıkış kodu {result.returncode}):\n{result.stderr[-STDERR_TAIL:]}"
        )

    track_dir = out_root / MODEL / audio_path.stem
    vocals = track_dir / "vocals.wav"
    no_vocals = track_dir / "no_vocals.wav"
    missing = [p for p in (vocals, no_vocals) if not p.is_file()]
    if missing:
        raise SeparationError(
            f"Demucs çıktı dosyaları bulunamadı: {', '.join(map(str, missing))}\n{result.stderr[-STDERR_TAIL:]}"
        )
    return vocals, no_vocals
