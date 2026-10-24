"""third_party/seed-vc altındaki Seed-VC'nin F0 koşullu şarkı modelini, kendi venv'i ile subprocess üzerinden çağırıp vokali referans sese dönüştürür."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from voice_cover.seedvc_env import SEEDVC_DIR, SEEDVC_PYTHON

STDERR_TAIL = 1500
DEFAULT_TIMEOUT = 1800
EXPECTED_SR = 44100
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")


class ConversionError(RuntimeError):
    """Seed-VC dönüştürmesi başarısız olduğunda fırlatılır."""


def _flag(value: bool) -> str:
    # inference.py bu argümanları str2bool ile okur
    return "True" if value else "False"


def build_command(
    vocals: Path,
    reference: Path,
    out_dir: Path,
    semitones: int,
    diffusion_steps: int,
    seedvc_dir: Path,
    seedvc_python: Path,
) -> list[str]:
    """inference.py komut satırını liste olarak kurar (shell=False, boşluklu yollar güvenli)."""
    return [
        str(seedvc_python),
        str(seedvc_dir / "inference.py"),
        "--source", str(vocals),
        "--target", str(reference),
        "--output", str(out_dir),
        "--diffusion-steps", str(int(diffusion_steps)),
        "--length-adjust", "1.0",
        "--inference-cfg-rate", "0.7",
        "--f0-condition", _flag(True),
        "--auto-f0-adjust", _flag(False),
        "--semi-tone-shift", str(int(semitones)),
        "--fp16", _flag(True),
    ]


def _check_output(path: Path) -> None:
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except RuntimeError as e:
        raise ConversionError(f"Seed-VC çıktısı okunamadı: {path}\n{e}") from e
    if info.channels != 1 or info.samplerate != EXPECTED_SR:
        raise ConversionError(
            f"Seed-VC çıktısı beklenen biçimde değil (mono, {EXPECTED_SR} Hz bekleniyordu): "
            f"{info.channels} kanal, {info.samplerate} Hz, dosya {path}"
        )


def convert_vocals(
    vocals: Path,
    reference: Path,
    workdir: Path,
    semitones: int = 0,
    diffusion_steps: int = 30,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    seedvc_dir: Path = SEEDVC_DIR,
    seedvc_python: Path = SEEDVC_PYTHON,
) -> Path:
    """Vokali referans sese dönüştürür ve oluşan mono 44.1 kHz wav dosyasının yolunu döndürür."""
    seedvc_dir = Path(seedvc_dir).resolve()
    seedvc_python = Path(seedvc_python)
    if not seedvc_python.is_file() or not (seedvc_dir / "inference.py").is_file():
        raise ConversionError(
            f"Seed-VC kurulu değil (python: {seedvc_python}, repo: {seedvc_dir}). "
            "Önce 'python scripts/setup_seedvc.py' çalıştır."
        )

    # cwd Seed-VC klasörüne geçtiği için bütün yollar mutlak olmalı
    vocals = Path(vocals).resolve()
    reference = Path(reference).resolve()
    for label, p in (("vokal", vocals), ("referans", reference)):
        if not p.is_file():
            raise ConversionError(f"{label} dosyası bulunamadı: {p}")

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="seedvc_", dir=workdir))  # her çağrıda yeni ve boş

    cmd = build_command(vocals, reference, out_dir, semitones, diffusion_steps, seedvc_dir, seedvc_python)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    try:
        result = subprocess.run(
            cmd,
            cwd=str(seedvc_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ConversionError(f"Seed-VC {timeout:.0f} saniye içinde bitmedi, işlem durduruldu.") from e

    if result.returncode != 0:
        stderr = result.stderr or ""
        tail = stderr[-STDERR_TAIL:]
        if any(m in stderr for m in OOM_MARKERS):
            raise ConversionError(
                "GPU belleği yetmedi, daha kısa şarkı ya da referans dene.\n" + tail
            )
        raise ConversionError(f"Seed-VC hata verdi (çıkış kodu {result.returncode}):\n{tail}")

    # Çıktı adını tahmin etme: klasörde oluşan tek wav'ı al
    wavs = sorted(out_dir.glob("*.wav"))
    if len(wavs) != 1:
        raise ConversionError(
            f"Seed-VC çıktı klasöründe 1 wav bekleniyordu, {len(wavs)} bulundu: {out_dir}\n"
            f"{(result.stderr or '')[-STDERR_TAIL:]}"
        )
    _check_output(wavs[0])
    return wavs[0]
