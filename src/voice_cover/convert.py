"""third_party/seed-vc altındaki Seed-VC'nin F0 koşullu şarkı modelini, kendi venv'i ile subprocess üzerinden çağırıp vokali referans sese dönüştürür."""

from __future__ import annotations

import collections
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from voice_cover.seedvc_env import SEEDVC_DIR, SEEDVC_PYTHON

STDERR_TAIL = 1500
DEFAULT_TIMEOUT = 1800
EXPECTED_SR = 44100
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")
# inference.py:402 -> print(f"RTF: {...}")  (stdout, tek satır, düz float)
RTF_RE = re.compile(r"^RTF:\s*([-+0-9.eE]+)\s*$")


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


def _run_streaming(
    cmd: list[str], cwd: str, env: dict, timeout: float, log_path: Path | None
) -> tuple[int, list[str], str, float]:
    """Seed-VC'yi Popen ile çalıştırır, stdout/stderr'i satır satır okur (log_path verilirse dosyaya yazar).

    Dönüş: (returncode, stdout satırları, stderr metni (son kısım), geçen süre).
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    t0 = time.perf_counter()
    stdout_lines: list[str] = []
    stderr_lines: collections.deque[str] = collections.deque(maxlen=2000)  # bellek sınırı, sadece son kısım gerekir
    lock = threading.Lock()
    log_fh = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("a", encoding="utf-8")
        log_fh.write(f"$ {' '.join(cmd)}\n")
        log_fh.flush()

    def pump(stream, name: str, sink) -> None:
        for line in iter(stream.readline, ""):
            sink.append(line)
            if log_fh is not None:
                with lock:
                    log_fh.write(f"[{time.perf_counter() - t0:7.1f}s {name}] {line.rstrip()}\n")
                    log_fh.flush()
        stream.close()

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, "stdout", stdout_lines), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, "stderr", stderr_lines), daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        proc.kill()
        proc.wait()
        for t in threads:
            t.join(timeout=5)
        if log_fh is not None:
            log_fh.write(f"[{time.perf_counter() - t0:7.1f}s] ZAMAN AŞIMI ({timeout:.0f} sn), süreç durduruldu\n")
            log_fh.close()
        raise ConversionError(f"Seed-VC {timeout:.0f} saniye içinde bitmedi, işlem durduruldu.") from e

    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    if log_fh is not None:
        log_fh.write(f"[{elapsed:7.1f}s] çıkış kodu {returncode}\n")
        log_fh.close()
    return returncode, stdout_lines, "".join(stderr_lines), elapsed


def _parse_rtf(stdout_lines: list[str]) -> float | None:
    for line in reversed(stdout_lines):
        m = RTF_RE.match(line.strip())
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


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
    log_path: Path | None = None,
    return_summary: bool = False,
) -> Path | tuple[Path, dict]:
    """Vokali referans sese dönüştürür ve oluşan mono 44.1 kHz wav dosyasının yolunu döndürür.

    log_path verilirse Seed-VC'nin stdout/stderr'i çalışırken satır satır (geçen süreyle) bu dosyaya yazılır.
    return_summary=True ise (yol, özet) döner. Özet: {"rtf", "elapsed_s", "log_path"}.
    """
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
    if log_path is not None:
        log_path = Path(log_path).resolve()

    cmd = build_command(vocals, reference, out_dir, semitones, diffusion_steps, seedvc_dir, seedvc_python)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    returncode, stdout_lines, stderr, elapsed = _run_streaming(cmd, str(seedvc_dir), env, timeout, log_path)

    if returncode != 0:
        tail = stderr[-STDERR_TAIL:]
        if any(m in stderr for m in OOM_MARKERS):
            raise ConversionError(
                "GPU belleği yetmedi, daha kısa şarkı ya da referans dene.\n" + tail
            )
        raise ConversionError(f"Seed-VC hata verdi (çıkış kodu {returncode}):\n{tail}")

    # Çıktı adını tahmin etme: klasörde oluşan tek wav'ı al
    wavs = sorted(out_dir.glob("*.wav"))
    if len(wavs) != 1:
        raise ConversionError(
            f"Seed-VC çıktı klasöründe 1 wav bekleniyordu, {len(wavs)} bulundu: {out_dir}\n"
            f"{stderr[-STDERR_TAIL:]}"
        )
    _check_output(wavs[0])

    if return_summary:
        summary = {"rtf": _parse_rtf(stdout_lines), "elapsed_s": elapsed, "log_path": log_path}
        return wavs[0], summary
    return wavs[0]
