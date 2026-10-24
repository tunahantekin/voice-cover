"""Deneme betiği: ref.wav'ı hazırlar, source.wav'ı doğrudan vokal olarak Seed-VC'ye verir, süre ve tepe VRAM'i ölçer.

Kullanım:
    python scripts/try_convert.py --out outputs/try1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from voice_cover.audio_utils import AudioError, prepare_reference  # noqa: E402
from voice_cover.convert import ConversionError, convert_vocals  # noqa: E402
from voice_cover.seedvc_env import SEEDVC_DIR  # noqa: E402

CKPT_DIR = SEEDVC_DIR / "checkpoints"


def dir_size_mb(path: Path) -> float:
    # HF önbelleği snapshots/ altında blobs/'a symlink koyar. realpath ile her gerçek dosyayı bir kez say.
    seen: set[str] = set()
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            real = os.path.realpath(os.path.join(dirpath, f))
            if real in seen:
                continue
            seen.add(real)
            try:
                total += os.path.getsize(real)
            except OSError:
                pass
    return total / 1024 / 1024


def vram_used_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class Monitor(threading.Thread):
    """Her 2 sn'de VRAM ve checkpoints boyutunu okur, tepe VRAM'i ve indirmenin bittiği anı kaydeder."""

    def __init__(self, t0: float, interval: float = 2.0):
        super().__init__(daemon=True)
        self.t0, self.interval = t0, interval
        self.stop_event = threading.Event()
        self.peak_vram = 0
        self.ckpt_start = dir_size_mb(CKPT_DIR)
        self.ckpt_last = self.ckpt_start
        self.last_growth_t = 0.0

    def run(self) -> None:
        while not self.stop_event.is_set():
            t = time.perf_counter() - self.t0
            vram = vram_used_mb()
            ckpt = dir_size_mb(CKPT_DIR)
            if vram is not None:
                self.peak_vram = max(self.peak_vram, vram)
            if ckpt > self.ckpt_last + 0.5:
                self.last_growth_t = t
            self.ckpt_last = ckpt
            print(f"[mon t={t:6.1f}s] vram={vram} MB  checkpoints={ckpt:8.1f} MB", flush=True)
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        self.join()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(ROOT / "samples" / "ref.wav"))
    ap.add_argument("--source", default=str(ROOT / "samples" / "source.wav"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--semitones", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        info = prepare_reference(Path(args.ref), out / "ref_prepared.wav")
    except AudioError as e:
        print(f"REFERANS HATASI: {e}", flush=True)
        return 1
    print(f"referans hazır: {info['path']} süre={info['duration']:.2f} sn rms={info['rms_db']:.1f} dB", flush=True)

    t0 = time.perf_counter()
    mon = Monitor(t0)
    print(f"başlangıç: vram={vram_used_mb()} MB checkpoints={mon.ckpt_start:.1f} MB", flush=True)
    mon.start()
    try:
        result, summary = convert_vocals(
            Path(args.source), info["path"], out,
            semitones=args.semitones, diffusion_steps=args.steps,
            log_path=out / "seedvc.log", return_summary=True,
        )
    except ConversionError as e:
        mon.stop()
        elapsed = time.perf_counter() - t0
        print(f"\nDÖNÜŞTÜRME HATASI ({elapsed:.1f} sn, tepe VRAM {mon.peak_vram} MB, son indirme artışı t={mon.last_growth_t:.1f}s):\n{e}", flush=True)
        print(f"Seed-VC log: {out / 'seedvc.log'}", flush=True)
        return 1
    mon.stop()
    elapsed = time.perf_counter() - t0

    data, sr = sf.read(str(result), dtype="float32", always_2d=True)
    downloaded = mon.ckpt_last - mon.ckpt_start
    print("\n===== SONUÇ =====", flush=True)
    print(f"toplam süre       : {elapsed:.1f} sn")
    print(f"indirilen         : {downloaded:.1f} MB (son artış t={mon.last_growth_t:.1f} sn)")
    print(f"tepe VRAM         : {mon.peak_vram} MB")
    print(f"Seed-VC RTF       : {summary['rtf']}  (Seed-VC süreci {summary['elapsed_s']:.1f} sn)")
    print(f"Seed-VC log       : {summary['log_path']}")
    print(f"çıktı             : {result}")
    print(f"çıktı süre/sr/kanal: {data.shape[0] / sr:.2f} sn / {sr} Hz / {data.shape[1]}")
    print(f"NaN / Inf         : {bool(np.isnan(data).any())} / {bool(np.isinf(data).any())}  tepe={np.abs(data).max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
