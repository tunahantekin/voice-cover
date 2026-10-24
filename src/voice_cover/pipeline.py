"""Doğrulama, ayırma, dönüştürme ve miks adımlarını sırayla çalıştırır, iş bitince yüklenen sesleri siler."""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from voice_cover.audio_utils import estimate_median_f0, load_audio, prepare_reference, suggest_semitone_shift
from voice_cover.convert import convert_vocals
from voice_cover.fetch import resolve_song
from voice_cover.mix import mix_tracks
from voice_cover.seedvc_env import PROJECT_ROOT
from voice_cover.separate import separate

DEFAULT_WORK_ROOT = PROJECT_ROOT / "outputs" / "jobs"
DEFAULT_MAX_SECONDS = 30.0
MAX_OCTAVE = 2

ProgressFn = Callable[[float, str], None]


@dataclass
class PipelineResult:
    output: Path
    info: dict = field(default_factory=dict)


def _noop(_: float, __: str) -> None:
    pass


def pick_window(vocals: Path, max_seconds: float) -> tuple[float, float]:
    """Vokal enerjisinin en yüksek olduğu ``max_seconds`` uzunluğundaki pencereyi bulur. (başlangıç, süre) sn."""
    y, sr = load_audio(vocals, mono=True)
    total = len(y) / sr
    if total <= max_seconds:
        return 0.0, total
    win, hop = int(max_seconds * sr), sr
    starts = range(0, len(y) - win + 1, hop)
    energy = [float(np.mean(np.square(y[s : s + win], dtype=np.float64))) for s in starts]
    best = int(np.argmax(energy))
    return best * hop / sr, max_seconds


def cut(src: Path, dst: Path, start_s: float, duration_s: float) -> Path:
    import soundfile as sf

    info = sf.info(str(src))
    start = int(start_s * info.samplerate)
    frames = int(duration_s * info.samplerate)
    data, sr = sf.read(str(src), start=start, frames=frames, dtype="float32", always_2d=True)
    sf.write(str(dst), data, sr, subtype="PCM_16")
    return dst


def auto_octave(ref_f0: float | None, src_f0: float | None) -> int:
    """Referans ile kaynak arasındaki perde farkına en yakın oktavı seçer (±MAX_OCTAVE)."""
    shift = suggest_semitone_shift(ref_f0, src_f0, limit=12 * MAX_OCTAVE)
    return int(max(-MAX_OCTAVE, min(MAX_OCTAVE, round(shift / 12))))


def cleanup(paths: list[Path]) -> None:
    """Dosya ve klasörleri sessizce siler (yoksa geçer)."""
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        except OSError:
            pass


def cleanup_old_jobs(work_root: Path = DEFAULT_WORK_ROOT, max_age_s: float = 3600) -> int:
    """``max_age_s`` saniyeden eski iş klasörlerini siler, silinen sayısını döndürür."""
    work_root = Path(work_root)
    if not work_root.is_dir():
        return 0
    now, removed = time.time(), 0
    for d in work_root.glob("job_*"):
        try:
            if d.is_dir() and now - d.stat().st_mtime > max_age_s:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            pass
    return removed


def run_pipeline(
    reference: Path,
    song_file: Path | None = None,
    song_url: str | None = None,
    *,
    key_shift: int = 0,
    octave_shift: int | None = None,
    diffusion_steps: int = 30,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    work_root: Path | None = None,
    delete_inputs: bool = False,
    keep_intermediates: bool = False,
    progress: ProgressFn | None = None,
) -> PipelineResult:
    """Uçtan uca: referans hazırla -> şarkıyı al -> Demucs -> pencere seç -> Seed-VC -> miks.

    octave_shift=None ise oktav otomatik seçilir. Vokal toplam ``key_shift + 12*octave_shift`` yarım ton,
    enstrümantal yalnızca ``key_shift`` kaydırılır. Başarıda iş klasöründe yalnızca cover.wav kalır.
    delete_inputs=True ise verilen referans ve şarkı dosyaları da (ör. arayüzün geçici yüklemeleri) silinir.
    """
    report = progress or _noop
    work_root = Path(work_root or DEFAULT_WORK_ROOT)
    work_root.mkdir(parents=True, exist_ok=True)
    wd = Path(tempfile.mkdtemp(prefix="job_", dir=work_root))
    timings: dict[str, float] = {}
    output = wd / "cover.wav"
    success = False

    def step(name: str, frac: float, msg: str):
        report(frac, msg)
        timings[name] = time.perf_counter()

    def done(name: str):
        timings[name] = round(time.perf_counter() - timings[name], 1)

    try:
        step("reference", 0.02, "Ses kaydın kontrol ediliyor")
        ref = prepare_reference(Path(reference), wd / "ref.wav")
        done("reference")

        step("fetch", 0.06, "Şarkı alınıyor")
        song = resolve_song(wd, file_path=song_file, url=song_url)
        done("fetch")

        step("separate", 0.12, "Vokal ve müzik ayrılıyor (Demucs)")
        vocals_full, inst_full = separate(song, wd)
        done("separate")

        step("window", 0.25, "Dönüştürülecek bölüm seçiliyor")
        start, length = pick_window(vocals_full, max_seconds)
        vocals = cut(vocals_full, wd / "vocals_cut.wav", start, length)
        inst = cut(inst_full, wd / "inst_cut.wav", start, length)
        done("window")

        step("pitch", 0.28, "Perde analiz ediliyor")
        ref_f0 = estimate_median_f0(ref["path"])
        src_f0 = estimate_median_f0(vocals)
        octave = auto_octave(ref_f0, src_f0) if octave_shift is None else int(octave_shift)
        total = int(key_shift) + 12 * octave
        done("pitch")

        step("convert", 0.32, "Sesin dönüştürülüyor (Seed-VC, uzun sürer)")
        converted, summary = convert_vocals(
            vocals, ref["path"], wd,
            semitones=total, diffusion_steps=diffusion_steps,
            log_path=wd / "seedvc.log", return_summary=True,
        )
        done("convert")

        step("mix", 0.95, "Müzikle birleştiriliyor")
        mix_tracks(converted, inst, output, key_shift=int(key_shift))
        done("mix")
        report(1.0, "Bitti")

        success = True
        info = {
            "ref_duration": round(ref["duration"], 2),
            "ref_rms_db": round(ref["rms_db"], 1),
            "ref_f0": ref_f0,
            "src_f0": src_f0,
            "key_shift": int(key_shift),
            "octave_shift": octave,
            "octave_auto": octave_shift is None,
            "total_semitones": total,
            "window_start": round(start, 1),
            "window_seconds": round(length, 1),
            "diffusion_steps": diffusion_steps,
            "rtf": summary.get("rtf"),
            "timings": timings,
        }
        return PipelineResult(output=output, info=info)
    finally:
        if delete_inputs:
            cleanup([Path(reference)] + ([Path(song_file)] if song_file else []))
        if not keep_intermediates:
            if success:
                cleanup([p for p in wd.iterdir() if p != output])
            else:
                cleanup([wd])
