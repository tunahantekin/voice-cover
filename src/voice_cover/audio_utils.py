"""Ses okuma, referans kaydını doğrulayıp hazırlama, medyan F0 tahmini ve yarım ton kaydırma önerisi."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

REF_SR = 44100
TRIM_TOP_DB = 30
MAX_REF_SECONDS = 30.0
MIN_REF_SECONDS = 8.0
MIN_RMS_DB = -45.0
CLIP_LEVEL = 0.99
MAX_CLIP_RATIO = 0.01
TARGET_PEAK = 0.95

F0_SR = 16000
F0_FMIN = 65.0
F0_FMAX = 1000.0
MIN_VOICED_FRAMES = 20


class AudioError(ValueError):
    """Kullanıcıya olduğu gibi gösterilebilecek Türkçe mesajlı ses hatası."""


def load_audio(path: Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Sesi float32 olarak okur. mono=True ise (n,), değilse (n, kanal) döndürür. sr verilirse yeniden örnekler."""
    path = Path(path)
    data: np.ndarray | None = None
    try:
        import soundfile as sf

        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n, kanal)
    except Exception:
        try:
            import librosa

            y, file_sr = librosa.load(str(path), sr=None, mono=False)
            data = np.atleast_2d(y).T  # (kanal, n) -> (n, kanal)
        except Exception as e:
            raise AudioError(f"Ses dosyası okunamadı: {path.name}") from e

    if data is None or data.size == 0:
        raise AudioError(f"Ses dosyası okunamadı: {path.name}")

    if sr is not None and sr != file_sr:
        import librosa

        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr, axis=0)
        file_sr = sr

    if mono:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(file_sr)


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * math.log10(rms) if rms > 0 else float("-inf")


def prepare_reference(src: Path, dst: Path) -> dict:
    """Referans kaydını doğrular ve dst'ye 44.1 kHz mono, tepe 0.95 olarak yazar.

    Sıra: mono + 44.1 kHz okuma, baş/son sessizlik kırpma, 30 sn sınırı, kontroller, tepe normalizasyonu.
    """
    import librosa
    import soundfile as sf

    y, sr = load_audio(src, sr=REF_SR, mono=True)
    trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    trimmed = trimmed[: int(MAX_REF_SECONDS * sr)]

    duration = len(trimmed) / sr
    rms_db = _rms_db(trimmed)

    if rms_db < MIN_RMS_DB:
        raise AudioError(
            f"Kayıt çok sessiz (ortalama seviye {rms_db:.1f} dB). Mikrofona daha yakın ve daha yüksek sesle konuşarak tekrar kaydet."
        )
    if duration < MIN_REF_SECONDS:
        raise AudioError(
            f"Kayıt çok kısa ({duration:.1f} sn). Sessizlikler çıkarıldıktan sonra en az {MIN_REF_SECONDS:.0f} saniye konuşma gerekiyor."
        )
    clip_ratio = float(np.mean(np.abs(trimmed) >= CLIP_LEVEL))
    if clip_ratio > MAX_CLIP_RATIO:
        raise AudioError(
            f"Kayıtta cızırtı (clipping) var (örneklerin %{clip_ratio * 100:.1f}'i tepe seviyede). Mikrofondan biraz uzaklaşıp tekrar kaydet."
        )

    peak = float(np.max(np.abs(trimmed)))
    out = trimmed * (TARGET_PEAK / peak)

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), out.astype(np.float32), sr, subtype="PCM_16")
    return {"duration": duration, "rms_db": rms_db, "path": dst}


def estimate_median_f0(path: Path, max_seconds: float = 30.0) -> float | None:
    """Sesin ortasından en fazla max_seconds alıp pyin ile medyan F0'ı (Hz) bulur. Yeterli sesli çerçeve yoksa None."""
    import librosa

    y, sr = load_audio(path, sr=F0_SR, mono=True)
    max_len = int(max_seconds * sr)
    if len(y) > max_len:
        start = (len(y) - max_len) // 2
        y = y[start : start + max_len]

    f0, voiced, _ = librosa.pyin(y, fmin=F0_FMIN, fmax=F0_FMAX, sr=sr)
    voiced_f0 = f0[voiced & ~np.isnan(f0)]
    if voiced_f0.size < MIN_VOICED_FRAMES:
        return None
    return float(np.median(voiced_f0))


def suggest_semitone_shift(ref_f0: float | None, src_f0: float | None, limit: int = 12) -> int:
    """Kaynak perdesini referansa yaklaştırmak için yarım ton önerir: round(12*log2(ref/src)), [-limit, limit] arası."""
    if ref_f0 is None or src_f0 is None or ref_f0 <= 0 or src_f0 <= 0:
        return 0
    shift = round(12 * math.log2(ref_f0 / src_f0))
    return int(max(-limit, min(limit, shift)))
