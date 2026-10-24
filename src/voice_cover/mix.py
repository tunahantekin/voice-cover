"""Dönüştürülmüş vokali enstrümantalle birleştirip son miksi üretir.

Perde kuralı: vokal ``key_shift + 12*octave_shift`` yarım ton kaydırılmış gelir (Seed-VC içinde).
Enstrümantal yalnızca ``key_shift`` kadar kaydırılır. Oktav kaydırma notaları değiştirmediği için
enstrümantalin oktav kaydırılmasına gerek yoktur.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MIX_SR = 44100
TARGET_LUFS = -14.0
PEAK_CEILING = 0.98
VOCAL_OVER_INST_DB = 1.0  # vokal, enstrümantalin RMS'inden bu kadar önde
MAX_VOCAL_GAIN_DB = 24.0  # neredeyse sessiz vokalde gürültünün aşırı yükseltilmesini önler


class MixError(ValueError):
    """Miks adımında kullanıcıya gösterilebilecek hata."""


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def shift_pitch(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """(n,) ya da (n, kanal) sesi tempo değiştirmeden yarım ton kaydırır."""
    if semitones == 0:
        return audio
    import librosa

    y = audio.T if audio.ndim == 2 else audio  # librosa (kanal, n) bekler
    shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(semitones))
    return (shifted.T if audio.ndim == 2 else shifted).astype(np.float32)


def mix_tracks(
    vocals: Path,
    instrumental: Path,
    out_path: Path,
    key_shift: int = 0,
    vocal_gain_db: float = 0.0,
    target_lufs: float = TARGET_LUFS,
) -> Path:
    """Vokal (mono) ile enstrümantali (stereo/mono) 44.1 kHz stereo olarak birleştirir.

    Enstrümantal ``key_shift`` kadar kaydırılır, vokal seviyesi enstrümantale göre ayarlanır,
    toplam ses yüksekliği ``target_lufs`` değerine çekilir ve tepe ``PEAK_CEILING`` altında tutulur.
    """
    from voice_cover.audio_utils import load_audio
    import soundfile as sf

    voc, _ = load_audio(vocals, sr=MIX_SR, mono=True)
    inst, _ = load_audio(instrumental, sr=MIX_SR, mono=False)
    if inst.shape[1] == 1:
        inst = np.repeat(inst, 2, axis=1)
    elif inst.shape[1] > 2:
        inst = inst[:, :2]

    inst = shift_pitch(inst, MIX_SR, key_shift)

    n = min(len(voc), len(inst))
    if n < MIX_SR // 2:
        raise MixError("Miks için ses çok kısa (yarım saniyeden az).")
    voc, inst = voc[:n], inst[:n]

    inst_rms, voc_rms = _rms(inst), _rms(voc)
    if voc_rms > 0 and inst_rms > 0:
        gain = (inst_rms / voc_rms) * 10 ** ((VOCAL_OVER_INST_DB + vocal_gain_db) / 20)
        voc = voc * min(gain, 10 ** (MAX_VOCAL_GAIN_DB / 20))
    mix = inst + voc[:, None]

    try:
        import pyloudnorm as pyln

        loudness = pyln.Meter(MIX_SR).integrated_loudness(mix)
        if np.isfinite(loudness):
            mix = pyln.normalize.loudness(mix, loudness, target_lufs)
    except Exception:
        pass  # ölçülemezse yalnızca tepe sınırı uygulanır

    peak = float(np.max(np.abs(mix)))
    if peak > PEAK_CEILING:
        mix = mix * (PEAK_CEILING / peak)
    if not np.all(np.isfinite(mix)):
        raise MixError("Miks sırasında geçersiz değer (NaN/Inf) oluştu.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mix.astype(np.float32), MIX_SR, subtype="PCM_16")
    return out_path
