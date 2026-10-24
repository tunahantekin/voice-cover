"""audio_utils testleri. Yalnızca sentetik sinyaller kullanılır, telifli ses yok."""

from __future__ import annotations

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")

from voice_cover.audio_utils import (  # noqa: E402
    AudioError,
    estimate_median_f0,
    load_audio,
    prepare_reference,
    suggest_semitone_shift,
)

SR = 44100


def speech_like(seconds: float, sr: int = SR, peak: float = 0.5) -> np.ndarray:
    """3 sinüs karışımı + hece benzeri genlik zarfı (4 Hz), sıfıra inmez."""
    t = np.arange(int(seconds * sr)) / sr
    tone = np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 360 * t) + 0.25 * np.sin(2 * np.pi * 540 * t)
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4 * t)
    x = tone * env
    return (peak * x / np.abs(x).max()).astype(np.float32)


def write(path, data, sr=SR):
    sf.write(str(path), data, sr, subtype="FLOAT")
    return path


# ---------- load_audio ----------

def test_load_audio_mono_and_stereo(tmp_path):
    stereo = np.stack([speech_like(1.0, 22050), 0.5 * speech_like(1.0, 22050)], axis=1)
    p = write(tmp_path / "stereo.wav", stereo, 22050)

    y, sr = load_audio(p)
    assert sr == 22050 and y.shape == (22050,) and y.dtype == np.float32

    y2, _ = load_audio(p, mono=False)
    assert y2.shape == (22050, 2) and y2.dtype == np.float32


def test_load_audio_resamples(tmp_path):
    p = write(tmp_path / "a.wav", speech_like(1.0, 22050), 22050)
    y, sr = load_audio(p, sr=16000)
    assert sr == 16000
    assert abs(len(y) - 16000) <= 2


def test_load_audio_unreadable_raises(tmp_path):
    p = tmp_path / "bozuk.wav"
    p.write_bytes(b"bu bir ses dosyasi degil")
    with pytest.raises(AudioError, match="Ses dosyası okunamadı: bozuk.wav"):
        load_audio(p)


# ---------- prepare_reference ----------

def test_prepare_reference_success_output_format(tmp_path):
    src = write(tmp_path / "ref.wav", np.stack([speech_like(12.0)] * 2, axis=1))  # stereo giriş
    dst = tmp_path / "out" / "ref_prepared.wav"
    info = prepare_reference(src, dst)

    assert info["path"] == dst and dst.is_file()
    assert 11.5 <= info["duration"] <= 12.0
    assert info["rms_db"] > -45
    meta = sf.info(str(dst))
    assert meta.samplerate == 44100 and meta.channels == 1
    out, _ = sf.read(str(dst), dtype="float32")
    assert abs(np.abs(out).max() - 0.95) < 0.01


def test_prepare_reference_trims_leading_silence(tmp_path):
    x = np.concatenate([np.zeros(2 * SR, dtype=np.float32), speech_like(10.0)])
    src = write(tmp_path / "ref.wav", x)
    info = prepare_reference(src, tmp_path / "out.wav")
    assert abs(info["duration"] - 10.0) < 0.1


def test_prepare_reference_caps_at_30_seconds(tmp_path):
    src = write(tmp_path / "ref.wav", speech_like(35.0))
    info = prepare_reference(src, tmp_path / "out.wav")
    assert abs(info["duration"] - 30.0) < 1e-6


def test_prepare_reference_too_short(tmp_path):
    src = write(tmp_path / "ref.wav", speech_like(5.0))
    with pytest.raises(AudioError, match="çok kısa"):
        prepare_reference(src, tmp_path / "out.wav")


def test_prepare_reference_too_quiet(tmp_path):
    rng = np.random.default_rng(0)
    src = write(tmp_path / "ref.wav", (0.001 * rng.standard_normal(12 * SR)).astype(np.float32))
    with pytest.raises(AudioError, match="çok sessiz"):
        prepare_reference(src, tmp_path / "out.wav")


def test_prepare_reference_clipping(tmp_path):
    x = np.clip(speech_like(12.0, peak=1.0) * 2.0, -1.0, 1.0)
    src = write(tmp_path / "ref.wav", x)
    with pytest.raises(AudioError, match="clipping"):
        prepare_reference(src, tmp_path / "out.wav")


def test_prepare_reference_does_not_write_on_error(tmp_path):
    src = write(tmp_path / "ref.wav", speech_like(5.0))
    dst = tmp_path / "out.wav"
    with pytest.raises(AudioError):
        prepare_reference(src, dst)
    assert not dst.exists()


# ---------- estimate_median_f0 ----------

def test_estimate_median_f0_sine_220(tmp_path):
    t = np.arange(2 * SR) / SR
    p = write(tmp_path / "sine.wav", (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    f0 = estimate_median_f0(p)
    assert f0 is not None and abs(f0 - 220) <= 5


def test_estimate_median_f0_silence_returns_none(tmp_path):
    p = write(tmp_path / "sessiz.wav", np.zeros(2 * SR, dtype=np.float32))
    assert estimate_median_f0(p) is None


# ---------- suggest_semitone_shift ----------

@pytest.mark.parametrize(
    "ref,src,limit,expected",
    [
        (440.0, 220.0, 12, 12),   # referans bir oktav yukarıda
        (220.0, 220.0, 12, 0),
        (None, 220.0, 12, 0),
        (220.0, None, 12, 0),
        (880.0, 110.0, 12, 12),   # +36 -> limit
        (110.0, 880.0, 12, -12),  # -36 -> limit
        (440.0, 220.0, 5, 5),
        (233.08, 220.0, 12, 1),   # ~1 yarım ton
    ],
)
def test_suggest_semitone_shift(ref, src, limit, expected):
    assert suggest_semitone_shift(ref, src, limit=limit) == expected
