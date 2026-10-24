"""mix.py testleri: sentetik sinyaller, model yok."""

from __future__ import annotations

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")

from voice_cover.mix import MixError, mix_tracks  # noqa: E402

SR = 44100


def sine(freq, seconds, sr=SR, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(x, sr=SR):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(spec)]


def test_mix_output_format_and_level(tmp_path):
    voc = tmp_path / "v.wav"
    inst = tmp_path / "i.wav"
    sf.write(str(voc), sine(300, 3.0), SR)
    sf.write(str(inst), np.stack([sine(110, 3.2), sine(110, 3.2)], axis=1), SR)
    out = mix_tracks(voc, inst, tmp_path / "out" / "mix.wav")

    data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    assert sr == SR and data.shape[1] == 2
    assert abs(data.shape[0] - 3.0 * SR) <= 1          # kısa olanın uzunluğu
    assert np.all(np.isfinite(data))
    assert np.abs(data).max() <= 0.98 + 1e-3


def test_mono_instrumental_becomes_stereo(tmp_path):
    voc, inst = tmp_path / "v.wav", tmp_path / "i.wav"
    sf.write(str(voc), sine(300, 2.0), SR)
    sf.write(str(inst), sine(110, 2.0), SR)
    data, _ = sf.read(str(mix_tracks(voc, inst, tmp_path / "m.wav")), always_2d=True)
    assert data.shape[1] == 2


def test_key_shift_moves_instrumental_pitch(tmp_path):
    voc, inst = tmp_path / "v.wav", tmp_path / "i.wav"
    sf.write(str(voc), np.zeros(2 * SR, dtype=np.float32), SR)  # sessiz vokal: yalnızca enstrümantal ölçülür
    sf.write(str(inst), sine(440, 2.0), SR)
    data, _ = sf.read(str(mix_tracks(voc, inst, tmp_path / "m.wav", key_shift=12)), dtype="float32", always_2d=True)
    f = dominant_freq(data[SR // 2 : SR + SR // 2, 0])
    assert abs(f - 880) < 15


def test_zero_key_shift_keeps_instrumental_pitch(tmp_path):
    voc, inst = tmp_path / "v.wav", tmp_path / "i.wav"
    sf.write(str(voc), np.zeros(2 * SR, dtype=np.float32), SR)
    sf.write(str(inst), sine(440, 2.0), SR)
    data, _ = sf.read(str(mix_tracks(voc, inst, tmp_path / "m.wav", key_shift=0)), dtype="float32", always_2d=True)
    assert abs(dominant_freq(data[SR // 2 : SR + SR // 2, 0]) - 440) < 15


def test_near_silent_vocal_gain_is_capped(tmp_path):
    voc, inst = tmp_path / "v.wav", tmp_path / "i.wav"
    sf.write(str(voc), sine(300, 2.0, amp=1e-5), SR)       # neredeyse sessiz (gürültü seviyesi) vokal
    sf.write(str(inst), sine(440, 2.0), SR)
    data, _ = sf.read(str(mix_tracks(voc, inst, tmp_path / "m.wav")), dtype="float32", always_2d=True)
    assert abs(dominant_freq(data[SR // 2 : SR + SR // 2, 0]) - 440) < 15  # vokal enstrümanı bastırmadı


def test_too_short_raises(tmp_path):
    voc, inst = tmp_path / "v.wav", tmp_path / "i.wav"
    sf.write(str(voc), sine(300, 0.2), SR)
    sf.write(str(inst), sine(110, 0.2), SR)
    with pytest.raises(MixError, match="çok kısa"):
        mix_tracks(voc, inst, tmp_path / "m.wav")
