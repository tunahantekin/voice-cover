"""separate() için uçtan uca test: sentetik sinyali Demucs ile gerçekten ayırır."""

from __future__ import annotations

import importlib.util
import time

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(importlib.util.find_spec("demucs") is None, reason="demucs kurulu değil"),
]

SR = 44100
DURATION = 10.0


def make_test_signal(path) -> np.ndarray:
    """Birkaç sinüs akoru + hafif gürültüden oluşan 10 sn'lik stereo sinyal (telifsiz)."""
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * DURATION)) / SR
    chords = [(261.63, 329.63, 392.00), (220.00, 261.63, 329.63), (174.61, 220.00, 261.63), (196.00, 246.94, 293.66)]
    seg = len(t) // len(chords)
    mono = np.zeros_like(t)
    for i, chord in enumerate(chords):
        sl = slice(i * seg, len(t) if i == len(chords) - 1 else (i + 1) * seg)
        mono[sl] = sum(np.sin(2 * np.pi * f * t[sl]) for f in chord) / len(chord)
    left = 0.5 * mono + 0.01 * rng.standard_normal(len(t))
    right = 0.45 * mono + 0.01 * rng.standard_normal(len(t))
    audio = np.stack([left, right], axis=1).astype(np.float32)
    sf.write(path, audio, SR)
    return audio


def test_separate_produces_valid_stems(tmp_path):
    from voice_cover.separate import separate

    src = tmp_path / "synthetic.wav"
    audio_in = make_test_signal(src)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    start = time.perf_counter()
    vocals, no_vocals = separate(src, tmp_path)
    elapsed = time.perf_counter() - start
    print(f"\n[separate] cihaz={device} (demucs varsayılanı) süre={elapsed:.1f} sn")

    in_dur = len(audio_in) / SR
    for path in (vocals, no_vocals):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        dur = data.shape[0] / sr
        print(
            f"[separate] {path.name}: sr={sr} kanal={data.shape[1]} süre={dur:.3f} sn "
            f"(giriş {in_dur:.3f}) NaN={bool(np.isnan(data).any())} max|x|={np.abs(data).max():.4f}"
        )
        assert sr == SR
        assert data.shape[1] == 2
        assert abs(dur - in_dur) < 0.05
        assert not np.isnan(data).any()
        assert data.size > 0

    # Sentetik sinyalde vokal yok, enerjinin büyük kısmı no_vocals'ta olmalı
    nv, _ = sf.read(no_vocals, dtype="float32")
    assert np.abs(nv).max() > 0.01
