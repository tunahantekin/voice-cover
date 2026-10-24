"""pipeline.py testleri: Demucs ve Seed-VC mock'lanır, geri kalan adımlar gerçek çalışır."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")

from voice_cover import pipeline  # noqa: E402
from voice_cover.pipeline import auto_octave, cleanup_old_jobs, pick_window, run_pipeline  # noqa: E402
from voice_cover.separate import SeparationError  # noqa: E402

SR = 44100


def tone(freq, seconds, amp=0.4):
    t = np.arange(int(seconds * SR)) / SR
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def inputs(tmp_path):
    ref = tmp_path / "ref.wav"
    song = tmp_path / "song.wav"
    sf.write(str(ref), tone(150, 10.0), SR)                         # konuşma perdesi ~150 Hz
    sf.write(str(song), np.stack([tone(300, 8.0)] * 2, axis=1), SR)  # şarkı ~300 Hz
    return ref, song


@pytest.fixture
def fakes(monkeypatch):
    calls = {}

    def fake_separate(song, wd):
        d = wd / "sep"
        d.mkdir()
        data, sr = sf.read(str(song), dtype="float32", always_2d=True)
        v, i = d / "vocals.wav", d / "no_vocals.wav"
        loud = data.copy()
        loud[: 2 * SR] *= 0.05                                   # ilk 2 sn sessiz: pencere sonradan seçilmeli
        sf.write(str(v), loud, sr)
        sf.write(str(i), 0.2 * data, sr)
        calls["separate"] = song
        return v, i

    def fake_convert(vocals, reference, workdir, semitones=0, diffusion_steps=30, *, log_path=None, return_summary=False, **_):
        calls["convert"] = {"semitones": semitones, "steps": diffusion_steps, "log_path": log_path}
        data, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
        out = workdir / "seedvc_x" / "vc.wav"
        out.parent.mkdir()
        sf.write(str(out), data.mean(axis=1), sr)
        return (out, {"rtf": 12.5, "elapsed_s": 1.0, "log_path": log_path}) if return_summary else out

    monkeypatch.setattr(pipeline, "separate", fake_separate)
    monkeypatch.setattr(pipeline, "convert_vocals", fake_convert)
    return calls


def test_run_pipeline_end_to_end(tmp_path, inputs, fakes):
    ref, song = inputs
    progress = []
    result = run_pipeline(ref, song_file=song, key_shift=2, max_seconds=4, diffusion_steps=15,
                          work_root=tmp_path / "jobs", progress=lambda f, m: progress.append((f, m)))

    assert result.output.name == "cover.wav" and result.output.is_file()
    assert [p.name for p in result.output.parent.iterdir()] == ["cover.wav"]   # ara dosyalar silindi
    data, sr = sf.read(str(result.output), always_2d=True)
    assert sr == SR and data.shape[1] == 2 and abs(data.shape[0] / SR - 4.0) < 0.05

    info = result.info
    assert info["octave_auto"] is True and info["octave_shift"] == -1          # 150 Hz vs 300 Hz -> -1 oktav
    assert info["key_shift"] == 2 and info["total_semitones"] == 2 - 12
    assert fakes["convert"]["semitones"] == -10 and fakes["convert"]["steps"] == 15
    assert fakes["convert"]["log_path"].name == "seedvc.log"
    assert info["window_start"] >= 2.0                                          # sessiz baş atlandı
    assert info["rtf"] == 12.5
    assert progress[0][0] < progress[-1][0] == 1.0
    assert ref.exists() and song.exists()                                       # delete_inputs=False


def test_manual_octave_overrides_auto(tmp_path, inputs, fakes):
    ref, song = inputs
    result = run_pipeline(ref, song_file=song, octave_shift=0, max_seconds=4, work_root=tmp_path / "jobs")
    assert result.info["octave_auto"] is False and result.info["total_semitones"] == 0
    assert fakes["convert"]["semitones"] == 0


def test_delete_inputs_removes_uploads(tmp_path, inputs, fakes):
    ref, song = inputs
    run_pipeline(ref, song_file=song, max_seconds=4, work_root=tmp_path / "jobs", delete_inputs=True)
    assert not ref.exists() and not song.exists()


def test_failure_removes_job_dir_and_inputs(tmp_path, inputs, fakes, monkeypatch):
    ref, song = inputs

    def boom(song, wd):
        raise SeparationError("demucs patladı")

    monkeypatch.setattr(pipeline, "separate", boom)
    with pytest.raises(SeparationError):
        run_pipeline(ref, song_file=song, work_root=tmp_path / "jobs", delete_inputs=True)
    assert list((tmp_path / "jobs").iterdir()) == []
    assert not ref.exists() and not song.exists()


def test_pick_window_finds_loud_part(tmp_path):
    x = np.concatenate([np.zeros(5 * SR), tone(200, 3.0), np.zeros(2 * SR)]).astype(np.float32)
    p = tmp_path / "v.wav"
    sf.write(str(p), x, SR)
    start, length = pick_window(p, 3.0)
    assert length == 3.0 and 4.0 <= start <= 6.0


def test_pick_window_short_audio(tmp_path):
    p = tmp_path / "v.wav"
    sf.write(str(p), tone(200, 2.0), SR)
    assert pick_window(p, 30.0) == (0.0, pytest.approx(2.0))


@pytest.mark.parametrize("ref,src,expected", [(133.0, 309.0, -1), (220.0, 220.0, 0), (110.0, 440.0, -2),
                                              (440.0, 110.0, 2), (None, 300.0, 0)])
def test_auto_octave(ref, src, expected):
    assert auto_octave(ref, src) == expected


def test_cleanup_old_jobs(tmp_path):
    old, new = tmp_path / "job_old", tmp_path / "job_new"
    old.mkdir(), new.mkdir()
    past = time.time() - 7200
    os.utime(old, (past, past))
    assert cleanup_old_jobs(tmp_path, max_age_s=3600) == 1
    assert not old.exists() and new.exists()
