"""app.py testleri: arayüz kurulur ama başlatılmaz, run_pipeline mock'lanır."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("voice_cover_app", ROOT / "app.py")
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)

from voice_cover.audio_utils import AudioError  # noqa: E402
from voice_cover.pipeline import PipelineResult  # noqa: E402

INFO = {"window_start": 52.0, "window_seconds": 30.0, "ref_f0": 133.0, "src_f0": 309.0, "octave_shift": -1,
        "octave_auto": True, "key_shift": 0, "rtf": 16.4, "timings": {"convert": 490.0}}


def no_progress(*a, **k):
    pass


def call(**kw):
    args = dict(ref_path="ref.wav", song_path="song.mp3", song_url="", consent=True, key_shift=0,
                octave_choice="Otomatik", steps=30, max_seconds=30, progress=no_progress)
    args.update(kw)
    return app.handle(**args)


def test_build_app_returns_blocks():
    assert isinstance(app.build_app(), gr.Blocks)


@pytest.mark.parametrize("kw,msg", [
    ({"consent": False}, "onay"),
    ({"ref_path": None}, "sesini kaydet"),
    ({"song_path": None, "song_url": "  "}, "şarkı dosyası"),
])
def test_handle_validates_inputs(monkeypatch, kw, msg):
    monkeypatch.setattr(app, "run_pipeline", lambda *a, **k: pytest.fail("pipeline çağrılmamalı"))
    with pytest.raises(gr.Error, match=msg):
        call(**kw)


def test_handle_success_passes_options(monkeypatch, tmp_path):
    seen = {}

    def fake(ref, **kw):
        seen.update(kw, ref=ref)
        return PipelineResult(output=tmp_path / "cover.wav", info=INFO)

    monkeypatch.setattr(app, "run_pipeline", fake)
    monkeypatch.setattr(app, "cleanup_old_jobs", lambda: 0)
    path, md = call(octave_choice="-1 oktav", key_shift=2, steps=15, max_seconds=20)
    assert path == str(tmp_path / "cover.wav")
    assert "RTF 16.4" in md and "133 Hz" in md
    assert seen["octave_shift"] == -1 and seen["key_shift"] == 2 and seen["diffusion_steps"] == 15
    assert seen["max_seconds"] == 20.0 and seen["delete_inputs"] is True


def test_handle_user_error_becomes_gr_error(monkeypatch):
    def fail(*a, **k):
        raise AudioError("Kayıt çok kısa (3.0 sn).")

    monkeypatch.setattr(app, "run_pipeline", fail)
    monkeypatch.setattr(app, "cleanup_old_jobs", lambda: 0)
    with pytest.raises(gr.Error, match="çok kısa"):
        call()
