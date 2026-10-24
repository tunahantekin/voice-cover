"""fetch.py testleri: dosya doğrulama, link izin listesi, yt-dlp sahte modülle."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from voice_cover import fetch  # noqa: E402
from voice_cover.fetch import FetchError, copy_upload, fetch_from_url, resolve_song, validate_url  # noqa: E402

SR = 44100


def wav(path, seconds=1.0):
    sf.write(str(path), np.zeros(int(SR * seconds), dtype=np.float32), SR)
    return path


# ---------- dosya yükleme ----------

def test_copy_upload_ok(tmp_path):
    src = wav(tmp_path / "Şarkı Adı.WAV")
    dst = copy_upload(src, tmp_path / "job")
    assert dst.name == "song.wav" and dst.is_file()
    assert src.is_file()  # kaynak silinmez, kopyalanır


def test_copy_upload_missing(tmp_path):
    with pytest.raises(FetchError, match="bulunamadı"):
        copy_upload(tmp_path / "yok.mp3", tmp_path / "job")


def test_copy_upload_bad_extension(tmp_path):
    p = tmp_path / "belge.pdf"
    p.write_bytes(b"x")
    with pytest.raises(FetchError, match="desteklenmiyor"):
        copy_upload(p, tmp_path / "job")


def test_copy_upload_too_big(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "MAX_FILE_MB", 0.01)
    with pytest.raises(FetchError, match="çok büyük"):
        copy_upload(wav(tmp_path / "a.wav", 1.0), tmp_path / "job")


def test_copy_upload_too_long(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "MAX_DURATION_S", 1)
    with pytest.raises(FetchError, match="çok uzun"):
        copy_upload(wav(tmp_path / "a.wav", 2.0), tmp_path / "job")


def test_copy_upload_unreadable(tmp_path):
    p = tmp_path / "bozuk.mp3"
    p.write_bytes(b"ses degil")
    with pytest.raises(FetchError, match="okunamadı"):
        copy_upload(p, tmp_path / "job")


# ---------- link doğrulama ----------

@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abc",
    "https://youtu.be/abc",
    "https://m.youtube.com/watch?v=abc",
    "https://music.youtube.com/watch?v=abc",
    "https://soundcloud.com/a/b",
])
def test_validate_url_allows(url):
    assert validate_url(url) == url


@pytest.mark.parametrize("url", [
    "", "youtube.com/watch?v=abc", "file:///etc/passwd", "ftp://youtube.com/x",
    "http://localhost:8080/x", "http://169.254.169.254/latest", "https://evilyoutube.com/x",
    "https://youtube.com.evil.com/x",
])
def test_validate_url_rejects(url):
    with pytest.raises(FetchError):
        validate_url(url)


# ---------- yt-dlp (sahte) ----------

class FakeYDL:
    duration = 120
    downloads = []

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        return {"duration": FakeYDL.duration}

    def download(self, urls):
        FakeYDL.downloads.append(urls)
        out = self.opts["outtmpl"].replace("%(ext)s", "wav")
        wav(out)


@pytest.fixture
def fake_ytdlp(monkeypatch):
    FakeYDL.duration, FakeYDL.downloads = 120, []
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    return FakeYDL


def test_fetch_from_url_downloads_wav(tmp_path, fake_ytdlp):
    out = fetch_from_url("https://youtu.be/abc", tmp_path / "job")
    assert out.name == "song.wav" and out.is_file()
    assert fake_ytdlp.downloads == [["https://youtu.be/abc"]]


def test_fetch_from_url_rejects_long_before_download(tmp_path, fake_ytdlp):
    fake_ytdlp.duration = 3600
    with pytest.raises(FetchError, match="çok uzun"):
        fetch_from_url("https://youtu.be/abc", tmp_path / "job")
    assert fake_ytdlp.downloads == []


# ---------- resolve_song ----------

def test_resolve_song_prefers_file(tmp_path, fake_ytdlp):
    out = resolve_song(tmp_path / "job", file_path=wav(tmp_path / "a.wav"), url="https://youtu.be/abc")
    assert out.name == "song.wav"
    assert fake_ytdlp.downloads == []


def test_resolve_song_requires_input(tmp_path):
    with pytest.raises(FetchError, match="yükle ya da link"):
        resolve_song(tmp_path / "job")
