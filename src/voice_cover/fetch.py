"""Şarkıyı yüklenen dosyadan (ana yol) ya da linkten (ikincil yol, yt-dlp) alıp çalışma klasörüne getirir."""

from __future__ import annotations

from pathlib import Path


def fetch_from_url(url: str, out_dir: Path) -> Path:
    raise NotImplementedError


def resolve_song(out_dir: Path, file_path: Path | None = None, url: str | None = None) -> Path:
    raise NotImplementedError
