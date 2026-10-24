"""Şarkıyı yüklenen dosyadan (ana yol) ya da linkten (ikincil yol, yt-dlp) alıp çalışma klasörüne getirir."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".webm"}
MAX_FILE_MB = 50
MAX_DURATION_S = 600
# Link yolu yalnızca bu sitelere izin verir (rastgele/iç adreslere istek atılmasını önler)
ALLOWED_HOSTS = ("youtube.com", "youtu.be", "soundcloud.com")


class FetchError(ValueError):
    """Şarkı alınamadığında kullanıcıya gösterilebilecek Türkçe mesajlı hata."""


def _duration_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        try:
            import librosa

            return float(librosa.get_duration(path=str(path)))
        except Exception:
            return None


def copy_upload(file_path: Path, out_dir: Path) -> Path:
    """Yüklenen dosyayı doğrulayıp çalışma klasörüne 'song<uzantı>' olarak kopyalar."""
    src = Path(file_path)
    if not src.is_file():
        raise FetchError("Yüklenen şarkı dosyası bulunamadı.")
    ext = src.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FetchError(f"Bu dosya türü desteklenmiyor ({ext or 'uzantısız'}). Desteklenenler: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    size_mb = src.stat().st_size / 1024 / 1024
    if size_mb > MAX_FILE_MB:
        raise FetchError(f"Dosya çok büyük ({size_mb:.0f} MB). En fazla {MAX_FILE_MB} MB yüklenebilir.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"song{ext}"
    shutil.copyfile(src, dst)

    duration = _duration_seconds(dst)
    if duration is None:
        raise FetchError("Şarkı dosyası okunamadı. Bozuk olabilir ya da desteklenmeyen bir biçimde.")
    if duration > MAX_DURATION_S:
        raise FetchError(f"Şarkı çok uzun ({duration / 60:.1f} dk). En fazla {MAX_DURATION_S // 60} dakika olabilir.")
    return dst


def validate_url(url: str) -> str:
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise FetchError("Geçerli bir link gir (https://...).")
    host = parsed.hostname.lower()
    if not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
        raise FetchError("Şimdilik yalnızca YouTube ve SoundCloud linkleri destekleniyor.")
    return url


def fetch_from_url(url: str, out_dir: Path) -> Path:
    """Linkteki sesi yt-dlp ile indirip 'song.wav' olarak kaydeder."""
    url = validate_url(url)
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise FetchError("Link desteği için yt-dlp kurulu değil.") from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}

    try:
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise FetchError("Linkteki ses bilgisi alınamadı. Link doğru mu, video herkese açık mı?") from e

    duration = (info or {}).get("duration")
    if duration and duration > MAX_DURATION_S:
        raise FetchError(f"Şarkı çok uzun ({duration / 60:.1f} dk). En fazla {MAX_DURATION_S // 60} dakika olabilir.")

    opts = {
        **base_opts,
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "song.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise FetchError("Şarkı linkten indirilemedi.") from e

    out = out_dir / "song.wav"
    if not out.is_file():
        raise FetchError("Şarkı indirildi ama ses dosyası oluşturulamadı (ffmpeg kurulu mu?).")
    return out


def resolve_song(out_dir: Path, file_path: Path | None = None, url: str | None = None) -> Path:
    """Dosya verilmişse onu (ana yol), yoksa linki kullanır."""
    if file_path:
        return copy_upload(Path(file_path), out_dir)
    if url and url.strip():
        return fetch_from_url(url, out_dir)
    raise FetchError("Bir şarkı dosyası yükle ya da link gir.")
