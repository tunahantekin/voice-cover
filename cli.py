"""voice-cover komut satırı arayüzü.

Örnek:
    python cli.py --ref samples/ref.wav --song samples/song_full.wav --out outputs/cover.wav
    python cli.py --ref samples/ref.wav --url "https://www.youtube.com/watch?v=..." --octave -1 --steps 15
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from voice_cover.audio_utils import AudioError  # noqa: E402
from voice_cover.convert import ConversionError  # noqa: E402
from voice_cover.fetch import FetchError  # noqa: E402
from voice_cover.mix import MixError  # noqa: E402
from voice_cover.pipeline import run_pipeline  # noqa: E402
from voice_cover.separate import SeparationError  # noqa: E402


def parse_octave(value: str) -> int | None:
    if value == "auto":
        return None
    try:
        return int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("--octave 'auto' ya da -2..2 arası bir tam sayı olmalı") from e


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Şarkıyı senin sesinle söylet.")
    ap.add_argument("--ref", required=True, help="kendi ses kaydın")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--song", help="şarkı dosyası")
    src.add_argument("--url", help="şarkı linki (YouTube/SoundCloud)")
    ap.add_argument("--out", help="sonucun kopyalanacağı yol (verilmezse outputs/jobs altında kalır)")
    ap.add_argument("--key", type=int, default=0, help="ton kaydırma, vokal + müzik (yarım ton)")
    ap.add_argument("--octave", type=parse_octave, default=None, help="'auto' ya da -2..2 (yalnızca vokal)")
    ap.add_argument("--steps", type=int, default=30, help="difüzyon adımı (10-30)")
    ap.add_argument("--max-seconds", type=float, default=30.0, help="dönüştürülecek süre")
    args = ap.parse_args(argv)

    def progress(frac: float, msg: str) -> None:
        print(f"[{frac * 100:3.0f}%] {msg}", flush=True)

    try:
        result = run_pipeline(
            Path(args.ref),
            song_file=Path(args.song) if args.song else None,
            song_url=args.url,
            key_shift=args.key,
            octave_shift=args.octave,
            diffusion_steps=args.steps,
            max_seconds=args.max_seconds,
            progress=progress,
        )
    except (AudioError, FetchError, SeparationError, ConversionError, MixError) as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 1

    output = result.output
    if args.out:
        dst = Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output, dst)
        output = dst
    print(f"\nSonuç: {output}")
    print(json.dumps(result.info, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
