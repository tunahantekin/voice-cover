"""Seed-VC'yi third_party/seed-vc altına klonlar ve bağımlılıklarını kurar (Windows dahil her yerde çalışır)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Plachtaa/seed-vc.git"
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "third_party" / "seed-vc"


def main() -> int:
    if shutil.which("git") is None:
        print("git bulunamadı, önce git kurun.", file=sys.stderr)
        return 1

    if (TARGET / ".git").exists():
        print(f"Seed-VC zaten var, klonlama atlandı: {TARGET}")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        print(f"Seed-VC klonlanıyor: {TARGET}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(TARGET)], check=True)

    req = TARGET / "requirements.txt"
    if not req.exists():
        print(f"requirements.txt bulunamadı: {req}", file=sys.stderr)
        return 1

    print("Seed-VC bağımlılıkları kuruluyor...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)
    print("Tamam.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
