"""Seed-VC'nin konumunu, sabitlenen commit'ini ve ayrı venv'inin python yolunu tek yerde tanımlar."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEEDVC_REPO_URL = "https://github.com/Plachtaa/seed-vc.git"
SEEDVC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
SEEDVC_DIR = PROJECT_ROOT / "third_party" / "seed-vc"
SEEDVC_VENV = SEEDVC_DIR / ".venv"


def venv_python(venv: Path) -> Path:
    """Bir venv'in python yolunu işletim sistemine göre döndürür."""
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


SEEDVC_PYTHON = venv_python(SEEDVC_VENV)
