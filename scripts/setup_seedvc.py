"""Seed-VC'yi sabit bir commit'te third_party/seed-vc altına klonlar ve ona ait ayrı bir Python 3.10 venv'i kurar.

Seed-VC'nin kendi requirements.txt dosyası KULLANILMAZ (nightly index satırları tüm kurulumu bozar).
Yalnızca inference.py'nin (F0 koşullu yol) gerçekten import ettiği paketler kurulur.

Kullanım:
    python scripts/setup_seedvc.py                 # kur
    python scripts/setup_seedvc.py --dry-run       # sadece ne yapacağını yazdır
    python scripts/setup_seedvc.py --only-missing  # mevcut venv'e yalnızca eksik/farklı sürümdeki paketleri kur

Import-time bağımlılık haritası (inference.py'den modül seviyesinde izlendi, torch.* alt modülleri dahil):
    Seed-VC (25 yerel dosya)
        torch, torchaudio, numpy, librosa, scipy, yaml, munch, einops, tqdm, huggingface_hub
        matplotlib            modules/bigvgan/utils.py:6
        dac.nn.quantize       modules/length_regulator.py:7
        transformers          inference.py:135 (fonksiyon içi, Whisper yolu her çalıştırmada yüklenir)
        torch alt modülleri   nn, nn.functional, nn.parameter, nn.utils, utils.checkpoint, utils.data,
                              distributions.uniform (hepsi torch içinde, ek paket yok)
    dac/__init__.py  ->  import audiotools, dac.model, dac.utils, dac.nn.loss
        argbind               dac/utils/__init__.py:3
        ffmpy                 audiotools/core/ffmpeg.py:7
        flatten_dict          audiotools/core/util.py:17
        importlib_resources   audiotools/core/playback.py:13
        julius                audiotools/core/audio_signal.py:12
        randomname            audiotools/ml/experiment.py:13
        rich                  audiotools/ml/decorators.py:9
        soundfile             audiotools/core/audio_signal.py:14
        tensorboard           audiotools/ml/decorators.py:23 (DOLAYLI: from torch.utils.tensorboard import SummaryWriter)
        protobuf              tensorboard bağımlılığı
        torch alt modülleri   distributed, nn.parallel, utils.data.distributed (torch içinde, ek paket yok)
    Lazy (yalnızca çağrılırsa): onnxruntime (rmvpe, DirectML yolu), IPython, pesq, pystoi, visqol, requests
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_cover.seedvc_env import (  # noqa: E402
    SEEDVC_COMMIT,
    SEEDVC_DIR,
    SEEDVC_PYTHON,
    SEEDVC_REPO_URL,
    SEEDVC_VENV,
)

LOG_FILE = ROOT / "setup_seedvc.log"
# numpy / torch / torchaudio sürümlerini kilitler, her pip adımına -c ile verilir
CONSTRAINTS_FILE = ROOT / "scripts" / "seedvc-constraints.txt"

# GTX 1650 (sm_75): cu121 ve cu124 build'lerinin ikisi de sm_75 içeriyor, sürücü (610.x) ikisini de destekliyor.
# cu121, torch 2.4.0 ile daha uzun süre test edilmiş varsayılan build olduğu için seçildi.
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
TORCH_PKGS = ["torch==2.4.0", "torchaudio==2.4.0"]
NUMPY_PIN = "numpy==1.26.4"

# inference.py + modules/ import zincirinden çıkarılan minimal liste
SEEDVC_PKGS = [
    "transformers==4.46.3",        # inference.py (WhisperModel, AutoFeatureExtractor), modules/commons.py
    "tokenizers==0.20.3",          # transformers 4.46.3 bunu ister (<0.21)
    "huggingface_hub==0.36.2",     # hf_utils.py, modules/bigvgan/bigvgan.py (transformers için <1.0)
    "librosa==0.10.2.post1",       # inference.py, modules/audio.py, modules/rmvpe.py, modules/bigvgan/meldataset.py
    "numba==0.60.0",               # librosa bağımlılığı, numpy 1.26 ile uyumlu
    "scipy==1.13.1",               # modules/audio.py, modules/rmvpe.py, modules/hifigan/generator.py, modules/bigvgan/*
    "soundfile==0.12.1",           # torchaudio 2.4 kayıt backend'i, audiotools
    "einops==0.8.0",               # modules/encodec.py, dac.nn.quantize
    "munch==4.0.0",                # modules/commons.py
    "PyYAML==6.0.2",               # inference.py, argbind
    "matplotlib==3.10.9",          # modules/bigvgan/utils.py (modül seviyesinde import)
    "tqdm==4.70.1",                # modules/flow_matching.py, modules/bigvgan/meldataset.py
    # descript-audio-codec ve audiotools --no-deps kurulduğu için import anında gerekenler:
    "argbind==0.3.9",              # dac/utils/__init__.py
    "ffmpy==1.0.0",                # audiotools/core/ffmpeg.py
    "flatten-dict==0.5.0",         # audiotools/core/util.py
    "importlib-resources==7.1.0",  # audiotools/core/playback.py
    "julius==0.2.8",               # audiotools/core/*
    "randomname==0.2.1",           # audiotools/ml/experiment.py
    "rich==15.0.0",                # audiotools/ml/decorators.py
    "tensorboard==2.21.0",         # audiotools/ml/decorators.py:23 (torch.utils.tensorboard üzerinden)
    "protobuf==7.36.2",            # tensorboard bağımlılığı (>=6.31.1,<8), abi3 wheel
]
# dac.nn.quantize için gerekli. Tam bağımlılıkları (protobuf<3.20, tensorboard, ipython...) gereksiz.
NO_DEPS_PKGS = ["descript-audio-codec==1.0.0", "descript-audiotools==0.7.2"]

VERIFY_CODE = r"""
import sys, torch, torchaudio, numpy
print("python    :", sys.version.split()[0])
print("torch     :", torch.__version__, "| torchaudio:", torchaudio.__version__, "| numpy:", numpy.__version__)
print("cuda      :", torch.cuda.is_available(), "| build:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu       :", torch.cuda.get_device_name(0), "| sm_75 destekli:", "sm_75" in torch.cuda.get_arch_list())
print("audio backends:", torchaudio.list_audio_backends())
import inference                                   # modül seviyesindeki importlar (model indirmez)
from transformers import AutoFeatureExtractor, WhisperModel
from modules.rmvpe import RMVPE
from modules.bigvgan import bigvgan
from modules.campplus.DTDNN import CAMPPlus
import dac.nn.quantize
print("inference import kontrolü: OK")
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], dry: bool, *, title: str, cwd: Path | None = None, to_log: bool = True) -> None:
    shown = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    if dry:
        if "\n" in shown:  # çok satırlı -c kodunu kısalt
            shown = shown.split(" -c ")[0] + " -c <VERIFY_CODE>"
        log(f"[dry-run] {title}\n          {shown}" + (f"\n          (cwd={cwd})" if cwd else ""))
        return
    log(f"==> {title}")
    if not to_log:
        subprocess.run(cmd, cwd=cwd, check=True)
        return
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S} | {title}\n$ {shown}\n")
        fh.flush()
        result = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"HATA: '{title}' başarısız (çıkış kodu {result.returncode}). Ayrıntı: {LOG_FILE}")
    log("    tamam")


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_versions(vpy: str) -> dict[str, str]:
    """Venv'de kurulu paketleri {normalize_ad: sürüm} olarak döndürür. Venv yoksa boş sözlük."""
    if not Path(vpy).exists():
        return {}
    out = subprocess.run([vpy, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return {}
    return {_canon(d["name"]): d["version"] for d in json.loads(out.stdout)}


def missing_specs(specs: list[str], installed: dict[str, str]) -> list[str]:
    """'paket==sürüm' listesinden kurulu olmayanları ya da sürümü farklı olanları döndürür (+cu121 eki yok sayılır)."""
    result = []
    for spec in specs:
        name, ver = spec.split("==")
        have = installed.get(_canon(name))
        if have is None or (have != ver and have.split("+")[0] != ver):
            result.append(spec)
    return result


def find_python310() -> str | None:
    candidates = [["py", "-3.10"]] if os.name == "nt" else [["python3.10"]]
    for c in candidates:
        if shutil.which(c[0]) is None:
            continue
        try:
            out = subprocess.run(c + ["-c", "import sys;print(sys.executable)"], capture_output=True, text=True)
        except OSError:
            continue
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return None


def ensure_repo(dry: bool) -> None:
    git = ["git", "-C", str(SEEDVC_DIR)]
    if (SEEDVC_DIR / ".git").exists():
        head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == SEEDVC_COMMIT:
            log(f"Seed-VC zaten var ve commit doğru ({SEEDVC_COMMIT[:12]}), klonlama atlandı.")
            return
        log(f"Seed-VC var ama commit farklı ({head[:12] or '?'}), sabit commit'e geçilecek.")
        run(git + ["fetch", "--depth", "1", "origin", SEEDVC_COMMIT], dry, title="Seed-VC commit fetch", to_log=False)
        run(git + ["checkout", "--detach", SEEDVC_COMMIT], dry, title="Seed-VC commit checkout", to_log=False)
        return
    if not dry:
        SEEDVC_DIR.mkdir(parents=True, exist_ok=True)
    run(git + ["init", "-q"], dry, title="Seed-VC git init", to_log=False)
    run(git + ["remote", "add", "origin", SEEDVC_REPO_URL], dry, title="Seed-VC remote ekle", to_log=False)
    run(git + ["fetch", "--depth", "1", "origin", SEEDVC_COMMIT], dry, title="Seed-VC commit fetch", to_log=False)
    run(git + ["checkout", "--detach", "FETCH_HEAD"], dry, title="Seed-VC checkout", to_log=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="kurmadan ne yapılacağını yazdır")
    ap.add_argument("--only-missing", action="store_true",
                    help="mevcut venv'de yalnızca eksik ya da sürümü farklı paketleri kur (pip güncellemesini atla)")
    args = ap.parse_args()
    dry, only_missing = args.dry_run, args.only_missing

    if shutil.which("git") is None:
        log("HATA: git bulunamadı, önce git kurun.")
        return 1

    log(f"Proje kökü   : {ROOT}")
    log(f"Seed-VC      : {SEEDVC_DIR} @ {SEEDVC_COMMIT}")
    log(f"Seed-VC venv : {SEEDVC_VENV}")
    log(f"Venv python  : {SEEDVC_PYTHON}")
    log(f"Log dosyası  : {LOG_FILE}\n")

    # 1) Repo
    ensure_repo(dry)

    # 2) Python 3.10 + venv
    py310 = find_python310()
    if py310 is None:
        log("HATA: Python 3.10 bulunamadı. Windows'ta 'py -0p' ile kontrol edin, "
            "python.org'dan 3.10 kurun ya da 'uv python install 3.10' kullanın.")
        return 1
    log(f"Python 3.10  : {py310}")
    if SEEDVC_PYTHON.exists():
        log("Seed-VC venv zaten var, yeniden oluşturulmayacak.")
    else:
        run([py310, "-m", "venv", str(SEEDVC_VENV)], dry, title="venv oluştur (Python 3.10)")

    vpy = str(SEEDVC_PYTHON)
    pip = [vpy, "-m", "pip", "install", "--disable-pip-version-check", "-c", str(CONSTRAINTS_FILE)]
    installed = installed_versions(vpy) if only_missing else {}

    def pip_step(specs: list[str], extra: list[str], title: str) -> None:
        if only_missing:
            specs = missing_specs(specs, installed)
            if not specs:
                log(f"--  {title}: hepsi kurulu, atlandı")
                return
            title = f"{title} [eksikler: {', '.join(specs)}]"
        run(pip + extra + specs, dry, title=title)

    # 3) Paketler (sırayla)
    if not only_missing:
        run([vpy, "-m", "pip", "install", "--upgrade", "pip"], dry, title="pip güncelle")
    pip_step(TORCH_PKGS, ["--index-url", TORCH_INDEX], "torch 2.4.0 + torchaudio 2.4.0 (cu121)")
    pip_step([NUMPY_PIN], [], "numpy 1.26.4")
    pip_step(SEEDVC_PKGS, [], "Seed-VC minimal paketleri")
    pip_step(NO_DEPS_PKGS, ["--no-deps"], "descript-audio-codec + audiotools (--no-deps)")

    # 4) Doğrulama (model indirmez, sadece import)
    run([vpy, "-c", VERIFY_CODE], dry, cwd=SEEDVC_DIR, title="doğrulama: sürümler + inference importları", to_log=False)

    log("\n[dry-run] Hiçbir şey kurulmadı." if dry else "\nKurulum tamamlandı.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
