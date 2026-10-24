"""convert_vocals() için hızlı testler. Seed-VC ve model ağırlıkları KULLANILMAZ, subprocess.run mock'lanır."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from voice_cover import convert  # noqa: E402
from voice_cover.convert import ConversionError, convert_vocals  # noqa: E402

SR = 44100


def write_wav(path: Path, channels: int = 1, sr: int = SR, seconds: float = 0.2) -> None:
    n = int(sr * seconds)
    data = np.zeros((n, channels), dtype=np.float32) if channels > 1 else np.zeros(n, dtype=np.float32)
    sf.write(str(path), data, sr)


@pytest.fixture
def env(tmp_path):
    """Boşluklu yollarla sahte Seed-VC kurulumu ve giriş dosyaları."""
    base = tmp_path / "klasör boşluk"
    seedvc_dir = base / "seed-vc"
    seedvc_dir.mkdir(parents=True)
    (seedvc_dir / "inference.py").write_text("# sahte\n", encoding="utf-8")
    py = seedvc_dir / ".venv" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_bytes(b"")
    vocals = base / "vocals.wav"
    ref = base / "referans ses.wav"
    write_wav(vocals)
    write_wav(ref)
    return {"seedvc_dir": seedvc_dir, "py": py, "vocals": vocals, "ref": ref, "workdir": base / "work"}


class FakeRun:
    """subprocess.run yerine geçer: çağrıyı kaydeder, --output klasörüne istenen sayıda wav yazar."""

    def __init__(self, n_outputs=1, returncode=0, stderr="", channels=1, sr=SR, raise_timeout=False):
        self.n_outputs, self.returncode, self.stderr = n_outputs, returncode, stderr
        self.channels, self.sr, self.raise_timeout = channels, sr, raise_timeout
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.raise_timeout:
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        out = Path(cmd[cmd.index("--output") + 1])
        for i in range(self.n_outputs):
            write_wav(out / f"vc_out_{i}.wav", channels=self.channels, sr=self.sr)
        return subprocess.CompletedProcess(cmd, self.returncode, stdout="", stderr=self.stderr)


def run_convert(env, fake, monkeypatch, **kw):
    monkeypatch.setattr(convert.subprocess, "run", fake)
    return convert_vocals(
        env["vocals"], env["ref"], env["workdir"],
        seedvc_dir=env["seedvc_dir"], seedvc_python=env["py"], **kw,
    )


def arg(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_command_arguments_cwd_and_python(env, monkeypatch):
    fake = FakeRun()
    run_convert(env, fake, monkeypatch, semitones=-2, diffusion_steps=25, timeout=99)
    cmd, kw = fake.calls[0]

    assert cmd[0] == str(env["py"])
    assert cmd[1] == str(env["seedvc_dir"].resolve() / "inference.py")
    assert arg(cmd, "--source") == str(env["vocals"].resolve())
    assert arg(cmd, "--target") == str(env["ref"].resolve())
    assert Path(arg(cmd, "--output")).is_absolute()
    assert arg(cmd, "--diffusion-steps") == "25"
    assert arg(cmd, "--length-adjust") == "1.0"
    assert arg(cmd, "--inference-cfg-rate") == "0.7"
    assert arg(cmd, "--f0-condition") == "True"
    assert arg(cmd, "--auto-f0-adjust") == "False"
    assert arg(cmd, "--semi-tone-shift") == "-2"
    assert arg(cmd, "--fp16") == "True"
    assert all(isinstance(c, str) for c in cmd)

    assert kw["cwd"] == str(env["seedvc_dir"].resolve())
    assert kw["shell"] is False
    assert kw["timeout"] == 99
    assert kw["env"]["PYTHONIOENCODING"] == "utf-8"
    assert kw["env"]["PYTHONUTF8"] == "1"


def test_success_returns_single_wav(env, monkeypatch):
    out = run_convert(env, FakeRun(), monkeypatch)
    assert out.is_file() and out.suffix == ".wav"
    info = sf.info(str(out))
    assert info.channels == 1 and info.samplerate == SR


def test_each_call_uses_new_empty_output_dir(env, monkeypatch):
    fake = FakeRun()
    a = run_convert(env, fake, monkeypatch)
    b = run_convert(env, fake, monkeypatch)
    assert a.parent != b.parent
    assert a.parent.parent == b.parent.parent == env["workdir"].resolve()


def test_nonzero_returncode_raises_with_stderr_tail(env, monkeypatch):
    stderr = "x" * 3000 + "SON_SATIR_HATA"
    with pytest.raises(ConversionError) as ei:
        run_convert(env, FakeRun(n_outputs=0, returncode=1, stderr=stderr), monkeypatch)
    msg = str(ei.value)
    assert "çıkış kodu 1" in msg
    assert "SON_SATIR_HATA" in msg
    assert "x" * 1600 not in msg  # yalnızca son 1500 karakter


def test_cuda_oom_has_clear_message(env, monkeypatch):
    stderr = "Traceback...\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
    with pytest.raises(ConversionError, match="GPU belleği yetmedi"):
        run_convert(env, FakeRun(n_outputs=0, returncode=1, stderr=stderr), monkeypatch)


@pytest.mark.parametrize("n", [0, 2])
def test_zero_or_multiple_outputs_raise(env, monkeypatch, n):
    with pytest.raises(ConversionError, match=f"1 wav bekleniyordu, {n} bulundu"):
        run_convert(env, FakeRun(n_outputs=n), monkeypatch)


@pytest.mark.parametrize("channels,sr", [(2, SR), (1, 22050)])
def test_wrong_output_format_raises(env, monkeypatch, channels, sr):
    with pytest.raises(ConversionError, match="beklenen biçimde değil"):
        run_convert(env, FakeRun(channels=channels, sr=sr), monkeypatch)


def test_timeout_raises(env, monkeypatch):
    with pytest.raises(ConversionError, match="saniye içinde bitmedi"):
        run_convert(env, FakeRun(raise_timeout=True), monkeypatch, timeout=5)


@pytest.mark.parametrize("missing", ["python", "inference"])
def test_seedvc_not_installed(env, monkeypatch, missing):
    if missing == "python":
        env["py"].unlink()
    else:
        (env["seedvc_dir"] / "inference.py").unlink()
    fake = FakeRun()
    with pytest.raises(ConversionError, match="setup_seedvc.py"):
        run_convert(env, fake, monkeypatch)
    assert fake.calls == []  # subprocess hiç çağrılmadı


def test_missing_input_file_raises(env, monkeypatch):
    env["vocals"].unlink()
    with pytest.raises(ConversionError, match="vokal dosyası bulunamadı"):
        run_convert(env, FakeRun(), monkeypatch)
