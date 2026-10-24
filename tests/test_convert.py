"""convert_vocals() için hızlı testler. Seed-VC ve model ağırlıkları KULLANILMAZ, subprocess.Popen mock'lanır."""

from __future__ import annotations

import io
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


class FakePopen:
    """subprocess.Popen yerine geçen fabrika: çağrıyı kaydeder, --output klasörüne istenen sayıda wav yazar,
    stdout/stderr'i StringIO olarak sunar."""

    def __init__(self, n_outputs=1, returncode=0, stdout="", stderr="", channels=1, sr=SR, raise_timeout=False):
        self.n_outputs, self.returncode = n_outputs, returncode
        self.stdout_text, self.stderr_text = stdout, stderr
        self.channels, self.sr, self.raise_timeout = channels, sr, raise_timeout
        self.calls = []
        self.wait_timeouts = []
        self.killed = False

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        out = Path(cmd[cmd.index("--output") + 1])
        if not self.raise_timeout:
            for i in range(self.n_outputs):
                write_wav(out / f"vc_out_{i}.wav", channels=self.channels, sr=self.sr)
        return _Proc(self)


class _Proc:
    def __init__(self, fake: FakePopen):
        self.fake = fake
        self.stdout = io.StringIO(fake.stdout_text)
        self.stderr = io.StringIO(fake.stderr_text)

    def wait(self, timeout=None):
        self.fake.wait_timeouts.append(timeout)
        if self.fake.raise_timeout and not self.fake.killed:
            raise subprocess.TimeoutExpired("seedvc", timeout)
        return self.fake.returncode

    def kill(self):
        self.fake.killed = True


def run_convert(env, fake, monkeypatch, **kw):
    monkeypatch.setattr(convert.subprocess, "Popen", fake)
    return convert_vocals(
        env["vocals"], env["ref"], env["workdir"],
        seedvc_dir=env["seedvc_dir"], seedvc_python=env["py"], **kw,
    )


def arg(cmd, name):
    return cmd[cmd.index(name) + 1]


# ---------- mevcut davranış (13 test) ----------

def test_command_arguments_cwd_and_python(env, monkeypatch):
    fake = FakePopen()
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
    assert fake.wait_timeouts[0] == 99
    assert kw["env"]["PYTHONIOENCODING"] == "utf-8"
    assert kw["env"]["PYTHONUTF8"] == "1"


def test_success_returns_single_wav(env, monkeypatch):
    out = run_convert(env, FakePopen(), monkeypatch)
    assert isinstance(out, Path)
    assert out.is_file() and out.suffix == ".wav"
    info = sf.info(str(out))
    assert info.channels == 1 and info.samplerate == SR


def test_each_call_uses_new_empty_output_dir(env, monkeypatch):
    fake = FakePopen()
    a = run_convert(env, fake, monkeypatch)
    b = run_convert(env, fake, monkeypatch)
    assert a.parent != b.parent
    assert a.parent.parent == b.parent.parent == env["workdir"].resolve()


def test_nonzero_returncode_raises_with_stderr_tail(env, monkeypatch):
    stderr = "x" * 3000 + "SON_SATIR_HATA"
    with pytest.raises(ConversionError) as ei:
        run_convert(env, FakePopen(n_outputs=0, returncode=1, stderr=stderr), monkeypatch)
    msg = str(ei.value)
    assert "çıkış kodu 1" in msg
    assert "SON_SATIR_HATA" in msg
    assert "x" * 1600 not in msg  # yalnızca son 1500 karakter


def test_cuda_oom_has_clear_message(env, monkeypatch):
    stderr = "Traceback...\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
    with pytest.raises(ConversionError, match="GPU belleği yetmedi"):
        run_convert(env, FakePopen(n_outputs=0, returncode=1, stderr=stderr), monkeypatch)


@pytest.mark.parametrize("n", [0, 2])
def test_zero_or_multiple_outputs_raise(env, monkeypatch, n):
    with pytest.raises(ConversionError, match=f"1 wav bekleniyordu, {n} bulundu"):
        run_convert(env, FakePopen(n_outputs=n), monkeypatch)


@pytest.mark.parametrize("channels,sr", [(2, SR), (1, 22050)])
def test_wrong_output_format_raises(env, monkeypatch, channels, sr):
    with pytest.raises(ConversionError, match="beklenen biçimde değil"):
        run_convert(env, FakePopen(channels=channels, sr=sr), monkeypatch)


def test_timeout_raises(env, monkeypatch):
    fake = FakePopen(raise_timeout=True)
    with pytest.raises(ConversionError, match="saniye içinde bitmedi"):
        run_convert(env, fake, monkeypatch, timeout=5)
    assert fake.killed  # süreç öldürüldü


@pytest.mark.parametrize("missing", ["python", "inference"])
def test_seedvc_not_installed(env, monkeypatch, missing):
    if missing == "python":
        env["py"].unlink()
    else:
        (env["seedvc_dir"] / "inference.py").unlink()
    fake = FakePopen()
    with pytest.raises(ConversionError, match="setup_seedvc.py"):
        run_convert(env, fake, monkeypatch)
    assert fake.calls == []  # subprocess hiç çağrılmadı


def test_missing_input_file_raises(env, monkeypatch):
    env["vocals"].unlink()
    with pytest.raises(ConversionError, match="vokal dosyası bulunamadı"):
        run_convert(env, FakePopen(), monkeypatch)


# ---------- log_path ve özet ----------

SEEDVC_STDOUT = "Vocoder model total parameters: 1.23M\nRTF: 16.834\n"
SEEDVC_STDERR = "100%|##########| 30/30 [00:42<00:00,  1.40s/it]\n"


def test_log_path_writes_stdout_and_stderr(env, monkeypatch, tmp_path):
    log = tmp_path / "loglar" / "seedvc.log"
    run_convert(env, FakePopen(stdout=SEEDVC_STDOUT, stderr=SEEDVC_STDERR), monkeypatch, log_path=log)
    text = log.read_text(encoding="utf-8")
    assert text.startswith("$ ")                      # çalıştırılan komut
    assert "stdout] RTF: 16.834" in text
    assert "stderr] 100%|##########| 30/30" in text
    assert "çıkış kodu 0" in text


def test_log_path_none_writes_no_log(env, monkeypatch, tmp_path):
    run_convert(env, FakePopen(stdout=SEEDVC_STDOUT, stderr=SEEDVC_STDERR), monkeypatch)
    assert list(tmp_path.rglob("*.log")) == []


def test_log_written_on_error_and_tail_in_message(env, monkeypatch, tmp_path):
    log = tmp_path / "hata.log"
    stderr = "y" * 2000 + "\nRuntimeError: bozuldu\n"
    with pytest.raises(ConversionError, match="RuntimeError: bozuldu") as ei:
        run_convert(env, FakePopen(n_outputs=0, returncode=1, stderr=stderr), monkeypatch, log_path=log)
    assert len(str(ei.value)) < 1600 + 100
    assert "çıkış kodu 1" in log.read_text(encoding="utf-8")


def test_return_summary_parses_rtf(env, monkeypatch, tmp_path):
    log = tmp_path / "s.log"
    path, summary = run_convert(env, FakePopen(stdout=SEEDVC_STDOUT), monkeypatch, log_path=log, return_summary=True)
    assert path.is_file()
    assert summary["rtf"] == pytest.approx(16.834)
    assert summary["elapsed_s"] >= 0
    assert summary["log_path"] == log.resolve()


def test_return_summary_without_rtf_line(env, monkeypatch):
    _, summary = run_convert(env, FakePopen(stdout="başka bir satır\n"), monkeypatch, return_summary=True)
    assert summary["rtf"] is None and summary["log_path"] is None
