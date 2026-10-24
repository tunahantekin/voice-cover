"""voice-cover Gradio arayüzü: kendi sesini kaydet, şarkı ekle, şarkıyı senin sesinle dinle."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from voice_cover.audio_utils import AudioError  # noqa: E402
from voice_cover.convert import ConversionError  # noqa: E402
from voice_cover.fetch import FetchError  # noqa: E402
from voice_cover.mix import MixError  # noqa: E402
from voice_cover.pipeline import cleanup_old_jobs, run_pipeline  # noqa: E402
from voice_cover.separate import SeparationError  # noqa: E402

SENTENCES = [
    "Bugün hava güzel olduğu için sabah erkenden kalkıp sahilde uzun bir yürüyüş yaptım.",
    "Çocukluğumda dedemin bahçesinde ağaçlara tırmanır, kirazları dalından toplayıp yerdim.",
    "Şarkı söylemeyi her zaman sevdim ama sesimi kayda almak benim için ilk kez oluyor.",
]

CONSENT_TEXT = (
    "Kaydettiğim ses bana ait (ya da sahibinden açık izin aldım). Sesimin bu işlem için kullanılmasını kabul ediyorum. "
    "Yüklediğim dosyaların işlem bitince silineceğini ve sonucu ticari amaçla kullanmayacağımı biliyorum."
)

OCTAVE_CHOICES = ["Otomatik", "-2 oktav", "-1 oktav", "0", "+1 oktav", "+2 oktav"]
OCTAVE_MAP = {"Otomatik": None, "-2 oktav": -2, "-1 oktav": -1, "0": 0, "+1 oktav": 1, "+2 oktav": 2}

USER_ERRORS = (AudioError, FetchError, SeparationError, ConversionError, MixError)


def _summary_md(info: dict) -> str:
    ref_f0 = f"{info['ref_f0']:.0f} Hz" if info.get("ref_f0") else "ölçülemedi"
    src_f0 = f"{info['src_f0']:.0f} Hz" if info.get("src_f0") else "ölçülemedi"
    octave = f"{info['octave_shift']:+d}" + (" (otomatik)" if info.get("octave_auto") else "")
    rtf = f"{info['rtf']:.1f}" if info.get("rtf") else "-"
    conv = info.get("timings", {}).get("convert", "-")
    return (
        f"**Dönüştürülen bölüm:** {info['window_start']:.0f}. saniyeden itibaren {info['window_seconds']:.0f} sn  \n"
        f"**Perde:** senin sesin {ref_f0}, şarkı {src_f0} · oktav {octave} · ton {info['key_shift']:+d}  \n"
        f"**Süre:** dönüştürme {conv} sn (RTF {rtf})"
    )


def handle(ref_path, song_path, song_url, consent, key_shift, octave_choice, steps, max_seconds, progress=gr.Progress()):
    """Arayüzün tek işlem fonksiyonu. Kullanıcı hatalarını gr.Error olarak gösterir."""
    if not consent:
        raise gr.Error("Devam etmek için onay kutusunu işaretle.", title="Onay gerekli")
    if not ref_path:
        raise gr.Error("Önce ekrandaki cümleleri okuyup sesini kaydet.", title="Ses kaydı yok")
    if not song_path and not (song_url or "").strip():
        raise gr.Error("Bir şarkı dosyası yükle ya da link gir.", title="Şarkı yok")

    cleanup_old_jobs()  # bir saatten eski sonuçları sil
    try:
        result = run_pipeline(
            Path(ref_path),
            song_file=Path(song_path) if song_path else None,
            song_url=song_url or None,
            key_shift=int(key_shift),
            octave_shift=OCTAVE_MAP.get(octave_choice),
            diffusion_steps=int(steps),
            max_seconds=float(max_seconds),
            delete_inputs=True,
            progress=lambda frac, msg: progress(frac, desc=msg),
        )
    except USER_ERRORS as e:
        raise gr.Error(str(e), title="İşlem tamamlanamadı", duration=None) from e
    return str(result.output), _summary_md(result.info)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="voice-cover", delete_cache=(1800, 3600)) as demo:
        gr.Markdown("# voice-cover\nŞarkıyı **senin sesinle** dinle. Eğitim yok, birkaç cümlelik kayıt yeterli.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Bu cümleleri sesli oku\n" + "\n".join(f"> {s}" for s in SENTENCES))
                ref = gr.Audio(
                    sources=["microphone", "upload"], type="filepath",
                    label="Ses kaydın (sessiz ortamda, 10-20 sn)",
                )
            with gr.Column():
                gr.Markdown("### 2. Şarkıyı ekle")
                song = gr.Audio(sources=["upload"], type="filepath", label="Şarkı dosyası (mp3, wav, m4a...)")
                with gr.Accordion("Ya da link ver (YouTube / SoundCloud)", open=False):
                    url = gr.Textbox(label="Şarkı linki", placeholder="https://www.youtube.com/watch?v=...")

        with gr.Accordion("Gelişmiş ayarlar", open=False):
            with gr.Row():
                octave = gr.Dropdown(OCTAVE_CHOICES, value="Otomatik", label="Oktav (yalnızca vokal)",
                                     info="Otomatik: sesinin perdesine en yakın oktav seçilir")
                key = gr.Slider(-6, 6, value=0, step=1, label="Ton (vokal + müzik, yarım ton)")
            with gr.Row():
                steps = gr.Slider(10, 30, value=30, step=1, label="Kalite adımı",
                                  info="Düşük = hızlı, yüksek = daha temiz")
                max_seconds = gr.Slider(10, 60, value=30, step=5, label="Dönüştürülecek süre (sn)",
                                        info="Vokalin en yoğun bölümü seçilir")

        consent = gr.Checkbox(value=False, label=CONSENT_TEXT)
        run = gr.Button("Şarkıyı benim sesimle söylet", variant="primary")

        gr.Markdown("### 3. Sonuç")
        out_audio = gr.Audio(type="filepath", label="Sonuç", interactive=False)
        out_info = gr.Markdown()

        run.click(
            handle,
            inputs=[ref, song, url, consent, key, octave, steps, max_seconds],
            outputs=[out_audio, out_info],
        )
        gr.Markdown(
            "<small>Yalnızca kendi sesini ya da izin aldığın birinin sesini kullan. Yüklenen sesler işlem bitince silinir, "
            "sonuçlar en geç bir saat içinde sunucudan kaldırılır. Ticari olmayan kullanım içindir. "
            "Şarkıların hakları sahiplerine aittir.</small>"
        )
    return demo


def main() -> None:
    demo = build_app()
    demo.queue(default_concurrency_limit=1, max_size=8)  # tek GPU, işler sırayla
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        max_file_size="50mb",
    )


if __name__ == "__main__":
    main()
