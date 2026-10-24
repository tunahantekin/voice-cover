# voice-cover

Ekrandaki 2-3 cümleyi okuyup sesini kaydediyorsun, bir şarkı ekliyorsun. Çıktı, o şarkının senin sesinle söylenmiş hali. Eğitim yok, zero-shot voice conversion kullanılıyor.

## Nasıl çalışır

1. Ses kaydın kontrol edilir (süre, sessizlik, cızırtı) ve hazırlanır.
2. Demucs şarkıyı vokal ve müzik olarak ayırır.
3. Vokalin en yoğun bölümü seçilir (varsayılan 30 sn).
4. Seed-VC vokali senin sesine dönüştürür. Oktav, sesinin perdesine göre otomatik seçilir.
5. Vokal müzikle yeniden birleştirilir. Ton değiştirilirse müzik de aynı kadar kaydırılır.

> Durum: çalışan ilk sürüm. GTX 1650'de 30 sn yaklaşık 8 dakika sürüyor.

## Kurulum

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128   # ortamına uygun sürüm
pip install -r requirements-dev.txt
python scripts/setup_seedvc.py      # Seed-VC'yi ayrı bir venv'e kurar (Python 3.10 gerekir)
```

## Kullanım

```bash
python app.py                                                  # arayüz: http://127.0.0.1:7860
python cli.py --ref ses.wav --song sarki.mp3 --out cover.wav   # komut satırı
pytest -m "not slow"                                           # hızlı testler
```

## Kullanılan bileşenler ve lisanslar

| Bileşen | Görev | Lisans |
|---|---|---|
| Seed-VC | Ses dönüştürme | GPL-3.0 (kod ve model ağırlıkları). Kod kopyalanmaz, `third_party/` altına klonlanıp subprocess ile çağrılır. |
| BigVGAN (nvidia/bigvgan_v2_44khz_128band_512x) | Vocoder | MIT |
| whisper-small (openai) | İçerik çıkarımı | Apache-2.0 |
| CAM++ (funasr/campplus) | Konuşmacı gömmesi | Apache-2.0 |
| RMVPE (lj1995/VoiceConversionWebUI) | Perde (F0) çıkarımı | MIT + "yalnızca araştırma amaçlı" ek madde. Ticari kullanıma uygun değil. Yazılımı kullanan ve üretilen sesleri yayan kişi sorumludur. |
| Demucs | Vokal ayırma | MIT |
| Gradio | Arayüz | Apache-2.0 |
| yt-dlp | Linkten şarkı indirme | Unlicense |
| librosa | Ses işleme | ISC |
| pyloudnorm | Ses yüksekliği | MIT |
| soundfile, NumPy, PyTorch, torchaudio | Ses okuma/yazma, hesaplama | BSD-3-Clause |

## Lisans

Bu projenin kendi kodu [GPL-3.0](LICENSE) lisanslıdır. Yukarıdaki bileşenler kendi lisanslarına tabidir.

## Ticari olmayan kullanım

Bu proje yalnızca ticari olmayan, kişisel ve araştırma amaçlı kullanım içindir. RMVPE'nin "yalnızca araştırma amaçlı" ek maddesi nedeniyle projeyi ya da ürettiği sesleri ticari bir üründe veya hizmette kullanma. Seed-VC'nin GPL-3.0 lisansı ayrıca, dağıtılan türev çalışmaların kaynak kodunun aynı lisansla paylaşılmasını şart koşar.

## Etik ve gizlilik

Yalnızca kendi sesini ya da izin aldığın birinin sesini kullan. Yüklenen sesler işlem bitince silinir ve arayüz kullanmadan önce rıza onayı ister. Şarkıların telif hakları sahiplerine aittir.
