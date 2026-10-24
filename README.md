# voice-cover

Ekrandaki 2-3 cümleyi okuyup sesini kaydediyorsun, bir şarkı ekliyorsun. Çıktı, o şarkının senin sesinle söylenmiş hali. Eğitim yok, zero-shot voice conversion kullanılıyor.

## Nasıl çalışır

1. Referans ses doğrulanır.
2. Demucs şarkıyı vokal ve enstrümantal olarak ayırır.
3. Seed-VC vokali senin sesine dönüştürür.
4. Vokal ve enstrümantal yeniden miks edilir.

> Durum: iskelet aşaması, iş mantığı henüz yazılmadı.

## Kurulum

```bash
pip install torch            # ortamına uygun sürüm
pip install -r requirements.txt
python scripts/setup_seedvc.py
python app.py
```

## Kullanılan bileşenler ve lisanslar

| Bileşen | Görev | Lisans |
|---|---|---|
| Seed-VC | Ses dönüştürme | GPL-3.0 (kod ve model ağırlıkları). Kod kopyalanmaz, `third_party/` altına klonlanıp subprocess ile çağrılır. |
| BigVGAN (nvidia/bigvgan_v2_44khz_128band_512x) | Vocoder | MIT |
| whisper-small (openai) | İçerik çıkarımı | Apache-2.0 |
| CAM++ (funasr/campplus) | Konuşmacı gömmesi | Apache-2.0 |
| RMVPE (lj1995/VoiceConversionWebUI) | Perde (F0) çıkarımı | MIT + "yalnızca araştırma amaçlı" ek madde. Ticari kullanıma uygun değil. Yazılımı kullanan ve üretilen sesleri yayan kişi sorumludur. |
| Demucs | Vokal ayırma | Doğrulanacak |
| Gradio | Arayüz | Doğrulanacak |
| yt-dlp | Linkten şarkı indirme | Doğrulanacak |
| librosa | Ses işleme | Doğrulanacak |
| pyloudnorm | Ses yüksekliği | Doğrulanacak |

## Ticari olmayan kullanım

Bu proje yalnızca ticari olmayan, kişisel ve araştırma amaçlı kullanım içindir. RMVPE'nin "yalnızca araştırma amaçlı" ek maddesi nedeniyle projeyi ya da ürettiği sesleri ticari bir üründe veya hizmette kullanma. Seed-VC'nin GPL-3.0 lisansı ayrıca, dağıtılan türev çalışmaların kaynak kodunun aynı lisansla paylaşılmasını şart koşar.

## Etik ve gizlilik

Yalnızca kendi sesini ya da izin aldığın birinin sesini kullan. Yüklenen sesler işlem bitince silinir ve arayüz kullanmadan önce rıza onayı ister. Şarkıların telif hakları sahiplerine aittir.
