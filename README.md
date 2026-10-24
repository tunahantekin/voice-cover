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

## Bileşenler ve lisanslar

| Bileşen | Görev | Lisans |
|---|---|---|
| Seed-VC | Ses dönüştürme | GPL-3.0 (kopyalanmaz, `third_party/` altına klonlanıp subprocess ile çağrılır) |
| Demucs | Vokal ayırma | MIT |
| Gradio | Arayüz | Apache-2.0 |
| yt-dlp | Linkten şarkı indirme | Unlicense |
| librosa | Ses işleme | ISC |
| pyloudnorm | Ses yüksekliği | MIT |

## Etik ve gizlilik

Yalnızca kendi sesini ya da izin aldığın birinin sesini kullan. Yüklenen sesler işlem bitince silinir ve arayüz kullanmadan önce rıza onayı ister. Şarkıların telif hakları sahiplerine aittir.
