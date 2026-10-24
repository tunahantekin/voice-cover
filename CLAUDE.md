# voice-cover

Kullanıcı ekranda yazan 2-3 cümleyi okuyup ses kaydı verir, sonra bir şarkı ekler. Çıktı, şarkının kullanıcının sesiyle söylenmiş halidir.

## Girdiler

- Referans ses: ekrandaki 2-3 cümlenin okunmasıyla alınan kayıt.
- Şarkı: dosya yükleme ana yol, link ikincil yol.

## Model

- Zero-shot voice conversion, eğitim yok.
- Model: Seed-VC (github.com/Plachtaa/seed-vc), şarkı için F0 koşullu model.
- Repo arşivlenmiş ve GPL-3.0 lisanslı. Kodunu projeye KOPYALAMA. `third_party/` altına klonla ve subprocess ile çağır.

## Pipeline

1. Referans ses doğrulama
2. Demucs ile ayırma (`htdemucs`, `--two-stems vocals`)
3. Seed-VC ile vokal dönüştürme
4. Miks

Vokal yarım ton kaydırılırsa enstrümantal de aynı miktarda kaydırılmalı.

## Kod yapısı

Kod arayüzden bağımsız modüller halinde olmalı.

```
src/voice_cover/
  audio_utils.py
  fetch.py
  separate.py
  convert.py
  mix.py
  pipeline.py
cli.py
app.py
third_party/seed-vc/   (klon, kopya değil)
```

## Arayüz ve yayın

- Gradio, mobil tarayıcıda da çalışmalı.
- Hugging Face Space'e yayınlanacak.
- Yedek olarak Colab notebook.

## Gizlilik

- Yüklenen sesler işlem bitince silinmeli.
- Arayüzde rıza onayı olmalı.
