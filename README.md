# 🇺🇿 Whisper Uzbek Speech-to-Text Fine-Tuning
[![Google Colab](https://img.shields.io/badge/Open%20in-Colab-orange?logo=googlecolab)]([https://colab.research.google.com](https://github.com/AbduxoliqUzoqov/speech_whisper/blob/main/testTronsform.ipynb))

Ushbu loyiha OpenAI **Whisper-Small** modelini o'zbek tili nutqini tanish uchun maxsus fine-tune qilishga mo'ljallangan. Loyiha Google Colab kabi resurslari cheklangan muhitlarda samarali ishlash uchun **Incremental Learning** (bosqichma-bosqich o'qitish) tizimi bilan jihozlangan.

## 🚀 Asosiy xususiyatlar

*   **Incremental Learning (Bosqichma-bosqich o'qitish)**: Colab-ning vaqt cheklovlaridan o'tish uchun modelni qismlarga bo'lib o'qitish imkoniyati.
*   **Automatic Cleanup**: Training tugagach, keraksiz checkpointlarni o'chirib, faqat eng yaxshi (Best Model) versiyasini saqlash.
*   **Anti-Hallucination**: "Sizlar" kabi asossiz takrorlanadigan so'zlarni kamaytirish uchun *Label Smoothing* va *Weight Decay* algoritmlari.
*   **Robust Text Normalizer**: O'zbek nutqidagi maxsus harflarni (o‘, g‘, sh, ch) to'g'ri qayta ishlash.
*   **8-bit Optimization**: VRAM-ni tejash uchun `bitsandbytes` orqali 8-bitli optimizatsiya.

## 🛠 Colab-da ishga tushirish (To'liq ketma-ketlik)

### 1. Loyihani birinchi marta yuklab olish (Faqat 1 marta)
```bash
!git clone https://github.com/AbduxoliqUzoqov/speech_whisper.git
%cd speech_whisper
```

### 2. Loyihani yangilash (Har safar yangi kod chiqqanda)
```bash
!git pull origin main
```

### 3. Kutubxonalarni o'rnatish
```bash
!pip install -U datasets transformers[torch] librosa evaluate jiwer accelerate bitsandbytes ffmpeg-python
```

## 📈 Modelni o'qitish (Incremental Training)

### 1. Birinchi o'qitish bosqichi
```python
from train_whisper import WhisperUzbekManager

manager = WhisperUzbekManager("openai/whisper-small")
train_ds, test_ds = manager.prepare_dataset(num_samples=2000, seed=1)
model = manager.load_model()
trainer = manager.run_training(train_ds, test_ds, model)
manager.save_final_model(trainer, "whisper-uz-v1")
```

### 2. O'qitishni yangi ma'lumotlar bilan davom ettirish
```python
# Avvalgi o'qitilgan modelni asos qilib olamiz
manager = WhisperUzbekManager("./whisper-uz-v1") 
# Yangi seed (seed=2) - yangi tasodifiy ma'lumotlarni beradi
train_ds, test_ds = manager.prepare_dataset(num_samples=2000, seed=2)
model = manager.load_model()
trainer = manager.run_training(train_ds, test_ds, model)
manager.save_final_model(trainer, "whisper-uz-v2")
```

## 🧪 Test qilish

```python
# Modelni audio fayl orqali tekshirish
# model_path: o'qitilgan model papkasi
# audio_input: fayl yo'li, numpy array yoki URL
result = manager.quick_test("./whisper-uz-v2", "mening_ovozim.wav")
print(f"Natija: {result}")
```

## 📂 Loyiha strukturasi

*   `train_whisper.py`: Asosiy boshqaruv klassi (`WhisperUzbekManager`) joylashgan fayl.
*   `README.md`: Loyiha haqida ma'lumot.
*   `results/`: Training vaqtidagi checkpointlar (avtomatik tozalanadi).

## 🔄 Ish jarayoni (Workflow)

| Bosqich | Vazifasi | Natija |
| :--- | :--- | :--- |
| **Data Prep** | 2000 ta tasodifiy audio tanlash | `train_ds`, `test_ds` |
| **Training** | Whisper-Small modelini o'qitish | `checkpoint-XXX` |
| **Cleanup** | Faqat eng yaxshi modelni ajratib olish | `whisper-uz-v1` |
| **Next Step** | Yangi ma'lumot bilan davom ettirish | `whisper-uz-v2` |

## 📊 Natijalar (Benchmark)
Loyiha 2000 ta sample bilan o'qitilganda quyidagi natijalarni ko'rsatdi:
- **WER (Word Error Rate):** ~35.6%
- **Training Time:** ~60-70 min (Google Colab T4 GPU)

---
**Muallif:** [Abduxoliq Uzoqov](https://github.com/AbduxoliqUzoqov)
