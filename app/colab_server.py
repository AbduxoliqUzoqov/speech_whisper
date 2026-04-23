# colab_server.py
# Bu fayldagi kodni to'lig'icha ko'chirib, Google Colab'dagi bitta katakka (cell) yozib ishlating!

import os
import os
import sys
import shutil
import subprocess
import nest_asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch

# 1. Kutubxonalarni avtomatik o'rnatish
print("Kutubxonalar o'rnatilmoqda...")
os.system("pip install fastapi uvicorn python-multipart nest-asyncio transformers datasets evaluate jiwer accelerate bitsandbytes librosa ffmpeg-python")

# 2. Cloudflared binar faylini yuklab olish va o'rnatish
if not os.path.exists("cloudflared"):
    print("Cloudflare o'rnatilmoqda...")
    os.system("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared")
    os.system("chmod +x cloudflared")

# 3. FastAPI ilovasini yaratish va CORS (Ruxsat) berish
app = FastAPI(title="Whisper Uzbek Real-time API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# API ishga tushishidan oldin Modelni XOTIRAGA BIR MARTA yuklab olamiz
# ---------------------------------------------------------
model_path = "./whisper-uz-v1" 
if not os.path.exists(model_path):
    print(f"DIQQAT: {model_path} topilmadi! Standart 'openai/whisper-small' yuklanmoqda...")
    model_path = "openai/whisper-small"

print(f"Model ({model_path}) xotiraga yuklanmoqda, iltimos kuting...")
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_path, language="Uzbek", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    print("Model muvaffaqiyatli yuklandi! 🚀")
except Exception as e:
    print(f"Model yuklashda xato: {e}")
    processor = None
    model = None

@app.get("/")
def home():
    return {"status": "success", "message": "Whisper API ishlayapti!"}

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    print(f"Yangi audio qabul qilindi: {file.filename}")
    
    if model is None or processor is None:
        return {"fayl_nomi": file.filename, "matn": "Xatolik: Model xotiraga yuklanmagan!"}

    # 1. Kelgan webm/audio faylni vaqtincha saqlash
    webm_location = f"temp_{file.filename}"
    
    with open(webm_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Librosa yordamida audioni numpy array ga o'g'iramiz
        import librosa
        audio_array, sr = librosa.load(webm_location, sr=16000)
        
        # 3. Model orqali sof tensorlar bilan tarjima qilish (Pipeline xatosini aylanib o'tish)
        input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="uzbek", task="transcribe")
        
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=225)
        matn = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
    except Exception as e:
        print(f"Xatolik yuz berdi: {e}")
        import traceback
        traceback.print_exc()
        matn = f"Xatolik yuz berdi: {e}"
        
    finally:
        # 4. Vaqtinchalik audio fayllarni tozalash
        if os.path.exists(webm_location): os.remove(webm_location)
        if os.path.exists(wav_location): os.remove(wav_location)
            
    return {"fayl_nomi": file.filename, "matn": matn}

# 4. Cloudflare Tunnel'ni orqa fonda ishga tushirish funksiyasi
def start_cloudflared(port):
    print("Cloudflare Tunnel ishga tushirilmoqda...")
    # cloudflared ni orqa fonda (background) port 8000 ga ulab ishga tushiramiz
    proc = subprocess.Popen(['./cloudflared', 'tunnel', '--url', f'http://127.0.0.1:{port}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    import time
    time.sleep(3) # Loglar yozilishi uchun biroz kutamiz
    
    # Loglarni o'qib, URL manzilni topamiz
    import urllib.request
    try:
        # trycloudflare o'z logini shunday o'qiydi
        with open("/content/nohup.out", "a") as f:
            pass # just to make sure it exists
    except:
        pass

    # Cloudflare URL ni chiqarib berish kodi
    print("\n" + "="*60)
    print("API MANZILI QIDIRILMOQDA...")
    print("Agar link chiqmasa, terminalda cloudflared loglaridan .trycloudflare.com linkini toping.")
    
    import re
    import threading
    
    def read_stderr():
        for line in proc.stderr:
            line_str = line.decode('utf-8')
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line_str)
            if match:
                print("\n🔥 SIZNING API MANZILINGIZ (Veb-saytga nusxalang!):")
                print(match.group(0))
                print("="*60 + "\n")
                break
                
    threading.Thread(target=read_stderr, daemon=True).start()
    return proc

# 5. Serverni ishga tushirish
if __name__ == "__main__":
    start_cloudflared(8000)
    nest_asyncio.apply()
    print("FastAPI serveri 8000-portda ishga tushdi.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
