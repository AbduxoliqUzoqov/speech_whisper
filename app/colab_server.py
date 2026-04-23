# colab_server.py
# Bu fayldagi kodni to'lig'icha ko'chirib, Google Colab'dagi bitta katakka (cell) yozib ishlating!

import os
import sys
import shutil
import subprocess
import nest_asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Asosiy papkadagi train_whisper faylini chaqirish uchun yo'lni qo'shamiz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.getcwd())

# 1. Kutubxonalarni avtomatik o'rnatish (Hamma kerakli kutubxonalar)
print("Kutubxonalar o'rnatilmoqda...")
os.system("pip install fastapi uvicorn python-multipart nest-asyncio transformers datasets evaluate jiwer accelerate bitsandbytes librosa")

try:
    from train_whisper import WhisperUzbekManager
except Exception as e:
    print(f"\n❌ XATOLIK: train_whisper.py yuklanmadi. Sababi: {e}\n")

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

# API ishga tushishidan oldin Modelni (Manager) xotiraga yuklab olamiz
model_path = "./whisper-uz-v1"
print(f"Model ({model_path}) xotiraga yuklanmoqda, iltimos kuting...")
try:
    manager = WhisperUzbekManager(model_id=model_path)
except Exception as e:
    print(f"Model yuklashda xato: {e}")
    manager = None

@app.get("/")
def home():
    return {"status": "success", "message": "Whisper API ishlayapti!"}

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    print(f"Yangi audio qabul qilindi: {file.filename}")
    
    if manager is None:
        return {"fayl_nomi": file.filename, "matn": "Xatolik: Model xotiraga yuklanmagan!"}

    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # MANAGER ORQALI TARJIMA QILISH
        matn = manager.quick_test(model_path=model_path, audio_input=file_location)
        
    except Exception as e:
        print(f"Xatolik yuz berdi: {e}")
        matn = f"Xatolik yuz berdi: {e}"
        
    finally:
        # Vaqtinchalik audio faylni o'chirib yuborish
        if os.path.exists(file_location):
            os.remove(file_location)
            
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
