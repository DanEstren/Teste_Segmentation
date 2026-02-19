import cv2
import threading
import time
import numpy as np

latest_frame = None
lock = threading.Lock() 
running = True

def camera_thread_logic():
    global latest_frame, running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    while running:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame
        else:
            print("⚠️ Câmera perdeu sinal. Reconectando em 2s...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def start_camera_loop():
    # Removi o 'async', pois threads rodam fora do event loop do asyncio
    t = threading.Thread(target=camera_thread_logic, daemon=True)
    t.start()
    print("📸 Câmera iniciada em Thread separada")

def get_frame():
    """Retorna uma CÓPIA segura do frame atual usando o Lock"""
    global latest_frame
    with lock:
        if latest_frame is not None:
            return latest_frame.copy() # O .copy() evita corromper a memória
        return None