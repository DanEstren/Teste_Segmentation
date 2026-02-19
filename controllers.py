import os
import cv2
import numpy as np
import json
import asyncio
from ultralytics import FastSAM
import camera

os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

class InferenceService:
    def __init__(self):
        self.model = FastSAM('FastSAM-s.pt')
        # REMOVIDO: self.camera = cv2.VideoCapture(0) 
        self.current_crop = None 

    def update_crop(self, box_data):
        self.current_crop = box_data

    async def generate_inference_stream(self):
        while True:
            # Pegamos o frame usando a função segura e com .copy()
            frame = camera.get_frame()
            
            if frame is None:
                await asyncio.sleep(0.05) 
                continue

            if self.current_crop:
                x = self.current_crop['x']
                y = self.current_crop['y']
                w = self.current_crop['w']
                h = self.current_crop['h']
                
                img_h, img_w = frame.shape[:2]
                x, y = max(0, x), max(0, y)
                w, h = min(w, img_w - x), min(h, img_h - y)

                crop_img = frame[y:y+h, x:x+w]

                if crop_img.size > 0:
                    results = self.model(crop_img, device='cpu', conf=0.4, verbose=False)
                    
                    if results[0].masks is not None:
                        mask = results[0].masks.data[0].cpu().numpy()
                        mask = cv2.resize(mask, (w, h))
                        
                        color_mask = np.zeros_like(crop_img)
                        color_mask[mask > 0.5] = [0, 255, 0] 
                        
                        blended_crop = cv2.addWeighted(crop_img, 1, color_mask, 0.5, 0)
                        frame[y:y+h, x:x+w] = blended_crop

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Pequeno respiro para o AsyncIO não travar
            await asyncio.sleep(0.01)

inference_service = InferenceService()