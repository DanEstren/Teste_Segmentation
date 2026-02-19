import os
import cv2
import numpy as np
import json
import asyncio
import time
from ultralytics import SAM
import camera

os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

class InferenceService:
    def __init__(self):
        # Carregue o modelo e envie para a GPU corretamente
        self.model = SAM("sam2.1_t.pt").to(device='cuda')
        self.current_crop = None 
        
        # --- Estados do Cache e Freeze ---
        self.is_frozen = False
        self.frozen_clean_frame = None       # Imagem original limpa (para salvar)
        self.frozen_processed_frame = None   # Imagem com o desenho verde em cima (para exibir)
        self.current_labels = []             # Guarda as coordenadas calculadas
        self.needs_processing = False        # Trava: Só roda a IA se for True

    def update_crop(self, box_data):
        self.current_crop = box_data
        # O usuário mandou uma nova caixa! Liberamos a trava para a IA rodar 1 vez
        self.needs_processing = True

    def toggle_freeze(self):
        """Congela ou descongela o vídeo"""
        self.is_frozen = not self.is_frozen
        
        if self.is_frozen:
            self.frozen_clean_frame = camera.get_frame()
            if self.frozen_clean_frame is not None:
                # Inicia o frame processado igual ao limpo
                self.frozen_processed_frame = self.frozen_clean_frame.copy()
            
            self.current_labels = []
            self.needs_processing = False
        else:
            # Limpa tudo da memória ao descongelar
            self.frozen_clean_frame = None
            self.frozen_processed_frame = None
            self.current_crop = None
            self.current_labels = []
            self.needs_processing = False
            
        return self.is_frozen

    def save_dataset(self):
        """Salva a imagem limpa e as labels no disco"""
        if not self.is_frozen or self.frozen_clean_frame is None:
            return {"status": "error", "message": "A tela precisa estar congelada."}
        if not self.current_labels:
            return {"status": "error", "message": "Nenhum objeto segmentado."}

        timestamp = int(time.time())
        filename = f"img_{timestamp}"

        img_path = f"dataset/images/{filename}.jpg"
        label_path = f"dataset/labels/{filename}.txt"

        # Salva o frame LIMPO
        cv2.imwrite(img_path, self.frozen_clean_frame)
        
        with open(label_path, "w") as f:
            f.write("\n".join(self.current_labels))

        return {"status": "success", "message": f"Salvo como {filename}"}

    async def generate_inference_stream(self):
        while True:
            # SE A TELA ESTIVER CONGELADA
            if self.is_frozen and self.frozen_clean_frame is not None:
                
                # SÓ RODA A IA SE A TRAVA ESTIVER LIBERADA (Usuário acabou de desenhar)
                if self.needs_processing and self.current_crop:
                    # Copia do frame limpo para podermos desenhar do zero sem borrar a imagem
                    processing_frame = self.frozen_clean_frame.copy()
                    
                    x = self.current_crop['x']
                    y = self.current_crop['y']
                    w = self.current_crop['w']
                    h = self.current_crop['h']
                    
                    img_h, img_w = processing_frame.shape[:2]
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, img_w - x), min(h, img_h - y)
                    
                    crop_img = processing_frame[y:y+h, x:x+w]
                    
                    if crop_img.size > 0:
                        # Roda o SAM2
                        results = self.model(crop_img, conf=0.4, verbose=False)
                        self.current_labels = []
                        
                        if results[0].masks is not None:
                            mask = results[0].masks.data[0].cpu().numpy()
                            # --- FIX DO SAM2 (bool para uint8) ---
                            mask = mask.astype(np.uint8) 
                            
                            mask = cv2.resize(mask, (w, h))
                            
                            color_mask = np.zeros_like(crop_img)
                            color_mask[mask > 0] = [0, 255, 0] 
                            blended_crop = cv2.addWeighted(crop_img, 1, color_mask, 0.5, 0)
                            processing_frame[y:y+h, x:x+w] = blended_crop
                            
                            mask_points = results[0].masks.xy 
                            for segment in mask_points:
                                global_segment = []
                                for point in segment:
                                    norm_x = (point[0] + x) / img_w
                                    norm_y = (point[1] + y) / img_h
                                    global_segment.extend([norm_x, norm_y])
                                line = f"0 {' '.join(map(str, global_segment))}"
                                self.current_labels.append(line)
                                
                    cv2.rectangle(processing_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    
                    # Atualiza o frame em cache com os novos desenhos e FECHA A TRAVA
                    self.frozen_processed_frame = processing_frame
                    self.needs_processing = False 
                
                # Pega a imagem que já está pronta na memória (Custo de CPU/GPU zero)
                frame = self.frozen_processed_frame
            
            # SE A TELA NÃO ESTIVER CONGELADA
            else:
                # Apenas exibe o vídeo rodando liso sem inferência
                frame = camera.get_frame()
            
            if frame is None:
                await asyncio.sleep(0.05)
                continue
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Coloquei o sleep em 0.03 para limitar o stream a ~30FPS. Economiza muita CPU.
            await asyncio.sleep(0.03) 

inference_service = InferenceService()