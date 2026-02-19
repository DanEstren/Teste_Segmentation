import os
import cv2
import numpy as np
import time
import asyncio
from ultralytics import FastSAM # Voltamos para o FastSAM!
import camera

# Garante que as pastas existam
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

class InferenceService:
    def __init__(self):
        # Inicia o FastSAM (a flag de GPU vai na hora da inferência)
        self.model = FastSAM('FastSAM-s.pt')
        self.current_crop = None 
        
        # --- Estados do Cache e Freeze ---
        self.is_frozen = False
        self.frozen_clean_frame = None       # Imagem original limpa (para salvar)
        self.frozen_processed_frame = None   # Imagem com a máscara verde (para exibir)
        self.current_labels = []             # Guarda as coordenadas calculadas no formato YOLO
        self.needs_processing = False        # Trava: Só roda a IA se for True

    def update_crop(self, box_data):
        self.current_crop = box_data
        # O usuário desenhou uma nova caixa. Liberamos a trava para a IA rodar 1 vez!
        self.needs_processing = True

    def toggle_freeze(self):
        """Congela ou descongela o vídeo"""
        self.is_frozen = not self.is_frozen
        
        if self.is_frozen:
            self.frozen_clean_frame = camera.get_frame()
            if self.frozen_clean_frame is not None:
                self.frozen_processed_frame = self.frozen_clean_frame.copy()
            
            self.current_labels = []
            self.needs_processing = False
        else:
            self.frozen_clean_frame = None
            self.frozen_processed_frame = None
            self.current_crop = None
            self.current_labels = []
            self.needs_processing = False
            
        return self.is_frozen

    def save_dataset(self):
        """Salva a imagem limpa e as labels no formato YOLO"""
        if not self.is_frozen or self.frozen_clean_frame is None:
            return {"status": "error", "message": "A tela precisa estar congelada."}
        if not self.current_labels:
            return {"status": "error", "message": "Nenhum objeto segmentado."}

        timestamp = int(time.time())
        filename = f"img_{timestamp}"

        img_path = f"dataset/images/{filename}.jpg"
        label_path = f"dataset/labels/{filename}.txt"

        # Salva o frame LIMPO (sem os desenhos verdes/vermelhos)
        cv2.imwrite(img_path, self.frozen_clean_frame)
        
        with open(label_path, "w") as f:
            f.write("\n".join(self.current_labels))

        return {"status": "success", "message": f"Salvo como {filename}"}

    async def generate_inference_stream(self):
        while True:
            # ==========================================================
            # 1. MODO CONGELADO (Onde a mágica acontece)
            # ==========================================================
            if self.is_frozen and self.frozen_clean_frame is not None:
                
                # Se a trava estiver aberta (usuário acabou de recortar)
                if self.needs_processing and self.current_crop:
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
                        # --- PRÉ-PROCESSAMENTO: CLAHE (Destaca as bordas) ---
                        lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
                        l_channel, a_channel, b_channel = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        cl = clahe.apply(l_channel)
                        limg = cv2.merge((cl, a_channel, b_channel))
                        enhanced_crop = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                        # --- INFERÊNCIA COM FASTSAM ---
                        # Passamos o 'enhanced_crop' pro modelo enxergar melhor.
                        # device=0 força o uso da GPU (se disponível) ou 'cpu'
                        try:
                            results = self.model(enhanced_crop, device=0, conf=0.4, verbose=False)
                        except Exception:
                            # Fallback caso device=0 falhe no seu PyTorch
                            results = self.model(enhanced_crop, device='cpu', conf=0.4, verbose=False)

                        self.current_labels = []
                        
                        if results[0].masks is not None:
                            masks_xy = results[0].masks.xy 
                            masks_data = results[0].masks.data
                            
                            center_x = w / 2
                            center_y = h / 2
                            best_mask_idx = -1
                            min_dist = float('inf')
                            
                            # --- FILTRO POR CENTRO DE MASSA (Centróide) ---
                            for i, segment in enumerate(masks_xy):
                                if len(segment) == 0: 
                                    continue
                                
                                # Calcula o centroide usando Momentos Espaciais do OpenCV
                                M = cv2.moments(segment)
                                if M['m00'] != 0:
                                    cx = int(M['m10'] / M['m00'])
                                    cy = int(M['m01'] / M['m00'])
                                else:
                                    cx, cy = segment[0][0], segment[0][1]
                                
                                # Distância Euclidiana até o centro do crop
                                dist = (cx - center_x)**2 + (cy - center_y)**2
                                
                                if dist < min_dist:
                                    min_dist = dist
                                    best_mask_idx = i
                            
                            # --- APLICA APENAS A MELHOR MÁSCARA ---
                            if best_mask_idx != -1:
                                # Extrai os dados da máscara vencedora
                                mask = masks_data[best_mask_idx].cpu().numpy()
                                
                                # Garante que está no formato correto pro OpenCV
                                if mask.dtype == bool:
                                    mask = mask.astype(np.uint8)
                                    
                                mask = cv2.resize(mask, (w, h))
                                color_mask = np.zeros_like(crop_img)
                                color_mask[mask > 0.5] = [0, 255, 0] 
                                
                                # Desenha a máscara no crop ORIGINAL (não no enhanced)
                                blended_crop = cv2.addWeighted(crop_img, 1, color_mask, 0.5, 0)
                                processing_frame[y:y+h, x:x+w] = blended_crop
                                
                                # Recalcula as coordenadas pro tamanho original da tela (1280x960)
                                best_segment = masks_xy[best_mask_idx]
                                global_segment = []
                                for point in best_segment:
                                    norm_x = (point[0] + x) / img_w
                                    norm_y = (point[1] + y) / img_h
                                    global_segment.extend([norm_x, norm_y])
                                    
                                line = f"0 {' '.join(map(str, global_segment))}"
                                self.current_labels.append(line)
                                
                    # Desenha a caixa vermelha de feedback visual
                    cv2.rectangle(processing_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    
                    # Salva o frame processado no cache e FECHA A TRAVA!
                    self.frozen_processed_frame = processing_frame
                    self.needs_processing = False 
                
                # Exibe o frame congelado (com ou sem máscara) sem gastar CPU/GPU
                frame = self.frozen_processed_frame
            
            # ==========================================================
            # 2. MODO VÍDEO NORMAL (Rodando solto)
            # ==========================================================
            else:
                frame = camera.get_frame()
            
            if frame is None:
                await asyncio.sleep(0.05)
                continue
                
            # Comprime pra JPG e envia via WebSocket/MJPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Trava o FPS do stream para não estourar o uso da CPU (aprox. 30 FPS)
            await asyncio.sleep(0.03) 

inference_service = InferenceService()