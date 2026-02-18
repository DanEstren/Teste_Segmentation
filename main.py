import os
import cv2
import numpy as np
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import FastSAM #, YOLO

app = FastAPI()

# Configuração de CORS e arquivos estáticos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Criar pastas se não existirem
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

# Carregar modelo (Pode usar FastSAM-s.pt ou yolov8n-seg.pt)
# Para segmentação genérica, o FastSAM é melhor, mas o v8-seg é mais rápido para teste.
# model = YOLO('yolov8n-seg.pt')  
model = FastSAM('FastSAM-s.pt')

@app.post("/segment")
async def segment_crop(
    file: UploadFile = File(...),
    box: str = Form(...) # Recebe string JSON com {x, y, w, h}
):
    # 1. Ler a imagem enviada
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    org_h, org_w = original_img.shape[:2]

    # 2. Processar o Bounding Box do Frontend
    crop_data = json.loads(box)
    x, y, w, h = int(crop_data['x']), int(crop_data['y']), int(crop_data['w']), int(crop_data['h'])

    # Validar limites
    x = max(0, x); y = max(0, y)
    w = min(w, org_w - x); h = min(h, org_h - y)

    # 3. Fazer o Crop (Recorte)
    crop_img = original_img[y:y+h, x:x+w]

    # 4. Rodar a IA APENAS no Crop
    # retina_masks=True garante melhor qualidade na segmentação
    results = model(crop_img, retina_masks=True) 

    segments_to_save = []

    # 5. Processar resultados e converter coordenadas
    if results[0].masks is not None:
        # Pega os polígonos (pontos x,y) na escala do CROP
        masks = results[0].masks.xy 
        
        for segment in masks:
            # --- AQUI É O PULO DO GATO ---
            # O 'segment' tem coordenadas (0,0) até (w,h) do crop.
            # Precisamos somar o offset (x,y) inicial do crop.
            
            global_segment = []
            for point in segment:
                gx = point[0] + x  # Soma o offset X
                gy = point[1] + y  # Soma o offset Y
                
                # Normalizar para formato YOLO (0.0 a 1.0 relativo à imagem ORIGINAL)
                norm_x = gx / org_w
                norm_y = gy / org_h
                global_segment.extend([norm_x, norm_y])
            
            # Formato YOLO Segmentation: <class_id> <x1> <y1> <x2> <y2> ...
            # Vamos assumir classe 0 para tudo neste exemplo
            line = f"0 {' '.join(map(str, global_segment))}"
            segments_to_save.append(line)

    # 6. Salvar no disco
    filename = file.filename.split('.')[0]
    
    # Salva Imagem Original
    img_path = f"dataset/images/{filename}.jpg"
    cv2.imwrite(img_path, original_img)
    
    # Salva Label (.txt)
    label_path = f"dataset/labels/{filename}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(segments_to_save))

    return {
        "status": "success", 
        "segments_found": len(segments_to_save),
        "saved_at": img_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)