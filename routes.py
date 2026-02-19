from fastapi import APIRouter, Form, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import json

# Importa a instância do serviço que criamos acima
from controllers import inference_service

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# A rota que você propôs! Ela fica "presa" enviando frames infinitamente
@router.get("/video_feed_inference")
async def video_feed_inference():
    return StreamingResponse(
        inference_service.generate_inference_stream(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Rota para o Front-end avisar: "Ei, o usuário desenhou um quadrado aqui!"
@router.post("/set_target")
async def set_target(box: str = Form(...)):
    try:
        box_data = json.loads(box)
        # Atualiza a variável lá dentro da classe
        inference_service.update_crop(box_data)
        return {"status": "success", "message": "Alvo atualizado para inferência"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    # --- NOVAS ROTAS ---
@router.post("/toggle_freeze")
async def toggle_freeze():
    is_frozen = inference_service.toggle_freeze()
    return {"status": "success", "is_frozen": is_frozen}

@router.post("/save_dataset")
async def save_dataset():
    result = inference_service.save_dataset()
    return result