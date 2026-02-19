from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routes import router
import webbrowser, threading, uvicorn
from camera import start_camera_loop # Importa a função de start

app = FastAPI(title="FastSAM Auto-Annotator API")

PORT = 7080

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# --- STARTUP EVENT ---
@app.on_event("startup")
def startup_event():
    start_camera_loop() # Inicia a leitura da webcam em background

def abrir_navegador():
    webbrowser.open(f"http://localhost:{PORT}")

if __name__ == "__main__":
    threading.Timer(0.5, abrir_navegador).start()
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)