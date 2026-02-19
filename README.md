# FastSAM Auto-Annotator 🎯

Uma ferramenta **Human-in-the-loop** para criação acelerada de datasets de segmentação. O usuário seleciona a região de interesse (Bounding Box) no frontend, e o backend utiliza o **FastSAM (Segment Anything Model)** para gerar máscaras de segmentação precisas, convertendo-as automaticamente para o sistema de coordenadas da imagem original.

## 🚀 Funcionalidades

* **Seleção Interativa:** Interface Web simples para desenhar Bounding Boxes sobre imagens.
* **Segmentação Assistida por IA:** Utiliza o modelo `FastSAM-s.pt` (ou `yolov8-seg`) para segmentar objetos dentro do crop.
* **Mapeamento de Coordenadas:** Algoritmo inteligente que traduz a máscara do "crop" de volta para a resolução original da imagem.
* **Dataset Ready:** Salva automaticamente:
* Imagem original em `dataset/images/`
* Labels no formato YOLO Segmentation em `dataset/labels/`



## 🛠️ Arquitetura e Lógica

O diferencial deste projeto é a preservação da resolução. Ao invés de redimensionar a imagem inteira para a entrada da IA (o que causaria perda de detalhes em objetos pequenos), o sistema funciona assim:

1. **Crop:** O Frontend envia apenas as coordenadas e a imagem original.
2. **Inference:** O Backend recorta a imagem em alta resolução.
3. **Segmentation:** O FastSAM processa apenas o recorte (maximiza a densidade de pixels).
4. **Recalculation:** As coordenadas da máscara  são convertidas para  usando o offset do crop:



## 📦 Estrutura do Projeto

```bash
meu_projeto/
│
├── dataset/             # Dados gerados (ignorado no git)
│   ├── images/          # Imagens originais salvas
│   └── labels/          # Arquivos .txt com segmentação YOLO
│
├── static/
│   └── index.html       # Frontend (Canvas + JS)
│
├── weights/             # Pesos do modelo
│   └── FastSAM-s.pt     # (Baixado automaticamente ou manual)
│
├── main.py              # Backend FastAPI
├── requirements.txt     # Dependências
└── README.md            # Documentação

```

## 🔧 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/fastsam-annotator.git
cd fastsam-annotator

```

### 2. Crie um ambiente virtual (Recomendado)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

```

### 3. Instale as dependências

Crie um arquivo `requirements.txt` com o conteúdo abaixo e instale:

```text
fastapi
uvicorn
python-multipart
opencv-python
ultralytics
supervision
numpy

```

```bash
pip install -r requirements.txt

```

### 4. Baixe o Modelo

O código baixará automaticamente o `FastSAM-s.pt` na primeira execução, ou você pode baixá-lo manualmente e colocar na raiz.

## ▶️ Como Usar

1. **Inicie o Servidor:**
```bash
uvicorn main:app --reload

```


2. **Acesse a Interface:**
Abra o navegador em `http://127.0.0.1:8000/static/index.html`.
3. **Fluxo de Trabalho:**
* Clique em "Escolher arquivo" e carregue uma imagem.
* Desenhe um retângulo vermelho ao redor do objeto que deseja segmentar.
* Clique em **"Enviar Crop para Segmentar"**.
* Verifique a pasta `dataset/labels/` para ver o arquivo `.txt` gerado.



## ⚙️ Configuração do Modelo (Main.py)

No arquivo `main.py`, você pode alternar entre modelos dependendo da necessidade:

```python
# Para objetos genéricos (Recomendado)
from ultralytics import FastSAM
model = FastSAM('FastSAM-s.pt')

# Para objetos comuns (COCO Dataset: Carro, Pessoa, etc.)
# from ultralytics import YOLO
# model = YOLO('yolov8n-seg.pt')

```

## 📝 Formato de Saída (Labels)

Os arquivos `.txt` são salvos no formato padrão YOLO Segmentation:

```text
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

```

* Tudo normalizado entre 0 e 1.
* `class_id` padrão é `0`.

## 🤝 Contribuição

Sinta-se à vontade para abrir Issues ou Pull Requests para melhorar a interface do frontend ou adicionar suporte a múltiplas classes.

---

**Desenvolvido com FastAPI e Ultralytics.**
