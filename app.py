
import os
import sys
import uvicorn
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add current directory to path so we can import from infer.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import infer
except ImportError:
    # If app.py is run from root, infer should be importable
    # But just in case
    import importlib.util
    spec = importlib.util.spec_from_file_location("infer", "infer.py")
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)

# Global model variables
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    print("Loading models... This may take a while.")
    
    import torch
    use_gpu = torch.cuda.is_available()
    print(f"Device detected: {'GPU' if use_gpu else 'CPU'}")
    
    # Paths (Hardcoded or Configurable)
    FRCNN_PATH = "models/feature_extracting/pretrained_model/faster_rcnn_res101_vg.pth"
    RELTR_PATH = "data/RelTR_ckpt/checkpoint0149.pth"
    CAPTION_PATH = "result/final_term/log_lstm_reltr/model-best.pth"
    INFOS_PATH = "result/final_term/log_lstm_reltr/infos_reltr-best.pkl"
    
    try:
        # Load Faster R-CNN
        print("Loading Faster R-CNN...")
        models['frcnn'] = infer.load_frcnn(FRCNN_PATH, use_gpu)
        
        # Load RelTR (Always load for demo power)
        print("Loading RelTR...")
        models['reltr'] = infer.load_reltr(RELTR_PATH, use_gpu)
        
        # Load Caption Model
        print("Loading Caption Model...")
        models['caption_model'], models['vocab'], models['opt'] = infer.load_captioner(CAPTION_PATH, INFOS_PATH, use_gpu)
        
        print("-" * 50)
        print("Models loaded successfully!")
        print("Access the Web App here: http://localhost:8000")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error loading models: {e}")
        # We don't exit here so the app can still start (maybe for debugging frontend), 
        # but prediction will fail.
    
    yield
    
    # Clean up (if needed)
    models.clear()

app = FastAPI(title="VisionText AI", lifespan=lifespan)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Files (Frontend)
# Ensure 'static' directory exists
if not os.path.exists("static"):
    os.makedirs("static")

from fastapi.responses import FileResponse

# 1. Mount static assets (CSS, JS) at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Serve index.html at root
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# 3. Explicitly handle other static files if needed at root (like favicon)
@app.get("/{filename}.css")
async def read_css(filename: str):
    return FileResponse(f"static/{filename}.css")

@app.get("/{filename}.js")
async def read_js(filename: str):
    return FileResponse(f"static/{filename}.js")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save temp file
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        import torch
        use_gpu = torch.cuda.is_available()
        
        # Inference Flow
        # 1. Feature Extraction (Visual)
        visual_feats = infer.get_features(models['frcnn'], temp_filename, use_gpu)
        
        # 2. Feature Extraction (Scene Graph)
        rel_feats = infer.get_reltr_features(models['reltr'], temp_filename, use_gpu)
        
        # 3. Generate Caption
        caption = infer.generate_caption(
            models['caption_model'], 
            models['vocab'], 
            visual_feats, 
            rel_feats=rel_feats, 
            use_gpu=use_gpu
        )
        
        # Cleanup
        os.remove(temp_filename)
        
        return {"caption": caption}
        
    except Exception as e:
        # Initial cleanup if failed
        if os.path.exists(f"temp_{file.filename}"):
            os.remove(f"temp_{file.filename}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
