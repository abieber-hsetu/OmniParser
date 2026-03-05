'''
python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05
'''

import sys
import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
import argparse
import uvicorn

os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.1'
os.environ['FLAGS_selected_gpus'] = '0'
os.environ['FLAGS_cudnn_deterministic'] = 'True'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.omniparser import Omniparser

def parse_arguments():
    parser = argparse.ArgumentParser(description='Omniparser API')
    parser.add_argument('--som_model_path', type=str, default='../../weights/icon_detect_v1_5/model_v1_5.pt', help='Path to the som model')
    parser.add_argument('--caption_model_name', type=str, default='florence2', help='Name of the caption model')
    parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence', help='Path to the caption model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.05, help='Threshold for box detection')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host for the API')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API')
    args = parser.parse_args()
    return args

args = parse_arguments()
config = vars(args)

app = FastAPI()
omniparser = Omniparser(config)

class ParseRequest(BaseModel):
    base64_image: str

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()
    
    # Unsere 3 Variablen abgreifen
    dino_labled_img, parsed_content_list, coordinates = omniparser.parse(parse_request.base64_image)
    
    latency = time.time() - start
    print('time:', latency)
    
    # --- NEU: Koordinaten sicher für JSON machen ---
    safe_coords = coordinates
    if hasattr(coordinates, 'tolist'):
        # Wenn es ein einzelnes großes Numpy-Array/Tensor ist
        safe_coords = coordinates.tolist()
    elif isinstance(coordinates, dict):
        # Wenn es ein Dictionary ist (z.B. {0: [x,y,w,h], 1: ...})
        safe_coords = {str(k): v.tolist() if hasattr(v, 'tolist') else v for k, v in coordinates.items()}
    elif isinstance(coordinates, list):
        # Wenn es eine Liste mit Numpy-Arrays darin ist
        safe_coords = [c.tolist() if hasattr(c, 'tolist') else c for c in coordinates]
        
    return {
        "som_image_base64": dino_labled_img, 
        "parsed_content_list": parsed_content_list, 
        "boxes": safe_coords,  # <-- Hier nutzen wir die sichere Version
        "latency": latency
    }

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)
