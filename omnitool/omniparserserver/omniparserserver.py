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
import io
from PIL import Image, ImageDraw, ImageFont
import base64
import re


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.omniparser import Omniparser


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
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.03, help='Threshold for box detection')
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
    print('Start synchronisiertes Parsing...')
    start = time.time()
    
    # 1. Originalbild laden
    img_data = base64.b64decode(parse_request.base64_image)
    original_img = Image.open(io.BytesIO(img_data)).convert("RGB")
    width, height = original_img.size

    # 2. OmniParser Rohdaten abgreifen
    # Wir ignorieren das vor-beschriftete Bild vom Modell!
    _, raw_texts, raw_coords = omniparser.parse(parse_request.base64_image)

    # 3. Elemente für die räumliche Sortierung sammeln
    elements = []
    for i in range(len(raw_texts)):
        if isinstance(raw_coords, list):
            coord = raw_coords[i] if i < len(raw_coords) else [0,0,0,0]
        else:
            coord = raw_coords.get(str(i), raw_coords.get(i, [0,0,0,0]))
        
        # Alten ID-Präfix aus dem Text entfernen
        clean_text = re.sub(r'^Text Box ID \d+: ', '', str(raw_texts[i])).strip()
        
        elements.append({
            "y": coord[1], 
            "x": coord[0], 
            "content": clean_text,
            "box": coord
        })

    # 4. RÄUMLICHE SORTIERUNG: Oben-Links nach Unten-Rechts (Leserichtung)
    # So wird ID 0 fast immer der Papierkorb oben links
    sorted_elements = sorted(elements, key=lambda e: (e["y"], e["x"]))

    # 5. BILD NEU ZEICHNEN & LISTEN AUFBAUEN
    draw = ImageDraw.Draw(original_img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    final_content_list = []
    sync_screen_info = []
    final_coords = {}

    for i, el in enumerate(sorted_elements):
        new_id = i
        box = el["box"]
        
        # Koordinaten-Skalierung (von 0-1 auf Pixel)
        x0, y0 = box[0] * width, box[1] * height
        x1 = x0 + (box[2] * width) if box[2] < 1.0 else box[2] * width
        y1 = y0 + (box[3] * height) if box[3] < 1.0 else box[3] * height
        
        # Sicherheitshalber min/max für PIL
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        # A) Rahmen & ID-Label auf das Bild zeichnen
        draw.rectangle([xmin, ymin, xmax, ymax], outline="#00FF00", width=2)
        label = str(new_id)
        t_bbox = draw.textbbox((xmin, ymin), label, font=font)
        draw.rectangle([t_bbox[0]-2, t_bbox[1]-2, t_bbox[2]+2, t_bbox[3]+2], fill="#00FF00")
        draw.text((xmin, ymin), label, fill="black", font=font)

        # B) Daten für Rückgabe speichern
        final_content_list.append(el["content"])
        sync_screen_info.append(f"ID {new_id}: {el['content']}")
        final_coords[str(new_id)] = box

    # 6. Fertiges Bild zurück nach Base64
    buffered = io.BytesIO()
    original_img.save(buffered, format="PNG")
    som_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "som_image_base64": som_base64, 
        "parsed_content_list": final_content_list, 
        "screen_info": "\n".join(sync_screen_info),
        "boxes": final_coords,
        "latency": time.time() - start
    }

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)
