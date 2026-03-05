import requests
import base64
from pathlib import Path
from tools.screen_capture import get_screenshot
from agent.llm_utils.utils import encode_image

OUTPUT_DIR = "./tmp/outputs"

class OmniParserClient:
    def __init__(self, url: str) -> None:
        self.url = url

    def __call__(self):
        # 1. Screenshot erstellen
        screenshot, screenshot_path = get_screenshot()
        screenshot_path = str(screenshot_path)
        image_base64 = encode_image(screenshot_path)
        
        # 2. Daten vom Server abrufen
        response = requests.post(self.url, json={"base64_image": image_base64})
        response_json = response.json()
        
        # Latenz-Anzeige beibehalten
        if 'latency' in response_json:
            print('omniparser latency:', response_json['latency'])

        # --- WICHTIG: UUID & PFADE (Deine Original-Logik + Fix) ---
        screenshot_path_uuid = Path(screenshot_path).stem.replace("screenshot_", "")
        # Wir setzen beide Keys, damit sowohl deine Pfade als auch der Agent (KeyError) zufrieden sind
        response_json['screenshot_uuid'] = screenshot_path_uuid
        response_json['screenshot_path_uuid'] = screenshot_path_uuid

        # --- S.O.M. BILD SPEICHERN (Deine Original-Logik) ---
        if 'som_image_base64' in response_json:
            som_image_data = base64.b64decode(response_json['som_image_base64'])
            som_screenshot_path = f"{OUTPUT_DIR}/screenshot_som_{screenshot_path_uuid}.png"
            with open(som_screenshot_path, "wb") as f:
                f.write(som_image_data)
        
        # 3. Metadaten zuweisen
        response_json['width'] = screenshot.size[0]
        response_json['height'] = screenshot.size[1]
        response_json['original_screenshot_base64'] = image_base64
        
        # 4. Koordinaten retten und Text formatieren
        # Wir übergeben das gesamte JSON an reformat, damit dort alles geregelt wird
        response_json = self.reformat_messages(response_json)
        
        return response_json
    
    def reformat_messages(self, response_json: dict):
        screen_info = ""
        content_list = response_json.get("parsed_content_list", [])
        
        final_coordinates = []
        
        for idx, element in enumerate(content_list):
            if isinstance(element, dict):
                # Koordinaten für den Executor sichern
                coords = element.get("bbox") or element.get("box") or [0, 0, 0, 0]
                final_coordinates.append(coords)
                
                # Text für die KI bauen
                content = element.get("content", "")
                e_type = element.get("type", "element")
                screen_info += f'ID: {idx}, {e_type}: {content}\n'
                
            elif isinstance(element, str):
                screen_info += f'ID: {idx}, {element}\n'
                # Platzhalter, falls es nur ein String ist
                final_coordinates.append(None)

        # --- VERBESSERTER FIX FÜR DIE LEERE LISTE ---
        if any(c is not None for c in final_coordinates):
            response_json['coordinates'] = [c if c else [0,0,0,0] for c in final_coordinates]
        else:
            # Fallback auf verschiedene mögliche Keys, die OmniParser-Forks gerne nutzen
            response_json['coordinates'] = (
                response_json.get('boxes') or 
                response_json.get('bboxes') or 
                response_json.get('coordinate') or 
                []
            )

        # --- NEU: SICHERHEITS-CHECK & DEBUGGING ---
        if not response_json.get('coordinates'):
            print(f">>> CLIENT ERROR: Keine Koordinaten vom Server erhalten!")
            print(f">>> EMPFANGENE KEYS VOM SERVER: {list(response_json.keys())}")

        # Finale Text-Info für das LLM speichern
        response_json['screen_info'] = screen_info
        return response_json