"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""
from collections.abc import Callable

try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

from anthropic import APIResponse
from anthropic.types import (
    TextBlock,
)
from anthropic.types.beta import (
    BetaContentBlock,
    BetaMessage,
    BetaMessageParam
)
from tools import ToolResult

from agent.llm_utils.omniparserclient import OmniParserClient
from agent.anthropic_agent import AnthropicActor
from agent.vlm_agent import VLMAgent
from agent.vlm_agent_with_orchestrator import VLMOrchestratedAgent
from executor.anthropic_executor import AnthropicExecutor
from executor.openai_executor import OpenAIExecutor
import time
import requests
import io
from PIL import Image, ImageDraw, ImageChops, ImageStat
import cv2
import numpy as np
import base64

BETA_FLAG = "computer-use-2024-10-22"

class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OPENAI = "openai"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
    APIProvider.OPENAI: "gpt-5.4",
}

def get_checkbox_status(box, img_bytes):
    """
    Prüft, ob eine Checkbox angehakt ist (unterstützt blaue/schwarze Haken).
    """
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    crop = img.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))
    
    # In HSV umwandeln für bessere Farberkennung
    hsv = np.array(crop.convert('HSV'))
    
    # Bereich für Blau (Hott-Therm Theme)
    # Blau hat meist einen hohen Sättigungsgrad
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Bereich für Schwarz/Dunkelgrau
    # Schwarz hat eine sehr niedrige Helligkeit (V)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # Kombinierte Maske: Ist ein Haken in Blau ODER Schwarz vorhanden?
    combined_mask = cv2.bitwise_or(blue_mask, black_mask)
    
    # Wenn mehr als 5% der Box "farbig" (Haken) sind, gilt sie als CHECKED
    active_pixels = np.count_nonzero(combined_mask)
    total_pixels = combined_mask.size
    return (active_pixels / total_pixels) > 0.05

def wait_for_ui_change(vm_url, timeout=45, sensitivity_threshold=10.0):
    """
    Zwei-Phasen-Wächter: Wartet zuerst auf eine Änderung (Popup) und 
    danach auf eine Stabilität des Bildschirms (Fertig geladen).
    """
    print(f"👀 Starte smarten Zwei-Phasen-Wächter (max {timeout}s)...")
    
    try:
        # Baseline = Der nackte Desktop im Moment des Klicks
        resp = requests.get(f"http://{vm_url}/screenshot", timeout=5)
        baseline_img = Image.open(io.BytesIO(resp.content)).convert('RGB')
    except Exception as e:
        print(f"⚠️ Konnte Baseline-Screenshot nicht laden: {e}. Nutze statische Pause.")
        time.sleep(5)
        return

    start_time = time.time()
    phase = 1 # 1 = Warten auf erste Änderung (Popup), 2 = Warten auf ruhiges Bild (Fertig)
    stable_count = 0
    last_img = baseline_img
    
    while time.time() - start_time < timeout:
        time.sleep(2.0) # 2 Sekunden Takt, um Flackern zu überbrücken
        
        try:
            resp = requests.get(f"http://{vm_url}/screenshot", timeout=5)
            current_img = Image.open(io.BytesIO(resp.content)).convert('RGB')
            
            width, height = baseline_img.size
            draw_base = ImageDraw.Draw(baseline_img)
            draw_curr = ImageDraw.Draw(current_img)
            draw_last = ImageDraw.Draw(last_img)
            
            # Taskleiste schwärzen
            draw_base.rectangle([0, height - 40, width, height], fill="black")
            draw_curr.rectangle([0, height - 40, width, height], fill="black")
            draw_last.rectangle([0, height - 40, width, height], fill="black")

            # Diff zur Baseline (Wie stark weicht es vom originalen Desktop ab?)
            diff_baseline = ImageChops.difference(baseline_img, current_img)
            stat_baseline = ImageStat.Stat(diff_baseline.convert('L'))
            ratio_baseline = (stat_baseline.mean[0] / 255) * 100
            
            if phase == 1:
                if ratio_baseline > sensitivity_threshold:
                    print(f"✨ Ladebildschirm/Popup erkannt ({ratio_baseline:.2f}%). Wechsle in Stabilitäts-Check...")
                    phase = 2
                    stable_count = 0
                else:
                    print(f"⏳ Warte auf Programmstart... (Änderung zum Desktop: {ratio_baseline:.2f}%)")
                    
            elif phase == 2:
                # Diff zum VORHERIGEN Frame (Bewegt sich gerade noch was?)
                diff_consecutive = ImageChops.difference(last_img, current_img)
                stat_consecutive = ImageStat.Stat(diff_consecutive.convert('L'))
                ratio_consecutive = (stat_consecutive.mean[0] / 255) * 100
                
                if ratio_consecutive < 0.5: # Kaum Änderungen = Das Bild steht still
                    # ANTI-DESKTOP-TRICK: Ist es wirklich die App oder nur wieder der leere Desktop?
                    if ratio_baseline > 3.0: 
                        stable_count += 1
                        print(f"🛑 Bild ist stabil ({stable_count}/2).")
                        if stable_count >= 2: # 2 mal hintereinander stabil (ca. 4 Sekunden Stillstand)
                            print("✅ Programm ist vollständig geladen und Einsatzbereit!")
                            time.sleep(1) # Kurzer Sicherheitspuffer
                            return
                    else:
                        print("⚠️ Bild ist stabil, sieht aber wieder aus wie der Desktop. Popup hat sich geschlossen. Warte auf Hauptfenster...")
                        stable_count = 0 # Counter resetten
                else:
                    print(f"⏳ Programm lädt noch / Animationen laufen (Bewegung: {ratio_consecutive:.2f}%)")
                    stable_count = 0
                    
            last_img = current_img.copy()
            
        except Exception as e:
            print(f"Fehler beim dynamischen Polling: {e}")
            pass

    print(f"⏱️ Timeout von {timeout}s erreicht. Gehe davon aus, dass die UI fertig ist.")

def get_checkbox_status(box, img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    width, height = img.size
    
    # Koordinaten-Sicherheit
    x, y, w, h = [int(v) for v in box] # Sicherstellen, dass es Integers sind
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = min(w, width - x)
    h = min(h, height - y)
    
    if w <= 5 or h <= 5: return False
        
    crop = img.crop((x, y, x + w, y + h))
    
    # Konvertierung für OpenCV
    hsv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2HSV)
    
    # BLAU-DOMINANZ-MASKE (Für Hott-Therm Checkboxen)
    # Sucht nach dem Blau, das die Checkbox ausfüllt
    lower_blue = np.array([100, 100, 50]) 
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Debug: Speichern im aktuellen Arbeitsverzeichnis
    debug_path = os.path.join(os.getcwd(), f"debug_box_{x}_{y}.png")
    cv2.imwrite(debug_path, mask)
    
    # Logik: Wenn > 30% der Box blau sind, ist sie aktiv
    ratio = np.count_nonzero(mask) / mask.size
    return ratio > 0.30

# Funktion für Erkennung des Grundriss in HottCad
def get_floorplan_contours_cv(image_bytes):
    """
    Analysiert den CAD-Grundriss direkt aus dem RAM-Stream des Screenshots,
    gibt die Ecken für den OmniParser zurück UND speichert ein Debug-Bild für die Masterarbeit.
    """
    # 1. Bild aus dem RAM dekodieren
    nparr = np.frombuffer(image_bytes.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return None

    # --- NEU: Kopie des Vollbildes für die Masterarbeit anlegen ---
    thesis_img = img.copy()

    # 2. Exakter Zuschnitt
    crop_img = img[325:775, 450:1100]
    
    # 3. Vorverarbeitung & Morphologisches Schließen (7x7 Pinsel)
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((7,7), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 4. Konturen finden
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid_corners = []

    # 5. Ecken mit getunten Parametern filtern
    for contour in contours:
        if cv2.contourArea(contour) > 100 or cv2.arcLength(contour, True) > 100:
            epsilon = 10.0  
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            
            for point in approx_polygon:
                x, y = point[0]
                is_duplicate = False
                
                # Duplikat-Filter (Radius 40)
                for (vx, vy) in valid_corners:
                    if abs(x - vx) < 40 and abs(y - vy) < 40:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    valid_corners.append((x, y))

    # 6. Koordinaten umrechnen UND auf das Vollbild zeichnen
    points = {}
    for i, (x, y) in enumerate(valid_corners):
        # Reale Koordinaten auf dem Vollbild berechnen
        real_x = x + 450
        real_y = y + 325
        corner_name = f"C{i+1}"
        
        # Für den OmniParser speichern
        points[corner_name] = (real_x, real_y)
        
        # --- NEU: Markierungen für die Masterarbeit zeichnen ---
        # Zeichnet einen massiven roten Punkt (Farbe in BGR: Blau=0, Grün=0, Rot=255)
        cv2.circle(thesis_img, (real_x, real_y), 6, (0, 0, 255), -1)
        # Schreibt den Namen (z.B. C1) in Blau leicht versetzt daneben
        cv2.putText(thesis_img, corner_name, (real_x + 10, real_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --- NEU: Das fertige Bild auf der Festplatte speichern ---
    # Überschreibt bei jedem Loop diese Datei mit dem aktuellsten Stand
    cv2.imwrite("cv_thesis_export.png", thesis_img)
    
    # Wichtig: Den Stream-Zeiger zurücksetzen, falls der Code danach nochmal das Bild liest
    image_bytes.seek(0)

    return points

def force_maximize_active_window():
    """
    Sendet einen Befehl an den Windows-Agenten, um das aktuell 
    aktive Fenster (egal welches Programm) zu maximieren.
    """
    try:
        # Port 5055 ist die Schnittstelle zu deiner Windows-VM
        url = "http://127.0.0.1:5055/execute" 
        
        # Dieses PS-Skript greift das aktive Fenster via Win32 API und maximiert es
        payload = {
            "mode": "gui",
            "action": "maximize",
            "targets": ["HottCAD", "Hott-Therm", "Lüftungskonzept Wohnen"]
        }
        
        requests.post(url, json=payload, timeout=3)
    except Exception as e:
        print(f"DEBUG: Globales Maximieren fehlgeschlagen: {e}")

def sampling_loop_sync(
    *,
    model: str,
    provider: APIProvider | None,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = 2,
    max_tokens: int = 4096,
    omniparser_url: str,
    save_folder: str = "./uploads",
    instruction_steps = [],
    current_step_index = 0,
    windows_agent_url: str = "127.0.0.1:5055"
):
    print(f"DEBUG-START: Das gewählte Modell ist: '{model}'")
    """
    Synchronous agentic sampling loop for the assistant/tool interaction of computer use.
    """
    print('in sampling_loop_sync, model:', model)
    omniparser_client = OmniParserClient(url=f"http://{omniparser_url}/parse/")
    if model == "claude-3-5-sonnet-20241022":
        # Register Actor and Executor
        actor = AnthropicActor(
            model=model, 
            provider=provider,
            api_key=api_key, 
            api_response_callback=api_response_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images
        )
    elif model in set(["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", "omniparser + qwen2.5vl", "omniparser + gpt-5.4"]):
        actor = VLMAgent(
            model=model,
            provider=provider,
            api_key=api_key,
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images
        )
    elif model in set(["omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated", "omniparser + R1-orchestrated", "omniparser + qwen2.5vl-orchestrated", "omniparser + gpt-5.4-orchestrated"]):
        actor = VLMOrchestratedAgent(
            model=model,
            provider=provider,
            api_key=api_key,
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
            save_folder=save_folder
        )
        actor.instruction_steps = instruction_steps
        actor.current_step_index = current_step_index
    else:
        raise ValueError(f"Model {model} not supported")
    anthropic_executor = AnthropicExecutor(
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
    )
    print(f"Model Inited: {model}, Provider: {provider}")
    
    tool_result_content = None
    
    print(f"Start the message loop. User messages: {messages}")
    
    if model == "claude-3-5-sonnet-20241022": # Anthropic loop
        while True:
            force_maximize_active_window()
            parsed_screen = omniparser_client()
            screen_info_block = TextBlock(text='Below is the structured accessibility information of the current UI screen, which includes text and icons you can operate on, take these information into account when you are making the prediction for the next action. Note you will still need to take screenshot to get the image: \n' + parsed_screen['screen_info'], type='text')
            screen_info_dict = {"role": "user", "content": [screen_info_block]}
            messages.append(screen_info_dict)
            tools_use_needed = actor(messages=messages)

            for message, tool_result_content in anthropic_executor(tools_use_needed, messages):
                yield message
        
            if not tool_result_content:
                return messages

            messages.append({"content": tool_result_content, "role": "user"})
    
    elif model in set(["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", "omniparser + qwen2.5vl", "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated", "omniparser + R1-orchestrated", "omniparser + qwen2.5vl-orchestrated", "omniparser + gpt-5.4-orchestrated"]):
        from executor.openai_executor import OpenAIExecutor
        openai_executor = OpenAIExecutor(output_callback, tool_output_callback)

        last_raw_img = None
        skip_counter = 0
        MAX_SKIPS = 5

        # max waits for llm
        consecutive_waits = 0
        MAX_WAITS = 2

        while True:

            time.sleep(0.5)
                
            # A. Screenshot machen und durch OmniParser analysieren lassen
            try:
                parsed_screen = omniparser_client()
            except Exception as e:
                print(f"⚠️ OmniParser temporär nicht erreichbar: {e}. Warte 2s und retry...")
                time.sleep(2)
                continue

            screen_text = str(parsed_screen.get("screen_info", "")).lower()

            raw_img_bytes = base64.b64decode(parsed_screen['original_screenshot_base64'])

            # Durchlaufe alle erkannten Boxen
            lines = parsed_screen.get("screen_info", "").split('\n')
            updated_info = []

            for line in lines:
                # 1. ID in der Zeile finden
                found_id = None
                for box_id in parsed_screen.get("boxes", {}).keys():
                    if f"ID: {box_id}" in line:
                        found_id = box_id
                        break
                
                # 2. Nur wenn wir eine ID gefunden UND es eine Checkbox ist, Status abrufen
                if found_id and "checkbox" in line.lower():
                    box = parsed_screen["boxes"][found_id]
                    is_checked = get_checkbox_status(box, raw_img_bytes)
                    status_text = " (Status: CHECKED)" if is_checked else " (Status: UNCHECKED)"
                    line = line.replace(f"ID: {found_id}", f"ID: {found_id}{status_text}")
                
                updated_info.append(line)

            # Bereinigte screen_info zurückschreiben
            parsed_screen["screen_info"] = '\n'.join(updated_info)

            is_hottcad_active = "hottcad" in screen_text and "grundriss" in screen_text

            if is_hottcad_active:
                img_data = base64.b64decode(parsed_screen['original_screenshot_base64'])
                # Nur dann führen wir die CPU-intensive Bildverarbeitung aus
                floorplan_pois = get_floorplan_contours_cv(io.BytesIO(img_data))
            else:
                floorplan_pois = None
                print("CV-Modus pausiert: HottCAD nicht im Fokus.")

            if floorplan_pois:
                # 1. SOM-Bild decodieren, um darauf zu zeichnen
                try:
                    som_data = base64.b64decode(parsed_screen["som_image_base64"])
                    som_img = Image.open(io.BytesIO(som_data))
                    draw_som = ImageDraw.Draw(som_img)
                except Exception as e:
                    print(f"⚠️ Konnte SOM-Bild für CV-Markierungen nicht laden: {e}")
                    som_img = None

                # 2. Wir vergeben feste Phantom-IDs (ab 901), um nicht mit OmniParser-IDs zu kollidieren
                base_id = 901
                for corner_name, (x, y) in floorplan_pois.items():
                    str_id = str(base_id)
                    
                    # Dem LLM den Text/Semantik geben
                    parsed_screen["screen_info"] += f"\n[{str_id}] Floorplan Corner {corner_name}"
                    
                    # Dem Executor die Klick-Koordinaten geben [x, y, breite, höhe]
                    parsed_screen["boxes"][str_id] = [int(x), int(y), 5, 5]
                    
                    # 3. CV-Punkt & ID auf das Bild zeichnen
                    if som_img:
                        # Roter Punkt
                        r = 4 # Radius
                        draw_som.ellipse((x - r, y - r, x + r, y + r), fill='red', outline='white')
                        
                        # ID (z.B. 901) daneben schreiben (wie OmniParser es tun würde)
                        # Leicht versetzt, damit es den Punkt nicht verdeckt
                        text_x, text_y = x + 6, y - 10
                        
                        # Dunkelroter Hintergrund für den Text (für Lesbarkeit)
                        # Wir schätzen die Textgröße grob ab (ca. 20x12 Pixel)
                        draw_som.rectangle([text_x - 2, text_y - 2, text_x + 22, text_y + 12], fill=(139, 0, 0))
                        draw_som.text((text_x, text_y), str_id, fill="white")

                    base_id += 1
                
                # 4. Verändertes Bild wieder encodieren und speichern
                if som_img:
                    buffered = io.BytesIO()
                    som_img.save(buffered, format="PNG")
                    parsed_screen["som_image_base64"] = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # B. KI-Entscheidung einholen
            # actor() ruft deinen VLMOrchestratedAgent auf
            tools_use_needed, vlm_response_json = actor(messages=messages, parsed_screen=parsed_screen)

            if vlm_response_json and vlm_response_json.get("Action") == "finished":
                print("✅ Testlauf erfolgreich beendet! Beende Loop.")
                yield tools_use_needed 
                break
                
            # --- SICHERUNG 1: Schutz vor Absturz (AttributeError) ---
            # Wenn das Modell gar kein Tool-Calling-Objekt zurückgibt (nur Text/Reasoning),
            # ist tools_use_needed = None. Wir fangen das hier ab.
            if tools_use_needed is None:
                print("⚠️ Warning: VLM provided reasoning but no valid JSON action. Forcing retry...")
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "ERROR: You provided reasoning but did not execute a physical action (Box ID). "
                                    "You MUST perform a physical interaction (e.g., left_click, type) using a valid Box ID "
                                    "from the screen parsing to proceed. Provide your action in the required JSON format."
                        }
                    ]
                })
                continue # Wir überspringen den Executor und lassen die KI neu entscheiden
            
            action_name = vlm_response_json.get("Action")
            reasoning = vlm_response_json.get("Reasoning", "")
            wait_time = vlm_response_json.get("post_action_wait", 0)

            if action_name == "left_click" and "checkbox" in vlm_response_json.get("Reasoning", "").lower():
                print("🧠 Checkbox-Logik erkannt: Aktiviere stabilen Wächter...")
                wait_for_ui_change(vm_url=vm_ip_and_port, timeout=10, sensitivity_threshold=2.0)
            
            if action_name == "wait":
                # 1. OCR-Check: Ist gerade eine Installation oder ein Ladevorgang aktiv?
                is_installing = any(word in screen_text for word in [
                    "installieren", "installation", "fortschritt", 
                    "update download", "bitte warten", "kopieren"
                ])
                
                # 2. Notbremse aussetzen, wenn es einen guten Grund zum Warten gibt
                if "Lokaler Python-Blocker" in reasoning or wait_time >= 20 or is_installing:
                    print("⏳ Notbremse ausgesetzt: Installation/Ladevorgang erkannt (System darf warten...)")
                    # Zähler zurücksetzen, da dieses Warten völlig legitim ist!
                    consecutive_waits = 0 
                else:
                    # 3. Das ist ein "echtes", unbegründetes Wait vom LLM
                    consecutive_waits += 1 
                    
                    # (Limit auf 3 erhöht, da 2 oft etwas zu streng ist, falls die KI nur kurz etwas prüfen will)
                    if consecutive_waits >= 3: 
                        print(f"🛑 NOTBREMSE: LLM steckt in der Wait-Falle! ({consecutive_waits}x 'wait' in Folge). Breche Aufgabe ab.")
                        break
            else:
                consecutive_waits = 0

            # C. Ausführung durch den OpenAIExecutor
            tool_result_content = None 

            # Wir nutzen die for-Schleife für die Generator-Werte des Executors
            for loop_msg, tool_results in openai_executor(
                response=tools_use_needed, 
                messages=messages, 
                parsed_screen=parsed_screen, 
                vlm_response_json=vlm_response_json
            ):
                yield loop_msg
                tool_result_content = tool_results

            # --- NEU: DYNAMISCHE PAUSE NACH DER AUSFÜHRUNG ---
            # Wir warten hier, BEVOR der nächste Screenshot gemacht wird!
            wait_time = vlm_response_json.get("post_action_wait", 1)
            should_maximize = vlm_response_json.get("force_maximize", False) or (action_name == "left_click")
            if wait_time > 1:
                # Nutze die IP deines Flask-Servers auf der VM (z.B. 127.0.0.1:5050 oder die echte IP)
                vm_ip_and_port = "127.0.0.1:5055" 
                
                # Wenn es ein schwerer Klick war (wait_time == 20), nutzen wir die dynamische Beobachtung
                if wait_time >= 20:
                    wait_for_ui_change(vm_url=vm_ip_and_port, timeout=wait_time, sensitivity_threshold=5.0)
                    force_maximize_active_window()
                elif should_maximize:
                    time.sleep(2)
                    force_maximize_active_window()
                else:
                    # Für kleine 4-Sekunden Pausen (Ordner) reicht ein normaler Sleep
                    print(f"⏳ Kurze System-Pause: Warte {wait_time}s...")
                    time.sleep(wait_time)
            # --- SICHERUNG 2: Schutz vor vorzeitigem Abbruch ---
            # Wenn der Executor gelaufen ist, aber keine Ergebnisse (Klicks) erzielt hat,
            # verhindern wir hier das 'return' (Beenden der Schleife).
            if not tool_result_content:
                print("⚠️ Warning: No action was executed by the tools. Sending reminder to LLM...")
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "ERROR: No tool result was generated. Please ensure you select a visible Box ID "
                                    "and a valid action like 'left_click' or 'type'."
                        }
                    ]
                })
                continue # Schleife läuft weiter, statt mit 'return messages' zu sterben

            # D. Erfolgreiche Aktion: Ergebnisse an die Historie hängen
            # Wir hängen die Tool-Ergebnisse (Screenshot nach dem Klick etc.) an
            if isinstance(tool_result_content, list):
                messages.append({"role": "user", "content": tool_result_content})
            else:
                messages.append({"role": "user", "content": [tool_result_content]})
