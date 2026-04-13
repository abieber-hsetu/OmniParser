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


def wait_for_ui_change(vm_url, timeout=45, sensitivity_threshold=5.0):
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
    windows_agent_url: str = "127.0.0.1:5055"  # <--- NEU: Standardwert für lokales Testen
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

        while True:
            # Frame Skipping Definition
            try:
                # 1. Superschneller lokaler Screenshot (Dauert nur wenige Millisekunden)
                resp = requests.get(f"http://{windows_agent_url}/screenshot", timeout=5)
                current_raw_img = Image.open(io.BytesIO(resp.content)).convert('RGB')
                
                if last_raw_img is not None:
                    width, height = last_raw_img.size
                    draw_last = ImageDraw.Draw(last_raw_img)
                    draw_curr = ImageDraw.Draw(current_raw_img)
                    
                    # Taskleiste ausblenden (verhindert falsche Skips durch blinkende Uhr)
                    draw_last.rectangle([0, height - 40, width, height], fill="black")
                    draw_curr.rectangle([0, height - 40, width, height], fill="black")
                    
                    diff = ImageChops.difference(last_raw_img, current_raw_img)
                    stat = ImageStat.Stat(diff.convert('L'))
                    diff_ratio = (stat.mean[0] / 255) * 100
                    
                    # Wenn sich fast nichts geändert hat (< 0.1% der Pixel)
                    if diff_ratio < 0.1: 
                        skip_counter += 1
                        if skip_counter <= MAX_SKIPS:
                            print(f"💨 Fast-Skip ({skip_counter}/{MAX_SKIPS}): Keine UI-Änderung. Warte 1.5s...")
                            time.sleep(1.5)
                            continue # Bricht diesen Durchlauf ab und fängt die while-Schleife von vorne an!
                        else:
                            print(f"⚠️ Max Skips ({MAX_SKIPS}) erreicht. Erzwinge LLM-Analyse (Klick ging evtl. ins Leere).")
                            messages.append({
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text", 
                                        "text": "WARNUNG: Dein letzter Klick hat absolut nichts auf dem Bildschirm verändert. "
                                                "Hast du versehentlich auf einen AUSGEGRAUTEN (inaktiven) Button geklickt oder ein leeres Feld getroffen? "
                                                "Analysiere das Bild genau! Fülle erst fehlende Felder aus, bevor du es erneut versuchst."
                                    }
                                ]
                            })
                    
                    # Reset counter, wenn sich das Bild ändert oder wir das Limit erreicht haben
                    skip_counter = 0
                        
                # Aktuelles Bild für den nächsten Loop merken
                last_raw_img = current_raw_img.copy()

            except Exception as e:
                print(f"⚠️ Fast-Skip Fehler: {e}. Mache normal weiter.")
                
            # A. Screenshot machen und durch OmniParser analysieren lassen
            parsed_screen = omniparser_client()

            # --- START: TOKEN-OPTIMIERUNG ---
            raw_text = parsed_screen.get('screen_info', '')
            clean_elements = []
            for line in raw_text.strip().split('\n'):
                if ':' in line and ',' in line:
                    # Macht aus "ID: 0, Icon Box ID 0: Recycle Bin" -> "[0] Recycle Bin"
                    box_id = line.split(',')[0].replace('ID:', '').strip()
                    label = line.split(':', 2)[-1].strip()
                    clean_elements.append(f"[{box_id}] {label}")
            
            # Überschreibe die lange Liste mit dem extrem kurzen String
            parsed_screen['screen_info'] = " | ".join(clean_elements)
            # --- ENDE: TOKEN-OPTIMIERUNG ---
            
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
            if wait_time > 1:
                # Nutze die IP deines Flask-Servers auf der VM (z.B. 127.0.0.1:5050 oder die echte IP)
                vm_ip_and_port = "127.0.0.1:5055" 
                
                # Wenn es ein schwerer Klick war (wait_time == 20), nutzen wir die dynamische Beobachtung
                if wait_time >= 20:
                    wait_for_ui_change(vm_url=vm_ip_and_port, timeout=wait_time, sensitivity_threshold=5.0)
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
            messages.append({
                "role": "user",
                "content": tool_result_content,
            })

            # Optional: Kurze Pause, damit das System Zeit zum Rendern hat
            # time.sleep(1)

            # Die Tool-Ergebnisse (z.B. neue Screenshots) für die KI sichtbar machen
            messages.append({"content": tool_result_content, "role": "user"})
