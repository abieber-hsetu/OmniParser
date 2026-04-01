import json
from collections.abc import Callable
from typing import cast, Callable
import uuid
from PIL import Image, ImageDraw, ImageChops, ImageStat
import base64
from io import BytesIO
import copy
from pathlib import Path
from datetime import datetime
from anthropic import APIResponse
from anthropic.types import ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam, BetaUsage
from agent.llm_utils.oaiclient import run_oai_interleaved
from agent.llm_utils.groqclient import run_groq_interleaved
from agent.llm_utils.utils import is_image_path
import time
import re
import os
from rag_manager import HsetuRagManager
import hashlib
import io

def extract_data(input_string, data_type):
    # 1. Versuch: Standard-Markdown Extraktion (```json ... ```)
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    matches = re.findall(pattern, input_string, re.DOTALL)
    if matches:
        return matches[0][0].strip()
    
    # 2. Versuch: Falls es JSON ist, suche einfach nach der ersten { und der letzten }
    if data_type == "json":
        try:
            start_idx = input_string.find('{')
            end_idx = input_string.rfind('}')
            if start_idx != -1 and end_idx != -1:
                return input_string[start_idx:end_idx + 1].strip()
        except Exception:
            pass

    # 3. Fallback: Gib den Originalstring zurück
    return input_string.strip()

class VLMOrchestratedAgent:
    def _load_prompt(self, path):
        """Hilfsfunktion zum Laden des Prompts aus einer Datei"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"⚠️ Konnte Prompt-Datei nicht laden: {e}. Nutze Fallback-String.")
            return "Task: {task}. Please analyze the progress."

    def __init__(
        self,
        model: str, 
        provider: str, 
        api_key: str,
        output_callback: Callable, 
        api_response_callback: Callable,
        max_tokens: int = 350,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
        save_folder: str = None,
        rag_manager = HsetuRagManager(),
        pdf_program_name = None,
    ):
        if model == "omniparser + gpt-4o" or model == "omniparser + gpt-4o-orchestrated":
            self.model = "gpt-4o-2024-11-20"
        elif model == "omniparser + R1" or model == "omniparser + R1-orchestrated":
            self.model = "deepseek-r1-distill-llama-70b"
        elif model == "omniparser + qwen2.5vl" or model == "omniparser + qwen2.5vl-orchestrated":
            self.model = "qwen2.5-vl-72b-instruct"
        elif model == "omniparser + o1" or model == "omniparser + o1-orchestrated":
            self.model = "o1"
        elif model == "omniparser + o3-mini" or model == "omniparser + o3-mini-orchestrated":
            self.model = "o3-mini"
        # FIX HIER: Zuweisung hinzugefügt
        elif model == "omniparser + gpt-5.4" or model == "omniparser + gpt-5.4-orchestrated":
            self.model = "gpt-5.4"
        else:
            raise ValueError(f"Model {model} not supported")
        

        self.provider = provider
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback
        self.save_folder = save_folder
        self.rag_manager = rag_manager
        
        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0
        self.plan, self.ledger = None, None
        self.instruction_steps = []
        self.current_step_index = 0
        self.program_name = None

        # for tracking opened programs
        self.launched_programs = set()

        self.cached_programs = []
        self.cached_rag_context = ""
        self.last_rag_step_index = -1 # Merkt sich, für welchen Schritt wir zuletzt gesucht haben
        
        # Programme nur EINMAL beim Agenten-Start aus der DB laden
        if self.rag_manager and hasattr(self.rag_manager, 'vector_db') and self.rag_manager.vector_db:
            try:
                metas = self.rag_manager.vector_db.get(include=['metadatas'])['metadatas']
                self.cached_programs = list(set(m['program'] for m in metas if 'program' in m))
            except Exception as e:
                print(f"⚠️ Konnte Programme nicht aus DB cachen: {e}")

        # Ledger prompt
        self.ledger_prompt_path = os.path.join(os.path.dirname(__file__), 'ledger_prompt.txt')
        self.ledger_prompt_template = self._load_prompt(self.ledger_prompt_path)

        # system prompt
        system_path = os.path.join(os.path.dirname(__file__), 'system_prompt.txt')
        self.system_prompt_template = self._load_prompt(system_path)
    
    def _calculate_and_display_usage(self, token_usage):
        """Berechnet die Kosten basierend auf den verbrauchten Tokens."""
        if not token_usage:
            return 0.0

        # Beispiel-Preise (Stand 2026, Preise pro 1 Mio. Tokens)
        prices = {
            "gpt-5.4": {"input": 2.50, "output": 15.00},
            "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
            "o1": {"input": 15.00, "output": 60.00},
            "o3-mini": {"input": 1.10, "output": 4.40},
            "default": {"input": 5.00, "output": 15.00}
        }

        model_price = prices.get(self.model, prices["default"])
        
        # --- NEU: Kugelsicheres Auslesen der Tokens ---
        if isinstance(token_usage, int):
            # Fall 1: Es wurde nur eine nackte Zahl übergeben (wir nehmen es mangels Alternative als Input)
            input_tokens = token_usage
            output_tokens = 0
            print("⚠️ Warnung: token_usage war ein Integer. Kostenaufschlüsselung ist geschätzt.")
        elif isinstance(token_usage, dict):
            # Fall 2: Es ist ein Dictionary (beachtet verschiedene Bezeichnungen der APIs)
            input_tokens = token_usage.get('input_tokens', token_usage.get('prompt_tokens', 0))
            output_tokens = token_usage.get('output_tokens', token_usage.get('completion_tokens', 0))
        else:
            # Fall 3: Es ist ein OpenAI oder Anthropic Pydantic-Objekt
            input_tokens = getattr(token_usage, 'prompt_tokens', getattr(token_usage, 'input_tokens', 0))
            output_tokens = getattr(token_usage, 'completion_tokens', getattr(token_usage, 'output_tokens', 0))
        
        # Berechnung: (Tokens / 1.000.000) * Preis
        current_step_cost = (
            (input_tokens * model_price["input"]) / 1_000_000 +
            (output_tokens * model_price["output"]) / 1_000_000
        )

        self.total_token_usage += (input_tokens + output_tokens)
        self.total_cost += current_step_cost

        usage_str_console = (
            f"💰 **Kosten dieser Anfrage:** ${current_step_cost:.4f} "
            f"(In: {input_tokens} | Out: {output_tokens})\n"
            f"📈 **Gesamtkosten bisher:** ${self.total_cost:.4f} "
            f"(Gesamt-Tokens: {self.total_token_usage})"
        )

        # Für die Gradio UI (Sauberes HTML)
        usage_str_html = (
            f"💰 <strong>Kosten dieser Anfrage:</strong> ${current_step_cost:.4f} "
            f"(In: {input_tokens} | Out: {output_tokens})<br>"
            f"📈 <strong>Gesamtkosten bisher:</strong> ${self.total_cost:.4f} "
            f"(Gesamt-Tokens: {self.total_token_usage})"
        )

        # Ausgabe
        print(f"\n{usage_str_console}")
        if self.output_callback:
            self.output_callback(
                f'<div style="padding: 10px; '
                f'background-color: var(--background-fill-secondary, #1f2937); '
                f'border-left: 4px solid var(--color-accent, #3b82f6); '
                f'border-radius: 4px; '
                f'color: var(--body-text-color, #e5e7eb); '
                f'font-family: monospace; '
                f'margin-bottom: 10px; '
                f'line-height: 1.5; '
                f'box-shadow: 0 1px 3px rgba(0,0,0,0.1);">'
                f'{usage_str_html}</div>'
            )
        return current_step_cost

    def set_instructions(self, steps, program_name=None):
        self.instruction_steps = steps
        self.current_step_index = 0
        self.pdf_program_name = program_name

    def _get_image_difference_ratio(self, img1_base64, img2_base64):
        try:
            # Bilder laden und in RGB konvertieren
            img1 = Image.open(io.BytesIO(base64.b64decode(img1_base64))).convert('RGB')
            img2 = Image.open(io.BytesIO(base64.b64decode(img2_base64))).convert('RGB')
            
            # Taskleiste maskieren (Störfaktoren wie Uhrzeit entfernen)
            # Wir schwärzen die unteren 40 Pixel
            width, height = img1.size
            draw1 = ImageDraw.Draw(img1)
            draw2 = ImageDraw.Draw(img2)
            draw1.rectangle([0, height - 40, width, height], fill="black")
            draw2.rectangle([0, height - 40, width, height], fill="black")

            # Differenz berechnen
            diff = ImageChops.difference(img1, img2)
            stat = ImageStat.Stat(diff.convert('L'))
            
            # Prozentsatz der Abweichung berechnen
            return (stat.mean[0] / 255) * 100
        except Exception as e:
            print(f"⚠️ Fehler beim Bildvergleich: {e}")
            return 100 # Im Fehlerfall Fortschritt erzwingen
    
    def __call__(self, messages: list, parsed_screen: dict):

        pdf_program = getattr(self, 'pdf_program_name', None)
        sync_screen_info_str = parsed_screen.get("screen_info", "")
        all_screen_text = sync_screen_info_str.lower()
        current_program = "Unknown"

        # --- NEU: POPUP DIREKT ABFANGEN (SPART TOKENS) ---
        # --- NEU: POPUP DIREKT ABFANGEN (SPART TOKENS) ---
        if "systemcheck is running" in all_screen_text or "systemcheck" in all_screen_text:
            print("🛑 Lade-Popup 'SystemCheck' erkannt! Warte 5 Sekunden...")
            
            dummy_json = {
                "Action": "wait", 
                "Reasoning": "Das Lade-Popup 'SystemCheck' ist sichtbar. Ich warte auf das Hauptfenster.",
                "post_action_wait": 5 # Kurzer Check im 5-Sekunden-Takt
            }
            
            response_content = [
                BetaTextBlock(text="Reasoning: Loading popup detected. Waiting for main GUI...", type='text'),
                BetaToolUseBlock(
                    id=f'toolu_{uuid.uuid4()}', 
                    input={'action': 'wait'}, 
                    name='computer', 
                    type='tool_use'
                )
            ]
            dummy_message = BetaMessage(
                id=f'msg_{uuid.uuid4()}', 
                content=response_content, 
                model=self.model,
                role='assistant', 
                type='message',
                stop_reason='tool_use',
                usage=BetaUsage(input_tokens=0, output_tokens=0)
            )
            return dummy_message, dummy_json
        # --- ENDE POPUP ABFANGEN ---

        # Screenshot related vars
        screenshot_uuid = parsed_screen['screenshot_uuid']
        screenshot_path = os.path.join(self.save_folder, f"screenshot_{screenshot_uuid}.png")
        som_path = os.path.join(self.save_folder, f"som_screenshot_{screenshot_uuid}.png")
        current_img_base64 = parsed_screen.get('original_screenshot_base64', "")
        
        # 1. Schrittzähler und Fortschritts-Ledger initialisieren/aktualisieren
        if self.step_count == 0:
            plan = self._initialize_task(messages)
            self.output_callback(f'-- Plan: {plan} --', )
            messages.append({"role": "assistant", "content": plan})

            # --- NEU: EIGENEN PLAN IN SCHRITTE UMWANDELN ---
            if not self.instruction_steps:
                try:
                    parsed_plan = json.loads(plan)
                    if isinstance(parsed_plan, dict):
                        self.instruction_steps = [str(v) for v in parsed_plan.values()]
                    elif isinstance(parsed_plan, list):
                        self.instruction_steps = [str(item) for item in parsed_plan]
                    
                    self.current_step_index = 0
                    print(f"🎯 LLM Plan übernommen: {len(self.instruction_steps)} eigene Schritte geladen.")
                except Exception as e:
                    print(f"⚠️ Warnung: LLM Plan war kein sauberes JSON. Fallback auf Gesamtaufgabe. ({e})")
                    self.instruction_steps = [self._task]

        else:
            # 1. Den Ledger von der KI erstellen lassen (ist ein JSON-String)
            updated_ledger = self._update_ledger(messages)
            
            total_steps = len(self.instruction_steps)
            current_step_num = self.current_step_index + 1
            
            # --- NEU: SCHRITT WIRKLICH ERHÖHEN UND FERTIG-SIGNAL SENDEN ---
            try:
                ledger_dict = json.loads(updated_ledger)
                is_satisfied = ledger_dict.get("is_request_satisfied", {}).get("answer", False)
                
                if is_satisfied:
                    # Der LLM-Ledger sagt: Der aktuelle Schritt ist fertig!
                    if self.current_step_index < (total_steps - 1):
                        self.current_step_index += 1
                        current_step_num = self.current_step_index + 1
                        print(f"✅ Schritt {current_step_num - 1} abgeschlossen! Wechsle zu Schritt {current_step_num}/{total_steps}.")
                        
                        # OVERRIDE: Wir zwingen den Ledger für loop.py wieder auf "false", damit der Testlauf weitergeht
                        ledger_dict["is_request_satisfied"]["answer"] = False
                        old_reason = ledger_dict["is_request_satisfied"].get("reason", "")
                        ledger_dict["is_request_satisfied"]["reason"] = f"[Schritt {current_step_num - 1} beendet. Starte {current_step_num}] {old_reason}"
                        
                        # Aktualisiertes JSON zurück in einen String verwandeln
                        updated_ledger = json.dumps(ledger_dict, indent=4)
                    else:
                        print(f"🎉 Alle {total_steps} Schritte erfolgreich abgeschlossen! Testlauf beendet.")
                        
                        # --- SAUBERER ABBRUCH ---
                        dummy_json = {"Action": "finished", "Reasoning": "Alle Schritte wurden erfolgreich abgeschlossen!"}
                        response_content = [
                            BetaTextBlock(text="Reasoning: The task is completely finished. End of execution.", type='text')
                        ]
                        dummy_message = BetaMessage(
                            id=f'msg_{uuid.uuid4()}', 
                            content=response_content, 
                            model=self.model,
                            role='assistant', 
                            type='message',
                            stop_reason='end_turn',
                            usage=BetaUsage(input_tokens=0, output_tokens=0)
                        )
                        
                        self.output_callback(
                            f'<details>'
                            f'  <summary><strong>Task Progress Ledger (FINAL - ALL DONE)</strong></summary>'
                            f'  <div style="padding: 10px; background-color: #d1fae5; border-radius: 5px; margin-top: 5px;">'
                            f'    <pre>{updated_ledger}</pre>'
                            f'  </div>'
                            f'</details>',
                        )
                        return dummy_message, dummy_json

            except Exception as e:
                # Fallback, falls das LLM versehentlich kaputtes JSON geliefert hat
                if '"answer": true' in updated_ledger.lower() or '"answer":true' in updated_ledger.lower():
                    if self.current_step_index < (total_steps - 1):
                        self.current_step_index += 1
                        current_step_num = self.current_step_index + 1
                        print(f"✅ Schritt {current_step_num - 1} abgeschlossen (Fallback)! Wechsle zu Schritt {current_step_num}/{total_steps}.")
                        updated_ledger = updated_ledger.replace(': true', ': false').replace(':true', ':false')

            # UI Update
            self.output_callback(
                f'<details>'
                f'  <summary><strong>Task Progress Ledger (Schritt {current_step_num}/{total_steps})</strong></summary>'
                f'  <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 5px;">'
                f'    <pre>{updated_ledger}</pre>'
                f'  </div>'
                f'</details>',
            )
            messages.append({"role": "assistant", "content": updated_ledger})
            self.ledger = updated_ledger

        self.step_count += 1
        
        # 2. Screenshots für das Debugging lokal speichern
        with open(screenshot_path, "wb") as f:
            f.write(base64.b64decode(parsed_screen['original_screenshot_base64']))
        with open(som_path, "wb") as f:
            f.write(base64.b64decode(parsed_screen['som_image_base64']))

        # 3. Metadaten vom OmniParser extrahieren
        latency_omniparser = parsed_screen['latency']
        screen_width, screen_height = parsed_screen['width'], parsed_screen['height']

        content_list = parsed_screen.get("boxes", {})
        
        # 4. Dynamische Programmerkennung (Nutzt jetzt den pfeilschnellen Cache)
        if self.cached_programs:
            for prog in self.cached_programs:
                if prog.lower() in all_screen_text:
                    current_program = prog
                    break

        effective_program = current_program if current_program != "Unknown" else pdf_program
        
        # --- NEU: Instruktionen aus Anweisung.pdf einbeziehen ---
        current_instruction = ""
        instruction_header = ""
        search_query = self._task
        
        if self.instruction_steps and self.current_step_index < len(self.instruction_steps):
            current_instruction = self.instruction_steps[self.current_step_index]
            current_step = self.current_step_index + 1
            total_steps = len(self.instruction_steps)
            
            # QUERY-CLEANING: Verbessert für semantische Suche
            clean_instruction = current_instruction.split("|")[0].replace("BEFEHL:", "").strip()
            clean_instruction = re.sub(r"Schritt\s*\d+\s*–?", "", clean_instruction).strip()
            
            # Wir kombinieren das Hauptziel mit dem Teilschritt für besseren Kontext
            search_query = f"Aufgabe: {self._task}. Aktueller Schritt: {clean_instruction}"
            
            instruction_header = f"""
                ### 📋 AKTUELLE TEST-ANWEISUNG ({current_step} von {total_steps}):
                {current_instruction}

                ### ⚠️ TECHNISCHE PROTOKOLL-REGELN:
                1. Analysiere den Screenshot: Ist das ZIEL der Aufgabe bereits erreicht (z.B. das Programmfenster ist bereits vollständig geöffnet und sichtbar)? Wenn ja, klicke NICHT erneut, sondern beende den Schritt!
                2. Wenn das Ziel erreicht ist, schreibe zwingend das Codewort **SCHRITT_ABGESCHLOSSEN** ans Ende deines Reasonings.
                3. Wenn die Aufgabe noch nicht sichtbar ist, führe die nächsten notwendigen Klicks/Eingaben aus.
                """
        else:
            search_query = self._task if hasattr(self, '_task') else "Nächster Schritt"

        # 5. SMART RAG ABFRAGE (Nur wenn sich der Schritt geändert hat!)
        if self.rag_manager:
            if self.current_step_index != self.last_rag_step_index:
                print(f"🔍 RAG-Suche aktiv für neuen Schritt {self.current_step_index + 1}...")
                self.cached_rag_context = self.rag_manager.get_context(
                    query=search_query, 
                    program=effective_program
                )
                self.last_rag_step_index = self.current_step_index
            else:
                # Wir sparen uns die Suche und nutzen das Wissen vom vorherigen Frame
                pass 
                
        rag_context = self.cached_rag_context

        # 6. System-Prompt injizieren
        system = self._get_system_prompt(
            screen_info=sync_screen_info_str,
            current_step=instruction_header if instruction_header else "",
            rag_context=rag_context
        )

        # 7. VLM Vorbereitung (Bilder filtern und anhängen) Länge wird begrenzt auf die letzten 4 Nachrichten um Verwirrung und Loops zu vermeiden
        if len(messages) > 5:
            # Erstelle eine frische Liste, um die Haupt-Historie nicht zu beschädigen
            planner_messages = [messages[0]] + messages[-4:]
        else:
            # Bei den ersten Schritten kopieren wir einfach alles
            planner_messages = copy.deepcopy(messages)

        _remove_som_images(planner_messages)
        _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)


        # Bilder an die letzte Nachricht anhängen
        if isinstance(planner_messages[-1], dict):
            if not isinstance(planner_messages[-1]["content"], list):
                planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
            
            # Wir hängen die oben definierten Pfade an
            planner_messages[-1]["content"].append(som_path)

        # 8. VLM Aufruf (GPT, DeepSeek oder Qwen)
        start = time.time()
        if "gpt" in self.model or "o1" in self.model or "o3-mini" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                provider_base_url="https://api.openai.com/v1",
                temperature=0.2,
            )
        elif "r1" in self.model:
            vlm_response, token_usage = run_groq_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
            )
        elif "qwen" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=min(2048, self.max_tokens),
                provider_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0,
            )
        
        # Display Token Usage
        current_step_cost = self._calculate_and_display_usage(token_usage)
        
        latency_vlm = time.time() - start
        self.output_callback(
            f'<i>Step {self.step_count} | OmniParser: {latency_omniparser:.2f}s | '
            f'LLM: {latency_vlm:.2f}s | Cost: ${current_step_cost:.4f}</i>'
        )

        # 9. JSON-Parsing und Schritt-Fortschritt-Kontrolle
        try:
            json_match = re.search(r'(\{.*\})', vlm_response, re.DOTALL)
            clean_json = json_match.group(1) if json_match else vlm_response
            vlm_response_json = json.loads(clean_json)
            reasoning = vlm_response_json.get("Reasoning", vlm_response_json.get("reasoning", ""))
                
        except Exception:
            vlm_response_json = {"Reasoning": "JSON Formatfehler", "Action": "wait", "Box_ID": None}

        if vlm_response_json and 'Action' in vlm_response_json:
            action = vlm_response_json.get('Action', '').lower()
            box_id_val = vlm_response_json.get("Box_ID", vlm_response_json.get("Box ID"))
            
            clicked_label = ""
            if box_id_val is not None and content_list:
                raw_el = content_list.get(str(box_id_val))
                if isinstance(raw_el, list) and len(raw_el) > 1:
                    clicked_label = str(raw_el[1]).lower()
                elif isinstance(raw_el, dict):
                    clicked_label = raw_el.get('content', '').lower()

            # 2. Reasoning und eventuell getippten Text auslesen
            reasoning_text = str(vlm_response_json.get("Reasoning", "")).lower()
            typed_text = str(vlm_response_json.get("Text_content", vlm_response_json.get("value", ""))).lower()
            
            # --- START: GLOBALE PROGRAMMSTART-ERKENNUNG ---
            is_program_launch = False
            
            # Wir prüfen ALLE Aktionen, die das Programm starten könnten
            if action in ["double_click", "left_click", "type"]:
                
                # Hat die Aktion optisch oder inhaltlich mit unserem Programm zu tun?
                interacts_with_program = any(w in clicked_label for w in ["hott", "therm", "cad"]) or \
                                         any(w in typed_text for w in ["hott", "therm", "cad"])
                
                # Will er es wirklich STARTEN? (Unterscheidet Start-Versuche von normalen Klicks im laufenden Programm)
                intends_to_open = any(w in reasoning_text for w in ["start", "open", "öffne", "search", "suche", "shortcut", "icon", "run"])
                
                # Fallback: Wenn er "Hott" tippt (type), will er es zu 99% über die Windows-Suche öffnen
                if interacts_with_program and (intends_to_open or action == "type"):
                    is_program_launch = True

            if is_program_launch:
                if "app_launched" in self.launched_programs:
                    print(f"🛡️ BLOCKIERT: Agent wollte das Programm ein zweites Mal über '{action}' starten!")
                    
                    # Aktion hart auf "wait" überschreiben
                    vlm_response_json["Action"] = "wait"
                    if "Next Action" in vlm_response_json:
                        vlm_response_json["Next Action"] = "wait"
                    vlm_response_json["post_action_wait"] = 5
                    
                    # Systemwarnung in den Chatverlauf schießen
                    messages.append({
                        "role": "user",
                        "content": "SYSTEM OVERRIDE: Your action was BLOCKED. You already launched the application! It is currently loading in the background. DO NOT try to open it again via search, start menu, or desktop shortcut. Look for a loading popup or wait for the main window to appear."
                    })
                else:
                    print(f"🚩 SIGNAL: Erster Programmstart registriert (via {action})!")
                    self.launched_programs.add("app_launched")
                    vlm_response_json["post_action_wait"] = 8
            else:
                # Standard Wartezeit für alle normalen Klicks, die kein Start sind
                vlm_response_json["post_action_wait"] = 2
        
        # --- HIER STARTET DER NEUE BULLET-PROOF BLOCK ---
        next_action = vlm_response_json.get("Action", vlm_response_json.get("Next Action", "None"))
        reasoning_str = str(vlm_response_json.get("Reasoning", "")).upper()
        action_str = str(next_action).upper()
        
        # Wenn der Agent das Wort SCHRITT_ABGESCHLOSSEN nutzt, erhöhen wir SOFORT den Schritt!
        if "SCHRITT_ABGESCHLOSSEN" in reasoning_str or "SCHRITT_ABGESCHLOSSEN" in action_str or "COMPLETION" in action_str:
            print(f"🚀 AGENT-SIGNAL: Agent meldet Schritt {self.current_step_index + 1} als beendet!")
            total_steps = len(self.instruction_steps)
            
            if self.current_step_index < (total_steps - 1):
                self.current_step_index += 1
                print(f"✅ Wechsle sofort zu Schritt {self.current_step_index + 1}/{total_steps}.")
                next_action = "wait"
                vlm_response_json["Action"] = "wait"
            else:
                print(f"🎉 Alle {total_steps} Schritte abgeschlossen! Testlauf beendet.")
                next_action = "finished"
                vlm_response_json["Action"] = "finished"
        # --- ENDE DES BULLET-PROOF BLOCKS ---

        # 10. Koordinatenberechnung und Visualisierung
        img_to_show_base64 = parsed_screen["som_image_base64"]
        
        # FIX: Unterstütze beide Schreibweisen aus Prompt ("Box_ID") und Skript ("Box ID")
        box_id_val = vlm_response_json.get("Box_ID", vlm_response_json.get("Box ID"))
        
        if box_id_val is not None:
            try:
                # FIX: OmniParser liefert Boxen oft als Dictionary mit String-Keys (z.B. "3" statt 3)
                target_item = None
                if isinstance(content_list, dict):
                    target_item = content_list.get(str(box_id_val), content_list.get(int(box_id_val)))
                elif isinstance(content_list, list) and int(box_id_val) < len(content_list):
                    target_item = content_list[int(box_id_val)]

                # FIX: target_item kann ein Dict, eine Liste von Listen oder eine flache Liste sein!
                bbox = None
                if target_item:
                    if isinstance(target_item, dict) and "bbox" in target_item:
                        bbox = target_item["bbox"]
                    elif isinstance(target_item, (list, tuple)):
                        # Fall A: Verschachtelte Liste, z.B. [[x1, y1, x2, y2], "Text"]
                        if len(target_item) > 0 and isinstance(target_item[0], (list, tuple)) and len(target_item[0]) >= 4:
                            bbox = target_item[0][:4]
                        # Fall B: Flache Liste, z.B. [x1, y1, x2, y2]
                        elif len(target_item) >= 4 and isinstance(target_item[0], (int, float)):
                            bbox = target_item[:4]

                if bbox and len(bbox) >= 4:
                    x_val, y_val, val3, val4 = bbox[:4]
                    
                    # 1. Sind die Werte absolut (Pixel) oder normalisiert (0.0 bis 1.0)?
                    # Wir prüfen, ob alle Werte <= 1.5 sind (normalisiert)
                    is_normalized = all(isinstance(v, (int, float)) and v <= 1.5 for v in bbox[:4])
                    
                    if is_normalized:
                        # OmniParser liefert in der Regel [x, y, width, height]
                        # Der alte Code ((bbox[0] + bbox[2]) / 2) war für [xmin, ymin, xmax, ymax] gedacht
                        # und hat bei [x, y, w, h] die Koordinate verschoben: (x + w) / 2 statt x + (w / 2).
                        
                        # Sicherheits-Check: Falls x + width > 1.05, kann es sich nur um xmax handeln.
                        if x_val + val3 > 1.05 or y_val + val4 > 1.05:
                            center_x = (x_val + val3) / 2 * screen_width
                            center_y = (y_val + val4) / 2 * screen_height
                        else:
                            # Das korrekte Zentrum für das OmniParser Format: [x, y, width, height]
                            center_x = (x_val + (val3 / 2)) * screen_width
                            center_y = (y_val + (val4 / 2)) * screen_height
                    else:
                        # Absolute Pixelwerte 
                        if x_val + val3 > screen_width * 1.05 or y_val + val4 > screen_height * 1.05:
                            center_x = (x_val + val3) / 2
                            center_y = (y_val + val4) / 2
                        else:
                            center_x = x_val + (val3 / 2)
                            center_y = y_val + (val4 / 2)

                    vlm_response_json["box_centroid_coordinate"] = [int(center_x), int(center_y)]
                    
                    img_to_show_data = base64.b64decode(img_to_show_base64)
                    img_to_show = Image.open(BytesIO(img_to_show_data))
                    draw = ImageDraw.Draw(img_to_show)
                    x, y = vlm_response_json["box_centroid_coordinate"] 
                    draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill='red', outline='white', width=2)
                    
                    buffered = BytesIO()
                    img_to_show.save(buffered, format="PNG")
                    img_to_show_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"Visualisierungs-Fehler: {e}")

        # 11. UI Output für Gradio
        self.output_callback(f'<img src="data:image/png;base64,{img_to_show_base64}">', )
        self.output_callback(
            f'<details>'
            f'  <summary><strong>Parsed Screen Elements</strong></summary>'
            f'  <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 5px;">'
            f'    <pre>{sync_screen_info_str}</pre>'
            f'  </div>'
            f'</details>',
        )
        
        # 12. Tool-Antwort-Objekte zusammenstellen
        vlm_plan_str = f"Reasoning: {vlm_response_json.get('Reasoning', '')}"
        response_content = [BetaTextBlock(text=vlm_plan_str, type='text')]
        
        # FIX: Unterstütze beide Schreibweisen aus Prompt ("Action") und Skript ("Next Action")
        next_action = vlm_response_json.get("Action", vlm_response_json.get("Next Action", "None"))

        # Wenn die Aktion "type" ist, fokussiere das Textfeld ZWINGEND vorher mit einem Linksklick
        if next_action == "type" and 'box_centroid_coordinate' in vlm_response_json:
            response_content.append(BetaToolUseBlock(
                id=f'toolu_{uuid.uuid4()}',
                input={'action': 'left_click', 'coordinate': vlm_response_json["box_centroid_coordinate"]},
                name='computer', type='tool_use'))

        if next_action == "type":
            response_content.append(BetaToolUseBlock(
                id=f'toolu_{uuid.uuid4()}',
                # FIX: Unterstütze "Text_content" (Prompt) und "value" (Skript)
                input={'action': 'type', 'text': vlm_response_json.get("Text_content", vlm_response_json.get("value", ""))},
                name='computer', type='tool_use'))

        elif next_action.lower() in ["none", "wait", "verify", "check", "observe", "done", "validate_step", "validate", "verify_completion", "analyze"]:
            response_content.append(BetaToolUseBlock(
                id=f'toolu_{uuid.uuid4()}',
                input={'action': 'wait'},
                name='computer', type='tool_use'))
                
        else:
            # FIX: Hänge die Koordinaten zwingend an den tatsächlichen Klick-Befehl an!
            action_input = {'action': next_action}
            if 'box_centroid_coordinate' in vlm_response_json:
                action_input['coordinate'] = vlm_response_json['box_centroid_coordinate']

            response_content.append(BetaToolUseBlock(
                id=f'toolu_{uuid.uuid4()}',
                input=action_input,
                name='computer', type='tool_use'))

        # Finales Objekt erstellen
        response_message = BetaMessage(
            id=f'msg_{uuid.uuid4()}', 
            content=response_content, 
            model=self.model,
            role='assistant', 
            type='message',
            stop_reason='tool_use',
            usage=BetaUsage(input_tokens=0, output_tokens=0)
        )
        # --- START: ANALYTICS LOGGING (JSONL + CSV) ---
        try:
            if 'current_step_cost' not in locals():
                current_step_cost = 0.0
                
            log_file_jsonl = os.path.join(self.save_folder, "analytics_log.jsonl")
            log_file_csv = os.path.join(self.save_folder, "analytics_log.csv")
            
            ledger_str = ""
            if self.ledger:
                try:
                    ledger_str = json.dumps(json.loads(self.ledger), ensure_ascii=False)
                except Exception:
                    ledger_str = str(self.ledger).replace("\n", " ")
            
            # 1. Maschine-lesbares Format für JSONL (mit Punkten und echten Floats)
            log_entry_json = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "step": self.step_count,
                "current_instruction_index": self.current_step_index,
                "program": effective_program,
                "model": self.model,
                "action": next_action,
                "reasoning": reasoning_str.replace("\n", " "),
                "coordinates": str(vlm_response_json.get("box_centroid_coordinate", [])),
                "latency_omniparser_s": round(latency_omniparser, 2),
                "latency_vlm_s": round(latency_vlm, 2),
                "step_cost_usd": round(current_step_cost, 4),
                "total_cost_usd": round(self.total_cost, 4),
                "total_tokens": getattr(self, 'total_token_usage', 0),
                "ledger": ledger_str
            }
            
            # 2. Excel-freundliches Format für CSV (Zahlen zwingend mit Komma statt Punkt!)
            log_entry_csv = log_entry_json.copy()
            log_entry_csv["latency_omniparser_s"] = f"{latency_omniparser:.2f}".replace(".", ",")
            log_entry_csv["latency_vlm_s"] = f"{latency_vlm:.2f}".replace(".", ",")
            log_entry_csv["step_cost_usd"] = f"{current_step_cost:.4f}".replace(".", ",")
            log_entry_csv["total_cost_usd"] = f"{self.total_cost:.4f}".replace(".", ",")
            
            # Schreibe das ausfallsichere JSONL-Format
            with open(log_file_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry_json, ensure_ascii=False) + "\n")
                
            # Schreibe die menschenlesbare CSV-Tabelle für Excel
            import csv
            file_exists = os.path.isfile(log_file_csv)
            with open(log_file_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=log_entry_csv.keys(), delimiter=';')
                if not file_exists:
                    writer.writeheader() 
                writer.writerow(log_entry_csv)
                
        except Exception as e:
            print(f"⚠️ Analytics Logging fehlgeschlagen: {e}")
        # --- ENDE: ANALYTICS LOGGING ---

        return response_message, vlm_response_json


    def _api_response_callback(self, response: APIResponse):
        self.api_response_callback(response)

    def _get_system_prompt(self, screen_info: str = "", current_step: str = "", rag_context: str = ""):
        """Erstellt den System-Prompt und injiziert alle dynamischen Daten."""
        
        # 1. Template laden (aus self.system_prompt_template)
        template = self.system_prompt_template

        # 3. Den RAG-Context formatieren
        rag_text = ""
        if rag_context:
            rag_text = (
                "### STATISCHES NACHSCHLAGEWERK (NUR ZUM NACHSCHLAGEN):\n"
                "Das folgende Wissen dient nur der Orientierung (z.B. Wo finde ich ein Menü?).\n"
                "IGNORIERE alle Nummerierungen oder 'Schritte' innerhalb dieses Blocks für deine Planung!\n"
                f"INHALT:\n{rag_context}\n"
                "--- ENDE NACHSCHLAGEWERK ---"
            )

        # 4. Alles in das Template einsetzen
        try:
            return template.format(
                screen_info=screen_info,
                instruction_block=current_step,
                rag_block=rag_text
            )
        except Exception as e:
            print(f"⚠️ Fehler beim Formatieren: {e}")
            return template # Fallback auf Roh-Template

    def _initialize_task(self, messages: list):
        # 1. Text sicher extrahieren
        content_data = messages[0]["content"]
        full_content = content_data[0].get("text", "") if isinstance(content_data, list) else str(content_data)

        # 2. Task säubern (Dokumentation entfernen)
        if "BASIEREND AUF DIESEM WISSEN" in full_content:
            self._task = full_content.split("FOLGENDE AUFGABE AUS:")[-1].strip()
            # Historie reinigen
            if isinstance(content_data, list):
                messages[0]["content"][0]["text"] = self._task
            else:
                messages[0]["content"] = self._task
        else:
            self._task = full_content

        # 3. PLANER-FIX: Wir geben ihm die PDF-Schritte explizit mit!
        # Damit er nicht nur "Meta-Planung" betreibt.
        steps_summary = "\n".join([f"- {s}" for s in self.instruction_steps])
        active_steps = []
        skip_words = ["identify", "determine", "observe", "identifizieren", "feststellen"]
        for s in self.instruction_steps:
            if not any(word in s.lower() for word in skip_words):
                active_steps.append(s)
        
        # Falls alles gefiltert wurde, nimm das Original, sonst die gefilterte Liste
        final_steps = active_steps if active_steps else self.instruction_steps
        steps_summary = "\n".join([f"- {s}" for s in final_steps])
        
        planning_input = f"Task: {self._task}\n\nExecute these ACTION steps:\n{steps_summary}"

        plan_prompt = self._get_plan_prompt(planning_input)
        
        # 4. API Call
        planner_messages = copy.deepcopy(messages)
        planner_messages.append({"role": "user", "content": [{"type": "text", "text": plan_prompt}]})
        
        vlm_response, _ = run_oai_interleaved(
            messages=planner_messages,
            system="You are a strategic test planner. Convert the text steps into a JSON execution plan.",
            model_name=self.model,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            provider_base_url="https://api.openai.com/v1",
            temperature=0,
        )

        plan = extract_data(vlm_response, "json")
        
        # Speichern (wie gehabt)
        plan_path = os.path.join(self.save_folder, "plan.json")
        with open(plan_path, "w", encoding='utf-8') as f:
            f.write(plan)
            
        return plan

    def _update_ledger(self, messages):
    # 1. Den Prompt-Template aus der Textdatei laden
        try:
            with open(self.ledger_prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except Exception as e:
            print(f"⚠️ Konnte ledger_prompt.txt nicht laden: {e}")
            # Fallback auf den alten String, falls die Datei fehlt
            prompt_template = ORCHESTRATOR_LEDGER_PROMPT 

    # 2. Alle dynamischen Daten vorbereiten
        total_steps = len(self.instruction_steps)
        current_step_idx = self.current_step_index + 1
        
        # Sicherstellen, dass wir eine gültige Anweisung haben
        if self.instruction_steps and self.current_step_index < total_steps:
            current_instr = self.instruction_steps[self.current_step_index]
        else:
            current_instr = "Keine weiteren Schritte definiert."

        # 3. Den Prompt mit allen Variablen befüllen
        # WICHTIG: Deine Textdatei muss die Platzhalter {task}, {current_step}, 
        # {total_steps} und {current_instruction} enthalten!
        update_ledger_prompt = prompt_template.format(
            task=self._task,
            current_step_idx=current_step_idx,
            total_steps=total_steps,
            current_instruction=current_instr
        )

        # 4. Den Aufruf an das LLM durchführen
        input_message = copy.deepcopy(messages)
        input_message.append({"role": "user", "content": update_ledger_prompt})
        
        vlm_response, token_usage = run_oai_interleaved(
            messages=input_message,
            system="",  # Ledger braucht meist kein separates System-Prompt
            model_name=self.model,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            provider_base_url="https://api.openai.com/v1",
            temperature=0,
        )

        # 5. Das Ergebnis extrahieren (als JSON-String zurückgeben)
        updated_ledger = extract_data(vlm_response, "json")
        
        return updated_ledger
    
    def _get_plan_prompt(self, task):
        plan_prompt = f"""
        Create a high-level ACTION plan for the task: {task}
        
        CRITICAL RULES:
        1. Start DIRECTLY with the first physical interaction (e.g., "Double-click the icon" or "Type in search").
        2. Do NOT include 'Identify', 'Observe', or 'Determine' as separate steps. 
        3. Every step must describe a visible change or interaction on the screen.
        4. Keep it to max 5-7 steps.

        Output as a JSON dict:
        ```json
        {{
        "step 1": "First physical action to perform",
        "step 2": "..."
        }}
        ```
        """
        return plan_prompt

def _remove_som_images(messages):
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            msg["content"] = [
                cnt for cnt in msg_content 
                if not (isinstance(cnt, str) and 'som' in cnt and is_image_path(cnt))
            ]


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place
    """
    if images_to_keep is None:
        return messages

    total_images = 0
    for msg in messages:
        for cnt in msg.get("content", []):
            if isinstance(cnt, str) and is_image_path(cnt):
                total_images += 1
            elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                for content in cnt.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        total_images += 1

    images_to_remove = total_images - images_to_keep
    
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            new_content = []
            for cnt in msg_content:
                # Remove images from SOM or screenshot as needed
                if isinstance(cnt, str) and is_image_path(cnt):
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                # VLM shouldn't use anthropic screenshot tool so shouldn't have these but in case it does, remove as needed
                elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                    new_tool_result_content = []
                    for tool_result_entry in cnt.get("content", []):
                        if isinstance(tool_result_entry, dict) and tool_result_entry.get("type") == "image":
                            if images_to_remove > 0:
                                images_to_remove -= 1
                                continue
                        new_tool_result_content.append(tool_result_entry)
                    cnt["content"] = new_tool_result_content
                # Append fixed content to current message's content list
                new_content.append(cnt)
            msg["content"] = new_content