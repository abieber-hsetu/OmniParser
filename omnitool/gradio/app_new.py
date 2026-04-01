"""
The app contains:
- a new UI for the OmniParser AI Agent.
- 
python app_new.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000
"""

import os
import io
import shutil
import mimetypes
from datetime import datetime
try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum
from functools import partial
from pathlib import Path
from typing import cast, List, Optional
import argparse
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock
from loop import (
    APIProvider,
    sampling_loop_sync,
)
from tools import ToolResult
import requests
from requests.exceptions import RequestException
import base64
import subprocess
from dotenv import load_dotenv
from rag_manager import HsetuRagManager
import gc
import re
import html

# Setup and Check of tmp Path
def setup_gradio_temp():
    system_tmp = "/tmp/gradio"
    local_tmp = "./.gradio_tmp"
    
    try:
        # 1. Versuch: Teste, ob wir im System-Ordner schreiben dürfen
        # Wir versuchen, den Ordner zu erstellen oder eine Testdatei anzulegen
        os.makedirs(system_tmp, exist_ok=True)
        test_file = os.path.join(system_tmp, ".permissions_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        
        print(f"✅ System-Temp ({system_tmp}) ist bereit.")
        # Wenn alles okay ist, lassen wir alles beim Alten
        
    except (PermissionError, OSError):
        # 2. Fallback: Wenn der Zugriff verweigert wird
        print(f"⚠️ Zugriff auf {system_tmp} verweigert. Nutze lokalen Fallback: {local_tmp}")
        
        # Lokalen Ordner erstellen
        Path(local_tmp).mkdir(parents=True, exist_ok=True)
        
        # Gradio über die Umgebungsvariable umleiten
        os.environ["GRADIO_TEMP_DIR"] = os.path.abspath(local_tmp)

def get_safe_filepath(file_obj):
    """Robust way to get filepath across different Gradio versions"""
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict) and "path" in file_obj:
        return file_obj["path"]
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)

setup_gradio_temp()
import gradio as gr

load_dotenv() # Fix: missing parentheses added

rag_manager = HsetuRagManager()

CONFIG_DIR = Path("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

INTRO_TEXT = '''
<div style="text-align: center; margin-bottom: 10px;">
    <h2>OmniParser AI Agent</h2>
    <p>Turn any vision-language model into an AI agent. We currently support <b>OpenAI (4o/o1/o3-mini/gpt-5.4), DeepSeek (R1), Qwen (2.5VL) or Anthropic Computer Use (Sonnet)</b>.</p>
    <p>Type a message and press send to start OmniTool. Press stop to pause, and press the trash icon in the chat to clear the message history.</p>
    <p>You can also upload files for analysis using the file upload section.</p>
</div>
'''

def get_host_ip():
    try:
        return subprocess.check_output(['hostname', '-I']).decode('utf-8').split()[0]
    except:
        return "127.0.0.1"

def parse_arguments():
    host_ip = get_host_ip()
    parser = argparse.ArgumentParser(description="Gradio App")
    parser.add_argument("--windows_host_url", type=str, default=f"{host_ip}:8006")
    parser.add_argument("--omniparser_server_url", type=str, default="localhost:8000")
    parser.add_argument("--windows_agent_port", type=int, default=5055)
    parser.add_argument("--run_folder", type=str, default="./tmp/outputs")
    return parser.parse_args()
args = parse_arguments()

# Update upload folder from args if provided
RUN_FOLDER = Path(os.path.join(args.run_folder, datetime.now().strftime('%Y%m%d_%H%M')))
RUN_FOLDER.mkdir(parents=True, exist_ok=True)

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def load_existing_files():
    """Load all existing files from the uploads folder"""
    files = []
    if RUN_FOLDER.exists():
        for file_path in RUN_FOLDER.iterdir():
            if file_path.is_file():
                files.append(str(file_path))
    return files

def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "model" not in state:
        state["model"] = "omniparser + gpt-5.4-orchestrated"
    if "provider" not in state:
        state["provider"] = "openai"
    if "openai_api_key" not in state:  # Fetch API keys from environment variables
        state["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    if "anthropic_api_key" not in state:
        state["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
    if "api_key" not in state:
        state["api_key"] = ""
    if "auth_validated" not in state:
        state["auth_validated"] = False
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 2
    if 'chatbot_messages' not in state:
        state['chatbot_messages'] = []
    if 'stop' not in state:
        state['stop'] = False
    if 'instruction_steps' not in state:
        state['instruction_steps'] = []
    if 'current_step_index' not in state:
        state['current_step_index'] = 0
    if 'uploaded_files' not in state:
        state['uploaded_files'] = []  # Start with an empty list instead of loading existing files

async def main(state):
    """Render loop for Gradio"""
    setup_state(state)
    return "Setup completed"

def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key to continue."
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except DefaultCredentialsError:
            return "Your google cloud credentials are not set up correctly."

def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        print(f"Debug: Error loading {filename}: {e}")
    return None

def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        print(f"Debug: Error saving {filename}: {e}")

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output

def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    if message is None:
        return
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
    
        print(f"_render_message: {str(message)[:100]}")
        
        if isinstance(message, str):
            return message
        
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult)
            or message.__class__.__name__ == "ToolResult"
        )
        if not message or (
            is_tool_result
            and hide_images
            and not hasattr(message, "error")
            and not hasattr(message, "output")
        ):  # return None if hide_images is True
            return
        # render tool result
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"Error: {message.error}"
            if message.base64_image and not hide_images:
                return f'<img src="data:image/png;base64,{message.base64_image}">'

        elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
            # Format reasoning text 
            return f"Next step Reasoning: {message.text}"
        elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
            return None
        else:  
            return message

    def _truncate_string(s, max_length=500):
        """Truncate long strings for concise printing."""
        if isinstance(s, str) and len(s) > max_length:
            return s[:max_length] + "..."
        return s
    
    # processing Anthropic messages
    message = _render_message(message, hide_images)
    
    if sender == "bot":
        chatbot_state.append({"role": "assistant", "content": message})
    else:
        chatbot_state.append({"role": "user", "content": message})
    
    concise_state = [(_truncate_string(user_msg), _truncate_string(bot_msg))
                        for user_msg, bot_msg in chatbot_state]

def valid_params(user_input, state):
    errors = []
    
    # Extrahiere die IP aus der Host-URL (z.B. 172.18.0.1)
    host_ip = args.windows_host_url.split(':')[0]
    
    # Prüfe Windows Agent und OmniParser Server
    check_configs = [
        ('Windows Host', f"{host_ip}:{args.windows_agent_port}", "/probe"),
        ('OmniParser Server', args.omniparser_server_url, "/probe/") # Wichtig: Slash am Ende
    ]
    
    for server_name, url, endpoint in check_configs:
        try:
            full_url = f"http://{url.replace('http://', '')}{endpoint}"
            response = requests.get(full_url, timeout=3)
            if response.status_code != 200:
                errors.append(f"{server_name} antwortet mit {response.status_code}")
        except:
            errors.append(f"{server_name} ({url}) ist nicht erreichbar")
    
    if not state.get("api_key", "").strip():
        errors.append("LLM API Key fehlt")
    if not user_input:
        errors.append("Keine Anfrage eingegeben")
    
    return errors

# Verhindert das Nutzen des RAGs wenn diese Begriffe im Prompt vorkommen
system_keywords = ["öffne", "startmenü", "suche", "herunterladen", "installiere"]

def should_use_rag(user_prompt):
    # Einfacher Check: Wenn es nur um Windows-Basics geht, RAG weglassen
    if any(word in user_prompt.lower() for word in system_keywords):
        return False
    return True

def process_input(user_input, state):
    if not user_input:
        user_input = ""

    if state.get("stop"):
        state["stop"] = False

    errors = valid_params(user_input, state)
    if errors:
        raise gr.Error("Validation errors: " + ", ".join(errors))
    
    # Nachricht an den internen Log (für die KI)
    state["messages"].append({
        "role": "user",
        "content": [{"type": "text", "text": user_input}],
    })

    # Nachricht an den Chatbot (UI)
    state['chatbot_messages'].append({"role": "user", "content": user_input})
    
    def get_safe_chat():
        return [m for m in state['chatbot_messages'] if m is not None and m.get("content") is not None]

    yield get_safe_chat(), gr.update(choices=os.listdir(args.run_folder))

    if hasattr(state["model"], "set_instructions"):
        state["model"].set_instructions(
            steps=state.get("instruction_steps", []),
            program_name=state.get("program_name") 
        )

    # Run sampling_loop_sync
    for loop_msg in sampling_loop_sync(
        model=state["model"],
        provider=state["provider"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state['chatbot_messages'], hide_images=False),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        max_tokens=16384,
        omniparser_url=args.omniparser_server_url,
        instruction_steps=state.get("instruction_steps", []),
        current_step_index=state.get("current_step_index", 0),
        save_folder=str(RUN_FOLDER)
    ):  
        if state.get("stop"):
            break

        if loop_msg is None:
            file_choices_update = detect_new_files(state)
            yield gr.skip(), file_choices_update
            continue

        # Vor jedem UI-Update reinigen wir die Liste
        yield get_safe_chat(), gr.update()
            
    # Ende der Aufgabe
    print("End of task. Close the loop.")
    file_choices_update = detect_new_files(state)
    yield get_safe_chat(), file_choices_update

def stop_app(state):
    state["stop"] = True
    return "App stopped"

def get_header_image_base64():
    try:
        script_dir = Path(__file__).parent
        image_path = script_dir.parent.parent / "imgs" / "header_bar_thin.png"
        
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f'data:image/png;base64,{encoded_string}'
    except Exception as e:
        print(f"Failed to load header image: {e}")
        return None

def get_file_viewer_html(file_path=None):
    """Generate HTML to view a file based on its type"""
    if not file_path:
        return f'<iframe src="http://{args.windows_host_url}/vnc.html?view_only=1&autoconnect=1&resize=scale" width="100%" height="580" allow="fullscreen"></iframe>'
    
    file_path = Path(file_path)
    if not file_path.exists():
        return f'<div class="error-message">File not found: {file_path.name}</div>'
    
    mime_type, _ = mimetypes.guess_type(file_path)
    file_type = mime_type.split('/')[0] if mime_type else 'unknown'
    file_extension = file_path.suffix.lower()
    
    if file_type == 'image':
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f'<div class="file-viewer"><h3>{file_path.name}</h3><img src="data:{mime_type};base64,{encoded_string}" style="max-width:100%; max-height:500px;"></div>'
    
    elif file_extension in ['.txt', '.py', '.js', '.html', '.css', '.json', '.md', '.csv'] or file_type == 'text':
        try:
            content = file_path.read_text(errors='replace')
            content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            highlight_class = ""
            if file_extension == '.py': highlight_class = "language-python"
            elif file_extension == '.js': highlight_class = "language-javascript"
            elif file_extension == '.html': highlight_class = "language-html"
            elif file_extension == '.css': highlight_class = "language-css"
            elif file_extension == '.json': highlight_class = "language-json"
            
            return f'''
            <div class="file-viewer">
                <h3>{file_path.name}</h3>
                <pre class="{highlight_class}" style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto; max-height: 500px; white-space: pre-wrap;"><code>{content}</code></pre>
            </div>
            '''
        except UnicodeDecodeError:
            return f'<div class="error-message">Cannot display binary file: {file_path.name}</div>'
    
    elif file_extension == '.pdf':
        try:
            with open(file_path, "rb") as pdf_file:
                encoded_string = base64.b64encode(pdf_file.read()).decode()
                return f'''
                <div class="file-viewer">
                    <h3>{file_path.name}</h3>
                    <iframe src="data:application/pdf;base64,{encoded_string}" width="100%" height="500px" style="border: none;"></iframe>
                </div>
                '''
        except Exception as e:
            return f'<div class="error-message">Error displaying PDF: {str(e)}</div>'
    else:
        size_kb = file_path.stat().st_size / 1024
        return f'<div class="file-viewer"><h3>{file_path.name}</h3><p>File type: {mime_type or "Unknown"}</p><p>Size: {size_kb:.2f} KB</p><p>This file type cannot be displayed in the browser.</p></div>'

def handle_file_upload(files, state, progress=gr.Progress()):
    if not files:
        return gr.update(choices=[])
    
    docs_dir = Path("./docs")
    archive_dir = Path("./archive")
    
    docs_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_file_names = []

    progress(0, desc="Dateien werden vorbereitet...")
    for file_obj in files:
        # NEU: Auch hier der sichere Pfad
        file_path_str = get_safe_filepath(file_obj)
        file_name = Path(file_path_str).name
        
        uploaded_file_names.append(file_name)
        shutil.copy(file_path_str, RUN_FOLDER / file_name)
        shutil.copy(file_path_str, docs_dir / file_name)
        
        if file_path_str not in state['uploaded_files']:
            state['uploaded_files'].append(file_path_str)

    gc.collect() 
    progress(0.5, desc="KI lernt neue Inhalte...")
    try:
        rag_manager.update_database(docs_folder="./docs")
    except Exception as e:
        print(f"RAG-Fehler: {e}")
        gr.Warning(f"Fehler bei der Analyse: {e}")

    progress(0.9, desc="Verschiebe Dateien ins Archiv...")
    for file_name in uploaded_file_names:
        source = docs_dir / file_name
        destination = archive_dir / file_name
        
        try:
            if destination.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                destination = archive_dir / f"{timestamp}_{file_name}"
            shutil.move(str(source), str(destination))
        except Exception as e:
            print(f"Fehler beim Verschieben von {file_name}: {e}")

    gr.Info("✅ Neue Informationen wurden gelernt und archiviert.")
    all_file_choices = [(Path(p).name, p) for p in state['uploaded_files']]
    return gr.update(choices=all_file_choices)

def toggle_view(view_mode, file_path=None, state=None):
    file_choices_update = gr.update()
    if view_mode == "File Viewer" and state is not None:
        file_choices_update = detect_new_files(state)
    
    if view_mode == "OmniTool Computer":
        return get_file_viewer_html(), file_choices_update
    else:
        if file_path:
            return get_file_viewer_html(file_path), file_choices_update
        else:
            return get_file_viewer_html(), file_choices_update

def detect_new_files(state):
    new_files_count = 0
    if RUN_FOLDER.exists():
        current_files = set(state['uploaded_files'])
        for file_path in RUN_FOLDER.iterdir():
            if file_path.is_file():
                file_path_str = str(file_path)
                if file_path_str not in current_files:
                    state['uploaded_files'].append(file_path_str)
                    new_files_count += 1
    
    file_choices = [(Path(path).name, path) for path in state['uploaded_files']]
    return gr.update(choices=file_choices)

def refresh_files(state):
    return detect_new_files(state)


# =========================================================================
# GRADIO UI SETUP
# =========================================================================
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.HTML("""
        <style>
        .left-upload-area {
            flex-grow: 1 !important; 
            min-height: 250px !important;
            margin-bottom: 10px !important;
        }
        
        .fixed-header-text {
            margin-bottom: 5px !important;
            margin-top: 10px !important;
            font-weight: bold;
        }

        /* DIE EINZIGE SCROLL-BOX */
        .scroll-box {
            max-height: 400px !important; 
            overflow-y: auto !important; 
            border: 1px solid var(--border-color-primary, #4b5563) !important; 
            border-radius: 8px !important;
            background-color: transparent !important; 
            padding: 15px !important;
            box-sizing: border-box !important; 
        }

        .placeholder-center {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100px;
            color: #9ca3af; 
            font-style: italic;
            font-size: 1.2em;
        }

        .list-active {
            text-align: left;
            color: var(--body-text-color, #e5e7eb); 
            line-height: 1.6;
            padding-bottom: 10px !important; 
        }

        /* --- DER ULTIMATIVE KILL-SWITCH --- */
        /* Verbietet der äußeren Spalte und all ihren Hilfs-Containern JEDEN Scrollbalken! */
        .no-outer-scroll {
            overflow-y: hidden !important;
        }
        .no-outer-scroll > div,
        .no-outer-scroll > .wrap {
            overflow-y: hidden !important;
        }
        
        </style>
    """)
    state = gr.State({})
    
    setup_state(state.value)
    
    header_image = get_header_image_base64()
    if header_image:
        gr.HTML(f'<img src="{header_image}" alt="OmniTool Header" width="100%">', elem_classes="no-padding")
        gr.HTML('<h1 style="text-align: center; font-weight: normal; margin-bottom: 20px;">Omni<span style="font-weight: bold;">Tool</span></h1>')
    else:
        gr.Markdown("# OmniTool", elem_classes="text-center")

    if not os.getenv("HIDE_WARNING", False):
        gr.HTML(INTRO_TEXT, elem_classes="markdown-text")

    # Settings Section
    with gr.Accordion("Settings", open=True, elem_classes="accordion-header"): 
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    label="Model",
                    choices=["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", "omniparser + qwen2.5vl", "claude-3-5-sonnet-20241022", "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated", "omniparser + gpt-5.4-orchestrated", "omniparser + R1-orchestrated", "omniparser + qwen2.5vl-orchestrated"],
                    value="omniparser + gpt-5.4-orchestrated",
                    interactive=True,
                    container=True
                )
            with gr.Column():
                only_n_images = gr.Slider(
                    label="N most recent screenshots",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    interactive=True
                )
        with gr.Row():
            with gr.Column(1):
                provider = gr.Dropdown(
                    label="API Provider",
                    choices=[option.value for option in APIProvider],
                    value="openai",
                    interactive=False,
                    container=True
                )
            with gr.Column(2):
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    value=state.value.get("api_key", ""),
                    placeholder="Paste your API key here",
                    interactive=True,
                    container=True
                )

    # File Upload Section
    with gr.Accordion("File Upload & Management", open=True):
        with gr.Row(equal_height=True): 
            
            # LINKE SPALTE
            with gr.Column():
                file_upload = gr.File(
                    label="Upload Files", 
                    file_count="multiple", 
                    elem_classes="left-upload-area"
                )
                with gr.Row():
                    upload_button = gr.Button("Upload Files", variant="primary")
                    refresh_button = gr.Button("Refresh Files", variant="secondary")

            # RECHTE SPALTE (HIER WIRD DER KILL-SWITCH AKTIVIERT)
            with gr.Column(elem_classes="no-outer-scroll"):
                instruction_upload = gr.File(
                    label="Anweisung.pdf hochladen", 
                    file_count="single",
                    type="filepath" 
                )
                
                gr.Markdown("### 📋 Erfasste Testschritte", elem_classes="fixed-header-text")
                
                # DIE EINZIGE SCROLLBOX
                instruction_status = gr.HTML(
                    value='<div class="placeholder-center">Warte auf Anweisungs-PDF...</div>',
                    elem_id="instruction-steps-list",
                    elem_classes="scroll-box"
                )
        with gr.Row():
            view_file_dropdown = gr.Dropdown(
                label="View File",
                choices=[],
                interactive=True,
                container=True
            )
            view_toggle = gr.Radio(
                label="Display Mode",
                choices=["OmniTool Computer", "File Viewer"],
                value="OmniTool Computer",
                interactive=True
            )

    # Prompt Line
    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(
                show_label=False, 
                placeholder="Type a message to send to Omniparser + X ...", 
                container=False
            )
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary", elem_classes="primary-button")
        with gr.Column(scale=1, min_width=50):
            stop_button = gr.Button(value="Stop", variant="secondary", elem_classes="secondary-button")

    # Chat
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chatbot History", 
                autoscroll=True, 
                height=580,
                avatar_images=("👤", "🤖")
            )
        with gr.Column(scale=3):
            display_area = gr.HTML(
                get_file_viewer_html(),
                elem_classes="no-padding"
            )

    def update_model(model_selection, state):
        state["model"] = model_selection
        print(f"Model updated to: {state['model']}")
        
        if model_selection == "claude-3-5-sonnet-20241022":
            provider_choices = [option.value for option in APIProvider if option.value != "openai"]
        elif model_selection in set(["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated", "omniparser + gpt-5.4-orchestrated"]):
            provider_choices = ["openai"]
        elif model_selection == "omniparser + R1":
            provider_choices = ["groq"]
        elif model_selection == "omniparser + qwen2.5vl":
            provider_choices = ["dashscope"]
        else:
            provider_choices = [option.value for option in APIProvider]
        default_provider_value = provider_choices[0]

        provider_interactive = len(provider_choices) > 1
        api_key_placeholder = f"{default_provider_value.title()} API Key"

        state["provider"] = default_provider_value
        state["api_key"] = state.get(f"{default_provider_value}_api_key", "")

        provider_update = gr.update(
            choices=provider_choices,
            value=default_provider_value,
            interactive=provider_interactive
        )
        api_key_update = gr.update(
            placeholder=api_key_placeholder,
            value=state["api_key"]
        )

        return provider_update, api_key_update

    def update_only_n_images(only_n_images_value, state):
        state["only_n_most_recent_images"] = only_n_images_value
   
    def update_provider(provider_value, state):
        state["provider"] = provider_value
        state["api_key"] = state.get(f"{provider_value}_api_key", "")
        
        api_key_update = gr.update(
            placeholder=f"{provider_value.title()} API Key",
            value=state["api_key"]
        )
        return api_key_update
                
    def update_api_key(api_key_value, state):
        state["api_key"] = api_key_value
        state[f'{state["provider"]}_api_key'] = api_key_value

    def clear_chat(state):
        state["messages"] = []
        state["responses"] = {}
        state["tools"] = {}
        state['chatbot_messages'] = []
        return state['chatbot_messages']

    def view_file(file_path, view_mode):
        if view_mode == "File Viewer" and file_path:
            return get_file_viewer_html(file_path)
        elif view_mode == "OmniTool Computer":
            return get_file_viewer_html()
        else:
            return display_area.value

    def update_view_file_dropdown(uploaded_files):
        if not uploaded_files:
            return gr.update(choices=[])
        
        file_choices = [(Path(path).name, path) for path in uploaded_files]
        return gr.update(choices=file_choices)
    
    def get_safe_filepath(file_obj):
        """Kugelsicherer Weg, um den Dateipfad in ALLEN Gradio-Versionen zu bekommen"""
        if file_obj is None: return None
        if isinstance(file_obj, str): return file_obj
        if hasattr(file_obj, "path"): return file_obj.path # <-- WICHTIG FÜR GRADIO 4!
        if isinstance(file_obj, dict) and "path" in file_obj: return file_obj["path"]
        if hasattr(file_obj, "name"): return file_obj.name
        return str(file_obj)

    def handle_instruction_upload(file_path, state, progress=gr.Progress()):
        if not file_path:
            return state, '<div class="placeholder-center">Warte auf Anweisungs-PDF...</div>'
        
        try:
            progress(0, desc="Lese PDF-Struktur...")
            instruction_data = rag_manager.parse_instruction_pdf(file_path)
            list_of_steps = instruction_data.get("steps", [])
            prog_name = instruction_data.get("program_name", "Unbekanntes Programm")
            
        except Exception as e:
            print(f"Absturz im PDF Parser: {e}")
            return state, f'<div class="placeholder-center" style="color: #ef4444;">❌ Fehler beim Lesen: {str(e)}</div>'
        
        if not list_of_steps:
            return state, '<div class="placeholder-center" style="color: #ef4444;">❌ Keine Inhalte in der PDF gefunden.</div>'
        
        state["instruction_steps"] = list_of_steps
        state["current_step_index"] = 0
        
        progress(0.7, desc="Strukturiere Testplan...")
        
        summary = '<div class="list-active">'
        summary += f'<h3 style="margin-bottom: 5px;">✅ Fahrplan für {prog_name}</h3>'
        summary += '<hr style="border: 0; border-top: 1px solid #374151; margin-bottom: 15px;">'
        
        for i, step in enumerate(list_of_steps):
            clean_step = re.sub(r'(?<![•\-\*|])\n(?![•\-\*|])', ' ', step)
            clean_step = re.sub(r'\s+', ' ', clean_step).strip()
            
            import html
            clean_step = html.escape(clean_step)
            
            if any(marker in clean_step for marker in ['|', '•', '- ']):
                parts = re.split(r'\s*[|•]|\s+-\s+', clean_step)
                header = parts[0].strip()
                items = [p.strip() for p in parts[1:] if p.strip()]
                
                item_html = "".join([f'<li style="margin-bottom: 5px;">{it}</li>' for it in items])
                content_html = f'{header}<ul style="margin-top: 8px; padding-left: 20px; color: #9ca3af;">{item_html}</ul>'
            else:
                content_html = clean_step

            summary += f'''
                <div style="margin-bottom: 25px;">
                    <div style="font-weight: bold; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.05em;">
                        Schritt {i+1}
                    </div>
                    <div style="font-size: 1.05em; line-height: 1.6; margin-top: 4px;">
                        {content_html}
                    </div>
                </div>
            '''
        
        summary += '</div>'
        progress(1.0, desc="Testplan bereit.")
        return state, summary

    model.change(fn=update_model, inputs=[model, state], outputs=[provider, api_key])
    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    provider.change(fn=update_provider, inputs=[provider, state], outputs=api_key)
    api_key.change(fn=update_api_key, inputs=[api_key, state], outputs=None)
    chatbot.clear(fn=clear_chat, inputs=[state], outputs=[chatbot])

    upload_button.click(
        fn=handle_file_upload,
        inputs=[file_upload, state],
        outputs=[view_file_dropdown]
    )

    instruction_upload.change(
        fn=handle_instruction_upload, 
        inputs=[instruction_upload, state], 
        outputs=[state, instruction_status]
    )
    
    view_file_dropdown.change(
        fn=view_file,
        inputs=[view_file_dropdown, view_toggle],
        outputs=[display_area]
    )
    
    submit_button.click(process_input, [chat_input, state], [chatbot, view_file_dropdown])
    stop_button.click(stop_app, [state], None)
    
    view_toggle.change(
        fn=toggle_view, 
        inputs=[view_toggle, view_file_dropdown, state], 
        outputs=[display_area, view_file_dropdown]
    )
    
    refresh_button.click(fn=refresh_files, inputs=[state], outputs=[view_file_dropdown])
    
    js_refresh = """
    function() {
        const refreshInterval = setInterval(function() {
            const refreshButtons = document.querySelectorAll('button');
            for (const button of refreshButtons) {
                if (button.textContent.includes('Refresh Files')) {
                    button.click();
                    break;
                }
            }
        }, 5000);
        
        return () => clearInterval(refreshInterval);
    }
    """
    
    gr.HTML("<script>(" + js_refresh + ")();</script>")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7888)