import os
import logging
import argparse
import shlex
import subprocess
from flask import Flask, request, jsonify, send_file
import threading
import traceback
import pyautogui
from PIL import Image
from io import BytesIO
import sys
import datetime

# Erstellt die Log-Datei im selben Ordner wie diese main.py
log_path = os.path.join(os.path.dirname(__file__), "agent_hintergrund_log.txt")
log_file = open(log_path, "a", encoding="utf-8")

log_file.write(f"\n\n{'='*40}\n--- AGENT NEUSTART: {datetime.datetime.now()} ---\n{'='*40}\n")
log_file.flush()

# Leite ALLES (Prints und Fehler) in diese Datei um
sys.stdout = log_file
sys.stderr = log_file

def execute_anything(data):
    shell = data.get('shell', False)
    command = data.get('command', "")
    
    try:
        # TRICK: Wenn "start" im Befehl vorkommt, nutzen wir Popen (nicht-blockierend)
        if isinstance(command, str) and command.startswith("start"):
            subprocess.Popen(command, shell=True)
            return jsonify({'status': 'success', 'message': 'Program started in background'})
        
        # Normale Befehle (whoami, dir) bleiben synchron für die Rückgabe
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                shell=shell, text=True, timeout=120)
        return jsonify({'status': 'success', 'output': result.stdout})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def execute(data):
    """Implementierung der UI-Steuerung via PyAutoGUI."""
    try:
        action = data.get('action')
        # Wir unterstützen beides: 'coordinate' als Liste oder direkt 'x' und 'y'
        coords = data.get('coordinate')
        x = data.get('x')
        y = data.get('y')
        
        # Falls eine Liste [x, y] kommt, entpacken wir sie
        if coords and len(coords) == 2:
            x, y = coords[0], coords[1]

        # WICHTIG: Hier muss ein normales 'if' stehen, kein 'elif'!
        if action in ['left_click', 'mouse_click']:
            if x is not None and y is not None:
                pyautogui.click(x, y)
            else:
                pyautogui.click()
            return jsonify({'status': 'success', 'message': 'Clicked'})

        elif action == 'double_click':
            if x is not None and y is not None:
                pyautogui.doubleClick(x, y)
            else:
                pyautogui.doubleClick()
            return jsonify({'status': 'success', 'message': 'Double-clicked'})

        elif action == 'right_click':
            if x is not None and y is not None:
                pyautogui.rightClick(x, y)
            else:
                pyautogui.rightClick()
            return jsonify({'status': 'success', 'message': 'Right-clicked'})
        
        elif action == 'wait':
            time.sleep(3)
            return jsonify({'status': 'success', 'message': 'Waited and observed UI'})

        # --- TASTATUR AKTIONEN ---
        elif action in ['type', 'type_text']:
            text = data.get('text', '')
            pyautogui.write(text, interval=0.05)
            return jsonify({'status': 'success', 'message': f'Typed text'})

        elif action in ['key', 'key_combination']:
            keys = data.get('keys', [])
            # Falls nur eine einzelne Taste als String kommt
            if isinstance(keys, str):
                pyautogui.press(keys)
            else:
                pyautogui.hotkey(*keys)
            return jsonify({'status': 'success'})

        # --- FALLBACK ---
        return jsonify({'status': 'error', 'message': f'Action {action} unknown'}), 400

    except Exception as e:
        logger.error(f"Fehler in Execute: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


execute_impl = execute  # switch to execute_anything to allow any command. Please use with caution only for testing purposes.


parser = argparse.ArgumentParser()
parser.add_argument("--log_file", help="log file path", type=str,
                    default=os.path.join(os.path.dirname(__file__), "server.log"))
parser.add_argument("--port", help="port", type=int, default=5050)
args = parser.parse_args()

logging.basicConfig(filename=args.log_file,level=logging.DEBUG, filemode='w' )
logger = logging.getLogger('werkzeug')

app = Flask(__name__)

computer_control_lock = threading.Lock()

@app.route('/probe', methods=['GET'])
def probe_endpoint():
    print(">>> Probe-Anfrage erhalten! Sende Antwort...")
    return jsonify({"status": "Probe successful", "message": "Service is operational"}), 200

@app.route('/execute', methods=['POST'])
def execute_command():
    """Dynamischer Verteiler für Shell-Befehle oder UI-Aktionen."""
    with computer_control_lock:
        data = request.json
        # Wir schauen, welcher Modus im JSON steht. Standard ist 'gui'.
        mode = data.get('mode', 'gui')

        if mode == 'unsafe' or mode == 'shell':
            logger.info("Executing Shell Command...")
            return execute_anything(data)
        else:
            logger.info("Executing GUI Action...")
            return execute(data)

@app.route('/screenshot', methods=['GET'])
def capture_screen_with_cursor():    
    cursor_path = os.path.join(os.path.dirname(__file__), "cursor.png")
    screenshot = pyautogui.screenshot()
    cursor_x, cursor_y = pyautogui.position()
    cursor = Image.open(cursor_path)
    # make the cursor smaller
    cursor = cursor.resize((int(cursor.width / 1.5), int(cursor.height / 1.5)))
    screenshot.paste(cursor, (cursor_x, cursor_y), cursor)

    # Convert PIL Image to bytes and send
    img_io = BytesIO()
    screenshot.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=args.port, debug=True)