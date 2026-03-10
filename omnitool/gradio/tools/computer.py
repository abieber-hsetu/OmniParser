import base64
import time
try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

from typing import Literal, TypedDict

from PIL import Image

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .screen_capture import get_screenshot
import requests
import re

OUTPUT_DIR = "./tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
    "hover",
    "wait"
]


class Resolution(TypedDict):
    width: int
    height: int


MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]

class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    Adapted for Windows using 'pyautogui'.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self, is_scaling: bool = False):
        super().__init__()

        # Get screen width and height using Windows command
        self.display_num = None
        self.offset_x = 0
        self.offset_y = 0
        self.is_scaling = is_scaling
        self.width, self.height = self.get_screen_size()
        print(f"screen size: {self.width}, {self.height}")

        self.key_conversion = {"Page_Down": "pagedown",
                               "Page_Up": "pageup",
                               "Super_L": "win",
                               "Escape": "esc"}


    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        print(f"action: {action}, text: {text}, coordinate: {coordinate}, is_scaling: {self.is_scaling}")
        
        # 1. Koordinaten vorbereiten (für alle Maus-Aktionen)
        x, y = None, None
        if coordinate is not None:
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            
            if self.is_scaling:
                x, y = self.scale_coordinates(ScalingSource.API, coordinate[0], coordinate[1])
            else:
                x, y = coordinate

        # 2. KEY & TYPE Aktionen (Tastatur)
        if action in ("key", "type"):
            if text is None: raise ToolError(f"text is required for {action}")
            if action == "key":
                # ... (Deine bestehende Key-Logik ist okay)
                self.send_to_vm(f"pyautogui.hotkey(*{[k.strip().lower() for k in text.split('+')]})")
                return ToolResult(output=f"Pressed keys: {text}")
            elif action == "type":
                self.send_to_vm(f"pyautogui.click({x}, {y})") if x else self.send_to_vm("pyautogui.click()")
                self.send_to_vm(f"pyautogui.typewrite('{text}', interval=0.1)")
                self.send_to_vm("pyautogui.press('enter')")
                return ToolResult(output=text)

        # 3. MAUS Aktionen (Klicks & Bewegung)
        if action in ("mouse_move", "left_click", "right_click", "double_click", "middle_click", "left_click_drag"):
            
            # Für dragTo brauchen wir Start-Koordinaten
            if action == "left_click_drag":
                self.send_to_vm(f"pyautogui.dragTo({x}, {y}, duration=0.5)")
                return ToolResult(output=f"Dragged to ({x}, {y})")

            # MAPPING: Hier senden wir die Koordinaten MIT an den Agenten
            if action == "mouse_move":
                self.send_to_vm(f"pyautogui.moveTo({x}, {y}, duration=0.2)")
                return ToolResult(output=f"Moved mouse to ({x}, {y})")
            
            elif action == "left_click":
                self.send_to_vm(f"pyautogui.click({x}, {y})")
            elif action == "right_click":
                self.send_to_vm(f"pyautogui.rightClick({x}, {y})")
            elif action == "double_click":
                # WICHTIG: Hier klickt er jetzt wirklich auf das Icon!
                self.send_to_vm(f"pyautogui.doubleClick({x}, {y})")
            
            return ToolResult(output=f"Performed {action} at ({x}, {y})")

        # 4. SONSTIGE Aktionen (Screenshot, Wait, etc.)
        if action == "screenshot":
            return await self.screenshot()
        elif action == "wait":
            time.sleep(1)
            return ToolResult(output="Waited 1s")
            
        raise ToolError(f"Invalid action: {action}")

    def send_to_vm(self, action: str, mode: str = "shell"):
        """
        Sendet einen Befehl an den Windows-Agenten.
        'shell' (Standard): Startet einen neuen Python-Prozess in der VM (sicher, aber langsamer).
        'gui': Nutzt die interne Instanz des Agenten für sofortige Aktionen (schnell).
        """
        prefix = "import pyautogui; pyautogui.FAILSAFE = False;"
        parse = (action == "pyautogui.position()")
        
        # Payload-Konstruktion basierend auf dem Modus
        if mode == "shell":
            command_list = ["python", "-c", f"{prefix} {action}"]
            if parse:
                command_list[-1] = f"{prefix} print({action})"
            
            payload = {
                "mode": "shell",
                "command": command_list
            }
        else:
            # GUI-Modus: Hier schicken wir die Aktion direkt an den 'Butler'
            payload = {
                "mode": "gui",
                "action": action  # z.B. "mouse_move", "left_click", etc.
            }

        try:
            print(f"--- Sending to VM ({mode} mode) ---")
            # Port 5055 ist unser gemappter Agenten-Port
            response = requests.post(
                "http://localhost:5055/execute", 
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=90
            )
            
            # Kurze Pause, damit die VM Zeit hat, die UI-Änderung zu verarbeiten
            time.sleep(0.5) 
            
            if response.status_code != 200:
                print(f"Error: Agent responded with status {response.status_code}")
                raise ToolError(f"Failed to execute command. Status code: {response.status_code}")

            result_data = response.json()
            print(f"Action executed successfully.")

            # Falls wir die Position abfragen, parsen wir das Ergebnis aus dem Output
            if parse:
                output = result_data.get('output', '').strip()
                match = re.search(r'Point\(x=(\d+),\s*y=(\d+)\)', output)
                if not match:
                    raise ToolError(f"Could not parse coordinates from output: {output}")
                x, y = map(int, match.groups())
                return x, y
            
            return result_data

        except requests.exceptions.RequestException as e:
            print(f"Network Error: {str(e)}")
            raise ToolError(f"An error occurred while trying to execute the command: {str(e)}")

    async def screenshot(self):
        if not hasattr(self, 'target_dimension'):
            screenshot = self.padding_image(screenshot)
            self.target_dimension = MAX_SCALING_TARGETS["WXGA"]
        width, height = self.target_dimension["width"], self.target_dimension["height"]
        screenshot, path = get_screenshot(resize=True, target_width=width, target_height=height)
        time.sleep(0.7) # avoid async error as actions take time to complete
        return ToolResult(base64_image=base64.b64encode(path.read_bytes()).decode())

    def padding_image(self, screenshot):
        """Pad the screenshot to 16:10 aspect ratio, when the aspect ratio is not 16:10."""
        _, height = screenshot.size
        new_width = height * 16 // 10

        padding_image = Image.new("RGB", (new_width, height), (255, 255, 255))
        # padding to top left
        padding_image.paste(screenshot, (0, 0))
        return padding_image

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = None

        for target_name, dimension in MAX_SCALING_TARGETS.items():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                    self.target_dimension = target_dimension
                    # print(f"target_dimension: {target_dimension}")
                break

        if target_dimension is None:
            # TODO: currently we force the target to be WXGA (16:10), when it cannot find a match
            target_dimension = MAX_SCALING_TARGETS["WXGA"]
            self.target_dimension = MAX_SCALING_TARGETS["WXGA"]

        # should be less than 1
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)

    def get_screen_size(self):
        """Return width and height of the screen"""
        try:
            response = requests.post(
                f"http://localhost:5055/execute",
                headers={'Content-Type': 'application/json'},
                json={
                    "mode": "shell", # WICHTIG: Damit der Verteiler richtig abbiegt!
                    "command": ["python", "-c", "import pyautogui; print(pyautogui.size())"]
                },
                timeout=90
            )
            if response.status_code != 200:
                raise ToolError(f"Failed to get screen size. Status code: {response.status_code}")
            
            output = response.json()['output'].strip()
            match = re.search(r'Size\(width=(\d+),\s*height=(\d+)\)', output)
            if not match:
                raise ToolError(f"Could not parse screen size from output: {output}")
            width, height = map(int, match.groups())
            return width, height
        except requests.exceptions.RequestException as e:
            raise ToolError(f"An error occurred while trying to get screen size: {str(e)}")
