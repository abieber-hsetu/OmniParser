import asyncio
from typing import Any, cast
from anthropic.types.beta import (
    BetaContentBlock, 
    BetaMessage, 
    BetaMessageParam, 
    BetaToolResultBlockParam
)
from tools import ComputerTool, ToolCollection, ToolResult

class OpenAIExecutor:
    def __init__(self, output_callback, tool_output_callback):
        self.tool_collection = ToolCollection(ComputerTool())
        self.output_callback = output_callback
        self.tool_output_callback = tool_output_callback

    def __call__(self, response: BetaMessage, messages: list[BetaMessageParam], parsed_screen: dict, vlm_response_json: dict):
        new_message = {
            "role": "assistant",
            "content": cast(list[Any], response.content),
        }
        if new_message not in messages:
            messages.append(new_message)

        tool_result_content = []
        
        for content_block in cast(list[BetaContentBlock], response.content):
            # Zeige den Tool-Use-Block im Chat an
            self.output_callback(content_block, sender="bot")
            
            if content_block.type == "tool_use":
                tool_input = dict(content_block.input)
                
                # --- AKTIONEN ÜBERSETZEN ---
                action_map = {
                    "hover": "mouse_move",
                    "click": "left_click",
                    "scroll_down": "scroll_down",
                    "scroll_up": "scroll_up"
                }
                current_action = tool_input.get("action")
                
                if current_action in action_map:
                    tool_input["action"] = action_map[current_action]
                    print(f">>> EXECUTOR: Aktion '{current_action}' vorbereitet.")
                
                if "scroll" in current_action:
                    if "clicks" not in tool_input:
                        tool_input["clicks"] = -400 if "down" in current_action else 400

                # --- KOORDINATEN-BERECHNUNG ---
                raw_box_id = vlm_response_json.get("Box ID") or vlm_response_json.get("box_id")
                if raw_box_id is not None:
                    try:
                        idx = int(raw_box_id)
                        coords_list = parsed_screen.get("coordinates", [])
                        box = None
                        if isinstance(coords_list, dict):
                            box = coords_list.get(str(idx)) or coords_list.get(idx)
                        elif isinstance(coords_list, list) and idx < len(coords_list):
                            box = coords_list[idx]

                        if box is not None:
                            img_width = parsed_screen.get("width", 1280)
                            img_height = parsed_screen.get("height", 800)
                            if len(box) >= 4:
                                center_x = box[0] + (box[2] / 2)
                                center_y = box[1] + (box[3] * 0.3)
                            else:
                                center_x, center_y = box[0], box[1]

                            if center_x <= 1.0 and center_y <= 1.0:
                                center_x *= img_width
                                center_y *= img_height
                            tool_input['coordinate'] = (min(max(int(center_x), 0), img_width - 1), 
                                                        min(max(int(center_y), 0), img_height - 1))

                    except (ValueError, TypeError):
                        print(f">>> EXECUTOR ERROR: Box ID '{raw_box_id}' ist ungültig.")

                # --- TOOL AUSFÜHREN ---
                try:
                    result = asyncio.run(self.tool_collection.run(
                        name=content_block.name,
                        tool_input=tool_input,
                    ))
                except Exception as e:
                    result = ToolResult(error=str(e))

                # --- GEFILTERTES LOGGING ---
                # HIER wird der "Buchstabiersalat" bei 'type' unterdrückt
                is_type_action = (current_action == "type")
                if not (is_type_action and "Pressed keys" in str(result.output)):
                    self.output_callback(result, sender="bot")
                
                # Ergebnis für die KI-Historie
                res_block: BetaToolResultBlockParam = {
                    "type": "tool_result",
                    "content": self._format_tool_output(result),
                    "tool_use_id": content_block.id,
                    "is_error": result.error is not None,
                }
                tool_result_content.append(res_block)

            yield [None, None], tool_result_content

    def _format_tool_output(self, result: ToolResult):
        output_blocks = []
        if result.output:
            output_blocks.append({"type": "text", "text": result.output})
        if result.error:
            output_blocks.append({"type": "text", "text": f"Error: {result.error}"})
        if result.base64_image:
            output_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": result.base64_image,
                },
            })
        return output_blocks