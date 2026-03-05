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
            self.output_callback(content_block, sender="bot")
            
            if content_block.type == "tool_use":
                tool_input = dict(content_block.input)
                
                # --- AKTIONEN ÜBERSETZEN ---
                action_map = {
                    "hover": "mouse_move",
                    "click": "left_click"
                }
                current_action = tool_input.get("action")
                if current_action in action_map:
                    tool_input["action"] = action_map[current_action]
                    print(f">>> EXECUTOR TRANSLATION: '{current_action}' übersetzt in '{tool_input['action']}'")

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
                                # --- WIR WISSEN JETZT: Format ist [x, y, breite, höhe] ---
                                x_min, y_min = box[0], box[1]
                                width, height = box[2], box[3]

                                # Wir zielen auf das obere Drittel (30%), um immer das Icon zu treffen!
                                center_x = x_min + (width / 2)
                                center_y = y_min + (height * 0.3)
                            else:
                                center_x, center_y = box[0], box[1]

                            # Ratios in echte Pixel umrechnen
                            if center_x <= 1.0 and center_y <= 1.0:
                                center_x *= img_width
                                center_y *= img_height

                            # Sicherheitsnetz (Clamping)
                            safe_x = min(max(int(center_x), 0), img_width - 1)
                            safe_y = min(max(int(center_y), 0), img_height - 1)

                            tool_input['coordinate'] = (safe_x, safe_y)
                            print(f">>> EXECUTOR MATCH: Box {idx} (Rohdaten: {box}) -> Pixel {tool_input['coordinate']}")
                        else:
                            print(f">>> EXECUTOR ERROR: Box {idx} nicht gefunden.")
                    
                    except (ValueError, TypeError):
                        print(f">>> EXECUTOR ERROR: Box ID '{raw_box_id}' ist ungültig.")

                try:
                    result = asyncio.run(self.tool_collection.run(
                        name=content_block.name,
                        tool_input=tool_input,
                    ))
                except Exception as e:
                    result = ToolResult(error=str(e))

                self.output_callback(result, sender="bot")
                
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