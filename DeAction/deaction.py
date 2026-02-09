import ast
import base64
import copy
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
import tokenize
from io import BytesIO, StringIO

from PIL import Image, ImageDraw, ImageFont

from DeAction.base_guardrail import BaseGuardrail
from DeAction.prompts import FAST_CHECK_SYSTEM_PROMPT, SYSTEMATIC_ANALYSIS_SYSTEM_PROMPT
from DeAction.utils import encode_image, COORDINATE_ACTION_LABELS, ANNOTATION_COLORS


logger = logging.getLogger("desktopenv.agent.deaction")

COORDINATE_ACTION_LABELS = {
    "click": "click",
    "doubleClick": "double-click",
    "rightClick": "right-click",
    "middleClick": "middle-click",
    "moveTo": "move cursor",
    "moveRel": "move cursor (relative)",
    "dragTo": "drag",
    "dragRel": "drag (relative)",
    "mouseDown": "mouse down",
    "mouseUp": "mouse up",
}

ANNOTATION_COLORS = [
    (239, 71, 111),   # pink/red
    (63, 193, 201),   # teal
    (255, 196, 0),    # amber
    (87, 117, 144),   # slate
    (27, 153, 139),   # green
    (244, 162, 97),   # orange
]

class DeAction(BaseGuardrail):
    def __init__(
        self,
        model: str = "gpt-5",
        model_reasoning_effort: Optional[str] = None,
        fast_check_reasoning_effort: Optional[str] = "medium",
        max_retry: int = 3,
        fast_check_model: Optional[str] = None,
        annotate_actions: bool = True,
    ):
        """
        Initialize the DeAction systematic analysis.
        
        Args:
            model: Model to use for systematic analysis (format: provider|model or just model for azure)
            max_retry: Maximum number of retries when action is misaligned
            fast_check_model: Optional lightweight model spec used for fast alignment checks
            annotate_actions: Overlay coordinate annotations on screenshots when actions use explicit coordinates
        """
        systematic_analysis_disabled = model is None
        if isinstance(model, str) and model.strip().lower() in ("", "none"):
            systematic_analysis_disabled = True

        safe_model = model if isinstance(model, str) and model.strip() else "gpt-5"
        super().__init__(
            model=safe_model,
            model_reasoning_effort=model_reasoning_effort,
            max_retry=max_retry,
            logger=logger,
            guardrail_name="systematic_analysis"
        )
        self.annotate_actions = annotate_actions
        self.systematic_analysis_disabled = systematic_analysis_disabled
        self.fast_check_reasoning_effort = fast_check_reasoning_effort
        if self.systematic_analysis_disabled:
            self.logger.info("[DeAction] Systematic analysis model disabled; fast-check-only mode enabled")
        self.fast_check_client = None
        self.fast_check_provider = None
        self.fast_check_model = None
        self.system_prompt = SYSTEMATIC_ANALYSIS_SYSTEM_PROMPT

        fast_check_disabled = fast_check_model is None
        if isinstance(fast_check_model, str) and fast_check_model.strip().lower() == "none":
            fast_check_disabled = True

        if fast_check_disabled:
            self.logger.info("[DeAction] Fast check model disabled; skipping fast checks")
        else:
            fast_check_model_spec = fast_check_model
            self.fast_check_provider, self.fast_check_model = self._parse_model_spec(fast_check_model_spec)
            if (self.fast_check_provider, self.fast_check_model) == (self.provider, self.model):
                self.fast_check_client = self.client
            else:
                self.fast_check_client = self._create_client(self.fast_check_provider, self.fast_check_model)
            self.logger.info(
                "[DeAction] Initialized fast check model with provider %s and model %s",
                self.fast_check_provider,
                self.fast_check_model,
            )
    
    def _probe_image_bytes(self, payload: bytes) -> Dict[str, object]:
        info: Dict[str, object] = {"byte_len": len(payload)}
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            info["signature"] = "png"
        elif payload.startswith(b"\xff\xd8"):
            info["signature"] = "jpeg"
        else:
            info["signature"] = "unknown"
        try:
            with Image.open(BytesIO(payload)) as img:
                info["format"] = img.format
                info["mode"] = img.mode
                info["size"] = img.size
        except Exception as exc:
            info["pil_error"] = str(exc)
        return info

    def _summarize_image_payload(self, payload: object) -> Dict[str, object]:
        summary: Dict[str, object] = {"type": type(payload).__name__}
        if payload is None:
            return summary
        if isinstance(payload, bytes):
            summary.update(self._probe_image_bytes(payload))
            return summary
        if isinstance(payload, str):
            summary["str_len"] = len(payload)
            encoded = payload
            if payload.startswith("data:image"):
                _, _, encoded = payload.partition(",")
            summary["base64_len"] = len(encoded)
            try:
                decoded = base64.b64decode(encoded, validate=True)
                summary["decoded_len"] = len(decoded)
                summary.update(self._probe_image_bytes(decoded))
            except Exception as exc:
                summary["decode_error"] = str(exc)
            return summary
        return summary

    def _summarize_history_payloads(self, payloads: List[object], max_items: int = 3) -> Dict[str, object]:
        summary: Dict[str, object] = {"count": len(payloads)}
        if not payloads:
            return summary
        sample = [self._summarize_image_payload(payload) for payload in payloads[:max_items]]
        summary["sample"] = sample
        if len(payloads) > max_items:
            summary["omitted"] = len(payloads) - max_items
        return summary

    def supports_narrative_memory(self) -> bool:
        return True

    def preprocess_remove_comments(self, code: str) -> str:
        """Remove comments from Python code while preserving strings"""
        if not code or not isinstance(code, str):
            return code

        result = []
        sio = StringIO(code)

        for tok_type, tok_string, _, _, _ in tokenize.generate_tokens(sio.readline):
            # Skip comments
            if tok_type == tokenize.COMMENT:
                continue
            # Skip NL (standalone comment lines, etc.)
            if tok_type == tokenize.NL:
                continue
            result.append(tok_string)

        return "".join(result)

    def _node_to_number(self, node: ast.AST) -> Optional[float]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = self._node_to_number(node.operand)
            if operand is None:
                return None
            return operand if isinstance(node.op, ast.UAdd) else -operand
        return None

    def _extract_call_coordinates(self, call_node: ast.Call) -> Optional[Tuple[int, int]]:
        x_value = None
        y_value = None
        for keyword in call_node.keywords:
            if keyword.arg == "x":
                x_value = self._node_to_number(keyword.value)
            elif keyword.arg == "y":
                y_value = self._node_to_number(keyword.value)
        if x_value is None or y_value is None:
            positional_numbers = []
            for arg in call_node.args:
                arg_value = self._node_to_number(arg)
                if arg_value is not None:
                    positional_numbers.append(arg_value)
            if len(positional_numbers) >= 2:
                x_value = positional_numbers[0] if x_value is None else x_value
                y_value = positional_numbers[1] if y_value is None else y_value
        if x_value is None or y_value is None:
            return None
        return int(round(x_value)), int(round(y_value))

    def _extract_coordinate_actions(self, agent_output: str):
        if not agent_output or not isinstance(agent_output, str):
            return []
        try:
            tree = ast.parse(agent_output)
        except SyntaxError:
            return []

        attribute_aliases = {"pyautogui"}
        direct_call_map = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "pyautogui":
                        attribute_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "pyautogui":
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    alias_name = alias.asname or alias.name
                    direct_call_map[alias_name] = alias.name

        actions = []

        outer_self = self

        class CoordinateVisitor(ast.NodeVisitor):
            def visit_Call(self, node):  # noqa: N802
                target_name = None
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id in attribute_aliases:
                        target_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    original = direct_call_map.get(node.func.id)
                    if original:
                        target_name = original

                if target_name and target_name in COORDINATE_ACTION_LABELS:
                    coords = outer_self._extract_call_coordinates(node)
                    if coords:
                        actions.append(
                            {
                                "x": coords[0],
                                "y": coords[1],
                                "label": COORDINATE_ACTION_LABELS.get(target_name, target_name),
                            }
                        )
                self.generic_visit(node)

        CoordinateVisitor().visit(tree)
        return actions

    def _load_font(self, size: int, bold: bool = False):
        """Try to load a readable TrueType font; fall back to default bitmap font."""
        font_candidates = [
            "Arial Bold.ttf",
            "Arial.ttf",
            "HelveticaNeue.ttc",
            "Helvetica.ttc",
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        for font_name in font_candidates:
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _annotate_screenshot_with_actions(self, screenshot_data, agent_output: str):
        coordinate_actions = self._extract_coordinate_actions(agent_output)
        if not coordinate_actions:
            return None

        if isinstance(screenshot_data, bytes):
            image_bytes = screenshot_data
        elif isinstance(screenshot_data, str):
            try:
                image_bytes = base64.b64decode(screenshot_data)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("[DeAction] Failed to decode screenshot for annotation: %s", exc)
                return None
        else:
            return None

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("[DeAction] Failed to open screenshot for annotation: %s", exc)
            return None

        draw = ImageDraw.Draw(image, "RGBA")
        label_font = self._load_font(size=24, bold=True)
        number_font = self._load_font(size=18)
        radius = 16
        width, height = image.size

        def clamp(value, lower, upper):
            return max(lower, min(upper, value))

        def measure(text, font):
            if hasattr(draw, "textbbox"):
                box = draw.textbbox((0, 0), text, font=font)
                return box[2] - box[0], box[3] - box[1]
            return draw.textsize(text, font=font)

        for idx, action in enumerate(coordinate_actions, start=1):
            x = clamp(action["x"], 0, width - 1)
            y = clamp(action["y"], 0, height - 1)
            color = ANNOTATION_COLORS[(idx - 1) % len(ANNOTATION_COLORS)]
            circle_bounds = (x - radius, y - radius, x + radius, y + radius)
            draw.ellipse(circle_bounds, fill=color + (200,), outline=(255, 255, 255, 255), width=3)

            number_text = str(idx)
            number_w, number_h = measure(number_text, number_font)
            number_pos = (x - number_w / 2, y - number_h / 2)
            draw.text(number_pos, number_text, font=number_font, fill=(255, 255, 255, 255))

            label_text = f"{action['label']} ({action['x']}, {action['y']})"
            label_w, label_h = measure(label_text, label_font)
            text_x = x + radius + 6
            if text_x + label_w > width:
                text_x = max(0, x - radius - 6 - label_w)
            text_y = y - label_h / 2
            text_y = clamp(text_y, 0, max(0, height - label_h))

            draw.text(
                (text_x, text_y),
                label_text,
                font=label_font,
                fill=(220, 0, 0, 255),
                stroke_width=2,
                stroke_fill=(255, 255, 255, 200),
            )

        output = BytesIO()
        image.convert("RGB").save(output, format="PNG")
        return output.getvalue()

    def _run_fast_check(self, user_message_content):
        """Run the lightweight fast check model to decide if full analysis is needed."""
        if not self.fast_check_client:
            return None

        fast_check_messages = [
            {
                "role": "system",
                "content": FAST_CHECK_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": copy.deepcopy(user_message_content)
            }
        ]

        fast_check_response_content = ""
        try:
            logger.info(
                "[DeAction] Running fast check model %s|%s to check alignment",
                self.fast_check_provider,
                self.fast_check_model,
            )

            if self.fast_check_provider in ("azure", "openai", "local"):
                request_kwargs = {}
                if self.fast_check_reasoning_effort:
                    request_kwargs["reasoning_effort"] = self.fast_check_reasoning_effort
                fast_check_response_obj = self.fast_check_client.chat.completions.create(
                    model=self.fast_check_model,
                    messages=fast_check_messages,
                    **request_kwargs,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "fast_check_response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "thought": {"type": "string"},
                                    "align": {"type": "boolean"}
                                },
                                "required": ["align", "thought"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }
                    }
                )
                fast_check_response_content = fast_check_response_obj.choices[0].message.content
                fast_check_result = json.loads(fast_check_response_content)

            elif self.fast_check_provider == "together":
                fast_check_messages[0]["content"] += '\n\nIMPORTANT: Respond with a valid JSON object containing exactly these keys: "thought" (string) and "align" (boolean).'
                fast_check_response_obj = self.fast_check_client.chat.completions.create(
                    model=self.fast_check_model,
                    messages=fast_check_messages,
                    temperature=0.1,
                    max_tokens=1000
                )
                fast_check_response_content = fast_check_response_obj.choices[0].message.content
                try:
                    fast_check_result = json.loads(fast_check_response_content)
                except json.JSONDecodeError:
                    import re
                    json_match = re.search(r'\{[^{}]*"align"[^{}]*\}', fast_check_response_content, re.DOTALL)
                    if json_match:
                        fast_check_result = json.loads(json_match.group())
                    else:
                        fast_check_result = {
                            "thought": fast_check_response_content,
                            "align": "true" in fast_check_response_content.lower()
                        }

            elif self.fast_check_provider == "aws":
                claude_messages = []
                system_message = None

                for msg in fast_check_messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                        continue
                    claude_msg = {"role": msg["role"], "content": []}

                    for content in msg["content"]:
                        if content["type"] == "text":
                            claude_msg["content"].append({"type": "text", "text": content["text"]})
                        elif content["type"] == "image_url":
                            image_data = content["image_url"]["url"].replace("data:image/png;base64,", "")
                            claude_msg["content"].append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data
                                }
                            })
                    claude_messages.append(claude_msg)

                fast_check_json_instruction = '\n\nPlease respond with a JSON object containing "thought" (string) and "align" (boolean) keys.'
                system_message_text = system_message + fast_check_json_instruction if system_message else fast_check_json_instruction

                fast_check_response_obj = self.fast_check_client.messages.create(
                    model=self.fast_check_model,
                    max_tokens=1000,
                    system=system_message_text,
                    messages=claude_messages
                )

                fast_check_response_content = fast_check_response_obj.content[0].text

                try:
                    fast_check_result = json.loads(fast_check_response_content)
                except json.JSONDecodeError:
                    import re
                    json_match = re.search(r'\{[^{}]*"align"[^{}]*\}', fast_check_response_content, re.DOTALL)
                    if json_match:
                        fast_check_result = json.loads(json_match.group())
                    else:
                        fast_check_result = {
                            "thought": fast_check_response_content,
                            "align": "true" in fast_check_response_content.lower()
                        }
            else:
                raise ValueError(f"Unsupported fast check provider: {self.fast_check_provider}")

            fast_check_align_raw = fast_check_result.get("align")
            fast_check_align = (
                fast_check_align_raw.lower() == "true"
                if isinstance(fast_check_align_raw, str)
                else bool(fast_check_align_raw)
            )
            fast_check_thought = fast_check_result.get("thought", "")
            logger.info(f"[DeAction] Fast check response: {fast_check_response_content}")
            return fast_check_align, fast_check_thought

        except Exception as fast_check_error:
            logger.warning(
                f"[DeAction] Fast check model evaluation failed ({fast_check_error}); proceeding with full analysis"
            )
            return None

    def check_once(self, instruction: str, response: str, obs: Dict, agent_history: Dict, max_retries: int = 1) -> Tuple[bool, str]:
        """
        Check whether the agent's predicted action is aligned with the user's objective once.
        Returns: (is_misaligned: bool, thought: str)
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"[DeAction] Calling systematic analysis with {self.model} (attempt {attempt + 1})")
                
                system_prompt = self.system_prompt
                
                # Prepare previous actions blocks (interleaved with screenshots later)
                previous_actions_entries: List[str] = []
                thoughts = agent_history.get('thoughts') or []
                actions = agent_history.get('actions') or []
                narratives = agent_history.get('narratives') or []
                includes_current = agent_history.get('includes_current_action', False)

                history_actions = list(actions)
                if includes_current and history_actions:
                    history_actions = history_actions[:-1]

                history_thoughts = list(thoughts)
                if includes_current and history_thoughts:
                    history_thoughts = history_thoughts[:-1]

                previous_actions_text = ""
                if history_actions:
                    logger.info(f"[DeAction] Processing {len(history_actions)} previous actions")
                    for i, action in enumerate(history_actions):
                        if isinstance(action, list) and len(action) > 0:
                            action_text = str(action[0]) if action[0] else ""
                        else:
                            action_text = str(action) if action else ""
                        preprocessed_action = self.preprocess_remove_comments(action_text)
                        entry_lines = [f"Step {i+1} Action:\n{preprocessed_action}"]

                        if i < len(history_thoughts):
                            thought = history_thoughts[i]
                            if isinstance(thought, str) and thought.strip():
                                entry_lines.append(f"Agent Thought: {thought.strip()}")

                        if i < len(narratives):
                            narrative_entry = narratives[i]
                            narrative_text = ""
                            key_changes_text = ""
                            if isinstance(narrative_entry, dict):
                                narrative_text = narrative_entry.get("narrative") or ""
                                key_changes = narrative_entry.get("key_changes") or []
                                if isinstance(key_changes, list) and key_changes:
                                    key_changes_lines = [
                                        f"- {str(item).strip()}"
                                        for item in key_changes
                                        if str(item).strip()
                                    ]
                                    if key_changes_lines:
                                        key_changes_text = "\n".join(key_changes_lines)
                            elif isinstance(narrative_entry, str):
                                narrative_text = narrative_entry

                            if narrative_text.strip():
                                entry_lines.append(f"Narrative Summary: {narrative_text.strip()}")
                            if key_changes_text:
                                entry_lines.append(f"Key Changes:\n{key_changes_text}")

                        entry_text = "\n".join(entry_lines) + "\n"
                        previous_actions_entries.append(entry_text)
                        previous_actions_text += entry_text

                # logger.info("[MisStep Detect] input")
                # logger.info(user_input)

                screenshot_payload = None
                formatted_history_images: List[str] = []
                observation_type = agent_history.get('observation_type', 'screenshot')
                if observation_type in ["screenshot", "screenshot_a11y_tree"]:
                    screenshot_payload = obs.get("screenshot")
                    if self.annotate_actions and screenshot_payload is not None:
                        logger.info("[DeAction] annotate coordinates")
                        annotated = self._annotate_screenshot_with_actions(screenshot_payload, response)
                        if annotated is not None:
                            screenshot_payload = annotated
                    if screenshot_payload is None:
                        logger.error("[DeAction] Screenshot payload missing; cannot continue")
                        return False, ""

                    history_payloads = obs.get("history_screenshots") or []
                    if isinstance(history_payloads, (list, tuple)):
                        for payload in history_payloads:
                            if not payload:
                                continue
                            try:
                                if isinstance(payload, bytes):
                                    formatted_history_images.append(encode_image(payload))
                                elif isinstance(payload, str):
                                    encoded = payload
                                    if payload.startswith("data:image"):
                                        _, _, encoded = payload.partition(",")
                                    if encoded:
                                        formatted_history_images.append(encoded)
                                else:
                                    logger.warning(
                                        "[DeAction] Unsupported history screenshot payload type: %s",
                                        type(payload),
                                    )
                            except Exception as exc:
                                logger.warning("[DeAction] Failed to encode history screenshot: %s", exc)

                    if formatted_history_images:
                        # Build user message content with interleaved history screenshots.
                        user_message_content = [
                            {
                                "type": "text",
                                "text": f"**User Objective**: {instruction}\n\n**Previous Actions**:\n",
                            }
                        ]

                        if previous_actions_entries:
                            total_actions = len(previous_actions_entries)
                            history_count = len(formatted_history_images)
                            history_start = max(0, total_actions - history_count)
                            for idx, entry in enumerate(previous_actions_entries):
                                user_message_content.append({"type": "text", "text": entry})
                                history_idx = idx - history_start
                                if 0 <= history_idx < history_count:
                                    user_message_content.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{formatted_history_images[history_idx]}",
                                                "detail": "high",
                                            },
                                        }
                                    )
                        else:
                            user_message_content.append({"type": "text", "text": "None\n"})
                            user_message_content.append(
                                {
                                    "type": "text",
                                    "text": f"Previous screenshots ({len(formatted_history_images)}):",
                                }
                            )
                            for history_b64 in formatted_history_images:
                                user_message_content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{history_b64}",
                                            "detail": "high",
                                        },
                                    }
                                )

                        additional_note_for_coordinates_annotation = (
                            "To help you understand the coordinates involved in current action, "
                            "the coordinates correspond to the center of each numbered circular marker, "
                            "with accompanying red text labels for identification. Please use these markers "
                            "for understanding the coordinated in current action."
                        )
                        user_message_content.append(
                            {
                                "type": "text",
                                "text": (
                                    "**Current Action**:\n"
                                    f"{self.preprocess_remove_comments(response)}\n\n"
                                    f"**Current State**: "
                                    f"{additional_note_for_coordinates_annotation if self.annotate_actions else ''}"
                                ),
                            }
                        )
                    else:
                        # Keep original prompt structure when no history screenshots exist.
                        additional_note_for_coordinates_annotation = (
                            "To help you understand the coordinates involved in current action, "
                            "the coordinates correspond to the center of each numbered circular marker, "
                            "with accompanying red text labels for identification. Please use these markers "
                            "for understanding the coordinated in current action."
                        )
                        user_input = f"""**User Objective**: {instruction}

**Previous Actions**: 
{previous_actions_text if previous_actions_text else "None"}

**Current Action**: 
{self.preprocess_remove_comments(response)}

**Current State**: {additional_note_for_coordinates_annotation if self.annotate_actions else ""}
"""

                        user_message_content = [
                            {
                                "type": "text",
                                "text": user_input
                            }
                        ]

                    base64_image = (
                        encode_image(screenshot_payload)
                        if isinstance(screenshot_payload, bytes)
                        else screenshot_payload
                    )
                    user_message_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        }
                    )
                else:
                    logger.error(f"[DeAction] Observation type {observation_type} is not supported, skip")
                    return False, ""
                

                # print(len(user_message_content))
                # print(user_message_content[0])

                fast_check_result = self._run_fast_check(user_message_content)
                if self.systematic_analysis_disabled:
                    if fast_check_result:
                        fast_check_align, fast_check_thought = fast_check_result
                        if fast_check_align:
                            logger.info("[DeAction] Fast-check-only mode: action aligned")
                            return False, fast_check_thought
                        logger.info("[DeAction] Fast-check-only mode: treating action as misaligned")
                        return True, fast_check_thought
                    logger.warning("[DeAction] Fast-check-only mode without fast check result; skipping systematic analysis")
                    return False, ""
                if fast_check_result:
                    fast_check_align, fast_check_thought = fast_check_result
                    if fast_check_align:
                        logger.info("[DeAction] Fast check determined action is clearly aligned; skipping systematic analysis")
                        return False, fast_check_thought

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": copy.deepcopy(user_message_content)
                    }
                ]
                
                # Call appropriate API based on provider
                if self.provider in ("azure", "openai", "local"):
                    # Azure/OpenAI API call with optional reasoning effort and strict JSON schema
                    request_kwargs = {
                        "model": self.model,
                        "messages": messages,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                            "name": "systematic_analysis_response",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "thought": {"type": "string"},
                                        "conclusion": {"type": "boolean"}
                                    },
                                    "required": ["thought", "conclusion"],
                                    # "required": ["conclusion"],
                                    "additionalProperties": False
                                },
                                "strict": True
                            }
                        }
                    }
                    if self.model_reasoning_effort:
                        logger.info(f"[DeAction] setting reasoning effort to {self.model_reasoning_effort}")
                        request_kwargs["reasoning_effort"] = self.model_reasoning_effort
                    response_obj = self.client.chat.completions.create(**request_kwargs)
                    
                    response_content = response_obj.choices[0].message.content
                    systematic_analysis_result = json.loads(response_content)
                    
                elif self.provider == "together":
                    # Together AI - add JSON instruction to messages and try to parse response
                    # Add JSON format instruction to the system message
                    messages[0]["content"] += '\n\nIMPORTANT: Respond with a valid JSON object containing exactly these keys: "thought" (string) and "conclusion" (boolean).'
                    
                    response_obj = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=2000
                    )
                    
                    response_content = response_obj.choices[0].message.content
                    
                    # Try to extract JSON from Together AI response
                    try:
                        systematic_analysis_result = json.loads(response_content)
                    except json.JSONDecodeError:
                        # Try to find JSON in the response
                        import re
                        json_match = re.search(r'\{[^{}]*"thought"[^{}]*"conclusion"[^{}]*\}', response_content, re.DOTALL)
                        if json_match:
                            systematic_analysis_result = json.loads(json_match.group())
                        else:
                            # Fallback: create a structured response from text
                            systematic_analysis_result = {
                                "thought": response_content,
                                "conclusion": "true" in response_content.lower() or "misaligned" in response_content.lower()
                            }
                    
                elif self.provider == "aws":
                    # Claude API call - convert messages format
                    claude_messages = []
                    system_message = None
                    
                    for msg in messages:
                        if msg["role"] == "system":
                            system_message = msg["content"]
                        else:
                            claude_msg = {"role": msg["role"], "content": []}
                            
                            for content in msg["content"]:
                                if content["type"] == "text":
                                    claude_msg["content"].append({"type": "text", "text": content["text"]})
                                elif content["type"] == "image_url":
                                    # Convert image format for Claude
                                    image_data = content["image_url"]["url"].replace("data:image/png;base64,", "")
                                    claude_msg["content"].append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": image_data
                                        }
                                    })
                            claude_messages.append(claude_msg)
                    
                    # Add JSON format instruction to system message
                    json_instruction = '\n\nPlease respond with a JSON object containing "thought" (string) and "conclusion" (boolean) keys.'
                    system_message_text = system_message + json_instruction if system_message else json_instruction
                    
                    response_obj = self.client.messages.create(
                        model=self.model,
                        max_tokens=4000,
                        system=system_message_text,
                        messages=claude_messages
                    )
                    
                    response_content = response_obj.content[0].text
                    
                    # Try to extract JSON from Claude's response
                    try:
                        # First try to parse directly
                        systematic_analysis_result = json.loads(response_content)
                    except json.JSONDecodeError:
                        # If that fails, try to find JSON in the response
                        import re
                        json_match = re.search(r'\{[^{}]*"thought"[^{}]*"conclusion"[^{}]*\}', response_content, re.DOTALL)
                        if json_match:
                            systematic_analysis_result = json.loads(json_match.group())
                        else:
                            # Fallback: create a structured response from text
                            systematic_analysis_result = {
                                "thought": response_content,
                                "conclusion": "true" in response_content.lower() or "misaligned" in response_content.lower()
                            }
                
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                
                logger.info(f"[DeAction] Response: {response_content}")
                
                return systematic_analysis_result["conclusion"], systematic_analysis_result["thought"]
                # return systematic_analysis_result["conclusion"], ""
                
            except Exception as e:
                error_text = str(e).lower()
                if "invalid image" in error_text or "invalid_image" in error_text:
                    main_summary = self._summarize_image_payload(screenshot_payload)
                    history_summary = self._summarize_history_payloads(formatted_history_images)
                    logger.warning(
                        "[DeAction] Invalid image debug main=%s history=%s",
                        main_summary,
                        history_summary,
                    )
                logger.warning(f"[DeAction] Guardrail attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"[DeAction] Guardrail failed after {max_retries} attempts: {e}")
                    # In case of error, assume the action is not misaligned
                    return False, f"Error after {max_retries} attempts: {str(e)}"
                else:
                    # Wait before retry with exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"[DeAction] Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
