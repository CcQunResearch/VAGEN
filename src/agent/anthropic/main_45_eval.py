import base64
import os
import time
from typing import Any, cast, Optional, Dict
from PIL import Image
import io
import requests
import json
import copy
import traceback
import random
import time
import requests
from src.agent.anthropic.main import ANON_CLAUDE_45_OPENAI
from qwen_agent.agents.fncall_agent import FnCallAgent
import requests
import time
import re  
# import datetime


from src.agent.anthropic.tool_search import Search   
from src.agent.anthropic.tool_visit import Visit 
from src.agent.anthropic.custom_tools_registry import get_tool_definitions
from datetime import datetime

from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
)

 
from src.agent.anthropic.utils import (
    COMPUTER_USE_BETA_FLAG, 
    PROMPT_CACHING_BETA_FLAG,
    SYSTEM_PROMPT, 
    SYSTEM_PROMPT_SEARCH, 
    SYSTEM_PROMPT_WINDOWS, 
    APIProvider, 
    PROVIDER_TO_DEFAULT_MODEL_NAME, 
    get_model_name,
    _response_to_params, 
    _inject_prompt_caching, 
    _maybe_filter_to_n_most_recent_images, 
    _custom_maybe_filter_to_n_most_recent_images,
    _custom_maybe_filter_to_n_most_recent_messages
)

from src.agent.anthropic.util import (
    simplify_messages_for_json,
    load_trajectory_data,
    create_user_prompt,
    extract_image_context_from_message,
    save_agent_images,
    parse_agent_judgment,
    extract_tool_usage_summary,
    _build_eval_prompt,
    remove_images_and_add_note,
    count_images_in_content
)

import logging
logger = logging.getLogger("desktopenv.agent")
agent_logger = logging.getLogger("desktopagent.agent")

# 在文件开头添加以下代码来禁用OpenAI的调试输出
logging.getLogger("openai").setLevel(logging.ERROR)
# 如果需要禁用所有HTTP请求的日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# MAX_HISTORY = 10
# API_RETRY_TIMES = 500  
API_RETRY_TIMES = 5
API_RETRY_INTERVAL = 5

ANON_CLAUDE_37 = "anthropic.claude-3.7-sonnet"
ANON_CLAUDE_45_OPENAI = "claude-sonnet-4.5-openai"
ANON_CLAUDE_4_OPENAI = "claude-sonnet-4-openai"

def resize_image_to_1024_768(image_bytes: bytes) -> tuple[bytes, float, float]:
    img = Image.open(io.BytesIO(image_bytes))
    original_width, original_height = img.size
    new_width, new_height = 1280, 720
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img_resized.save(buffer, format='PNG')
    resized_bytes = buffer.getvalue()
    x_ratio = original_width / new_width
    y_ratio = original_height / new_height
    return resized_bytes, x_ratio, y_ratio


class AnthropicAgent45Eval:
    DEFAULT_TOOLS = [
        # "browse",
        # "search",
        # "visit",
        "execute_python",
        "execute_shell",
        "check_screenshot",
        # "read_pptx"  # ⭐ 添加到默认工具列表
    ]
    def __init__(self,
                 platform: str = "Ubuntu",
                 model: str = "ANON",
                 provider: APIProvider = APIProvider.ANON,
                 max_tokens: int = 4096,
                 api_url: str = "",
                 customer_tools = DEFAULT_TOOLS,
                 api_key: str = "",
                 system_prompt_suffix: str = "",
                 only_n_most_recent_images: Optional[int] = 10,
                 only_n_most_recent_images_eval: Optional[int] = 10,
                 action_space: str = "claude_computer_use",
                 screen_size: tuple[int, int] = (1920, 1080),
                 no_thinking: bool = False,
                 use_isp: bool = False,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 log_image_base64: bool = False,
                 *args, **kwargs
                 ):
        self.platform = platform
        self.action_space = action_space
        self.logger = logger
        self.class_name = self.__class__.__name__
        self.model_name = model
        if 'claude-offical-3.7' in self.model_name or "claude-3.7" in self.model_name:
            self.model_name = ANON_CLAUDE_37
        if 'claude-offical-4.5-openai' in self.model_name or "claude-4.5-openai" in self.model_name:
            self.model_name = ANON_CLAUDE_45_OPENAI
        if 'claude-offical-4-openai' in self.model_name or "claude-4-openai" in self.model_name:
            self.model_name = ANON_CLAUDE_4_OPENAI

        self.provider = provider
        self.max_tokens = max_tokens
        self.api_url = api_url
        self.api_key = api_key
        self.system_prompt_suffix = system_prompt_suffix
        self.only_n_most_recent_images = only_n_most_recent_images
        self.only_n_most_recent_images_eval = only_n_most_recent_images_eval
        self.messages: list[BetaMessageParam] = []
        self.image_names: list[str] = []
        self.screen_size = screen_size
        self.resize_factor = (
            screen_size[0] / 1280,  # Assuming 1280 is the base width
            screen_size[1] / 720   # Assuming 720 is the base height
        )
        self.log_image_base64 = log_image_base64
        self.use_openai_format = "openai" in model
        self.turn_count = 0
        self.custom_tools = get_tool_definitions(customer_tools)

        self.no_thinking = no_thinking
        self.use_isp = use_isp
        self.temperature = temperature
        self.top_p = top_p

        llm_cfg = {
            'model': model,
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10, 
                'temperature': 0.6, 
                'top_p': 0.95
            }, 
            'model_type': 'qwen_dashscope'
        }

        try:
            self.web_agent = FnCallAgent(
                function_list=["search","visit"],  
                llm=llm_cfg,
                system_message="You are a helpful assistant.",
                name="WebAgent",
                description="An agent that can perform searches"
            )

            logger.info("Search agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize search agent: {e}")
            self.web_agent = None

    def _get_sampling_params(self):
        """Get sampling parameters (temperature and/or top_p) - let API validate exclusivity"""
        params = {}
        if self.temperature is not None:
            params['temperature'] = self.temperature
        if self.top_p is not None:
            params['top_p'] = self.top_p
        return params
    def _extract_raw_response_string(self, response) -> str:
        """Extract and concatenate raw response content into a single string."""
        raw_response_str = ""
        if response.content:
            for block in response.content:
                if hasattr(block, 'text') and block.text:
                    raw_response_str += f"[TEXT] {block.text}\n"
                elif hasattr(block, 'thinking') and block.thinking:
                    raw_response_str += f"[THINKING] {block.thinking}\n"
                elif hasattr(block, 'name') and hasattr(block, 'input'):
                    raw_response_str += f"[TOOL_USE] {block.name}: {block.input}\n"
                else:
                    raw_response_str += f"[OTHER] {str(block)}\n"
        return raw_response_str.strip()

    def add_tool_result(self, tool_call_id: str, result: str, screenshot: bytes = None, screenshot_name: str = "screenshot_name"):
        """Add tool result to message history"""
        tool_result_content = [
            {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": result,
                # "content": [{"type": "text", "text": result}]
            }
        ]
        
        # Add screenshot if provided
        if screenshot is not None:
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
            tool_result_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png", 
                    "data": screenshot_base64
                }
            })
            # tool_result_content[0]["content"] = screenshot_base64
        
        self.messages.append({
            "role": "user",
            "content": tool_result_content
        })
        self.image_names.append(screenshot_name)
    
    # def parse_actions_from_tool_call(self, tool_call: Dict, screenshot_resize_ratio = 1.0) -> str:
    def parse_actions_from_tool_call(self, tool_call: Dict, screenshot_resize_ratio_x=1.0, screenshot_resize_ratio_y=1.0) -> str:
        result = ""
        function_args = (
            tool_call["input"]
        )
        
        action = function_args.get("action")
        if not action:
            action = tool_call.function.name
        action_conversion = {
            "left click": "click",
            "right click": "right_click"
        }
        action = action_conversion.get(action, action)

        # agent_logger.info(f"action={action}, function_args={function_args}, screenshot_resize_ratio={screenshot_resize_ratio}")
        
        text = function_args.get("text")
        coordinate = function_args.get("coordinate")
        start_coordinate = function_args.get("start_coordinate")
        scroll_direction = function_args.get("scroll_direction")
        scroll_amount = function_args.get("scroll_amount")
        duration = function_args.get("duration")
        
       
        if coordinate and (screenshot_resize_ratio_x and screenshot_resize_ratio_y):
            # 坐标从模型输入 映射回原始分辨率
            coordinate = (
                int(coordinate[0] * screenshot_resize_ratio_x),
                int(coordinate[1] * screenshot_resize_ratio_y)
            )
            agent_logger.info(f"coordinate={coordinate}, screenshot_resize_ratio={screenshot_resize_ratio_x} {screenshot_resize_ratio_y}")
        if start_coordinate and (screenshot_resize_ratio_x != 1.0 or screenshot_resize_ratio_y != 1.0):
            start_coordinate = (
                int(start_coordinate[0] * screenshot_resize_ratio_x),
                int(start_coordinate[1] * screenshot_resize_ratio_y)
            )
            logger.info(f"Resized start_coordinate={start_coordinate}")

        if action == "left_mouse_down":
            result += "pyautogui.mouseDown()\n"
        elif action == "left_mouse_up":
            result += "pyautogui.mouseUp()\n"
        
        elif action == "hold_key":
            if not isinstance(text, str) or duration is None:
                raise ValueError("'text' and 'duration' are required for hold_key action")
            
            keys = [k.strip().lower() for k in text.split('+')]
            for key in keys:
                result += f"pyautogui.keyDown('{key}')\n"
            result += f"time.sleep({duration})\n"
            for key in reversed(keys):
                result += f"pyautogui.keyUp('{key}')\n"
        # Handle mouse move and drag actions
        elif action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ValueError(f"coordinate is required for {action}")
            if text is not None:
                raise ValueError(f"text is not accepted for {action}")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ValueError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) for i in coordinate):
                raise ValueError(f"{coordinate} must be a tuple of ints")
            
            x, y = coordinate[0], coordinate[1]
            if action == "mouse_move":
                result += (
                    f"pyautogui.moveTo({x}, {y}, duration={duration or 0.5})\n"
                )
                expected_outcome = f"Mouse moved to ({x},{y})."
            elif action == "left_click_drag":
                # If start_coordinate is provided, validate and move to start before dragging
                if start_coordinate:
                    if not isinstance(start_coordinate, (list, tuple)) or len(start_coordinate) != 2:
                        raise ValueError(f"{start_coordinate} must be a tuple of length 2")
                    if not all(isinstance(i, int) for i in start_coordinate):
                        raise ValueError(f"{start_coordinate} must be a tuple of ints")
                    start_x, start_y = start_coordinate[0], start_coordinate[1]
                    result += (
                        f"pyautogui.moveTo({start_x}, {start_y}, duration={duration or 0.5})\n"
                    )
                result += (
                    f"pyautogui.dragTo({x}, {y}, duration={duration or 0.5})\n"
                )
                expected_outcome = f"Cursor dragged to ({x},{y})."
        # Handle keyboard actions
        elif action in ("key", "type"):
            if text is None:
                raise ValueError(f"text is required for {action}")
            if coordinate is not None:
                raise ValueError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ValueError(f"{text} must be a string")
            if action == "key":
                key_conversion = {
                    "page_down": "pagedown",
                    "page_up": "pageup",
                    "super_l": "win",
                    "super": "command",
                    "escape": "esc"
                }
                keys = text.split('+')
                for key in keys:
                    key = key.strip().lower()
                    key = key_conversion.get(key, key)
                    result += (f"pyautogui.keyDown('{key}')\n")
                for key in reversed(keys):
                    key = key.strip().lower()
                    key = key_conversion.get(key, key)
                    result += (f"pyautogui.keyUp('{key}')\n")
                expected_outcome = f"Key {key} pressed."
            elif action == "type":
                # 检测是否为 apt 命令
                is_apt_command = "apt-get install" in text or "apt install" in text
                
                # 可配置的镜像源和代理
                MIRROR_SOURCES = {
                    "ANON": {
                        "http_proxy": "ANON",
                        "https_proxy": "ANON",
                        "no_proxy": "ANON",
                        "apt_option": 'ANON'
                    },
                    "official": {
                        "http_proxy": "ANON",
                        "https_proxy": "ANON",
                        "no_proxy": "ANON",
                        "apt_option": 'ANON'
                    }
                }
                # 使用 ANON 作为默认镜像源
                current_mirror = MIRROR_SOURCES["official"]
                
                # 如果是 apt 命令，先设置代理
                if is_apt_command:
                    result += "# 设置代理以使用 ANON 镜像\n"
                    for key, value in current_mirror.items():
                        if key != "apt_option":  # 跳过 apt 选项
                            result += f"pyautogui.typewrite('export {key}={value}', interval=0.01)\n"
                            result += "pyautogui.press('enter')\n"
                    result += "time.sleep(0.5)\n"
                    result += "# 注意：apt 命令已配置代理\n"
                    text = text.replace(
                        "apt install",
                        'apt install -o Acquire::http::Proxy="ANON"'
                    ).replace(
                        "apt-get install",
                        'apt-get install -o Acquire::http::Proxy="ANON"'
                    )

                # 智能输入策略：根据文本长度和内容选择最优方法
                text_length = len(text)
                
                # if text_length > 500:
                #     # 长文本：使用剪贴板方式（最快）
                #     result += f"import pyperclip\n"
                #     # 转义文本中的引号和反斜杠
                #     escaped_text = text.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
                #     result += f'pyperclip.copy("""{escaped_text}""")\n'
                #     result += "pyautogui.hotkey('ctrl', 'v')\n"
                #     result += "time.sleep(0.1)\n"
                if text_length > 100:
                    # 中等文本：混合策略（平衡速度和准确性）
                    buffer = ""
                    for char in text:
                        if char in ['\n', '\\', '"', "'"]:
                            # 输出缓冲区中的普通字符
                            if buffer:
                                # 转义buffer中的特殊字符
                                escaped_buffer = buffer.replace('\\', '\\\\').replace("'", "\\'")
                                result += f"pyautogui.typewrite('{escaped_buffer}', interval=0.01)\n"
                                buffer = ""
                            
                            # 处理特殊字符
                            if char == '\n':
                                result += "pyautogui.press('enter')\n"
                            elif char == "'":
                                result += 'pyautogui.press("\'")\n'
                            elif char == '\\':
                                result += "pyautogui.press('\\\\')\n"
                            elif char == '"':
                                result += "pyautogui.press('\"')\n"
                        else:
                            buffer += char
                    
                    # 输出剩余的缓冲区内容
                    if buffer:
                        escaped_buffer = buffer.replace('\\', '\\\\').replace("'", "\\'")
                        result += f"pyautogui.typewrite('{escaped_buffer}', interval=0.01)\n"
                else:
                    # 短文本：逐字符输入（最准确）
                    for char in text:
                        if char == '\n':
                            result += "pyautogui.press('enter')\n"
                        elif char == "'":
                            result += 'pyautogui.press("\'")\n'
                        elif char == '\\':
                            result += "pyautogui.press('\\\\')\n"
                        elif char == '"':
                            result += "pyautogui.press('\"')\n"
                        else:
                            result += f"pyautogui.press('{char}')\n"
                
                expected_outcome = f"Text {text[:50]}{'...' if text_length > 50 else ''} written."

         

        # Handle scroll actions
        elif action == "scroll":
            if text is not None:
                result += (f"pyautogui.keyDown('{text.lower()}')\n")
            if coordinate is None:
                if scroll_direction in ("up", "down"):
                    result += (
                        f"pyautogui.scroll({scroll_amount if scroll_direction == 'up' else -scroll_amount})\n"
                    )
                elif scroll_direction in ("left", "right"):
                    result += (
                        f"pyautogui.hscroll({scroll_amount if scroll_direction == 'right' else -scroll_amount})\n"
                    )
            else:
                if scroll_direction in ("up", "down"):
                    x, y = coordinate[0], coordinate[1]
                    result += (
                        f"pyautogui.scroll({scroll_amount if scroll_direction == 'up' else -scroll_amount}, {x}, {y})\n"
                    )
                elif scroll_direction in ("left", "right"):
                    x, y = coordinate[0], coordinate[1]
                    result += (
                        f"pyautogui.hscroll({scroll_amount if scroll_direction == 'right' else -scroll_amount}, {x}, {y})\n"
                    )
            if text is not None:
                result += (f"pyautogui.keyUp('{text.lower()}')\n")
            expected_outcome = "Scroll action finished"

        # Handle click actions
        elif action in ("left_click", "right_click", "double_click", "middle_click", "left_press", "triple_click"):
            # Handle modifier keys during click if specified
            if text:
                keys = text.split('+')
                for key in keys:
                    key = key.strip().lower()
                    result += f"pyautogui.keyDown('{key}')\n"
            if coordinate is not None:
                x, y = coordinate
                if action == "left_click":
                    result += (f"pyautogui.click({x}, {y})\n")
                elif action == "right_click":
                    result += (f"pyautogui.rightClick({x}, {y})\n")
                elif action == "double_click":
                    result += (f"pyautogui.doubleClick({x}, {y})\n")
                elif action == "middle_click":
                    result += (f"pyautogui.middleClick({x}, {y})\n")
                elif action == "left_press":
                    result += (f"pyautogui.mouseDown({x}, {y})\n")
                    result += ("time.sleep(1)\n")
                    result += (f"pyautogui.mouseUp({x}, {y})\n")
                elif action == "triple_click":
                    result += (f"pyautogui.tripleClick({x}, {y})\n")
            else:
                if action == "left_click":
                    result += ("pyautogui.click()\n")
                elif action == "right_click":
                    result += ("pyautogui.rightClick()\n")
                elif action == "double_click":
                    result += ("pyautogui.doubleClick()\n")
                elif action == "middle_click":
                    result += ("pyautogui.middleClick()\n")
                elif action == "left_press":
                    result += ("pyautogui.mouseDown()\n")
                    result += ("time.sleep(1)\n")
                    result += ("pyautogui.mouseUp()\n")
                elif action == "triple_click":
                    result += ("pyautogui.tripleClick()\n")
            # Release modifier keys after click
            if text:
                keys = text.split('+')
                for key in reversed(keys):
                    key = key.strip().lower()
                    result += f"pyautogui.keyUp('{key}')\n"
            expected_outcome = "Click action finished"
            
        elif action == "wait":
            if duration is None: 
                result += "time.sleep(1)\n"
            else:
                result += f"time.sleep({duration})\n"
        elif action == "fail":
            result += "FAIL"
            expected_outcome = "Finished"
        elif action == "done":
            result += "DONE"
            expected_outcome = "Finished"
        elif action == "call_user":
            result += "CALL_USER"
            expected_outcome = "Call user"
        elif action == "screenshot":
            result += "pyautogui.sleep(0.1)\n"
            expected_outcome = "Screenshot taken"   
        # 未在代码中处理的动作
        elif action == "cursor_position":
            # 这个动作是获取信息，而不是执行动作，所以在这个函数中通常不生成代码
            # 可以考虑在另一个流程中处理它，或者直接返回一个注释
            result += "# Action 'cursor_position' is not an executable command in this context.\n"
        else:
            raise ValueError(f"Invalid action: {action}")
        
        return result
            
    def predict(self, task_instruction: str, obs: Dict = None, system: Any = None):
        self.turn_count += 1
        system = BetaTextBlockParam(
            type="text",
            text=f"{SYSTEM_PROMPT_WINDOWS if self.platform == 'Windows' else SYSTEM_PROMPT_SEARCH}{' ' + self.system_prompt_suffix if self.system_prompt_suffix else ''}"
        )
        
        image_meta = {}
        # resize screenshot if resize_factor is set
        if obs and "screenshot" in obs:
            # Convert bytes to PIL Image
            screenshot_bytes = obs["screenshot"]
            resized_bytes, x_ratio, y_ratio = resize_image_to_1024_768(screenshot_bytes)
            obs["screenshot"] = resized_bytes
            obs["screenshot_resize_ratio_x"] = x_ratio
            obs["screenshot_resize_ratio_y"] = y_ratio
            
            screenshot_image = Image.open(io.BytesIO(screenshot_bytes))

            original_width, original_height = screenshot_image.size #1920 * 1080
            # new_width, new_height = 1024, 768
            new_width, new_height = 1280, 720
            ratio_x, ratio_y = x_ratio, y_ratio
            image_meta = {
                "original_size": [original_width, original_height],
                "input_size": [new_width, new_height],
                "resize_ratio_x": ratio_x,
                "resize_ratio_y": ratio_y
            }
            logger.info(f"  original=({original_width}, {original_height}), new=({new_width}, {new_height}), resize_ratio_x={ratio_x}, resize_ratio_y={ratio_y}")
            

        if not self.messages:
            
            init_screenshot = obs
            init_screenshot_base64 = base64.b64encode(init_screenshot["screenshot"]).decode('utf-8')
            self.messages.append({
                "role": "user",
                "content": [
                    {
                    "type": "image",
                    "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": init_screenshot_base64,
                        },
                    },
                    {"type": "text", "text": task_instruction},
                ]
            })
            self.image_names.append(init_screenshot.get("screenshot_name", "screenshot_name"))
            
       

        # 检查消息列表是否为空
        if self.messages:
            # 反向遍历,找到最近一次 assistant 消息
            for i in range(len(self.messages) - 1, -1, -1):
                msg = self.messages[i]
                
                if msg.get("role") == "assistant":
                    # 获取该 assistant 消息的 content
                    last_message_content = msg.get("content", [])
                    
                    # 确保 content 是列表
                    if isinstance(last_message_content, str):
                        last_message_content = [{"type": "text", "text": last_message_content}]
                    
                    # 提取所有 computer 工具调用
                    tool_use_blocks = [
                        block for block in last_message_content 
                        if (
                            isinstance(block, dict) and
                            block.get("type") == "tool_use" and 
                            block.get("name") == "computer"
                        )
                    ]
                    
                    # 为每个 computer tool_use 添加 tool_result
                    for idx, tool_block in enumerate(tool_use_blocks):
                        tool_input = tool_block.get("input", {})
                        action = tool_input.get("action")
                        is_last_tool = idx == len(tool_use_blocks) - 1
                        
                        include_screenshot = None
                        
                        if obs:
                            if action == "screenshot":
                                # Screenshot action 总是包含截图
                                include_screenshot = obs.get("screenshot")
                            elif is_last_tool:
                                # 最后一个 tool 自动包含截图
                                include_screenshot = obs.get("screenshot")
                        
                        # 添加 tool_result
                        self.add_tool_result(
                            tool_call_id=tool_block["id"],
                            result="Success",
                            screenshot=include_screenshot,
                            screenshot_name=obs.get("screenshot_name", "screenshot") if include_screenshot else None
                        )
                    
                    break  # 只处理最近一次 assistant 消息

            
        enable_prompt_caching = False
        
        betas = [COMPUTER_USE_BETA_FLAG]
        # Add interleaved thinking beta if ISP is requested
        if self.use_isp:
            betas.append("interleaved-thinking-2025-05-14")
            logger.info(f"Added interleaved thinking beta. Betas: {betas}")
            
        image_truncation_threshold = 10
        print('self.provider', self.provider)
        if self.provider == APIProvider.ANTHROPIC:
            client = Anthropic(api_key=self.api_key, max_retries=4)
            enable_prompt_caching = True
        elif self.provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif self.provider == APIProvider.BEDROCK:
            client = AnthropicBedrock(
                aws_access_key=os.getenv('ACCESS_KEY_ID'),
                aws_secret_key=os.getenv('SECRET_ACCESS_KEY'),
                aws_region=os.getenv('DEFAULT_REGION'),
            )
        elif self.provider == APIProvider.ANON:
            pass

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            _inject_prompt_caching(self.messages)
            image_truncation_threshold = 20
            system["cache_control"] = {"type": "ephemeral"}

        if self.only_n_most_recent_images:
            _custom_maybe_filter_to_n_most_recent_images(
                self.messages,
                self.image_names,
                self.only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )
        # if len(self.messages) > 1:
        #     new_content = []
        #     for content in self.messages[0]["content"]:
        #         if content["type"] != "image":
        #             new_content.append(content)
        #     self.messages[0]["content"] = new_content

        system_prompt_to_add = system['text'] if self.turn_count == 1 else None
        simple_messages_for_export, _ = self.prepare_messages_for_export(system_prompt_to_add)

        
        tool_config = {
            'name': 'computer', 
            'type': 'computer_20250124', 
            'display_width_px': 1280, 
            'display_height_px': 720, 
            'display_number': 1
        }

        tools = [
            tool_config,
        ] if self.platform == 'Ubuntu' else [
            tool_config,
        ]

        # Configure thinking mode based on user preferences
        if self.no_thinking:
            # Disable thinking mode - omit the thinking parameter
            extra_body = {}
            actual_max_tokens = self.max_tokens  # Use default when no thinking
            logger.info("Thinking mode: DISABLED")
        else:
            # Enable thinking mode (regular or interleaved)
            # Use consistent 2048 budget for both regular and ISP thinking
            budget_tokens = 2048

            # For regular thinking: max_tokens > budget_tokens (API requirement)
            # For ISP: budget_tokens can exceed max_tokens (represents total across all thinking blocks)
            if self.max_tokens <= budget_tokens:
                required_max_tokens = budget_tokens + 500  # Give some headroom
                logger.warning(f"Regular thinking requires max_tokens > budget_tokens. Increasing max_tokens from {self.max_tokens} to {required_max_tokens}")
                actual_max_tokens = required_max_tokens
            else:
                actual_max_tokens = self.max_tokens
            tools  = tools + self.custom_tools

            extra_body = {
                # "thinking": {"type": "enabled", "budget_tokens": 1024}
                "thinking": {"type": "enabled", "budget_tokens": budget_tokens}
            }
            if self.use_isp:
                logger.info("Thinking mode: INTERLEAVED SCRATCHPAD (ISP)")
            else:
                logger.info("Thinking mode: REGULAR SCRATCHPAD")

        try:
            response = None
            
            for attempt in range(API_RETRY_TIMES):
                try:
                    if self.model_name in ["claude-3-7-sonnet-20250219", "claude-4-opus-20250514", "claude-4-sonnet-20250514"]:
                        response = client.beta.messages.create(
                            max_tokens=self.max_tokens,
                            messages=self.messages,
                            model=PROVIDER_TO_DEFAULT_MODEL_NAME[self.provider, self.model_name],
                            system=[system],
                            tools=tools,
                            betas=betas,
                            extra_body=extra_body
                        )
                    elif self.model_name == "claude-3-5-sonnet-20241022":
                        response = client.beta.messages.create(
                            max_tokens=self.max_tokens,
                            messages=self.messages,
                            model=PROVIDER_TO_DEFAULT_MODEL_NAME[self.provider, self.model_name],
                            system=[system],
                            tools=tools,
                            betas=betas,
                        )
                    elif self.model_name in  [ANON_CLAUDE_37, ANON_CLAUDE_45_OPENAI, ANON_CLAUDE_4_OPENAI]:
                        response = self.call_ANON_model(system, tools, betas, extra_body, obs["example_id"])

                    # logger.info(f"Response: {response}")
                    break  
                except (APIError, APIStatusError, APIResponseValidationError) as e:
                    error_msg = str(e)
                    logger.warning(f"Anthropic API error (attempt {attempt+1}/{API_RETRY_TIMES}): {error_msg}")
                    
                    if "25000000" in error_msg or "Member must have length less than or equal to" in error_msg:
                        logger.warning("Detected 25MB limit error, automatically reducing image count")
                        current_image_count = self.only_n_most_recent_images
                        new_image_count = max(1, current_image_count // 2)  # Keep at least 1 image
                        self.only_n_most_recent_images = new_image_count
                        
                        # _maybe_filter_to_n_most_recent_images(
                        #     self.messages,
                        #     new_image_count,
                        #     min_removal_threshold=image_truncation_threshold,
                        # )
                        _custom_maybe_filter_to_n_most_recent_images(
                            self.messages,
                            self.image_names,
                            new_image_count,
                            min_removal_threshold=image_truncation_threshold,
                        )
                        logger.info(f"Image count reduced from {current_image_count} to {new_image_count}")
                    
                    if attempt < API_RETRY_TIMES - 1:
                        interval = random.randint(10, 30)
                        logger.info(f"sleep {interval}s retry ANON call")
                        time.sleep(interval)
                    else:
                        raise  # All attempts failed, raise exception to enter existing except logic

        except (APIError, APIStatusError, APIResponseValidationError) as e:
            logger.exception(f"Anthropic API error: {str(e)}")
            try:
                logger.warning("Retrying with backup API key...")

                backup_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY_BACKUP"), max_retries=4)
                if self.model_name in ["claude-3-7-sonnet-20250219", "claude-4-opus-20250514", "claude-4-sonnet-20250514"]:
                    response = backup_client.beta.messages.create(
                        max_tokens=self.max_tokens,
                        messages=self.messages,
                        model=PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.ANTHROPIC, self.model_name],
                        system=[system],
                        tools=tools,
                        betas=betas,
                        extra_body=extra_body
                    )
                elif self.model_name == "claude-3-5-sonnet-20241022":
                    response = backup_client.beta.messages.create(
                        max_tokens=self.max_tokens,
                        messages=self.messages,
                        model=PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.ANTHROPIC, self.model_name],
                        system=[system],
                        tools=tools,
                        betas=betas,
                    )
                else:
                    raise
                logger.info("Successfully used backup API key")
            except Exception as backup_e:
                backup_error_msg = str(backup_e)
                logger.exception(f"Backup API call also failed: {backup_error_msg}")
                
                # Check if backup API also has 25MB limit error
                if "25000000" in backup_error_msg or "Member must have length less than or equal to" in backup_error_msg:
                    logger.warning("Backup API also encountered 25MB limit error, further reducing image count")
                    # Reduce image count by half again
                    current_image_count = self.only_n_most_recent_images
                    new_image_count = max(1, current_image_count // 2)  # Keep at least 1 image
                    self.only_n_most_recent_images = new_image_count
                    
                    # Reapply image filtering
                    # _maybe_filter_to_n_most_recent_images(
                    #     self.messages,
                    #     new_image_count,
                    #     min_removal_threshold=image_truncation_threshold,
                    # )
                    _custom_maybe_filter_to_n_most_recent_images(
                        self.messages,
                        self.image_names,
                        new_image_count,
                        min_removal_threshold=image_truncation_threshold,
                    )
                    logger.info(f"Backup API image count reduced from {current_image_count} to {new_image_count}")
                
                # return None, None
                raise backup_e

        except Exception as e:
            logger.exception(f"Error in Anthropic API: {str(e)}")
            logger.error(traceback.format_exc())
            # return None, None
            raise e
        if response is None:
            logger.error("Response is None after API call - this should not happen")
            return None, None

        response_params = _response_to_params(response)
        # logger.info(f"Received response params: {response_params}")

        raw_response_str = self._extract_raw_response_string(response)

        # Store response in message history
        self.messages.append({
            "role": "assistant",
            "content": response_params
        })
        self.image_names.append("no_screenshot")
        
        # 处理响应结果
        max_parse_retry = 3
        for parse_retry in range(max_parse_retry):
            actions: list[Any] = []
            reasonings: list[str] = []
            try:
                for content_block in response_params:
                    if content_block["type"] == "tool_use":
                        tool_name = content_block["name"]
                        tool_input = content_block["input"]
                        
                        # 创建基础 action 结构
                        base_action = {
                            "name": tool_name,
                            "input": cast(dict[str, Any], tool_input),
                            "id": content_block["id"],
                            "action_type": content_block.get("type"),
                            "raw_response": raw_response_str
                        }
                        
                        # 根据工具类型处理 command 字段
                        if tool_name == "computer":
                            # Computer 工具:解析为可执行命令
                            base_action["command"] = self.parse_actions_from_tool_call(
                                content_block,
                                obs.get("screenshot_resize_ratio_x", 1.0),
                                obs.get("screenshot_resize_ratio_y", 1.0)
                            )
                            logger.info(f"===Computer action: {tool_input}")
                            
                        elif tool_name == "browse_url":
                            # 浏览器打开网址
                            url = tool_input.get("url")
                            if not url:
                                raise ValueError("Tool 'browse_url' was called without a 'url'.")
                            base_action["action_type"] = "BROWSE_URL"
                            base_action["url"] = url
                            base_action["command"] = f"BROWSE_URL: {url}"
                            logger.info(f"===Browse action: url={url}")
                            
                        elif tool_name == "execute_python":
                            # 执行 Python 代码
                            code_to_execute = tool_input.get("code")
                            if not code_to_execute:
                                raise ValueError("Tool 'execute_python' was called without 'code'.")
                            base_action["action_type"] = "EXECUTE_PYTHON"
                            base_action["command"] = code_to_execute
                            logger.info(f"===Execute Python action: {code_to_execute}")
                            
                        elif tool_name == "execute_shell":
                            # 执行 Shell 命令
                            command_to_execute = tool_input.get("command")
                            if not command_to_execute:
                                raise ValueError("Tool 'execute_shell' was called without 'command'.")
                            base_action["action_type"] = "EXECUTE_SHELL"
                            base_action["command"] = command_to_execute
                            logger.info(f"===Execute Shell action: {command_to_execute}")
                            
                        elif tool_name in ["search", "visit"]:
                            # Web Agent 工具处理
                            try:
                                if self.web_agent is None:
                                    raise Exception("Web agent not initialized")
                                
                                # 调用 web agent 工具
                                result = self.web_agent._call_tool(tool_name, tool_input)
                                logger.info(f"Tool '{tool_name}' result: {result[:500]}...")
                                
                                # 添加工具结果到消息历史
                                self.add_tool_result(
                                    content_block["id"],
                                    f"Tool '{tool_name}' completed. Result: {result}",
                                    screenshot=None,
                                    screenshot_name=f"{tool_name}_result"
                                )
                                
                                base_action["action_type"] = tool_name.upper()
                                base_action["command"] = f"{tool_name.upper()}: {tool_input}"
                                base_action["processed"] = True
                                base_action["result"] = result
                                
                            except Exception as e:
                                logger.error(f"Tool '{tool_name}' execution failed: {e}")
                                self.add_tool_result(
                                    content_block["id"],
                                    f"Tool '{tool_name}' failed: {str(e)}",
                                    screenshot=None,
                                    screenshot_name=f"{tool_name}_error"
                                )
                                base_action["action_type"] = f"{tool_name.upper()}_ERROR"
                                base_action["command"] = f"{tool_name.upper()}_ERROR: {str(e)}"
                                base_action["processed"] = True
                                base_action["error"] = str(e)
                                
                        else:
                            # 其他未知工具的通用处理
                            base_action["command"] = f"{tool_name.upper()}: {str(tool_input)}"
                            logger.warning(f"Unknown tool '{tool_name}' with input: {tool_input}")
                        
                        actions.append(base_action)
                        
                    elif content_block["type"] == "text":
                        reasonings.append(content_block["text"])
                        
                    elif content_block["type"] == "server_tool_use":
                        se_input = content_block["input"]
                        logger.info(f"====get search result!!! {se_input}")
                
                # 处理 reasonings
                if isinstance(reasonings, list) and len(reasonings) > 0:
                    reasonings = reasonings[0]
                else:
                    reasonings = ""
                
                # 检查是否标记为 INFEASIBLE
                if raw_response_str and "[INFEASIBLE]" in raw_response_str:
                    logger.info("Detected [INFEASIBLE] pattern in response, triggering FAIL action")
                    actions = [{
                        "action_type": "FAIL",
                        "raw_response": raw_response_str
                    }]
                
                # 构建 info_dict
                info_dict = {
                    "messages_export": simple_messages_for_export,
                    "low_level_action": reasonings,
                    "pyautogui_actions": [a.get("command", "") for a in actions],
                    "response_text": json.dumps(response_params, ensure_ascii=False),
                    "image_meta": image_meta
                }
                
                logger.info(f"Received actions: {actions}")
                logger.info(f"Received reasonings: {reasonings}")
                
                # 如果没有 actions,返回 DONE
                if len(actions) == 0:
                    actions = [{
                        "action_type": "DONE",
                        "raw_response": raw_response_str
                    }]
                
                return reasonings, actions, info_dict
            except Exception as e:
                logger.warning(f"parse_actions_from_tool_call parsing failed (attempt {parse_retry+1}/3), will retry API request: {e}")
                # Remove the recently appended assistant message to avoid polluting history
                self.messages.pop()
                self.image_names.pop()
                # Retry API request
                response = None
                for attempt in range(API_RETRY_TIMES):
                    try:
                        if self.model_name in ["claude-3-7-sonnet-20250219", "claude-4-opus-20250514", "claude-4-sonnet-20250514"]:
                            response = client.beta.messages.create(
                                max_tokens=self.max_tokens,
                                messages=self.messages,
                                model=PROVIDER_TO_DEFAULT_MODEL_NAME[self.provider, self.model_name],
                                system=[system],
                                tools=tools,
                                betas=betas,
                                extra_body=extra_body
                            )
                        elif self.model_name == "claude-3-5-sonnet-20241022":
                            response = client.beta.messages.create(
                                max_tokens=self.max_tokens,
                                messages=self.messages,
                                model=PROVIDER_TO_DEFAULT_MODEL_NAME[self.provider, self.model_name],
                                system=[system],
                                tools=tools,
                                betas=betas,
                            )
                        elif self.model_name == ANON_CLAUDE_37:
                            response = self.call_ANON_model(system, tools, betas, extra_body, obs["example_id"])
                        elif self.model_name == ANON_CLAUDE_45_OPENAI or self.model_name == ANON_CLAUDE_4_OPENAI:
                            response = self.call_ANON_model(system, tools, betas, extra_body, obs["example_id"])

                        logger.info(f"Response: {response}")
                        break  # Success, exit retry loop
                    except (APIError, APIStatusError, APIResponseValidationError) as e2:
                        error_msg = str(e2)
                        logger.warning(f"Anthropic API error (attempt {attempt+1}/{API_RETRY_TIMES}): {error_msg}")
                        if attempt < API_RETRY_TIMES - 1:
                            time.sleep(API_RETRY_INTERVAL)
                        else:
                            raise
                response_params = _response_to_params(response)
                logger.info(f"Received response params: {response_params}")

                # Update raw response string for retry case (will be used in next loop iteration)
                raw_response_str = self._extract_raw_response_string(response)

                info_dict = {
                    "messages_export": simple_messages_for_export,
                    "low_level_action": "", # Could not parse reasonings here
                    "pyautogui_actions": [], # Could not parse actions here
                    "response_text": json.dumps(response_params, ensure_ascii=False),
                    "image_meta": image_meta
                }
                self.messages.append({
                    "role": "assistant",
                    "content": response_params
                })
                self.image_names.append("no_screenshot")
                if parse_retry == max_parse_retry - 1:
                    logger.error(f"parse_actions_from_tool_call parsing failed 3 times consecutively, terminating: {e}")
                    # actions = ["FAIL"]
                    actions = [{
                        "action_type": "FAIL",
                        "raw_response": f"Failed to parse actions from tool call after {max_parse_retry} attempts: {e}"
                    }]
                    return reasonings, actions, info_dict
    def reset(self, _logger=None, *args, **kwargs):
        """
        Reset the agent's state.
        """
        global agent_logger
        if _logger:
            agent_logger = _logger
        
        # 重置主消息历史
        self.messages = []
        self.image_names = []
        self.turn_count = 0
        
        # 重置评估消息历史(独立于主消息历史)
        self.eval_messages = []
        self.eval_image_names = []
        
        agent_logger.info(f"{self.class_name} reset.")

    def rewrite_model_message_without_image(self):
        simple_messages = []
        user_count = 0
        for message, image_name in zip(self.messages, self.image_names):
            if message["role"] == "user":
                user_count += 1 
            simple_messages.append(self.replace_sensitive_data(message, image_name))
        return simple_messages, user_count
        
    
    def replace_sensitive_data(self, data, image_name, parent_key=''):
        """
        递归遍历字典，如果某层存在 key='type' 且 value='image'，
        则在同一层中将 'source' 或 'data' 的 value 替换为 "base64"。
        返回新字典，不影响原数据。
        """
        if isinstance(data, list):
            new_list = [item for item in data]
            new_list[-1] = self.replace_sensitive_data(new_list[-1], image_name)
            return new_list
        
        if not isinstance(data, dict):
            return data

        # 深拷贝原始数据，避免修改原字典
        new_dict = copy.deepcopy(data)

        for key, value in new_dict.items():
            if key == 'type' and value == 'image':
                # 找到同一层中的 'source' 或 'data' 并替换
                k = "data"
                if k in new_dict:
                    # 如果是真实场景，这里可以替换为 base64.b64encode(value).decode('utf-8')
                    new_dict[k] = image_name  # 示例替换为字符串 "base64"
                k = "source"
                if k in new_dict:
                    new_dict[k] = self.replace_sensitive_data(new_dict[k], image_name, parent_key=k)
            elif key == 'media_type' and value == 'image/png':
                # 找到同一层中的 'source' 或 'data' 并替换
                k = "data"
                if k in new_dict:
                    # 如果是真实场景，这里可以替换为 base64.b64encode(value).decode('utf-8')
                    new_dict[k] = image_name  # 示例替换为字符串 "base64"
            elif key == 'type' and value == 'tool_result':
                # 找到同一层中的 'source' 或 'data' 并替换
                for k in ['content']:
                    if k in new_dict:
                        # 如果是真实场景，这里可以替换为 base64.b64encode(value).decode('utf-8')
                        new_dict[k] = image_name  # 示例替换为字符串 "base64"
            elif isinstance(value, dict):
                new_dict[key] = self.replace_sensitive_data(value, image_name, parent_key=key)
            elif isinstance(value, list):
                new_dict[key] = [self.replace_sensitive_data(item, image_name) if isinstance(item, dict) else item for item in value]

        return new_dict


    def _format_content_for_export(self, content, image_name):
        """
        Helper to format the 'content' part of a message for export.
        - Removes 'tool_result' blocks.
        - Transforms 'image' blocks to OpenAI 'image_url' format.
        """
        if not isinstance(content, list):
            return content

        new_content = []
        for content_block in content:
            if content_block.get("type") == "tool_result":
                continue  # Remove tool_result blocks
            
            elif content_block.get("type") == "image":
                # Transform to OpenAI format
                new_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_name}
                })
            else:
                # Keep other blocks like 'text' or 'tool_use'
                new_content.append(content_block)
        
        return new_content

    def prepare_messages_for_export(self, system_prompt_text=None):
        """
        Creates a copy of the message history for exporting.
        - Adds system prompt on the first turn.
        - Image blocks are replaced with OpenAI-style image_url paths.
        - tool_result blocks are removed.
        """
        export_messages = []
        if system_prompt_text:
            export_messages.append({"role": "system", "content": system_prompt_text})

        for message, image_name in zip(self.messages, self.image_names):
            # Deep copy to avoid modifying the original message history
            new_message = copy.deepcopy(message)
            
            # Process the content of the message using the new helper
            processed_content = self._format_content_for_export(new_message.get('content'), image_name)

            # Only add the message if its content is not empty after filtering
            if processed_content:
                new_message['content'] = processed_content
                export_messages.append(new_message)
            
        return export_messages, sum(1 for m in export_messages if m.get("role") == "user")

    def call_ANON_model(self, system, tools, betas, extra_body, example_id):
        if self.use_openai_format:
            return self._call_ANON_model_openai(system, tools, betas, extra_body, example_id)
        else:
            return self._call_ANON_model_anthropic(system, tools, betas, extra_body, example_id)

    def _convert_response_from_openai_to_anthropic(self, openai_response_json):
        content = []
        response_message = openai_response_json.get("choices", [{}])[0].get("message", {})
        
        if response_message.get("content"):
            content.append({
                "type": "text",
                "text": response_message["content"]
            })
        
        if response_message.get("tool_calls"):
            for tool_call in response_message["tool_calls"]:
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {} # Or handle error appropriately
                content.append({
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "input": arguments
                })

        return BetaMessage(
            id=openai_response_json.get("id", "unknown-id"),
            type="message",
            role="assistant",
            content=content,
            model=openai_response_json.get("model", ""),
            stop_reason="tool_use" if response_message.get("tool_calls") else "end_turn",
            stop_sequence=None,
            usage={
                "input_tokens": openai_response_json.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": openai_response_json.get("usage", {}).get("completion_tokens", 0)
            }
        )
 
    def _convert_messages_to_openai_format(self, messages, system_prompt):
        openai_messages = []
        if system_prompt:
             openai_messages.append({'role': 'system', 'content': system_prompt})

        for message in messages:
            role = message['role']
            content = message['content']

            if role == 'user':
                openai_content_parts = []
                tool_results = []
                
                for item in content:
                    if item['type'] == 'tool_result':
                        tool_results.append({
                            'role': 'tool',
                            'tool_call_id': item['tool_use_id'],
                            'content': str(item.get('content', ''))
                        })
                    elif item['type'] == 'image':
                        source = item['source']
                        if source['type'] == 'base64':
                            openai_content_parts.append({
                                'type': 'image_url',
                                'image_url': { 'url': f"data:{source['media_type']};base64,{source['data']}", "detail": "high" }
                            })
                    elif item['type'] == 'text':
                        openai_content_parts.append({'type': 'text', 'text': item['text']})

                openai_messages.extend(tool_results)
                if openai_content_parts:
                    openai_messages.append({'role': 'user', 'content': openai_content_parts})
            
            elif role == 'assistant':
                tool_calls = []
                text_content = ""
                has_tool_use = False
                for item in content:
                    if item['type'] == 'tool_use':
                        has_tool_use = True
                        tool_calls.append({
                            'id': item['id'],
                            'type': 'function',
                            'function': {
                                'name': item['name'],
                                'arguments': json.dumps(item['input'])
                            }
                        })
                    elif item['type'] == 'text':
                        text_content += item['text']
                
                assistant_message = {'role': 'assistant'}
                if text_content:
                    assistant_message['content'] = text_content

                if has_tool_use:
                    assistant_message['tool_calls'] = tool_calls
                    if not text_content:
                        assistant_message['content'] = None
                
                openai_messages.append(assistant_message)
        
        return openai_messages


    def _call_ANON_model_openai(self, system, tools, betas, extra_body, example_id, max_retries=3, retry_interval=5):
        user_count = sum(1 for message in self.messages if message.get("role") == "user")
        agent_logger.info(f"request ANON model (openai format) {self.model_name}, 第{user_count-1}轮")

        timestamp_ms = int(time.time() * 1000)
        trace_id = f"{example_id}_{user_count-1}_{timestamp_ms}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}", 
            "M-TraceId": trace_id
        }

        openai_messages = self._convert_messages_to_openai_format(self.messages, system["text"])
        simple_messages_for_log, _ = self.rewrite_model_message_without_image()
        agent_logger.info(f"request ANON model (openai format) {self.model_name}, 第{user_count-1}轮, message:\n {json.dumps(simple_messages_for_log, indent=2, ensure_ascii=False)}")

        payload = {
            "model": self.model_name.replace("-openai", ""),
            "max_tokens": self.max_tokens,
            "messages": openai_messages,
            "tools": tools,
            "anthropic_beta": betas,
        }

        logger.info(f"request ANON model (openai format) {self.model_name}, tools: {tools}, request M-TraceId={trace_id}")

        for attempt in range(max_retries):
            try:
                http_response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
                logger.info(f"request ANON model (openai format) {self.model_name}, request M-TraceId={trace_id}, Response status_code: {http_response.status_code}, reason: {http_response.reason}")

                if http_response.status_code == 429:
                    logger.error(f"Error response from API: {http_response.text}")
                    resp_json = http_response.json()
                    raise APIStatusError(
                        http_response.reason,
                        response=http_response,
                        body=resp_json
                    )

                if http_response.status_code == 400:
                    logger.error(f"400 Bad Request: {http_response.text}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying after 400 error... attempt {attempt+1}/{max_retries}")
                        time.sleep(retry_interval)
                        continue
                    else:
                        raise Exception(f"400 Bad Request: {http_response.text}")

                if http_response.status_code != 200:
                    logger.error(f"Error response from API: {http_response.text}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying after error {http_response.status_code}... attempt {attempt+1}/{max_retries}")
                        time.sleep(retry_interval)
                        continue
                    else:
                        raise Exception(f"{http_response.status_code} {http_response.reason}")

                response_json = http_response.json()
                logger.info(f"request ANON model (openai format) response: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
                return self._convert_response_from_openai_to_anthropic(response_json)

            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                logger.error(f"requests timeout/connection error: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying after timeout/connection error... attempt {attempt+1}/{max_retries}")
                    time.sleep(retry_interval)
                    continue
                else:
                    raise Exception(f"requests timeout/connection error after {max_retries} attempts: {str(e)}")
            except Exception as e:
                logger.error(f"Exception in _call_ANON_model_openai: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying after exception... attempt {attempt+1}/{max_retries}")
                    time.sleep(retry_interval)
                    continue
                else:
                    raise


    def _call_ANON_model_anthropic(self, system, tools, betas, extra_body, example_id):
        user_count = 0
        if self.log_image_base64:
            user_count = sum(1 for message in self.messages if message.get("role") == "user")
            agent_logger.info(f"request ANON model {self.model_name}, 第{user_count-1}轮, message:\n {json.dumps(self.messages, indent=2, ensure_ascii=False)}")
            trace_id = f"{trace_id}"
        else:
            simple_messages, user_count = self.rewrite_model_message_without_image()
            agent_logger.info(f"request ANON model {self.model_name}, 第{user_count-1}轮, message:\n {json.dumps(simple_messages, indent=2, ensure_ascii=False)}")

        timestamp_ms = int(time.time() * 1000)
        trace_id = f"{example_id}_{user_count-1}_{timestamp_ms}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}", 
            "M-TraceId": trace_id
        }
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
            "system": system["text"],
            "tools": tools,
            "anthropic_beta": betas,
            "thinking": extra_body["thinking"]
        }
        logger.info(f"request ANON model {self.model_name}, tools: {tools}, betas: {betas}, request M-TraceId={trace_id}")
        http_response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
        logger.info(f"request ANON model {self.model_name}, request M-TraceId={trace_id}, Response status_code: {http_response.status_code}, reason: {http_response.reason}, text: {http_response.json()}")
        if http_response.status_code == 429:
            raise APIStatusError(http_response.reason)
        if http_response.status_code != 200:
            raise Exception(http_response.reason)
        
        return BetaMessage.model_validate_json(http_response.text)

    def eval_task(self, env, task_config, trajectory, save_dir, max_round=40):
        """
        评估任务执行结果
        
        Args:
            env: 环境实例(用于执行需要交互的工具调用)
            task_config: 任务配置
            trajectory: 已执行的轨迹 {"screenshots": [], "actions": []}
            save_dir: 保存目录
            max_round: 最大评估轮数
            
        Returns:
            dict: 评估结果 {"success": bool, "score": float, "reason": str, "eval_steps": int}
        """
        logger.info("="*80)
        logger.info("Starting eval_task with Claude agent")
        logger.info(f"Max rounds: {max_round}")
        logger.info(f"Task: {task_config.get('instruction', 'N/A')}")
        logger.info("="*80)

        # 重置评估消息历史
        self.eval_messages = []
        self.eval_image_names = []
        betas = [COMPUTER_USE_BETA_FLAG]
        
        # 准备评估用的图片映射
        imgs_map = {}
        screenshots_for_tools = []
        
        # 保存轨迹中的截图到临时目录
        TOOL_CALL_IMG_TEMP_DIR = os.getenv("TOOL_CALL_IMG_TEMP", "./temp_eval_images")
        os.makedirs(TOOL_CALL_IMG_TEMP_DIR, exist_ok=True)

        # ⭐ 记录缩放比例(用于后续的坐标转换)
        eval_resize_ratio_x = 1.0
        eval_resize_ratio_y = 1.0
        
        for i, screenshot_bytes in enumerate(trajectory.get("screenshots", [])):
            step_name = f"step_{i + 1}"

            # ⭐ 缩放截图 
            resized_bytes, x_ratio, y_ratio = resize_image_to_1024_768(screenshot_bytes)
            
            # 记录第一张图片的缩放比例
            if i == 0:
                eval_resize_ratio_x = x_ratio
                eval_resize_ratio_y = y_ratio
                logger.info(f"Evaluation resize ratios: x={x_ratio:.4f}, y={y_ratio:.4f}")
            
            # 将缩放后的 bytes 转换为 PIL Image
            screenshot_img = Image.open(io.BytesIO(resized_bytes))
            screenshots_for_tools.append(screenshot_img)
            
            # 保存到临时文件
            path_name = f"eval_{step_name}_{int(time.time()*1000)}"
            image_path = os.path.join(TOOL_CALL_IMG_TEMP_DIR, f"{path_name}.png")
            screenshot_img.save(image_path)
            imgs_map[step_name] = image_path
            
            logger.info(f"Saved evaluation screenshot: {step_name} -> {image_path}")
        
        # 构建评估提示
        instruction = task_config.get("instruction", "")
        eval_prompt = _build_eval_prompt(instruction, task_config, trajectory, len(screenshots_for_tools))
        
        # 添加初始用户消息(包含任务描述和轨迹信息)
        initial_content = [
            {"type": "text", "text": eval_prompt}
        ]
        
        # 添加第一张和最后一张截图作为参考
        if screenshots_for_tools:
            # 第一张截图
            # first_screenshot_base64 = base64.b64encode(trajectory["screenshots"][0]).decode('utf-8')
            # initial_content.append({
            #     "type": "image",
            #     "source": {
            #         "type": "base64",
            #         "media_type": "image/png",
            #         "data": first_screenshot_base64
            #     }
            # })
            # self.eval_image_names.append("initial_screenshot")
            
            # 最后一张截图
            if len(trajectory["screenshots"]) > 0:
                # 缩放最后一张截图
                last_screenshot_resized, _, _ = resize_image_to_1024_768(trajectory["screenshots"][-1])
                last_screenshot_base64 = base64.b64encode(last_screenshot_resized).decode('utf-8')
                
                initial_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": last_screenshot_base64
                    }
                })
                self.eval_image_names.append("final_screenshot")

            # # 最后4张截图
            # if len(trajectory["screenshots"]) > 0:
            #     # 获取最后4张截图 (如果不足4张则取全部)
            #     recent_screenshots = trajectory["screenshots"][-4:]
                
            #     for idx, screenshot_bytes in enumerate(recent_screenshots):
            #         # 缩放截图
            #         resized_bytes, _, _ = resize_image_to_1024_768(screenshot_bytes)
            #         screenshot_base64 = base64.b64encode(resized_bytes).decode('utf-8')
                    
            #         initial_content.append({
            #             "type": "image",
            #             "source": {
            #                 "type": "base64",
            #                 "media_type": "image/png",
            #                 "data": screenshot_base64
            #             }
            #         })
            #     self.eval_image_names.append(f"final_five_screenshots")
    
        self.eval_messages.append({
            "role": "user",
            "content": initial_content
        })

        # 生成易读版本用于打印
        display_messages = self._format_eval_messages_for_display(self.eval_messages)
        logger.info(f"====eval_messages (formatted for display):\n{json.dumps(display_messages, indent=2, ensure_ascii=False)}")
        
        
        # 获取工具定义
        tool_config = {
            'name': 'computer', 
            'type': 'computer_20250124', 
            'display_width_px': 1280, 
            'display_height_px': 720, 
            'display_number': 1
        }

        tools = [
            tool_config,
        ] if self.platform == 'Ubuntu' else [
            tool_config,
        ]
        tools  = tools + self.custom_tools
        
        # 使用原始 SYSTEM_PROMPT
        system_prompt = SYSTEM_PROMPT_WINDOWS if self.platform == 'Windows' else SYSTEM_PROMPT
        if self.system_prompt_suffix:
            system_prompt += ' ' + self.system_prompt_suffix

        # ⭐ 添加评估特定的指导
        system_prompt += f"""

        EVALUATION MODE INSTRUCTIONS:
        ============================
        You are in evaluation mode with a maximum of {max_round} rounds. 
        **IMPORTANT**: Provide your final judgment BEFORE reaching the maximum rounds

        Final Judgment Format (REQUIRED):
        EVALUATION RESULT:
        Status: [SUCCESS or FAILURE]
        Confidence: [HIGH or MEDIUM or LOW]
        Reasoning: [Your detailed reasoning]

        ⚠️ If you don't provide the final judgment in the correct format, the evaluation will fail.
        """

        # ⭐ 创建评估结果保存目录
        eval_result_dir = os.path.join(save_dir, 'eval_result')
        os.makedirs(eval_result_dir, exist_ok=True)
        
        # 评估循环
        eval_step = 0
        evaluation_complete = False
        final_reasoning = []
        
        for round_num in range(max_round):
            eval_step += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluation Round {eval_step}/{max_round}")
            logger.info(f"{'='*60}")
            
            try:
                # 直接调用 _call_ANON_model_openai
                timestamp_ms = int(time.time() * 1000)
                example_id = task_config.get("id", "eval_task")
                trace_id = f"{example_id}_eval_{eval_step}_{timestamp_ms}"
                
                # 生成易读版本用于打印
                display_messages = self._format_eval_messages_for_display(self.eval_messages)
                logger.info(f"====eval_messages round_num:{round_num} (formatted for display):\n{json.dumps(display_messages, indent=2, ensure_ascii=False)}")
                
                # 转换评估消息为 OpenAI 格式
                openai_messages = self._convert_messages_to_openai_format(self.eval_messages, system_prompt)

                # 验证转换结果
                if not openai_messages:
                    logger.error("No valid messages after conversion, skipping this round")
                    break
                
                # 生成易读版本用于打印
                display_messages_openai = self._format_eval_messages_for_display(openai_messages)
                # logger.info(f"====openai eval_messages (formatted for display):\n{json.dumps(display_messages_openai, indent=2, ensure_ascii=False)}")
                
                logger.info(f"Calling ANON model for evaluation, trace_id={trace_id}")
                logger.info(f"Messages count: {len(openai_messages)}")
                
                logger.info(f"Calling ANON model for evaluation, trace_id={trace_id}")
                logger.info(f"Messages count: {len(openai_messages)}")
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}", 
                    "M-TraceId": trace_id
                }
                
                payload = {
                    "model": self.model_name.replace("-openai", ""),
                    "max_tokens": self.max_tokens,
                    "messages": openai_messages,
                    "anthropic_beta": [COMPUTER_USE_BETA_FLAG],
                }
                
                # 只有在有工具时才添加 tools 参数
                if tools:
                    payload["tools"] = tools
                
                # 添加采样参数
                sampling_params = self._get_sampling_params()
                payload.update(sampling_params)
                
                # 调用 API
                max_retries = 15
                retry_interval = 8
                response = None
                
                for attempt in range(max_retries):
                    try:
                        http_response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
                        
                        logger.info(f"Evaluation API response: status_code={http_response.status_code}")
                        
                        if http_response.status_code == 429:
                            logger.error(f"Rate limit error: {http_response.text}")
                            resp_json = http_response.json()
                            raise APIStatusError(
                                http_response.reason,
                                response=http_response,
                                body=resp_json
                            )
                        
                        if http_response.status_code == 400:
                            logger.error(f"400 Bad Request: {http_response.text}")
                            if attempt < max_retries - 1:
                                logger.info(f"Retrying after 400 error... attempt {attempt+1}/{max_retries}")
                                time.sleep(retry_interval)
                                continue
                            else:
                                raise Exception(f"400 Bad Request: {http_response.text}")
                        
                        if http_response.status_code == 500:
                            logger.error(f"====500 Internal Server Error: {http_response.text}")
                            if attempt < max_retries - 1:
                                logger.info(f"Retrying after 500 error... attempt {attempt+1}/{max_retries}")
                                time.sleep(retry_interval)
                                continue
                            else:
                                raise Exception(f"500 Internal Server Error: {http_response.text}")
                        
                        if http_response.status_code != 200:
                            logger.error(f"Error response from API: {http_response.text}")
                            if attempt < max_retries - 1:
                                logger.info(f"Retrying after error {http_response.status_code}... attempt {attempt+1}/{max_retries}")
                                time.sleep(retry_interval)
                                continue
                            else:
                                raise Exception(f"{http_response.status_code} {http_response.reason}")
                        
                        response_json = http_response.json()
                        logger.info(f"Evaluation response received successfully")
                        
                        # 转换为 Anthropic 格式
                        response = self._convert_response_from_openai_to_anthropic(response_json)
                        break
                    
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.error(f"Request error: {str(e)}")
                            time.sleep(retry_interval)
                            continue
                        else:
                            raise Exception(f"Request timeout after {max_retries} attempts: {str(e)}")
                        
                    # except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                    #     logger.error(f"Request timeout/connection error: {str(e)}")
                    #     if attempt < max_retries - 1:
                    #         logger.info(f"Retrying after timeout... attempt {attempt+1}/{max_retries}")
                    #         time.sleep(retry_interval)
                    #         continue
                    #     else:
                    #         raise Exception(f"Request timeout after {max_retries} attempts: {str(e)}")
                
                if response is None:
                    logger.error("Failed to get response from ANON model")
                    break

                logger.info(f"=====Response: {response}")
                
                # 提取响应内容
                response_params = _response_to_params(response)
                raw_response_str = self._extract_raw_response_string(response)
                
                # 添加助手响应到消息历史
                self.eval_messages.append({
                    "role": "assistant",
                    "content": response_params
                })
                self.eval_image_names.append("no_screenshot")
                
                # 收集推理文本
                for block in response_params:
                    if block.get("type") == "text":
                        final_reasoning.append(block.get("text", ""))
                
                # 处理工具调用
                tool_calls = [block for block in response_params if block.get("type") == "tool_use"]
                
                if not tool_calls:
                    # 没有工具调用,认为评估完成
                    logger.info("No more tool calls, evaluation complete")
                    evaluation_complete = True
                    break
                
                # 处理每个工具调用
                has_computer_action = False
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    tool_input = tool_call.get("input", {})
                    tool_id = tool_call.get("id")
                    
                    logger.info(f"Tool call: {tool_name}")
                    logger.info(f"  Input: {json.dumps(tool_input, ensure_ascii=False)}")
                    
                    # 根据工具名称判断是否需要与环境交互
                    if tool_name == "check_screenshot":
                        # 查看图片任务,不需要与环境交互
                        tool_result = self._read_key_images(tool_input, imgs_map)
                        logger.info(f"  ✓ Read screenshots from local files")
                        
                    elif tool_name == "computer":
                        # 需要与环境交互
                        tool_result = self._execute_computer_action(
                            tool_input, 
                            env,
                            eval_resize_ratio_x,
                            eval_resize_ratio_y
                        )
                        has_computer_action = True
                        logger.info(f"  ✓ Executed computer action with environment")

                        reward_save_dir = save_dir+'/eval_reault'
                        os.makedirs(reward_save_dir, exist_ok=True)  

                        #保存交互截图
                        action_timestamp = datetime.now().strftime("%Y%m%d@%H%M%S%f")
                        with open(os.path.join(reward_save_dir, f"round_{round_num + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                            _f.write(tool_result['screenshot'])

                    elif tool_name == "execute_python":
                        # ⭐ 执行 Python 代码
                        tool_result = self._execute_python_code(tool_input, env, eval_result_dir, round_num)
                        logger.info(f"  ✓ Executed Python code")
                        
                    elif tool_name == "execute_shell":
                        # ⭐ 执行 Shell 命令
                        tool_result = self._execute_shell_command(tool_input, env, eval_result_dir, round_num)
                        logger.info(f"  ✓ Executed Shell command")

                    # ⭐ 新增: 处理 read_pptx 工具
                    elif tool_name == "read_pptx":
                        # 读取 PPTX 文件
                        tool_result = self._read_pptx_file(tool_input, env, eval_result_dir, round_num)
                        logger.info(f"  ✓ Read PPTX file")
    
                    else:
                        # 其他工具,尝试作为查看图片处理
                        logger.warning(f"Unknown tool: {tool_name}, treating as screenshot reader")
                        tool_result = self._read_key_images(tool_input, imgs_map)
                    
                    # if round_num < max_round - 1:
                        # 添加工具结果到消息历史
                    self._add_tool_result_to_eval_messages(
                        tool_id=tool_id,
                        tool_result=tool_result
                    )
                
                # 如果执行了 computer 动作,可能需要额外的评估轮次
                if has_computer_action:
                    logger.info("Computer action executed, continuing evaluation...")
                    
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in evaluation round {eval_step}: {e}")
                logger.error(traceback.format_exc())
                break
        
        # 如果没有完成评估,请求最终判断
        if not evaluation_complete and eval_step >= 15:  # 改为 <=
            logger.warning("Evaluation not complete, requesting final judgment...")
            self.eval_messages = remove_images_and_add_note(self.eval_messages, self.only_n_most_recent_images_eval)
            self.eval_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Now, you don't need to go any further with the verification. Based on your previous analysis, judge whether the task has been completed and provide your final conclusion in the specified format."
                    }
                ]
            })
            self.eval_image_names.append("no_screenshot")
            
            try:
                timestamp_ms = int(time.time() * 1000)
                example_id = task_config.get("id", "eval_task")
                trace_id = f"{example_id}_eval_final_{timestamp_ms}"
                
                # 转换评估消息为 OpenAI 格式
                openai_messages = self._convert_messages_to_openai_format(self.eval_messages, system_prompt)
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}", 
                    "M-TraceId": trace_id
                }
                
                payload = {
                    "model": self.model_name.replace("-openai", ""),
                    "max_tokens": self.max_tokens,
                    "messages": openai_messages,
                    "anthropic_beta": [COMPUTER_USE_BETA_FLAG],
                }
                
                # 最终判断不需要工具
                # payload["tools"] = []
                
                # 添加采样参数
                sampling_params = self._get_sampling_params()
                payload.update(sampling_params)
                
                max_retries = 15
                retry_interval = 8
                
                for attempt in range(max_retries):
                    try:
                        http_response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
                        
                        if http_response.status_code == 429:
                            logger.error(f"Final judgment rate limit error: {http_response.text}")
                            if attempt < max_retries - 1:
                                logger.info(f"Retrying final judgment after 429... attempt {attempt+1}/{max_retries}")
                                time.sleep(retry_interval)
                                continue
                        
                        if http_response.status_code == 500:
                            logger.error(f"Final judgment 500 Internal Server Error: {http_response.text}")
                            if attempt < max_retries - 1:
                                logger.info(f"Retrying final judgment after 500... attempt {attempt+1}/{max_retries}")
                                time.sleep(retry_interval)
                                continue
                        
                        if http_response.status_code == 200:
                            response_json = http_response.json()
                            final_response = self._convert_response_from_openai_to_anthropic(response_json)
                            
                            if final_response:
                                final_params = _response_to_params(final_response)
                                for block in final_params:
                                    if block.get("type") == "text":
                                        final_reasoning.append(block.get("text", ""))
                                self.eval_messages.append({
                                    "role": "assistant",
                                    "content": final_params
                                })
                                self.eval_image_names.append("no_screenshot")
                                logger.info("Successfully received final judgment.")
                            # 成功获取结果，跳出重试循环
                            break
                        else:
                            logger.error(f"Final judgment request failed: {http_response.status_code} {http_response.text}")
                            if attempt < max_retries - 1:
                                logger.info(f"Retrying final judgment after error {http_response.status_code}... attempt {attempt+1}/{max_retries}")
                                time.sleep(retry_interval)
                                continue
                          
                    except Exception as e:
                        logger.error(f"Exception during final judgment request: {str(e)}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying final judgment after exception... attempt {attempt+1}/{max_retries}")
                            time.sleep(retry_interval)
                            continue
                        else:
                            logger.error(f"Final judgment failed after {max_retries} attempts.")
                # === 添加重试机制结束 ===
                    
            except Exception as e:
                logger.error(f"Error getting final judgment: {e}")
                logger.error(traceback.format_exc())
        
        # 解析最终结果
        combined_reasoning = "\n".join(final_reasoning)
        result = self._parse_eval_result(combined_reasoning, eval_step)
        
        # 保存评估日志
        self._save_eval_log(save_dir, self.eval_messages, result, task_config)
        
        logger.info("="*80)
        logger.info(f"Evaluation completed: {result}")
        logger.info("="*80)
        
        return result

    def _read_key_images(self, tool_input: dict, imgs_map: dict) -> dict:
    # def _read_key_images_for_eval(self, tool_input: dict, imgs_map: dict) -> dict:
        """
        读取关键截图并使用大模型进行分析(评估专用版本)
        返回格式适配多轮对话
        """
        image_names = tool_input.get("image_names", [])
        
        if not image_names:
            return {
                "success": False,
                "text": "No image names provided",
                "images": []
            }
        
        # 收集图片
        result_images = []
        loaded_info = []
        missing_info = []
        
        for img_name in image_names:
            if img_name in imgs_map:
                img_path = imgs_map[img_name]
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        result_images.append(img)
                        loaded_info.append(img_name)
                        logger.info(f"  ✓ Loaded: {img_name}")
                    except Exception as e:
                        missing_info.append(f"{img_name} (error: {str(e)})")
                        logger.error(f"  ✗ Error loading {img_name}: {e}")
                else:
                    missing_info.append(f"{img_name} (file not found)")
                    logger.warning(f"  ✗ File not found: {img_name}")
            else:
                missing_info.append(f"{img_name} (not in trajectory)")
                logger.warning(f"  ✗ Not in trajectory: {img_name}")
        
        if not result_images:
            error_text = "Failed to load any images:\n" + "\n".join(missing_info)
            if imgs_map:
                error_text += f"\n\nAvailable images: {list(imgs_map.keys())}"
            return {"success": False, "text": error_text, "images": []}
        
        # 构建结果文本
        result_text = f"Successfully loaded {len(result_images)} screenshot(s): {', '.join(loaded_info)}\n\n"
        result_text += "These are the screenshots at the requested steps. Please analyze them to check if the conditions are met."
        
        if missing_info:
            result_text += f"\n\nNote: Could not load {len(missing_info)} image(s): {', '.join(missing_info)}"
        
        return {
            "success": True,
            "text": result_text,
            "images": result_images
        }

    def _execute_computer_action(self, tool_input: dict, env, resize_ratio_x: float = 1.0, resize_ratio_y: float = 1.0) -> dict:
        """执行计算机操作(与环境交互)"""
        try:
            
            # ⭐ 构建 tool_call 结构(用于 parse_actions_from_tool_call)
            tool_call = {
                "name": "computer",
                "input": tool_input,
                "id": f"eval_action_{int(time.time()*1000)}"
            }
            
            # ⭐ 使用缩放比例解析动作(坐标会被正确转换)
            command = self.parse_actions_from_tool_call(
                tool_call,
                screenshot_resize_ratio_x=resize_ratio_x,
                screenshot_resize_ratio_y=resize_ratio_y
            )
            
            # 构建动作
            action = {
                "name": "computer",
                "input": tool_input,
                "id": tool_call["id"],
                "action_type": "tool_use",
                "command": command
            }
            
            logger.info(f"Executing computer action: {action['command'][:200]}")
            
            # 执行动作
            # obs, reward, done, info = env.step(action, sleep_after_execution=2)
            obs, reward, done, info = env.step(action)

            # ⭐ 获取原始截图
            original_screenshot = obs.get("screenshot")
            
            # ⭐ 对截图进行 resize(与评估流程保持一致)
            resized_screenshot = None
            if original_screenshot:
                try:
                    resized_bytes, _, _ = resize_image_to_1024_768(original_screenshot)
                    resized_screenshot = resized_bytes
                    logger.info(f"  ✓ Resized screenshot from original to 1280x720")
                except Exception as resize_error:
                    logger.error(f"  ✗ Failed to resize screenshot: {resize_error}")
                    # 如果 resize 失败,使用原始截图
                    resized_screenshot = original_screenshot
        
            
            result_text = f"Computer action executed successfully.\n"
            result_text += f"Reward: {reward}\n"
            result_text += f"Done: {done}\n"
            if info:
                result_text += f"Info: {str(info)[:200]}\n"
            
            return {
                "success": True,
                "text": result_text,
                "screenshot": resized_screenshot,  # ⭐ 返回缩放后的截图
                "images": []
            }
            
        except Exception as e:
            logger.error(f"Error executing computer action: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "text": f"Error executing computer action: {str(e)}",
                "screenshot": None,
                "images": []
            }

    # ⭐ 简化版:执行 Python 代码
    def _execute_python_code(self, tool_input: dict, env, save_dir: str, round_num: int) -> dict:
        """
        执行 Python 代码(与环境交互)
        
        Args:
            tool_input: 工具输入,包含 code 字段
            env: 环境实例
            save_dir: 保存目录
            round_num: 当前轮次
            
        Returns:
            dict: 执行结果
        """
        try:
            code_to_execute = tool_input.get("code")
            if not code_to_execute:
                return {
                    "success": False,
                    "text": "No Python code provided",
                    "images": []
                }
            
            # 构建动作
            action = {
                "name": "execute_python",
                "input": tool_input,
                "id": f"eval_python_{int(time.time()*1000)}",
                "action_type": "EXECUTE_PYTHON",
                "command": code_to_execute
            }
            
            logger.info(f"Executing Python code: {code_to_execute[:200]}...")
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            # ⭐ 直接使用 info 作为结果文本(不做提取)
            result_text = str(info) if info else "No output"
            
            logger.info(f"Python execution result: {result_text[:200]}...")
            
            # 保存执行结果到文件
            action_timestamp = datetime.now().strftime("%Y%m%d@%H%M%S%f")
            result_file = os.path.join(save_dir, f"python_round_{round_num + 1}_{action_timestamp}.txt")
            
            try:
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(f"Code:\n{code_to_execute}\n\n")
                    f.write(f"Result:\n{result_text}\n")
                logger.info(f"  ✓ Saved Python execution result: {result_file}")
            except Exception as e:
                logger.error(f"  ✗ Failed to save result: {e}")
            
            return {
                "success": True,
                "text": result_text,
                "images": []
                # ⭐ 不返回 screenshot
            }
            
        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "text": f"Error executing Python code: {str(e)}",
                "images": []
            }

    # ⭐ 简化版:执行 Shell 命令
    def _execute_shell_command(self, tool_input: dict, env, save_dir: str, round_num: int) -> dict:
        """
        执行 Shell 命令(与环境交互)
        
        Args:
            tool_input: 工具输入,包含 command 字段
            env: 环境实例
            save_dir: 保存目录
            round_num: 当前轮次
            
        Returns:
            dict: 执行结果
        """
        try:
            command_to_execute = tool_input.get("command")
            if not command_to_execute:
                return {
                    "success": False,
                    "text": "No shell command provided",
                    "images": []
                }
            
            # 构建动作
            action = {
                "name": "execute_shell",
                "input": tool_input,
                "id": f"eval_shell_{int(time.time()*1000)}",
                "action_type": "EXECUTE_SHELL",
                "command": command_to_execute
            }
            
            logger.info(f"Executing Shell command: {command_to_execute}")
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            # ⭐ 直接使用 info 作为结果文本(不做提取)
            result_text = str(info) if info else "No output"
            
            logger.info(f"Shell execution result: {result_text[:200]}...")
            
            # 保存执行结果到文件
            action_timestamp = datetime.now().strftime("%Y%m%d@%H%M%S%f")
            result_file = os.path.join(save_dir, f"shell_round_{round_num + 1}_{action_timestamp}.txt")
            
            try:
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(f"Command:\n{command_to_execute}\n\n")
                    f.write(f"Result:\n{result_text}\n")
                logger.info(f"  ✓ Saved Shell execution result: {result_file}")
            except Exception as e:
                logger.error(f"  ✗ Failed to save result: {e}")
            
            return {
                "success": True,
                "text": result_text,
                "images": []
                # ⭐ 不返回 screenshot
            }
            
        except Exception as e:
            logger.error(f"Error executing Shell command: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "text": f"Error executing Shell command: {str(e)}",
                "images": []
            }
    
    def _read_pptx_file(self, tool_input: dict, env, save_dir: str, round_num: int) -> dict:
        """
        读取 PPTX 文件详细信息(在环境中执行)
        
        Args:
            tool_input: 工具输入,包含 file_path, extract_level, slide_numbers
            env: 环境实例
            save_dir: 保存目录
            round_num: 当前轮次
            
        Returns:
            dict: 执行结果
        """
        try:
            file_path = tool_input.get("file_path")
            extract_level = tool_input.get("extract_level", "detailed")
            slide_numbers = tool_input.get("slide_numbers", None)
            
            if not file_path:
                return {
                    "success": False,
                    "text": "No file path provided",
                    "images": []
                }
            
            logger.info(f"Reading PPTX file in environment: {file_path}")
            logger.info(f"  Extract level: {extract_level}")
            if slide_numbers:
                logger.info(f"  Slide numbers: {slide_numbers}")
            
            # ⭐ 导入 ReadPPTX 工具并生成代码
            from src.agent.anthropic.tool_readPPT import ReadPPTX
            
            read_pptx_tool = ReadPPTX()
            python_code = read_pptx_tool.generate_execution_code(
                file_path=file_path,
                extract_level=extract_level,
                slide_numbers=slide_numbers
            )
            
            # 构建动作
            action = {
                "name": "execute_python",
                "input": {"code": python_code},
                "id": f"eval_read_pptx_{int(time.time()*1000)}",
                "action_type": "EXECUTE_PYTHON",
                "command": python_code
            }
            
            logger.info(f"Executing PPTX read code in environment (code length: {len(python_code)} chars)")
            
            # 在环境中执行
            obs, reward, done, info = env.step(action)
            
            # ⭐ 直接使用 info 作为结果文本
            result_text = str(info) if info else "No output from PPTX read operation"
            
            logger.info(f"PPTX read result: {result_text[:200]}...")
            
            # 保存执行结果到文件
            action_timestamp = datetime.now().strftime("%Y%m%d@%H%M%S%f")
            result_file = os.path.join(save_dir, f"pptx_read_round_{round_num + 1}_{action_timestamp}.txt")
            
            try:
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(f"File: {file_path}\n")
                    f.write(f"Extract Level: {extract_level}\n")
                    if slide_numbers:
                        f.write(f"Slide Numbers: {slide_numbers}\n")
                    f.write("\n" + "="*80 + "\n\n")
                    f.write(f"Python Code:\n{python_code}\n\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Result:\n{result_text}\n")
                logger.info(f"  ✓ Saved PPTX read result: {result_file}")
            except Exception as e:
                logger.error(f"  ✗ Failed to save result: {e}")
            
            return {
                "success": True,
                "text": result_text,
                "images": []
            }
            
        except Exception as e:
            logger.error(f"Error reading PPTX file: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "text": f"Error reading PPTX file: {str(e)}",
                "images": []
            }

    def _add_tool_result_to_eval_messages(self, tool_id: str, tool_result: dict):
        """添加工具结果到评估消息历史"""
        content = [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": tool_result.get("text", "Success")
            }
        ]
        
        # 添加图片(如果有)
        if tool_result.get("images"):
            for img in tool_result["images"]:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_base64
                    }
                })
            self.eval_image_names.append(f"tool_result_images_{len(self.eval_image_names)}")
        
        # 添加环境交互后的截图
        if tool_result.get("screenshot"):
            screenshot_base64 = base64.b64encode(tool_result["screenshot"]).decode()
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_base64
                }
            })
            self.eval_image_names.append(f"computer_action_screenshot_{len(self.eval_image_names)}")
        
        if not tool_result.get("images") and not tool_result.get("screenshot"):
            self.eval_image_names.append("no_screenshot")
        
        self.eval_messages = remove_images_and_add_note(self.eval_messages, max(self.only_n_most_recent_images_eval - count_images_in_content(content), 0))

        self.eval_messages.append({
            "role": "user",
            "content": content
        })


    def _parse_eval_result(self, reasoning_text: str, eval_steps: int) -> dict:
        """解析评估结果(基于格式化输出判断成功/失败)"""
        if not reasoning_text:
            return {
                "success": False,
                "score": 0.0,
                "reason": "No reasoning provided by evaluator",
                "eval_steps": eval_steps,
                "confidence": "UNKNOWN",
                "parse_method": "empty"
            }
        
        success = None
        confidence = "UNKNOWN"
        reasoning = reasoning_text
        parse_method = "failed"
        
        # 方法1: 解析完整的 EVALUATION RESULT 格式
        # eval_result_pattern = r"EVALUATION RESULT:\s*Status:\s*(SUCCESS|FAILURE)\s*Confidence:\s*(HIGH|MEDIUM|LOW)\s*Reasoning:\s*(.+?)(?=\n\n|\Z)"
        eval_result_pattern = r"EVALUATION RESULT:\s*Reasoning:\s*(.+?)\s*Status:\s*(SUCCESS|FAILURE)\s*Confidence:\s*(HIGH|MEDIUM|LOW)"
        match = re.search(eval_result_pattern, reasoning_text, re.IGNORECASE | re.DOTALL)
        
        if match:
            reasoning = match.group(1).strip()
            status = match.group(2).upper()
            confidence = match.group(3).upper()
            success = (status == "SUCCESS")
            parse_method = "full_format"
            logger.info(f"✓ Parsed formatted evaluation result: Status={status}, Confidence={confidence}")
        else:
            # 方法2: 尝试更宽松的格式匹配 - 只要有 Status 字段
            status_pattern = r"Status:\s*(SUCCESS|FAILURE)"
            status_match = re.search(status_pattern, reasoning_text, re.IGNORECASE)
            
            if status_match:
                status = status_match.group(1).upper()
                success = (status == "SUCCESS")
                parse_method = "partial_format"
                logger.info(f"✓ Parsed status from text: {status}")
                
                # 尝试提取 confidence
                confidence_pattern = r"Confidence:\s*(HIGH|MEDIUM|LOW)"
                confidence_match = re.search(confidence_pattern, reasoning_text, re.IGNORECASE)
                if confidence_match:
                    confidence = confidence_match.group(1).upper()
                else:
                    confidence = "MEDIUM"  # 默认中等置信度
                
                # 尝试提取 reasoning
                reasoning_pattern = r"Reasoning:\s*(.+?)(?=\n\n|\Z)"
                reasoning_match = re.search(reasoning_pattern, reasoning_text, re.IGNORECASE | re.DOTALL)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                else:
                    # 如果没有找到 Reasoning 字段,使用整个文本
                    reasoning = reasoning_text
        
        # 如果解析失败,返回失败结果
        if success is None:
            logger.error("Failed to parse evaluation result - no valid format found")
            logger.error(f"Raw output: {reasoning_text[:500]}...")
            return {
                "success": False,
                "score": 0.0,
                "reason": "Failed to parse evaluation result. The evaluator did not provide output in the required format.",
                "eval_steps": eval_steps,
                "confidence": "UNKNOWN",
                "parse_method": "failed",
                "raw_output": reasoning_text
            }
        
        # 计算分数
        score = 1.0 if success else 0.0
        
        # # 根据置信度调整分数(可选)
        # if confidence == "MEDIUM" and success:
        #     score = 0.9
        # elif confidence == "LOW" and success:
        #     score = 0.8
        
        logger.info(f"Evaluation result: success={success}, confidence={confidence}, score={score}, parse_method={parse_method}")
        
        return {
            "success": success,
            "score": score,
            "reason": reasoning,
            "eval_steps": eval_steps,
            "confidence": confidence,
            "parse_method": parse_method,
            "raw_output": reasoning_text
        }

    def _save_eval_log(self, save_dir: str, messages: list, result: dict, task_config: dict):
        """保存评估日志"""
        eval_log_path = os.path.join(save_dir, "cua_eval_log.json")

        # def save_string_to_file(content, file_path):
        #     """
        #     将字符串保存到指定路径的文件中。
            
        #     :param content: 要保存的字符串内容
        #     :param file_path: 目标文件的完整路径（如 'data/test.txt'）
        #     """
        #     try:
        #         # 'w' 表示写入模式（write），会覆盖原文件
        #         # encoding='utf-8' 确保中文字符不会乱码
        #         with open(file_path, 'w', encoding='utf-8') as f:
        #             f.write(content)
        #         print(f"成功保存到: {file_path}")
        #     except Exception as e:
        #         print(f"保存失败，错误原因: {e}")
        
        # 简化消息(移除 base64 图片)
        simple_messages = []
        for msg in messages:
            simple_msg = {"role": msg["role"]}
            if isinstance(msg.get("content"), list):
                simple_content = []
                for item in msg["content"]:
                    if item.get("type") == "image":
                        simple_content.append({"type": "image", "note": "[image_data_removed]"})
                    elif item.get("type") == "tool_result":
                        # 保留工具结果但移除图片数据
                        simple_item = {
                            "type": "tool_result",
                            "tool_use_id": item.get("tool_use_id"),
                            "content": item.get("content", "")[:500]
                        }
                        simple_content.append(simple_item)
                    else:
                        simple_content.append(item)
                simple_msg["content"] = simple_content
            else:
                simple_msg["content"] = msg.get("content")
            simple_messages.append(simple_msg)
        
        log_data = {
            "task_instruction": task_config.get("instruction", "N/A"),
            "task_id": task_config.get("id", "N/A"),
            "evaluation_result": result,
            "conversation_history": simple_messages,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name
        }
        
        with open(eval_log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        # # ⭐ 保存简化的 result_messages (图片替换为路径)
        # result_messages_file = save_dir +  "/result_messages.json"
        # print("===result_messages_file: ",result_messages_file)
        # simplified_messages = simplify_messages_for_json(all_eval_messages, "agent_images")
        
        # with open(result_messages_file, "w", encoding="utf-8") as f:
        #     json.dump(simplified_messages, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation log saved to: {eval_log_path}")
        
        # 同时保存一个可读的文本摘要
        summary_path = os.path.join(save_dir, "cua_eval_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("CUA Evaluation Summary\n")
            f.write("="*80 + "\n\n")
            f.write(f"Task ID: {task_config.get('id', 'N/A')}\n")
            f.write(f"Task: {task_config.get('instruction', 'N/A')}\n\n")
            f.write(f"Result: {'SUCCESS' if result['success'] else 'FAILURE'}\n")
            f.write(f"Score: {result['score']}\n")
            f.write(f"Confidence: {result.get('confidence', 'UNKNOWN')}\n")
            f.write(f"Parse Method: {result.get('parse_method', 'unknown')}\n")
            f.write(f"Evaluation Steps: {result['eval_steps']}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            f.write("Reasoning:\n")
            f.write("-"*80 + "\n")
            f.write(result['reason'])
            f.write("\n" + "="*80 + "\n")
            
            # 如果有原始输出,也保存
            if result.get('raw_output') and result['raw_output'] != result['reason']:
                f.write("\nRaw Output:\n")
                f.write("-"*80 + "\n")
                f.write(result['raw_output'])
                f.write("\n" + "="*80 + "\n")
        
        logger.info(f"Evaluation summary saved to: {summary_path}")

    def _format_eval_messages_for_display(self, messages: list) -> list:
        """
        将评估消息格式化为易读版本(移除 base64 图片数据)
        
        Args:
            messages: 原始消息列表
            
        Returns:
            list: 格式化后的消息列表
        """
        display_messages = []
        
        for msg_idx, msg in enumerate(messages):
            display_msg = {"role": msg.get("role", "unknown")}
            content = msg.get("content")
            
            if isinstance(content, str):
                # 简单文本内容
                display_msg["content"] = content
            elif isinstance(content, list):
                # 列表内容,需要处理每个元素
                display_content = []
                
                for item_idx, item in enumerate(content):
                    if not isinstance(item, dict):
                        display_content.append(item)
                        continue
                    
                    item_type = item.get("type")
                    
                    if item_type == "text":
                        # 文本块:保留完整内容
                        display_content.append({
                            "type": "text",
                            "text": item.get("text", "")[:500]  # 限制长度
                        })
                        
                    elif item_type == "image":
                        # 图片块:替换为占位符
                        source = item.get("source", {})
                        if source.get("type") == "base64":
                            display_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": source.get("media_type", "image/png"),
                                    "data": f"<BASE64_IMAGE_DATA_{msg_idx}_{item_idx}>"
                                }
                            })
                        else:
                            display_content.append({
                                "type": "image",
                                "note": f"<IMAGE_{msg_idx}_{item_idx}>"
                            })
                            
                    elif item_type == "tool_result":
                        # 工具结果:简化内容
                        tool_content = item.get("content", "")
                        if isinstance(tool_content, str):
                            simplified_content = tool_content[:200] + "..." if len(tool_content) > 200 else tool_content
                        else:
                            simplified_content = str(tool_content)[:200]
                        
                        display_content.append({
                            "type": "tool_result",
                            "tool_use_id": item.get("tool_use_id", "unknown"),
                            "content": simplified_content
                        })
                        
                    elif item_type == "tool_use":
                        # 工具调用:保留完整信息
                        display_content.append({
                            "type": "tool_use",
                            "id": item.get("id", "unknown"),
                            "name": item.get("name", "unknown"),
                            "input": item.get("input", {})
                        })
                        
                    else:
                        # 其他类型:保留原样但限制长度
                        display_item = dict(item)
                        if "data" in display_item and isinstance(display_item["data"], str) and len(display_item["data"]) > 100:
                            display_item["data"] = f"<DATA_{len(display_item['data'])}_BYTES>"
                        display_content.append(display_item)
                
                display_msg["content"] = display_content
            else:
                display_msg["content"] = str(content)[:200]
            
            display_messages.append(display_msg)
        
        return display_messages








if __name__ == "__main__":
    import argparse
    import sys
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def load_trajectory_from_folder(trajectory_dir: str) -> dict:
        """
        从文件夹加载轨迹数据
        
        Args:
            trajectory_dir: 轨迹文件夹路径(包含 traj.jsonl 和截图文件)
            
        Returns:
            dict: {"screenshots": [bytes], "actions": [dict]}
        """
        if not os.path.exists(trajectory_dir):
            raise FileNotFoundError(f"Directory not found: {trajectory_dir}")
        
        traj_file = os.path.join(trajectory_dir, "traj.jsonl")
        if not os.path.exists(traj_file):
            raise FileNotFoundError(f"traj.jsonl not found in: {trajectory_dir}")
        
        trajectory = {
            "screenshots": [],
            "actions": []
        }
        
        logger.info(f"Loading trajectory from: {trajectory_dir}")
        
        with open(traj_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    step_data = json.loads(line)
                    
                    # 读取截图
                    screenshot_file = step_data.get("screenshot_file")
                    if screenshot_file:
                        screenshot_path = os.path.join(trajectory_dir, screenshot_file)
                        if os.path.exists(screenshot_path):
                            with open(screenshot_path, 'rb') as img_f:
                                trajectory["screenshots"].append(img_f.read())
                            logger.info(f"  ✓ Loaded screenshot: {screenshot_file}")
                        else:
                            logger.warning(f"  ✗ Screenshot not found: {screenshot_path}")
                    
                    # 保存动作信息
                    action_info = {
                        "step_num": step_data.get("step_num"),
                        "action_timestamp": step_data.get("action_timestamp"),
                        "action": step_data.get("action", {}),
                        "response": step_data.get("response", ""),
                        "command": step_data.get("action", {}).get("command", ""),
                        "raw_response": step_data.get("action", {}).get("raw_response", ""),
                        "screenshot_file": screenshot_file
                    }
                    trajectory["actions"].append(action_info)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                except Exception as e:
                    logger.error(f"Line {line_num}: Error - {e}")
                    continue
        
        logger.info(f"✓ Loaded {len(trajectory['screenshots'])} screenshots and {len(trajectory['actions'])} actions")
        
        if len(trajectory["screenshots"]) == 0:
            raise ValueError("No valid screenshots found in trajectory")
        
        return trajectory
    
    def eval_task_from_local(agent, trajectory_dir: str, instruction: str, 
                            task_id: str = None, save_dir: str = None, 
                            max_round: int = 20) -> dict:
        """
        从本地轨迹文件夹进行评估(调试模式)
        
        Args:
            agent: AnthropicAgent45Eval 实例
            trajectory_dir: 轨迹文件夹路径
            instruction: 任务指令
            task_id: 任务ID(可选)
            save_dir: 保存目录(可选)
            max_round: 最大评估轮数
            
        Returns:
            dict: 评估结果
        """
        logger.info("="*80)
        logger.info("LOCAL DEBUG MODE: Evaluating from folder")
        logger.info(f"Trajectory directory: {trajectory_dir}")
        logger.info("="*80)
        
        # 设置默认值
        if task_id is None:
            task_id = os.path.basename(trajectory_dir.rstrip('/'))
        
        if save_dir is None:
            save_dir = os.path.join(trajectory_dir, "eval_results")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 加载轨迹
        try:
            trajectory = load_trajectory_from_folder(trajectory_dir)
            
            # 打印轨迹摘要
            logger.info("\nTrajectory Summary:")
            logger.info(f"  Total steps: {len(trajectory['actions'])}")
            if trajectory["actions"]:
                first_action = trajectory["actions"][0]
                last_action = trajectory["actions"][-1]
                logger.info(f"  First step: {first_action.get('action_timestamp', 'N/A')}")
                logger.info(f"  Last step: {last_action.get('action_timestamp', 'N/A')}")
                logger.info(f"  First response: {first_action.get('response', '')[:100]}")
                logger.info(f"  Last response: {last_action.get('response', '')[:100]}")
            
        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}")
            logger.error(traceback.format_exc())
            raise
        
        # 构建任务配置
        task_config = {
            "id": task_id,
            "instruction": instruction,
            "trajectory_dir": trajectory_dir,
            "num_steps": len(trajectory["actions"])
        }
        
        logger.info(f"\nTask Configuration:")
        logger.info(f"  ID: {task_id}")
        logger.info(f"  Instruction: {instruction}")
        logger.info(f"  Save directory: {save_dir}")
        logger.info("="*80)
        
        # 调用 eval_task (env=None 表示本地调试模式,不与环境交互)
        return agent.eval_task(
            env=None,
            task_config=task_config,
            trajectory=trajectory,
            save_dir=save_dir,
            max_round=max_round
        )
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Evaluate task execution from local trajectory")
    parser.add_argument("--trajectory_dir", type=str, required=True,
                       help="Path to trajectory directory (containing traj.jsonl and screenshots)")
    parser.add_argument("--instruction", type=str, required=True,
                       help="Task instruction to evaluate")
    parser.add_argument("--task_id", type=str, default=None,
                       help="Task ID (default: use folder name)")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save evaluation results (default: trajectory_dir/eval_results)")
    parser.add_argument("--max_round", type=int, default=20,
                       help="Maximum evaluation rounds (default: 20)")
    parser.add_argument("--model", type=str, default="claude-4.5-openai",
                       help="Model name (default: claude-4.5-openai)")
    parser.add_argument("--api_url", type=str, 
                       default=None,
                       help="API URL")
    parser.add_argument("--api_key", type=str,
                       default=None,
                       help="API Key")
    parser.add_argument("--platform", type=str, default="Ubuntu",
                       choices=["Ubuntu", "Windows"],
                       help="Platform (default: Ubuntu)")
    
    args = parser.parse_args()
    
    # 验证必需的参数
    if not os.path.exists(args.trajectory_dir):
        logger.error(f"Trajectory directory not found: {args.trajectory_dir}")
        sys.exit(1)
    
    # 创建 agent
    try:
        agent = AnthropicAgent45Eval(
            model=args.model,
            api_url=args.api_url,
            api_key=args.api_key,
            platform=args.platform,
            provider=APIProvider.ANON,
            max_tokens=4096,
            temperature=0.0,  # 评估时使用确定性输出
        )
        logger.info(f"✓ Agent initialized: {args.model}")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 执行评估
    try:
        result = eval_task_from_local(
            agent=agent,
            trajectory_dir=args.trajectory_dir,
            instruction=args.instruction,
            task_id=args.task_id,
            save_dir=args.save_dir,
            max_round=args.max_round
        )
        
        # 打印结果
        print("\n" + "="*80)
        print("EVALUATION RESULT")
        print("="*80)
        print(f"Success: {result['success']}")
        print(f"Score: {result['score']}")
        print(f"Confidence: {result.get('confidence', 'UNKNOWN')}")
        print(f"Parse Method: {result.get('parse_method', 'unknown')}")
        print(f"Evaluation Steps: {result['eval_steps']}")
        print(f"\nReasoning:")
        print("-"*80)
        print(result['reason'])
        print("="*80)
        
        # 退出码
        sys.exit(0 if result['success'] else 1)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)