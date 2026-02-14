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
from .utils import COMPUTER_USE_BETA_FLAG, PROMPT_CACHING_BETA_FLAG,SYSTEM_PROMPT, SYSTEM_PROMPT_WINDOWS, APIProvider, PROVIDER_TO_DEFAULT_MODEL_NAME, get_model_name
from .utils import _response_to_params, _inject_prompt_caching, _maybe_filter_to_n_most_recent_images, _custom_maybe_filter_to_n_most_recent_images

import logging
logger = logging.getLogger("desktopenv.agent")
agent_logger = logging.getLogger("desktopagent.agent")

# MAX_HISTORY = 10
# API_RETRY_TIMES = 500  
API_RETRY_TIMES = 10
API_RETRY_INTERVAL = 5

ANON_CLAUDE_37 = "anthropic.claude-3.7-sonnet"
ANON_CLAUDE_45_OPENAI = "claude-sonnet-4.5-openai"
ANON_CLAUDE_4_OPENAI = "claude-sonnet-4-openai"

def upload_image_to_s3(image_bytes: bytes, example_id: str, step_num: int) -> str:
    """
    上传图片到 S3 并返回预签名 URL
    
    Args:
        image_bytes: 图片字节数据
        example_id: 任务 ID
        step_num: 步骤编号
    
    Returns:
        预签名 URL
    """
    try:
        # 生成唯一的文件名
        timestamp = int(time.time() * 1000)
        image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        filename = f"{S3_PREFIX}{example_id}/step_{step_num}_{timestamp}_{image_hash}.png"
        
        # 上传到 S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=image_bytes,
            ContentType='image/png',
            # 可选：设置 ACL 为 public-read（如果需要公开访问）
            # ACL='public-read'
        )
        
        # 生成预签名 URL
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET_NAME,
                'Key': filename
            },
            ExpiresIn=S3_EXPIRATION
        )
        
        logger.info(f"Uploaded image to S3: {filename}, URL: {url[:100]}...")
        return url
    
    except ClientError as e:
        logger.error(f"Failed to upload image to S3: {e}")
        raise


def resize_image_to_1024_768(image_bytes: bytes) -> tuple[bytes, float, float]:
    img = Image.open(io.BytesIO(image_bytes))
    original_width, original_height = img.size
    # new_width, new_height = 1024, 768
    new_width, new_height = 1280, 720
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img_resized.save(buffer, format='PNG')
    resized_bytes = buffer.getvalue()
    x_ratio = original_width / new_width
    y_ratio = original_height / new_height
    return resized_bytes, x_ratio, y_ratio


class AnthropicAgent45:
    def __init__(self,
                 platform: str = "Ubuntu",
                 model: str = "claude-3-5-sonnet-20241022",
                 provider: APIProvider = APIProvider.ANON,
                 max_tokens: int = 4096,
                 api_url: str = "",
                 api_key: str = "",
                 system_prompt_suffix: str = "",
                 only_n_most_recent_images: Optional[int] = 10,
                 action_space: str = "claude_computer_use",
                 screen_size: tuple[int, int] = (1920, 1080),
                 no_thinking: bool = False,
                 use_isp: bool = False,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 log_image_base64: bool = False,
                 use_s3_for_images: bool = True,  # ⭐ 新增参数
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

        self.no_thinking = no_thinking
        self.use_isp = use_isp
        self.temperature = temperature
        self.top_p = top_p

        # ⭐ 新增：是否使用 S3 存储图片
        self.use_s3_for_images = use_s3_for_images
        self.current_example_id = None  # 用于生成 S3 文件名
        

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
        
        # ⭐⭐⭐ 修改：使用 S3 URL 而不是 base64 ⭐⭐⭐
        if screenshot is not None:
            if self.use_s3_for_images and self.current_example_id:
                # 上传到 S3 并获取 URL
                step_num = len([m for m in self.messages if m.get("role") == "user"])
                screenshot_url = upload_image_to_s3(screenshot, self.current_example_id, step_num)
                
                tool_result_content.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": screenshot_url
                    }
                })
            else:
                # 回退到 base64（如果 S3 不可用）
                screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
                tool_result_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png", 
                        "data": screenshot_base64
                    }
                })
        
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
        
        # resize coordinates if resize_factor is set
        # if coordinate and self.resize_factor:
        #     coordinate = (
        #         int(coordinate[0] * self.resize_factor[0]),
        #         int(coordinate[1] * self.resize_factor[1])
        #     )
        # if coordinate and screenshot_resize_ratio:
        #     coordinate = (
        #         int(coordinate[0] / screenshot_resize_ratio),
        #         int(coordinate[1] / screenshot_resize_ratio)
        #     )
        if coordinate and (screenshot_resize_ratio_x and screenshot_resize_ratio_y):
            # 坐标从模型输入(1024,768)映射回原始分辨率
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
                        'apt install -o Acquire::http::Proxy="http://archive.ubuntu.com"'
                    ).replace(
                        "apt-get install",
                        'apt-get install -o Acquire::http::Proxy="http://archive.ubuntu.com"'
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

            # elif action == "type":
            #     # result += (
            #     #     f"pyautogui.typewrite(\"\"\"{text}\"\"\", interval=0.01)\n"
            #     # )
            #     for char in text:
            #         if char == '\n':
            #             result += "pyautogui.press('enter')\n"
            #         elif char == "'":
            #             result += 'pyautogui.press("\'")\n'
            #         elif char == '\\':
            #             result += "pyautogui.press('\\\\')\n"
            #         elif char == '"':
            #             result += "pyautogui.press('\"')\n"
            #         else:
            #             result += f"pyautogui.press('{char}')\n"
            #     expected_outcome = f"Text {text} written."

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
            text=f"{SYSTEM_PROMPT_WINDOWS if self.platform == 'Windows' else SYSTEM_PROMPT}{' ' + self.system_prompt_suffix if self.system_prompt_suffix else ''}"
        )
        # ⭐ 设置当前任务 ID（用于 S3 文件名）
        if obs and "example_id" in obs:
            self.current_example_id = obs["example_id"]
        
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
            # ratio_x, ratio_y = x_ratio, y_ratio
            image_meta = {
                "original_size": [original_width, original_height],
                "input_size": [new_width, new_height],
                "resize_ratio_x": x_ratio,
                "resize_ratio_y": y_ratio
            }
            logger.info(f"  original=({original_width}, {original_height}), new=({new_width}, {new_height}), resize_ratio_x={x_ratio}, resize_ratio_y={y_ratio}")
            

        if not self.messages:
            
            init_screenshot = obs
            # init_screenshot_base64 = base64.b64encode(init_screenshot["screenshot"]).decode('utf-8')
            # ⭐⭐⭐ 修改：使用 S3 URL 而不是 base64 ⭐⭐⭐
            if self.use_s3_for_images and self.current_example_id:
                screenshot_url = upload_image_to_s3(init_screenshot["screenshot"], self.current_example_id, 0)
                image_content = {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": screenshot_url
                    }
                }
            else:
                init_screenshot_base64 = base64.b64encode(init_screenshot["screenshot"]).decode('utf-8')
                image_content = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": init_screenshot_base64,
                    }
                }
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
            
        #增加
        # if self.messages and "tool_use" in [content_block["type"] for content_block in self.messages[-1]["content"]]:
        #     self.add_tool_result(
        #         self.messages[-1]["content"][-1]["id"],
        #         f"Success",
        #         screenshot=obs.get("screenshot") if obs else None,
        #         screenshot_name=obs.get("screenshot_name", "screenshot_name")
        #     )
        # Add tool_result for ALL tool_use blocks in the last message
        if self.messages:
            last_message_content = self.messages[-1]["content"]
            tool_use_blocks = [block for block in last_message_content if block.get("type") == "tool_use"]

            for i, tool_block in enumerate(tool_use_blocks):
                tool_input = tool_block.get("input", {})
                action = tool_input.get("action")
                is_last_tool = i == len(tool_use_blocks) - 1

                include_screenshot = None

                if obs:
                    if action == "screenshot":
                        # Screenshot action always gets regular screenshot
                        include_screenshot = obs.get("screenshot")
                    elif is_last_tool:
                        # Auto-screenshot: last tool gets regular screenshot (unless it's zoom, handled above)
                        include_screenshot = obs.get("screenshot")

                self.add_tool_result(
                    tool_block["id"],
                    f"Success",
                    screenshot=include_screenshot
                )
            
            
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
                aws_secret_key=os.getenv('ECRET_ACCESS_KEY'),
                aws_region=os.getenv('DEFAULT_REGION'),
            )
        elif self.provider == APIProvider.ANON:
            pass

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            _inject_prompt_caching(self.messages)
            image_truncation_threshold = 20
            system["cache_control"] = {"type": "ephemeral"}

        # if self.only_n_most_recent_images:
        #     _maybe_filter_to_n_most_recent_images(
        #         self.messages,
        #         self.only_n_most_recent_images,
        #         min_removal_threshold=image_truncation_threshold,
        #     )

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
        

        max_parse_retry = 3
        for parse_retry in range(max_parse_retry):
            actions: list[Any] = []
            reasonings: list[str] = []
            try:
                for content_block in response_params:
                    if content_block["type"] == "tool_use":
                        actions.append({
                            "name": content_block["name"],
                            "input": cast(dict[str, Any], content_block["input"]),
                            "id": content_block["id"],
                            "action_type": content_block.get("type"),
                            # "command": self.parse_actions_from_tool_call(content_block, obs.get("screenshot_resize_ratio", 1.0))
                            "command": self.parse_actions_from_tool_call(
                                content_block,
                                obs.get("screenshot_resize_ratio_x", 1.0),
                                obs.get("screenshot_resize_ratio_y", 1.0)
                            ),
                            # "command": self.parse_actions_from_tool_call(content_block),
                            "raw_response": raw_response_str  # Add raw response to each action
                        })
                    elif content_block["type"] == "text":
                        reasonings.append(content_block["text"])
                if isinstance(reasonings, list) and len(reasonings) > 0:
                    reasonings = reasonings[0]
                else:
                    reasonings = ""
                # Check if the model indicated the task is infeasible
                if raw_response_str and "[INFEASIBLE]" in raw_response_str:
                    logger.info("Detected [INFEASIBLE] pattern in response, triggering FAIL action")
                    # Override actions with FAIL
                    actions = [{
                        "action_type": "FAIL",
                        "raw_response": raw_response_str
                    }]

                info_dict = {
                    "messages_export": simple_messages_for_export,
                    "low_level_action": reasonings,
                    "pyautogui_actions": [a.get("command", "") for a in actions],
                    "response_text": json.dumps(response_params, ensure_ascii=False),
                    "image_meta": image_meta
                }

                logger.info(f"Received actions: {actions}")
                logger.info(f"Received reasonings: {reasonings}")
                if len(actions) == 0:
                    # actions = ["DONE"]
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
                    if isinstance(reasonings, list) and len(reasonings) > 0:
                        reasonings = reasonings[0]
                    else:
                        reasonings = ""
                    return reasonings, actions, info_dict
    def reset(self, _logger = None, *args, **kwargs):
        """
        Reset the agent's state.
        """
        global agent_logger
        if _logger:
            agent_logger = _logger
        # else:
        #     logger = logging.getLogger("desktopenv.agent")
        self.messages = []
        self.image_names = []
        self.turn_count = 0
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
                        # ⭐⭐⭐ 支持 S3 URL ⭐⭐⭐
                        if source['type'] == 'url':
                            openai_content_parts.append({
                                'type': 'image_url',
                                'image_url': {
                                    'url': source['url'],
                                    "detail": "high"
                                }
                            })
                        elif source['type'] == 'base64':
                            openai_content_parts.append({
                                'type': 'image_url',
                                'image_url': {
                                    'url': f"data:{source['media_type']};base64,{source['data']}",
                                    "detail": "high"
                                }
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

    # def _call_ANON_model_openai(self, system, tools, betas, extra_body, example_id):
    #     user_count = sum(1 for message in self.messages if message.get("role") == "user")
    #     agent_logger.info(f"request ANON model (openai format) {self.model_name}, 第{user_count-1}轮")

    #     timestamp_ms = int(time.time() * 1000)
    #     trace_id = f"{example_id}_{user_count-1}_{timestamp_ms}"

    #     headers = {
    #         "Content-Type": "application/json",
    #         "Authorization": f"Bearer {self.api_key}", 
    #         "M-TraceId": trace_id
    #     }

    #     openai_messages = self._convert_messages_to_openai_format(self.messages, system["text"])
        
    #     # for logging without images
    #     simple_messages_for_log, _ = self.rewrite_model_message_without_image()
    #     agent_logger.info(f"request ANON model (openai format) {self.model_name}, 第{user_count-1}轮, message:\n {json.dumps(simple_messages_for_log, indent=2, ensure_ascii=False)}")
        
    #     payload = {
    #         "model": self.model_name.replace("-openai", ""),
    #         "max_tokens": self.max_tokens,
    #         "messages": openai_messages,
    #         "tools": tools,
    #         "anthropic_beta": betas,
    #     }
        
    #     logger.info(f"request ANON model (openai format) {self.model_name}, tools: {tools}, request M-TraceId={trace_id}")
    #     http_response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
    #     logger.info(f"request ANON model (openai format) {self.model_name}, request M-TraceId={trace_id}, Response status_code: {http_response.status_code}, reason: {http_response.reason}")
        
    #     if http_response.status_code == 429:
    #         # raise APIStatusError(http_response.reason)
    #         logger.error(f"Error response from API: {http_response.text}")
    #         resp_json = http_response.json()
    #         raise APIStatusError(
    #             http_response.reason,
    #             response=http_response,
    #             body=resp_json
    #         )

    #     if http_response.status_code != 200:
    #         logger.error(f"Error response from API: {http_response.text}")
    #         raise Exception(f"{http_response.status_code} {http_response.reason}")
        
    #     response_json = http_response.json()
    #     logger.info(f"request ANON model (openai format) response: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
    #     return self._convert_response_from_openai_to_anthropic(response_json)
    import requests
    import time

    def _call_ANON_model_openai(self, system, tools, betas, extra_body, example_id, max_retries=20, retry_interval=5):
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
        