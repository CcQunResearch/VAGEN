import os
import base64
from openai import OpenAI

# 1. 配置环境信息 (优先读取环境变量，否则使用默认值)
API_TYPE = "openai"
API_MODEL = "gemini-3-flash-preview" 
API_BASE_URL = "ANON"
API_KEY = "ANON"

# 初始化客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

def encode_image(image_source):
    """
    辅助函数：处理图片。
    如果是本地路径，转换为Base64；如果是URL，直接返回。
    """
    if image_source.startswith("http"):
        return image_source
    
    try:
        with open(image_source, "rb") as image_file:
            return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
    except Exception as e:
        print(f"图片读取失败: {e}")
        return None

def chat_with_model(messages, user_text, image_path=None):
    """
    调用模型的函数。
    
    Args:
        messages (list): 之前的对话历史列表 (也就是上下文)。
        user_text (str): 当前轮用户的文本输入。
        image_path (str, optional): 图片路径 (本地路径或URL)。默认为 None。
        
    Returns:
        str: 模型的回复内容。
        list: 更新后的对话历史。
    """
    
    # 构造当前轮的用户消息内容
    current_content = []
    
    # 1. 添加文本部分
    if user_text:
        current_content.append({
            "type": "text",
            "text": user_text
        })
    
    # 2. 添加图片部分 (如果有)
    if image_path:
        image_url = encode_image(image_path)
        if image_url:
            current_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    # detail 可选 "low", "high", 或 "auto"
                    "detail": "auto" 
                }
            })

    # 将当前构造好的消息加入历史记录
    messages.append({
        "role": "user",
        "content": current_content
    })

    try:
        # 发起 API 调用
        response = client.chat.completions.create(
            model=API_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=16384
        )

        # 获取模型回复
        assistant_reply = response.choices[0].message.content
        
        # 将模型回复也加入历史记录，形成闭环
        messages.append({
            "role": "assistant",
            "content": assistant_reply
        })
        
        return assistant_reply, messages

    except Exception as e:
        print(f"API 调用出错: {e}")
        # 如果出错，为了不影响下一轮，建议把刚才添加的 user message 移除，或者根据业务逻辑处理
        return None, messages