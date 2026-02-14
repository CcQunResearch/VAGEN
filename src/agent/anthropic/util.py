import os
# from arm_agent.agent_verl import VerlAgent
# from arm_agent.utils import encode_image_file
from dotenv import load_dotenv
import json
import time
import base64
import traceback
from pathlib import Path
from PIL import Image
import argparse
import re
import copy
from .api import chat_with_model


traj_summary_prompt_en = """<Instruction>
The reasoning content of a GUI Agent at every step of a task execution trajectory usually contains the following three parts (which may not exist simultaneously):
- State Observation: Describes the current screen environment state and the feedback from the previous operation (e.g., what is displayed on the screen, a certain window has been opened);
- Sub-goal Analysis: The Agent's content regarding plans, intentions, task decomposition, or self-correction for the current step or future operations (e.g., subjective reasoning like "The current goal is to enter the official website of PyCharm and then download its latest version");
- Action Description: A description of the atomic operation to be executed in the current step (e.g., "I need to click the save button to save the modified file").

You will be provided with the output content of every step of a GUI Agent's task execution trajectory. Please summarize the Agent's operation for each step, with the following requirements:
1. Summarize step by step; summarize the operation of each step into one sentence (in English), do not miss any step.
2. Only summarize contents related to "State Observation" and "Action Description", discarding contents related to "Sub-goal Analysis"; please refer to the example provided below for details.
3. Output according to the format specified in the example below; only output the summary, do not output any other irrelevant content.

<Example>
Model Output:
Step 3: [TEXT] Good! I can see your desktop with a notification about software updates. I'll help you install Spotify. The easiest way to install Spotify on Ubuntu is through Snap, which is already available on your system. Let me open a terminal and install it for you.
[TOOL_USE] computer: {{'action': 'key', 'text': 'ctrl+alt+t'}}

Summary:
Step 3: There is a software update notification on the desktop. The agent opened a terminal using the "ctrl+alt+t" hotkey.

<Agent Trajectory>
{}

Now, please complete the step-by-step summary of this GUI Agent trajectory based on the preceding information."""

# 4. In each intermediate step of your evaluation, you must provide a analysis or reasoning text before calling tools. In the final evaluation step, based on your analysis, determine if the task was completed successfully and provide your final judgment in the specified format. You MUST be extremely cautious and strict when judging a task as successfully executed.
def _build_eval_prompt(instruction: str, task_config: dict, trajectory: dict, num_screenshots: int) -> str:
    # """构建评估提示"""
    # actions_text = "\n".join([
    #     f"Step {i+1}: {str(action.get('raw_response', action))}"
    #     for i, action in enumerate(trajectory.get("actions", []))
    # ])
    # # 5. `read_pptx`: Read PowerPoint file details, include - Check slide count, Verify text content, Inspect font properties, Examine table data

    # summary_prompt = traj_summary_prompt_en.format(actions_text)

    # retry_time = 10
    # retry_internal = 8
    # for i in range(retry_time):
    #     actions_summary, chat_history = chat_with_model([{"role": "system", "content": "You are a helpful assistant."}], summary_prompt)

    #     if actions_summary:
    #         break
    #     else:
    #         time.sleep(retry_internal)

    # actions_summary = actions_summary.replace("Summary:", "").strip()

    actions_summary = "\n".join([
        f"Step {i+1}: {str(action.get('raw_response', action))}"
        for i, action in enumerate(trajectory.get("actions", []))
    ])
    
    prompt = f"""You are an expert evaluator for GUI automation tasks. Your job is to determine if the given task was successfully completed based on the execution trajectory.

**Available Tools:**
1. `check_screenshot`: View specific screenshot of one step from the trajectory (e.g., step_1, step_7, etc). Use this to examine key moments in the execution.
2. `computer`: Interact with the environment by GUI operations to verify the current state (if needed).
3. `execute_python`: Interact with the environment by python code to verify the current state (if needed).
4. `execute_shell`: Interact with the environment by bash code to verify the current state (if needed).
    
**Your Evaluation Process:**
1. First, you will be provided with the Task Instruction, Execution Trajectory, and the Last Screenshot of the last step. Please begin your evaluation process based on this information.
2. You can devise your own verification strategy. One suggested strategy is as follows: First, you can check whether there are any obvious errors or if the task can be directly judged as successful by reviewing the Execution Trajectory and the screenshot from the last step. Next, you may use the `check_screenshot` tool to examine the screenshot of a specific intermediate step to further verify the process. If you find that relying solely on the action contents and screenshots is insufficient to determine whether the task was completed, and you believe it is necessary to directly interact with the GUI environment for verification, you can use the `computer` tool to interact with the computer after `check_screenshot`.
3. When using the `computer` tool, you can directly check the system status, such as viewing local folders or querying system settings through the terminal to verify whether the system has reached the expected state.
4. Based on your analysis, determine if the task was completed successfully and provide your final judgment in the specified format. You MUST be extremely cautious and strict when judging a task as successfully executed.

**Judgment Criteria:**
- Was the task objective fully achieved?
- Are there any errors or incomplete steps?
- Does the final state match the expected outcome?

**IMPORTANT: Final Judgment Format**
When you have completed your evaluation, you MUST provide your final judgment in the following exact format:

EVALUATION RESULT:
Reasoning: Your detailed reasoning explaining why the task succeeded or failed
Status: SUCCESS or FAILURE
Confidence: HIGH or MEDIUM or LOW

Example of correct format:
EVALUATION RESULT:
Reasoning: The task was completed successfully. All required steps were executed correctly, and the final state matches the expected outcome.
Status: SUCCESS
Confidence: HIGH

**IMPORTANT: Tool Usage**
⚠️ **You MUST use the actual tool calling mechanism provided by the API.**
⚠️ **DO NOT write tool calls as text like "[Tool Use - tool_name]" or similar.**
⚠️ **Use the proper function calling format that the system understands.**

Please begin your evaluation by examining the key screenshots.

**Task Instruction:**
{instruction}

**Execution Trajectory:**
Total steps: {num_screenshots}
Actions taken:
{actions_summary}

**Last Screenshot (step {num_screenshots}):**"""
    
    return prompt


# def _build_eval_prompt(instruction: str, task_config: dict, trajectory: dict, num_screenshots: int) -> str:
#     """构建评估提示"""
#     # actions_text = "\n".join([
#     #     f"Step {i+1}: {str(action.get('raw_response', action))}"
#     #     for i, action in enumerate(trajectory.get("actions", []))
#     # ])
#     # # 5. `read_pptx`: Read PowerPoint file details, include - Check slide count, Verify text content, Inspect font properties, Examine table data

#     # summary_prompt = traj_summary_prompt_en.format(actions_text)

#     # actions_summary, chat_history = chat_with_model([{"role": "system", "content": "You are a helpful assistant."}], summary_prompt)
#     # actions_summary = actions_summary.replace("Summary:", "").strip()


#     actions_summary = "\n".join([
#         f"Step {i+1}: {str(action.get('raw_response', action))}"
#         for i, action in enumerate(trajectory.get("actions", []))
#     ])
    
#     prompt = f"""You are an expert evaluator for GUI automation tasks. Your job is to determine if the given task was successfully completed based on the execution trajectory.

# **Available Tools:**
# 1. `check_screenshot`: View specific screenshots from the trajectory (e.g., step_1, step_7,etc). Use this to examine key moments in the execution.
# 2. `computer`: Interact with the environment by GUI operations to verify the current state (if needed).
# 3. `execute_python`: Interact with the environment by python code to verify the current state (if needed).
# 4. `execute_shell`: Interact with the environment by bash code to verify the current state (if needed).
    
# **Your Evaluation Process:**
# 1. First, you will be provided with the Task Instruction, Execution Trajectory, and the Last Screenshots of the last five steps. Please begin your evaluation process based on this information.
# 2. You can devise your own verification strategy. One suggested strategy is as follows: First, you can check whether there are any obvious errors or if the task can be directly judged as successful by reviewing the Execution Trajectory and the screenshot from the last steps. Next, you may use the `check_screenshot` tool to examine the screenshot of a specific intermediate step to further verify the process. If you find that relying solely on the action contents and screenshots is insufficient to determine whether the task was completed, and you believe it is necessary to directly interact with the GUI environment for verification, you can use the `computer` tool to interact with the computer after `check_screenshot`.
# 3. When using the `computer` tool, you can directly check the system status, such as viewing local folders or querying system settings through the terminal to verify whether the system has reached the expected state.
# 4. Based on your analysis, determine if the task was completed successfully and provide your final judgment in the specified format. You MUST be extremely cautious and strict when judging a task as successfully executed.

# **Judgment Criteria:**
# - Was the task objective fully achieved?
# - Are there any errors or incomplete steps?
# - Does the final state match the expected outcome?

# **IMPORTANT: Final Judgment Format**
# When you have completed your evaluation, you MUST provide your final judgment in the following exact format:

# EVALUATION RESULT:
# Reasoning: Your detailed reasoning explaining why the task succeeded or failed
# Status: SUCCESS or FAILURE
# Confidence: HIGH or MEDIUM or LOW

# Example of correct format:
# EVALUATION RESULT:
# Reasoning: The task was completed successfully. All required steps were executed correctly, and the final state matches the expected outcome.
# Status: SUCCESS
# Confidence: HIGH

# **IMPORTANT: Tool Usage**
# ⚠️ **You MUST use the actual tool calling mechanism provided by the API.**
# ⚠️ **DO NOT write tool calls as text like "[Tool Use - tool_name]" or similar.**
# ⚠️ **Use the proper function calling format that the system understands.**

# Please begin your evaluation by examining the key screenshots.

# **Task Instruction:**
# {instruction}

# **Execution Trajectory:**
# Total steps: {num_screenshots}
# Actions taken:
# {actions_summary}

# **Last Screenshots (step {max(num_screenshots-3, 1)}-{num_screenshots}):**"""
    
#     return prompt


# def _build_eval_prompt(instruction: str, task_config: dict, trajectory: dict, num_screenshots: int) -> str:
#     """构建评估提示"""
#     # actions_text = "\n".join([
#     #     f"Step {i+1}: {str(action.get('raw_response', action))}"
#     #     for i, action in enumerate(trajectory.get("actions", []))
#     # ])
#     # # 5. `read_pptx`: Read PowerPoint file details, include - Check slide count, Verify text content, Inspect font properties, Examine table data

#     # summary_prompt = traj_summary_prompt_en.format(actions_text)

#     # actions_summary, chat_history = chat_with_model([{"role": "system", "content": "You are a helpful assistant."}], summary_prompt)
#     # actions_summary = actions_summary.replace("Summary:", "").strip()

#     actions_summary = "\n".join([
#         f"Step {i+1}: {str(action.get('raw_response', action))}"
#         for i, action in enumerate(trajectory.get("actions", []))
#     ])
    
#     prompt = f"""You are an expert evaluator for GUI automation tasks. Your job is to determine if the given task was successfully completed based on the execution trajectory.

# **Available Tools:**
# 1. `check_screenshot`: View specific screenshot of one step from the trajectory (e.g., step_1, step_7, etc). Use this to examine key moments in the execution.
# 2. `computer`: Interact with the environment by GUI operations to verify the current state (if needed).
# 3. `execute_python`: Interact with the environment by python code to verify the current state (if needed).
# 4. `execute_shell`: Interact with the environment by bash code to verify the current state (if needed).
    
# **Your Evaluation Process:**
# 1. First, you will be provided with the Task Instruction, Execution Trajectory, and the Last screenshot in the last step. Please begin your evaluation process based on this information.
# 2. You can devise your own verification strategy. One suggested strategy is as follows: First, you can check whether there are any obvious errors or if the task can be directly judged as successful by reviewing the Execution Trajectory and the screenshot from the last step. Next, you may use the `check_screenshot` tool to examine the screenshot of a specific intermediate step to further verify the process. If you find that relying solely on the action contents and screenshots is insufficient to determine whether the task was completed, and you believe it is necessary to directly interact with the GUI environment for verification, you can use the `computer` tool to interact with the computer after `check_screenshot`.
# 3. When using the `computer` tool, you can directly check the system status, such as viewing local folders or querying system settings through the terminal to verify whether the system has reached the expected state.
# 4. Based on your analysis, determine if the task was completed successfully and provide your final judgment in the specified format.

# **Judgment Criteria:**
# - Was the task objective fully achieved?
# - Are there any errors or incomplete steps?
# - Does the final state match the expected outcome?

# **IMPORTANT: Final Judgment Format**
# When you have completed your evaluation, you MUST provide your final judgment in the following exact format:

# EVALUATION RESULT:
# Reasoning: Your detailed reasoning explaining why the task succeeded or failed
# Status: SUCCESS or FAILURE
# Confidence: HIGH or MEDIUM or LOW

# Example of correct format:
# EVALUATION RESULT:
# Reasoning: The task was completed successfully. All required steps were executed correctly, and the final state matches the expected outcome.
# Status: SUCCESS
# Confidence: HIGH

# **IMPORTANT: Tool Usage**
# ⚠️ **You MUST use the actual tool calling mechanism provided by the API.**
# ⚠️ **DO NOT write tool calls as text like "[Tool Use - tool_name]" or similar.**
# ⚠️ **Use the proper function calling format that the system understands.**

# Please begin your evaluation by examining the key screenshots.

# **Task Instruction:**
# {instruction}

# **Execution Trajectory:**
# Total steps: {num_screenshots}
# Actions taken:
# {actions_summary}

# **Last screenshot:**"""
    
#     return prompt

def _build_eval_prompt_static(instruction: str, trajectory: dict, num_screenshots: int) -> str:
    """构建评估提示"""
    actions_summary = "\n".join([
        f"Step {i+1}: {str(action.get('raw_response', action))}"
        for i, action in enumerate(trajectory.get("actions", []))
    ])
    # 5. `read_pptx`: Read PowerPoint file details, include - Check slide count, Verify text content, Inspect font properties, Examine table data
    
    prompt = f"""You are an expert evaluator for GUI automation tasks. Your job is to determine if the given task was successfully completed based on the execution trajectory.

    **Available Tools:**
    1. `check_screenshot`: View specific screenshots from the trajectory (e.g., step_1, step_7,etc). Use this to examine key moments in the execution.
        

    **Your Evaluation Process:**
    1. First, you will be provided with the Task Instruction, Execution Trajectory, and the Last screenshot in the last step. Please begin your evaluation process based on this information.
    2. You can devise your own verification strategy. One suggested strategy is as follows: First, you can check whether there are any obvious errors or if the task can be directly judged as successful by reviewing the Execution Trajectory and the screenshot from the last step. Next, you may use the `check_screenshot` tool to examine the screenshot of a specific intermediate step to further verify the process. If you find that relying solely on the action contents and screenshots is insufficient to determine whether the task was completed, and you believe it is necessary to directly interact with the GUI environment for verification, you can use the `computer` tool to interact with the computer after `check_screenshot`.
    3. When using the `computer` tool, you can directly check the system status, such as viewing local folders or querying system settings through the terminal to verify whether the system has reached the expected state.
    4. Based on your analysis, determine if the task was completed successfully
    5. Provide your final judgment in the specified format

    **Judgment Criteria:**
    - Was the task objective fully achieved?
    - Are there any errors or incomplete steps?
    - Does the final state match the expected outcome?

    **IMPORTANT: Final Judgment Format**
    When you have completed your evaluation, you MUST provide your final judgment in the following exact format:

    EVALUATION RESULT:
    Status: SUCCESS or FAILURE
    Confidence: HIGH or MEDIUM or LOW
    Reasoning: Your detailed reasoning explaining why the task succeeded or failed

    Example of correct format:
    EVALUATION RESULT:
    Status: SUCCESS
    Confidence: HIGH
    Reasoning: The task was completed successfully. All required steps were executed correctly, and the final state matches the expected outcome.

    **IMPORTANT: Tool Usage**
    ⚠️ **You MUST use the actual tool calling mechanism provided by the API.**
    ⚠️ **DO NOT write tool calls as text like "[Tool Use - tool_name]" or similar.**
    ⚠️ **Use the proper function calling format that the system understands.**

    Please begin your evaluation by examining the key screenshots.

    **Task Instruction:**
    {instruction}

    **Execution Trajectory:**
    Total steps: {num_screenshots}
    Actions taken:
    {actions_summary}

    **Last screenshot:**
    """
    
    return prompt

def _build_eval_prompt_static_zh(instruction: str, trajectory: dict, num_screenshots: int) -> str:
    """构建评估提示(中文版)"""
    actions_summary = "\n".join([
        f"步骤 {i+1}: {str(action.get('raw_response', action))}"
        for i, action in enumerate(trajectory.get("actions", []))
    ])
    
    prompt = f"""你是一位 GUI 自动化任务的专家评估员。你的工作是根据执行轨迹判断给定任务是否成功完成。

    **可用工具:**
    1. `check_screenshot`: 查看轨迹中特定步骤的截图(例如 step_1, step_7 等)。使用此工具检查执行过程中的关键时刻。注意：一次最多查看3张截图

    **评估流程:**
    1. 首先,你将获得任务指令、执行轨迹和最后一步的截图。请基于这些信息开始评估。
    2. 你可以设计自己的验证策略。一个建议的策略如下:
    - 首先,通过查看执行轨迹和最后一步的截图,检查是否有明显的错误,或者任务是否可以直接判定为成功。
    - 如果根据已有信息不能判定任务是否执行成功,你可以使用 `check_screenshot` 工具查看特定中间步骤的截图,以进一步验证过程。
    3. 基于你的分析,最终判断任务是否成功完成。
    4. 以指定格式提供你的最终判断。

    **判断标准:**
    - 任务目标是否完全实现?
    - 是否存在错误或未完成的步骤?
    - 最终状态是否符合预期结果?

    **重要提示: 最终判断格式**
    当你完成评估后,你**必须**按照以下确切格式提供最终判断:

    EVALUATION RESULT:
    Status: SUCCESS 或 FAILURE
    Confidence: HIGH 或 MEDIUM 或 LOW
    Reasoning: 你的详细推理,解释任务为何成功或失败

    正确格式示例:
    EVALUATION RESULT:
    Status: SUCCESS
    Confidence: HIGH
    Reasoning: 任务成功完成。所有必需的步骤都正确执行,最终状态符合预期结果。

    请开始你的评估,首先检查关键截图。

    **任务指令:**
    {instruction}

    **执行轨迹:**
    总步骤数: {num_screenshots}
    执行的动作:
    {actions_summary}

    **最后一步的截图:**
"""
    
    return prompt

def _build_eval_prompt_zh(instruction: str, trajectory: dict, num_screenshots: int) -> str:
        """构建评估提示"""
        actions_summary = "\n".join([
            # f"Step {i+1}: {str(action.get('command', action))[:100]}"
            f"Step {i+1}: {str(action.get('command', action))}"
            for i, action in enumerate(trajectory.get("actions", []))
        ])
        
        prompt = f"""<Instruction>
        判断GUI Agent的任务执行是否成功是一个困难的挑战，仅依赖执行轨迹中的思考过程和screenshot是不充分的，某些任务成功与否的判断需要Verifier与GUI环境进行互动才能得以进行（比如检查文件是否存在、验证文件内容是否正确等等）。请你作为一个Verifier Agent来评估GUI Agent是否完成了用户query中指定的任务，你可以通过查看GUI Agent的任务执行轨迹与screenshot或者直接对GUI环境进行操作等方式来进行验证。具体要求如下：
        1. 最初你会被提供GUI Agent执行任务的每个step的Thought文本内容以及最终step的screenshot，同时你也被赋予了查看其他step的screenshot以及直接与GUI环境进行交互的权限，你需要据此以Agent的多步推理的形式完成验证过程。
        2. 你可以自行规划验证的策略。一种建议的策略是：首先，你可以根据GUI Agent的每个step的Thought内容与最终step的screenshot来查验执行的轨迹是否有明显错误或可直接判定成功，接着你可以通过请求查看中间某个step的screenshot来继续验证过程，如果仅依赖Thought文本和screenshot无法验证任务是否完成，且你认为需要直接操作GUI环境去进行验证，则你可以直接输出操作的action。
        3. 你每次只需要输出一个action。根据你输出的action，你将被提供你所请求的中间step的screenshot或者执行你的GUI action后的环境screenshot。
        4. 你的中间验证step的输出格式必须包含Thought和Action两个部分，其中Thought是你在当前的验证step下所做的验证计划以及对你想要采取的验证action的总结，Action是当前step下你想要采取的具体验证action，其格式必须符合下面的Action Space中规定的格式。
        5. 你的最终验证step的输出格式必须包含Thought和Judgment两个部分，Thought是你对你前面执行的验证step的总结，Judgment是你的最终判断，如果判断任务完成，Judgment为1，如果没有完成，则为0，如果你认为无法做出判断或者该任务不可验证，则为-1。

        When you have completed your evaluation, you MUST provide your final judgment in the following exact format:
        <EVALUATION RESULT Template>
        Status: SUCCESS or FAILURE
        Confidence: HIGH or MEDIUM or LOW
        Reasoning: Your detailed reasoning explaining why the task succeeded or failed

        Example of correct format:
        EVALUATION RESULT:
        Status: SUCCESS
        Confidence: HIGH
        Reasoning: The task was completed successfully. All required steps were executed correctly, and the final state matches the expected outcome.

        **IMPORTANT: 每次

        <User Query>
        {instruction}
        <Total steps>
        {num_screenshots}
        <GUI Agent Trajectory>
        {actions_summary}
        <Last screenshot>
        """
        
        return prompt

def simplify_messages_for_json(messages, image_dir_name="agent_images"):
    """
    简化消息以便保存为JSON，将base64图片替换为文件路径引用
    
    Args:
        messages: 原始消息列表
        image_dir_name: 图片目录名称（相对路径）
    
    Returns:
        list: 简化后的消息列表
    """
    simplified_messages = []
    
    # 追踪当前轮次和工具调用上下文（用于智能命名）
    current_round = 0
    current_tool_context = None
    
    for msg_idx, msg in enumerate(messages):
        simplified_msg = {
            "role": msg.get("role", "unknown"),
        }
        
        # 更新上下文
        if msg.get("role") == "assistant":
            current_round += 1
            content = msg.get("content", "")
            if isinstance(content, str) and '<tool_call>' in content:
                try:
                    tool_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
                    if tool_match:
                        tool_data = json.loads(tool_match.group(1))
                        tool_name = tool_data.get('name', '')
                        tool_args_str = tool_data.get('arguments', '{}')
                        if isinstance(tool_args_str, str):
                            tool_args = json.loads(tool_args_str)
                        else:
                            tool_args = tool_args_str
                        
                        current_tool_context = {
                            'tool_name': tool_name,
                            'tool_args': tool_args,
                            'round': current_round
                        }
                except:
                    pass
        
        # 处理内容
        content = msg.get("content", "")
        
        if isinstance(content, str):
            # 纯文本内容，直接保留
            simplified_msg["content"] = content
        
        elif isinstance(content, list):
            # 多模态内容，需要处理图片
            simplified_content = []
            img_in_msg_idx = 0
            
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    
                    if image_url and image_url.startswith("data:image"):
                        # 是 base64 图片，生成文件名引用
                        header = image_url.split(",", 0)[0] if "," in image_url else image_url
                        
                        # 提取图片格式
                        try:
                            ext_match = re.search(r'data:image/(\w+);', header)
                            ext = ext_match.group(1) if ext_match else "png"
                        except:
                            ext = "png"
                        
                        # ⭐ 智能命名（与 save_agent_images 保持一致）
                        if current_tool_context:
                            tool_name = current_tool_context['tool_name']
                            tool_args = current_tool_context['tool_args']
                            round_num = current_tool_context['round']
                            
                            if tool_name == 'read_key_image':
                                image_names = tool_args.get('image_names', [])
                                if img_in_msg_idx < len(image_names):
                                    img_name = f"{image_names[img_in_msg_idx]}_round{round_num}.{ext}"
                                else:
                                    img_name = f"read_round{round_num}_img{img_in_msg_idx}.{ext}"
                            
                            elif tool_name == 'image_zoom_in_tool':
                                source_image = tool_args.get('image', 'unknown')
                                bbox = tool_args.get('bbox_2d', [])
                                if bbox:
                                    bbox_str = f"_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                                else:
                                    bbox_str = ""
                                img_name = f"{source_image}_zoom{bbox_str}_round{round_num}.{ext}"
                            
                            else:
                                img_name = f"{tool_name}_round{round_num}_img{img_in_msg_idx}.{ext}"
                        else:
                            img_name = f"msg{msg_idx}_img{img_in_msg_idx}.{ext}"
                        
                        # 构建相对路径
                        image_path = f"{image_dir_name}/{img_name}"
                        
                        # 添加简化的图片引用
                        simplified_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_path
                            },
                            "_metadata": {
                                "original_format": "base64",
                                "size_info": f"{len(image_url)} bytes" if image_url else "unknown"
                            }
                        })
                        
                        img_in_msg_idx += 1
                    
                    else:
                        # 已经是文件路径，直接保留
                        simplified_content.append(item)
                
                else:
                    # 非图片内容，直接保留
                    simplified_content.append(item)
            
            simplified_msg["content"] = simplified_content
        
        else:
            # 其他类型的内容，直接保留
            simplified_msg["content"] = content
        
        simplified_messages.append(simplified_msg)
    
    return simplified_messages


# def load_trajectory_data(traj_dir):
#     """
#     加载轨迹数据
    
#     Args:
#         traj_dir: 轨迹目录路径
    
#     Returns:
#         tuple: (instruction, actions, screenshots, raw_steps)
#     """
#     traj_dir = Path(traj_dir)
    
#     # 1. 加载 traj.jsonl
#     traj_file = traj_dir / "traj.jsonl"
#     if not traj_file.exists():
#         raise FileNotFoundError(f"traj.jsonl not found in {traj_dir}")
    
#     with open(traj_file, "r", encoding="utf-8") as f:
#         steps = [json.loads(line) for line in f]
    
#     # 2. 提取 actions（更详细的格式）
#     actions = []
#     for i, step in enumerate(steps, 1):
#         action_info = step.get("action", {})
#         action_type = action_info.get("action_type", "unknown")
        
#         # 构建动作描述
#         if action_type == "tool_use":
#             action_name = action_info.get("name", "unknown")
#             action_input = action_info.get("input", {})
            
#             # 格式化动作输入
#             if action_name == "computer":
#                 # 对于 computer 工具，提取关键信息
#                 action_detail = action_input.get("action", "")
#                 coordinate = action_input.get("coordinate", [])
#                 if coordinate:
#                     action_str = f"Step {i}: {action_detail} at coordinate {coordinate}"
#                 else:
#                     action_str = f"Step {i}: {action_detail}"
#             else:
#                 action_str = f"Step {i}: {action_name}({json.dumps(action_input, ensure_ascii=False)})"
        
#         elif action_type == "DONE":
#             action_str = f"Step {i}: Task marked as DONE"
        
#         else:
#             action_str = f"Step {i}: {action_type}"
        
#         # 添加 agent 的响应（如果有）
#         response = step.get("response", "")
#         if response:
#             response_preview = response
#             action_str += f"\n  Agent's reasoning: {response_preview}"
        
#         actions.append(action_str)
    
#     # 3. 加载 screenshots
#     screenshots = []
#     for step in steps:
#         screenshot_file = step.get("screenshot_file")
#         if screenshot_file:
#             img_path = traj_dir / screenshot_file
#             if img_path.exists():
#                 screenshots.append(Image.open(img_path))
#             else:
#                 print(f"Warning: Screenshot not found: {img_path}")
#                 screenshots.append(None)
#         else:
#             screenshots.append(None)
    
#     # 4. 加载 instruction
#     instruction = None
#     config_file = traj_dir / "config.json"
#     if config_file.exists():
#         with open(config_file, "r", encoding="utf-8") as f:
#             config = json.load(f)
#             instruction = config.get("instruction")
    
#     # 如果没有 config.json，尝试从第一步提取
#     if not instruction and steps:
#         instruction = steps[0].get("instruction", "Unknown task")
    
#     return instruction, actions, screenshots, steps


import json
from pathlib import Path
from PIL import Image
import re


def load_trajectory_data(traj_dir,output_dir, max_action_length=200, max_thought_length=300):
    """
    加载轨迹数据（兼容多种格式）
    
    提取两部分信息：
    1. 动作代码（action 字段的原始内容）
    2. 自然语言推理（response 中的 Thought/think 部分）
    
    Args:
        traj_dir: 轨迹目录路径
        max_action_length: 动作代码最大长度（超过会截断）
        max_thought_length: 推理文本最大长度（超过会截断）
    
    Returns:
        tuple: (instruction, actions, screenshots, raw_steps)
            - instruction: 任务指令
            - actions: 格式化的动作列表（包含动作代码和推理）
            - screenshots: PIL Image 对象列表
            - raw_steps: 原始步骤数据（jsonl 解析结果）
    """
    traj_dir = Path(traj_dir)
    
    # 1. 加载 traj.jsonl
    traj_file = traj_dir / "traj.jsonl"
    if not traj_file.exists():
        raise FileNotFoundError(f"traj.jsonl not found in {traj_dir}")
    
    with open(traj_file, "r", encoding="utf-8") as f:
        steps = [json.loads(line) for line in f]
    
    if not steps:
        raise ValueError(f"Empty trajectory file: {traj_file}")
    
    # 2. 提取 actions（包含动作代码和推理）
    actions = []
    for i, step in enumerate(steps, 1):
        action_raw = step.get("action", "")
        response = step.get("response", "")
        
        # 构建动作描述
        action_parts = []
        action_parts.append(f"Step {i}:")
        
        # ========== 提取动作代码 ==========
        if action_raw:
            if isinstance(action_raw, dict):
                # 字典格式（旧版）
                action_code = json.dumps(action_raw, ensure_ascii=False, indent=2)
            elif isinstance(action_raw, str):
                # 字符串格式（新版）- 直接使用原始代码
                action_code = action_raw.strip()
            else:
                action_code = str(action_raw)
            
            # 如果代码太长，截断
            if len(action_code) > max_action_length:
                action_code = action_code[:max_action_length] + "..."
            
            action_parts.append(f"  Action: {action_code}")
        else:
            action_parts.append(f"  Action: [No action]")
        
        # ========== 提取自然语言推理 ==========
        if response:
            thought = _extract_thought(response)
            if thought:
                # 如果推理太长，截断
                if len(thought) > max_thought_length:
                    thought = thought[:max_thought_length] + "..."
                action_parts.append(f"  Thought: {thought}")
        
        actions.append("\n".join(action_parts))
    
    # 3. 加载 screenshots
    screenshots = []
    for step in steps:
        screenshot_file = step.get("screenshot_file")
        if screenshot_file:
            img_path = traj_dir / screenshot_file
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    screenshots.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load screenshot {img_path}: {e}")
                    screenshots.append(None)
            else:
                print(f"Warning: Screenshot not found: {img_path}")
                screenshots.append(None)
        else:
            print(f"Warning: No screenshot_file in step {step.get('step_num', '?')}")
            screenshots.append(None)
    
    # 4. 加载 instruction
    instruction = None
    
    # 方法1: 从 config.json 加载
    config_file = traj_dir / "config.json"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                instruction = config.get("instruction")
                # 保存配置
                with open(output_dir / "config.json", "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")
                
    
    # 方法2: 从第一步提取
    if not instruction and steps:
        instruction = steps[0].get("instruction")
    
    # 方法3: 从所有步骤中查找
    if not instruction:
        for step in steps:
            if "instruction" in step and step["instruction"]:
                instruction = step["instruction"]
                break
    
    # 兜底
    if not instruction:
        instruction = "No instruction found"
    
    return instruction, actions, screenshots, steps


def _extract_thought(response):
    """
    从 response 中提取自然语言推理部分
    
    支持的格式：
    1. ## Thought:\n...
    2. Thought:\n...
    3. <think>...</think>
    4. <think_never_used_...>...</think_never_used_...>
    
    Args:
        response: 原始响应字符串
    
    Returns:
        str: 提取的推理文本（不包含标签和代码）
    """
    if not response:
        return ""
    
    thought_text = ""
    
    # 方法1: 提取 <think_never_used_...> 标签内容
    think_tag_match = re.search(
        r'<think_never_used_[^>]+>(.*?)</think_never_used_[^>]+>',
        response,
        flags=re.DOTALL
    )
    if think_tag_match:
        thought_text = think_tag_match.group(1).strip()
    
    # 方法2: 提取 <think> 标签内容
    if not thought_text:
        think_match = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
        if think_match:
            thought_text = think_match.group(1).strip()
    
    # 方法3: 提取 ## Thought: 或 Thought: 部分
    if not thought_text:
        thought_match = re.search(
            r'(?:##\s*)?Thought:\s*(.*?)(?=\n(?:##|Action|$))',
            response,
            flags=re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought_text = thought_match.group(1).strip()
    
    # 方法4: 如果都没找到，尝试提取整个响应的开头部分（排除代码和标签）
    if not thought_text:
        # 移除代码块
        cleaned = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        # 移除 XML/HTML 风格的标签（包括 seed:tool_call 等）
        cleaned = re.sub(r'<[^>]+>.*?</[^>]+>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<[^>]+/>', '', cleaned)
        # 移除多余空行
        cleaned = re.sub(r'\n{2,}', '\n', cleaned).strip()
        
        # 如果清理后还有内容，使用它
        if cleaned and len(cleaned) > 10:  # 至少10个字符才算有效
            thought_text = cleaned
    
    # 最终清理：压缩多余的空白字符
    if thought_text:
        thought_text = re.sub(r'\s+', ' ', thought_text).strip()
    
    return thought_text



def create_user_prompt(instruction, actions, total_steps):
    """
    创建用户提示词（包含 actions 信息）
    
    Args:
        instruction: 任务指令
        actions: 动作列表
        total_steps: 总步数
    
    Returns:
        str: 格式化的提示词
    """
    # 格式化 actions
    actions_text = "\n".join(actions)
    
    user_prompt = f"""
**Your Role:**
You are an expert evaluator analyzing whether an agent successfully completed a GUI task. You have access to the agent's execution trajectory, including all actions taken and screenshots at each step.

**Task Instruction:**
{instruction}

**Agent's Execution Trajectory:**
The agent executed the following {total_steps} steps:

{actions_text}

---

**Your Analysis Process:**

You need to verify whether the task was successfully completed by examining the visual evidence. Follow this iterative process:

1. **First, analyze the trajectory:**
   - Review the task instruction carefully
   - Examine the sequence of actions the agent took
   - Identify which steps are critical for task completion
   - Determine what visual evidence would confirm success
   - List specific things you need to check (e.g., "Check if bookmark was added to bookmarks bar", "Verify dialog was closed", etc.)

2. **Next, gather visual evidence:**
   - Use the `read_key_image` tool to view screenshots at specific steps
     * Example: `read_key_image(image_names=["step_1", "step_5"])` to view steps 1 and 5
   - Use the `image_zoom_in_tool` to examine specific regions in detail
     * Example: `image_zoom_in_tool(image="step_5", bbox_2d=[0, 0, 500, 200])` to zoom into top-left region
   - Focus on areas where you expect to see changes or confirmations

3. **Then, review your findings:**
   - Carefully analyze what the tools show you
   - Compare screenshots before and after critical actions
   - Check if the expected visual changes occurred
   - Decide if you need to examine other steps or regions

**Continue this loop until you have sufficient evidence to make a confident judgment.**

---

**Final Output Format:**

Once your investigation is complete, provide your judgment in this format:

<answer>YES</answer> or <answer>NO</answer>

**Then provide detailed reasoning that includes:**
- **Summary of findings:** What visual evidence did you find?
- **Critical steps analyzed:** Which steps did you examine and why?
- **Key evidence:** What specific visual elements confirmed success or failure?
  * Example: "In step 5, the bookmark star icon is filled in blue, indicating the page was bookmarked"
  * Example: "The bookmarks bar shows the new bookmark with title 'XXX'"
- **Conclusion:** Clear explanation of why the task succeeded or failed

---

**Available Screenshots:**
You have access to screenshots from step_1 to step_{total_steps}.

**Now, begin your analysis. Start by examining the trajectory and deciding which steps to investigate first.**
"""
    return user_prompt


def extract_image_context_from_message(msg, msg_idx):
    """
    从消息中提取图片上下文信息，用于智能命名
    
    Args:
        msg: 消息对象
        msg_idx: 消息索引
    
    Returns:
        dict: 包含上下文信息的字典
    """
    context = {
        'tool_name': None,
        'step_numbers': [],
        'round_num': None,
    }
    
    # 尝试从前一条 assistant 消息中提取工具调用信息
    content = msg.get("content", "")
    
    # 如果是列表（多模态内容），查找文本部分
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                content = item.get("text", "")
                break
    
    if isinstance(content, str):
        # 提取 step 编号
        step_matches = re.findall(r'step_(\d+)', content.lower())
        if step_matches:
            context['step_numbers'] = sorted(set(map(int, step_matches)))
        
        # 识别工具名称
        if 'read_key_image' in content.lower():
            context['tool_name'] = 'read_key_image'
        elif 'zoom' in content.lower():
            context['tool_name'] = 'zoom_in'
    
    return context


def save_agent_images(messages, save_dir):
    """
    保存agent消息中的图片到文件，使用智能命名
    
    Args:
        messages: agent.run() 返回的消息列表
        save_dir: 保存目录
    
    Returns:
        tuple: (处理后的消息列表, 图片索引映射)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 清理旧图片
    for file in save_dir.glob("*.png"):
        file.unlink()
    for file in save_dir.glob("*.jpg"):
        file.unlink()
    
    # 追踪当前轮次和工具调用上下文
    current_round = 0
    current_tool_context = None
    image_index_map = {}  # 旧索引 -> 新文件名
    
    processed_messages = []
    
    for msg_idx, msg in enumerate(messages):
        processed_msg = msg.copy()
        
        # 更新上下文
        if msg.get("role") == "assistant":
            current_round += 1
            # 尝试从 assistant 消息中提取工具调用信息
            content = msg.get("content", "")
            if isinstance(content, str):
                # 查找工具调用
                if '<tool_call>' in content:
                    try:
                        tool_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
                        if tool_match:
                            tool_data = json.loads(tool_match.group(1))
                            tool_name = tool_data.get('name', '')
                            tool_args_str = tool_data.get('arguments', '{}')
                            if isinstance(tool_args_str, str):
                                tool_args = json.loads(tool_args_str)
                            else:
                                tool_args = tool_args_str
                            
                            current_tool_context = {
                                'tool_name': tool_name,
                                'tool_args': tool_args,
                                'round': current_round
                            }
                    except:
                        pass
        
        # 处理图片
        content = msg.get("content", "")
        
        if isinstance(content, list):
            processed_content = []
            img_in_msg_idx = 0  # 同一消息中的图片索引
            
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    
                    if image_url and image_url.startswith("data:image"):
                        try:
                            # 解析base64图片
                            header, b64_data = image_url.split(",", 1)
                            ext = header.split("/")[1].split(";")[0]
                            img_bytes = base64.b64decode(b64_data)
                            
                            # ⭐ 智能命名逻辑
                            if current_tool_context:
                                tool_name = current_tool_context['tool_name']
                                tool_args = current_tool_context['tool_args']
                                round_num = current_tool_context['round']
                                
                                if tool_name == 'read_key_image':
                                    # read_key_image 工具：使用 step_X 命名
                                    image_names = tool_args.get('image_names', [])
                                    if img_in_msg_idx < len(image_names):
                                        img_name = f"{image_names[img_in_msg_idx]}_round{round_num}.{ext}"
                                    else:
                                        img_name = f"read_round{round_num}_img{img_in_msg_idx}.{ext}"
                                
                                elif tool_name == 'image_zoom_in_tool':
                                    # zoom 工具：标注来源 step 和区域
                                    source_image = tool_args.get('image', 'unknown')
                                    bbox = tool_args.get('bbox_2d', [])
                                    if bbox:
                                        bbox_str = f"_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                                    else:
                                        bbox_str = ""
                                    img_name = f"{source_image}_zoom{bbox_str}_round{round_num}.{ext}"
                                
                                else:
                                    # 其他工具
                                    img_name = f"{tool_name}_round{round_num}_img{img_in_msg_idx}.{ext}"
                            else:
                                # 无上下文：使用消息索引
                                img_name = f"msg{msg_idx}_img{img_in_msg_idx}.{ext}"
                            
                            # 保存图片
                            img_path = save_dir / img_name
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                            
                            # 记录映射
                            image_index_map[f"msg{msg_idx}_img{img_in_msg_idx}"] = img_name
                            
                            # 替换URL为相对路径
                            processed_item = item.copy()
                            processed_item["image_url"] = {"url": img_name}
                            processed_content.append(processed_item)
                            
                            print(f"[Saved] {img_path}")
                            img_in_msg_idx += 1
                            
                        except Exception as e:
                            print(f"Error saving image: {e}")
                            traceback.print_exc()
                            processed_content.append(item)
                    else:
                        processed_content.append(item)
                else:
                    processed_content.append(item)
            
            processed_msg["content"] = processed_content
        
        processed_messages.append(processed_msg)
    
    return processed_messages, image_index_map


def parse_agent_judgment(final_response):
    """
    解析agent的最终判断
    
    Args:
        final_response: agent的最后一条消息内容
    
    Returns:
        dict: {'success': bool, 'confidence': str, 'reasoning': str}
    """
    import re
    
    response_lower = final_response.lower()
    
    # 方法1: 查找 <answer>YES</answer> 或 <answer>NO</answer>
    answer_match = re.search(r'<answer>\s*(yes|no)\s*</answer>', response_lower)
    if answer_match:
        success = (answer_match.group(1) == 'yes')
        return {
            'success': success,
            'confidence': 'high',
            'reasoning': final_response
        }
    
    # 方法2: 关键词匹配
    positive_keywords = [
        'successfully completed',
        'task was successful',
        'successfully executed',
        'task is complete',
        'yes, the task',
        'task succeeded',
    ]
    
    negative_keywords = [
        'not successful',
        'failed to complete',
        'task was not',
        'unsuccessful',
        'did not complete',
        'no, the task',
        'task failed',
    ]
    
    has_positive = any(kw in response_lower for kw in positive_keywords)
    has_negative = any(kw in response_lower for kw in negative_keywords)
    
    if has_positive and not has_negative:
        return {'success': True, 'confidence': 'medium', 'reasoning': final_response}
    
    if has_negative:
        return {'success': False, 'confidence': 'medium', 'reasoning': final_response}
    
    # 方法3: 无法确定
    return {'success': None, 'confidence': 'low', 'reasoning': final_response}


def extract_tool_usage_summary(messages, tool_call_recorder=None):
    """
    提取工具使用摘要
    
    Args:
        messages: agent 消息列表
        tool_call_recorder: ToolCallRecorder 实例（如果启用了日志）
    
    Returns:
        dict: 工具使用统计
    """
    import re
    
    # 优先使用 ToolCallRecorder 的数据
    if tool_call_recorder and hasattr(tool_call_recorder, 'tool_calls'):
        print("\n[DEBUG] Using ToolCallRecorder data for tool usage summary")
        tool_calls_data = tool_call_recorder.tool_calls
        
        # 统计各工具调用次数
        tool_call_counts = {}
        tool_specific_data = {}
        
        for call in tool_calls_data:
            tool_name = call['tool_name']
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            
            # 记录特定工具的详细信息
            if tool_name == 'read_key_image':
                if 'read_key_image_steps' not in tool_specific_data:
                    tool_specific_data['read_key_image_steps'] = []
                # 提取 image_names
                args = call.get('tool_args', {})
                image_names = args.get('image_names', [])
                tool_specific_data['read_key_image_steps'].extend(image_names)
            
            elif tool_name == 'image_zoom_in_tool':
                if 'zoomed_images' not in tool_specific_data:
                    tool_specific_data['zoomed_images'] = []
                # 提取 image 参数
                args = call.get('tool_args', {})
                image = args.get('image', '')
                if image:
                    tool_specific_data['zoomed_images'].append(image)
        
        result = {
            'read_key_image_calls': tool_call_counts.get('read_key_image', 0),
            'read_key_image_steps': list(set(tool_specific_data.get('read_key_image_steps', []))),
            'image_zoom_in_calls': tool_call_counts.get('image_zoom_in_tool', 0),
            'zoomed_images': list(set(tool_specific_data.get('zoomed_images', []))),
        }
        
        print(f"[DEBUG] Tool usage from recorder: {result}")
        return result
    
    # 回退：从消息中解析
    print("\n[DEBUG] Parsing tool usage from messages (ToolCallRecorder not available)")
    
    tool_calls = {
        'read_key_image': [],
        'image_zoom_in_tool': [],
    }
    
    for msg_idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        
        print(f"[DEBUG] Checking message {msg_idx}, content length: {len(content)}")
        
        # 方法1：查找 <tool_call> 标签
        tool_call_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
        print(f"[DEBUG] Found {len(tool_call_blocks)} <tool_call> blocks")
        
        for block in tool_call_blocks:
            # 解析 JSON
            try:
                tool_data = json.loads(block)
                tool_name = tool_data.get('name', '')
                tool_args = tool_data.get('arguments', {})
                
                print(f"[DEBUG] Parsed tool: {tool_name}, args type: {type(tool_args)}")
                
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                
                if tool_name == 'read_key_image':
                    image_names = tool_args.get('image_names', [])
                    print(f"[DEBUG] read_key_image with images: {image_names}")
                    tool_calls['read_key_image'].extend(image_names)
                
                elif tool_name == 'image_zoom_in_tool':
                    image = tool_args.get('image', '')
                    print(f"[DEBUG] image_zoom_in_tool with image: {image}")
                    if image:
                        tool_calls['image_zoom_in_tool'].append(image)
            except Exception as e:
                print(f"[DEBUG] Failed to parse tool call block: {e}")
                pass
        
        # 方法2：查找函数调用模式（备用）
        read_matches = re.findall(r'read_key_image.*?image_names.*?$$(.*?)$$', content, re.DOTALL)
        if read_matches:
            print(f"[DEBUG] Found {len(read_matches)} read_key_image matches (method 2)")
        
        for match in read_matches:
            steps = re.findall(r'["\']?(step_\d+)["\']?', match)
            tool_calls['read_key_image'].extend(steps)
        
        zoom_matches = re.findall(r'image_zoom_in_tool.*?image.*?["\'](\w+)["\']', content)
        if zoom_matches:
            print(f"[DEBUG] Found {len(zoom_matches)} image_zoom_in_tool matches (method 2)")
        
        tool_calls['image_zoom_in_tool'].extend(zoom_matches)
    
    result = {
        'read_key_image_calls': len(tool_calls['read_key_image']),
        'read_key_image_steps': list(set(tool_calls['read_key_image'])),
        'image_zoom_in_calls': len(tool_calls['image_zoom_in_tool']),
        'zoomed_images': list(set(tool_calls['image_zoom_in_tool'])),
    }
    
    print(f"[DEBUG] Final tool usage: {result}")
    return result


# def remove_images_and_add_note(messages):
#     """
#     遍历消息历史，移除图片数据，并在相邻的文本块中添加系统提示。
#     """
#     # 使用 deepcopy 防止修改原始数据
#     cleaned_messages = copy.deepcopy(messages)
    
#     # 提示语
#     system_note = "\n\n[System Note: Image data removed to save context window]"
    
#     for msg in cleaned_messages:
#         # 如果 content 不是列表（是纯字符串），说明没图片，跳过
#         if not isinstance(msg.get('content'), list):
#             continue
            
#         new_content_list = []
#         image_was_removed = False
        
#         # 1. 第一次遍历：分离图片和其他内容
#         for item in msg['content']:
#             if item.get('type') == 'image':
#                 image_was_removed = True
#                 # 图片项直接丢弃，不添加到新列表中
#             else:
#                 new_content_list.append(item)
        
#         # 2. 如果删除了图片，需要找到合适的地方插入提示语
#         if image_was_removed:
#             # 策略：追加到列表中的最后一个文本类元素（Text 或 Tool Result）
#             if new_content_list:
#                 last_item = new_content_list[-1]
                
#                 # 情况 A: 常规文本消息
#                 if last_item.get('type') == 'text':
#                     last_item['text'] = last_item['text'] + system_note
                    
#                 # 情况 B: Tool Result (通常 check_screenshot 的结果在这里)
#                 elif last_item.get('type') == 'tool_result':
#                     # tool_result 的 content 可能是字符串
#                     if isinstance(last_item.get('content'), str):
#                         last_item['content'] = last_item['content'] + system_note
#                     # 有些格式中 tool_result content 也是 list，这里简单处理字符串情况
            
#             else:
#                 # 极端情况：如果这条消息里只有图片，没有文字
#                 # 我们需要创建一个新的 text item 来承载这个 Note，否则消息内容为空会报错
#                 new_content_list.append({
#                     "type": "text",
#                     "text": system_note.strip() # 去掉开头的换行符
#                 })
        
#         # 更新消息的内容
#         msg['content'] = new_content_list
        
#     return cleaned_messages

# def remove_images_and_add_note(messages, keep_last_n_images=0):
#     """
#     遍历消息历史，移除图片数据，并在相邻的文本块中添加系统提示。
    
#     Args:
#         messages: 消息列表
#         keep_last_n_images (int): 从后往前保留的图片最大数量。
#                                   遵循“原子性原则”：如果单条消息内的图片数超过剩余配额，
#                                   则该条消息的所有图片都会被删除。
#     """
#     # 使用 deepcopy 防止修改原始数据
#     cleaned_messages = copy.deepcopy(messages)
    
#     # 提示语
#     system_note = "\n\n[System Note: Image data removed to save context window]"
    
#     # --- 步骤 1: 预计算哪些消息的图片需要保留 ---
#     # 记录允许保留图片的消息的索引 (indices)
#     indices_to_keep_images = set()
#     remaining_quota = keep_last_n_images

#     if remaining_quota > 0:
#         # 倒序遍历（从最新的消息开始检查）
#         for i in range(len(cleaned_messages) - 1, -1, -1):
#             msg = cleaned_messages[i]
#             content = msg.get('content')
            
#             # 如果 content 不是列表，说明没有混合内容，跳过
#             if not isinstance(content, list):
#                 continue
            
#             # 统计当前消息内的图片数量
#             img_count = sum(1 for item in content if item.get('type') == 'image')
            
#             if img_count == 0:
#                 continue
            
#             # 核心逻辑：原子性检查
#             # 只有当 剩余配额 >= 当前消息图片总数 时，才保留
#             if img_count <= remaining_quota:
#                 indices_to_keep_images.add(i)
#                 remaining_quota -= img_count
#             else:
#                 # 如果当前消息图片太多，配额不够，则该消息图片全删。
#                 # 并不停止循环，继续往前看是否有包含少量图片的消息能塞进剩余配额（贪心策略）
#                 pass
                
#             # 如果配额用完了，理论上可以提前 break，但为了严谨处理后续可能的 0 图消息，继续循环无妨
#             if remaining_quota <= 0:
#                 pass 

#     # --- 步骤 2: 执行清理和添加提示 ---
#     # 正序遍历处理
#     for i, msg in enumerate(cleaned_messages):
#         # 如果 content 不是列表（是纯字符串），说明没图片，跳过
#         if not isinstance(msg.get('content'), list):
#             continue
            
#         new_content_list = []
#         image_was_removed = False
        
#         # 判断当前消息是否在“保留白名单”中
#         should_keep_images = (i in indices_to_keep_images)
        
#         for item in msg['content']:
#             if item.get('type') == 'image':
#                 if should_keep_images:
#                     # 在白名单内，保留图片
#                     new_content_list.append(item)
#                 else:
#                     # 不在白名单，移除图片，并标记状态
#                     image_was_removed = True
#             else:
#                 new_content_list.append(item)
        
#         # 如果删除了图片，需要找到合适的地方插入提示语
#         if image_was_removed:
#             # 策略：追加到列表中的最后一个文本类元素（Text 或 Tool Result）
#             if new_content_list:
#                 last_item = new_content_list[-1]
                
#                 # 情况 A: 常规文本消息
#                 if last_item.get('type') == 'text':
#                     last_item['text'] = last_item['text'] + system_note
                    
#                 # 情况 B: Tool Result (通常 check_screenshot 的结果在这里)
#                 elif last_item.get('type') == 'tool_result':
#                     # tool_result 的 content 可能是字符串
#                     if isinstance(last_item.get('content'), str):
#                         last_item['content'] = last_item['content'] + system_note
#                     # 如果 tool_result content 是 list，这里暂不处理复杂嵌套，视具体 schema 而定
            
#             else:
#                 # 极端情况：如果这条消息里只有图片，没有文字
#                 # 我们需要创建一个新的 text item 来承载这个 Note
#                 new_content_list.append({
#                     "type": "text",
#                     "text": system_note.strip() # 去掉开头的换行符
#                 })
        
#         # 更新消息的内容
#         msg['content'] = new_content_list
        
#     return cleaned_messages

def remove_images_and_add_note(messages, keep_last_n_images=0):
    """
    遍历消息历史，移除图片数据，并在相邻的文本块中添加系统提示。
    
    修改规则：
    1. 总是保留第一条消息（content index 0）的图片。
    2. 总是保留 check_screenshot 工具返回结果中的图片（通过特定文本字符串匹配识别）。
    3. 上述保留的图片占用 keep_last_n_images 配额。
       - 如果受保护图片 > 配额，则全部保留受保护图片，剩余配额为0。
       - 如果受保护图片 < 配额，剩余配额用于从后往前保留最新的图片。
    """
    # 使用 deepcopy 防止修改原始数据
    cleaned_messages = copy.deepcopy(messages)
    
    # 提示语
    system_note = "\n\n[System Note: Image data removed to save context window]"
    # check_screenshot 识别特征句
    check_screenshot_marker = "These are the screenshots at the requested steps. Please analyze them to check if the conditions are met."
    
    # --- 辅助函数 ---
    
    def get_image_count(msg):
        """计算单条消息内的图片数量"""
        content = msg.get('content')
        if isinstance(content, list):
            return sum(1 for item in content if item.get('type') == 'image')
        return 0

    def is_check_screenshot_response(msg):
        """
        判断消息是否包含 check_screenshot 的特定回复文本。
        修改为：直接将 content 转换为字符串进行查找，简单且覆盖所有嵌套层级。
        """
        content = msg.get('content')
        # 只要目标句子出现在 content 的字符串表示中（无论是 key 还是 value，甚至是嵌套结构），即视为命中
        return check_screenshot_marker in str(content)

    # --- 步骤 1: 预计算哪些消息的图片需要保留 ---
    indices_to_keep_images = set()
    remaining_quota = keep_last_n_images

    # ---------------------------------------------------------
    # 1.1 优先处理“强制保留”的消息 (First Content & Check Screenshot)
    # ---------------------------------------------------------
    for i, msg in enumerate(cleaned_messages):
        img_count = get_image_count(msg)
        if img_count == 0:
            continue

        is_protected = False
        
        # 规则 A: 第一条消息 (Index 0)
        if i == 0:
            is_protected = True
        # 规则 B: Check Screenshot 回复 (使用新的简化版判断函数)
        elif is_check_screenshot_response(msg):
            is_protected = True
            
        if is_protected:
            indices_to_keep_images.add(i)
            # 扣除配额（允许透支，稍后修正）
            remaining_quota -= img_count

    # 如果强制保留的图片已经把配额用光（甚至透支），修正为 0，不再保留后续普通图片
    if remaining_quota < 0:
        remaining_quota = 0

    # ---------------------------------------------------------
    # 1.2 处理剩余配额：倒序遍历保留最新的图片
    # ---------------------------------------------------------
    if remaining_quota > 0:
        for i in range(len(cleaned_messages) - 1, -1, -1):
            # 如果该消息已经是受保护消息，跳过，避免重复计算
            if i in indices_to_keep_images:
                continue
            
            msg = cleaned_messages[i]
            img_count = get_image_count(msg)
            
            if img_count == 0:
                continue
            
            # 原子性检查：配额足够才保留
            if img_count <= remaining_quota:
                indices_to_keep_images.add(i)
                remaining_quota -= img_count
            else:
                break
            
            # 如果配额耗尽，停止查找
            if remaining_quota <= 0:
                break
    
    print(f"Indices keeping images: {indices_to_keep_images}") # 调试用打印

    # --- 步骤 2: 执行清理和添加提示 ---
    for i, msg in enumerate(cleaned_messages):
        if not isinstance(msg.get('content'), list):
            continue
            
        new_content_list = []
        image_was_removed = False
        
        should_keep_images = (i in indices_to_keep_images)
        
        for item in msg['content']:
            if item.get('type') == 'image':
                if should_keep_images:
                    new_content_list.append(item)
                else:
                    image_was_removed = True
            else:
                new_content_list.append(item)
        
        # 如果删除了图片，需要插入提示语
        if image_was_removed:
            if new_content_list:
                last_item = new_content_list[-1]
                # 策略：追加到最后一个文本类元素
                if last_item.get('type') == 'text':
                    last_item['text'] = last_item['text'] + system_note
                elif last_item.get('type') == 'tool_result':
                    if isinstance(last_item.get('content'), str):
                        last_item['content'] = last_item['content'] + system_note
                    # 也可以选择处理 tool_result content 为 list 的情况，视需求而定
            else:
                # 只有图片的情况
                new_content_list.append({
                    "type": "text",
                    "text": system_note.strip()
                })
        
        msg['content'] = new_content_list
        
    return cleaned_messages

def count_images_in_content(content):
    """
    计算单个消息内容(content)中的图片数量。
    
    Args:
        content: 消息的内容字段，可能是字符串(String)或列表(List[Dict])。
        
    Returns:
        int: 图片的数量。
    """
    # 如果 content 是字符串、None 或其他非列表类型，说明肯定没有图片结构
    if not isinstance(content, list):
        return 0
        
    # 遍历列表，统计 type 为 'image' 的字典项数量
    # 增加 isinstance(item, dict) 判断以防止列表里混入非字典元素导致报错
    return sum(1 for item in content if isinstance(item, dict) and item.get('type') == 'image')