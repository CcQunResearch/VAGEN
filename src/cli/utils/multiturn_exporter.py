import os
import json
import datetime
from typing import List, Dict, Any, Optional


def make_assistant_message(text: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": text if isinstance(text, str) else json.dumps(text, ensure_ascii=False),
    }


def append_multi_turn_sample(
    output_dir: str,
    messages: List[Dict[str, Any]],
    assistant_response: str,
    meta: Optional[Dict[str, Any]] = None,
    filename: str = "multi_turn.jsonl",
    steps: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    以标准多轮对话格式写入一条样本：{"messages": [...], "meta": {...}}
    - messages: 使用 openai 风格的多模态结构，user.content 使用相对路径 image_url
    - assistant_response: 本轮模型输出，将追加为最后一条 assistant 消息
    - meta: 额外信息（step、reward、done、action、screenshot 等）
    """

    sample = {
        "messages": list(messages) + [make_assistant_message(assistant_response)],
        "meta": meta or {},
    }
    if steps is not None:
        sample["steps"] = steps

    out_path = os.path.join(output_dir, filename)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False, indent=2))
        f.write("\n")



def assemble_messages_from_infodicts(info_dicts: List[Dict[str, Any]]) -> (List[Dict[str, Any]], str):
    """
    基于每步的 info_dict 列表组装“去重后”的顶层串联：
    - 仅保留首个 system（如有）
    - 每步取该轮的最后一个 user 作为输入
    - 前 n-1 轮追加当轮 assistant 文本；最后一轮的 assistant 作为最终 assistant 返回
    - 返回 (messages_without_final_assistant, final_assistant_text)
    """
    if not info_dicts:
        return [], ""
    messages: List[Dict[str, Any]] = []
    system_added = False
    final_assistant = ""
    total = len(info_dicts)
    for idx, info in enumerate(info_dicts):
        msgs = list(info.get("messages_export", []) if isinstance(info, dict) else [])
        # 首步添加 system 一次
        if not system_added and msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            messages.append(msgs[0])
            system_added = True
        # 取最后一个 user
        user_msg = None
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                user_msg = m
                break
        if user_msg is not None:
            messages.append(user_msg)
        # assistant 文本
        resp_text = info.get("response_text") if isinstance(info, dict) else None
        if resp_text is None:
            resp_text = ""
        if idx < total - 1:
            messages.append({"role": "assistant", "content": resp_text})
        else:
            final_assistant = resp_text
    return messages, final_assistant


def export_from_collected_if_success(
    output_dir: str,
    info_dicts: List[Dict[str, Any]],
    result_value: float,
    success_threshold: float = 0.0,
    filename: str = "multi_turn.jsonl",
) -> bool:
    if result_value is None or float(result_value) <= float(success_threshold):
        return False

    messages, final_assistant = assemble_messages_from_infodicts(info_dicts)
    if not messages or not final_assistant:
        return False
    # 组织步骤级数据，明确每步输入/输出
    steps_payload: List[Dict[str, Any]] = []
    # 按步骤汇总低层动作、pyautogui 动作与截图到 meta 序列
    low_level_actions_seq: List[Any] = []
    pyautogui_actions_seq: List[List[Any]] = []
    screenshots_seq: List[List[str]] = []
    image_meta_seq: List[Optional[Dict[str, Any]]] = []

    # 预先收集 step_0 的截图（若存在）
    try:
        if isinstance(output_dir, str) and os.path.isdir(output_dir):
            matched0 = [name for name in os.listdir(output_dir)
                        if isinstance(name, str)
                        and name.startswith("step_0_")
                        and name.endswith(".png")]
            matched0.sort()
            # 去重并保持顺序，过滤空值
            deduped0: List[str] = []
            seen0 = set()
            for _u0 in matched0:
                try:
                    _s0 = str(_u0).strip()
                except Exception:
                    _s0 = ""
                if not _s0:
                    continue
                if _s0 not in seen0:
                    seen0.add(_s0)
                    deduped0.append(_s0)
            if deduped0:
                screenshots_seq.append(deduped0)
                image_meta_seq.append(None) # Add a placeholder for step 0
    except Exception:
        pass
    for idx, info in enumerate(info_dicts):
        msgs = list(info.get("messages_export", []) if isinstance(info, dict) else [])
        resp = info.get("response_text") if isinstance(info, dict) else None
        if not isinstance(resp, str):
            resp = json.dumps(resp, ensure_ascii=False) if resp is not None else ""
        steps_payload.append({
            "step_index": idx + 1,
            "input_messages": msgs,
            "assistant": resp,
        })
        # 汇总 low_level_action
        low_level_actions_seq.append(info.get("low_level_action") if isinstance(info, dict) else None)
        # 汇总 pyautogui_actions（保证为列表）
        py_actions = []
        if isinstance(info, dict):
            val = info.get("pyautogui_actions")
            if isinstance(val, list):
                py_actions = val
            elif val is None:
                py_actions = []
            else:
                py_actions = [val]
        pyautogui_actions_seq.append(py_actions)
        # 直接从输出目录收集该步的截图：匹配 step_{n}_*.png
        screenshots_for_step: List[str] = []
        try:
            step_no = idx + 1
            if isinstance(output_dir, str) and os.path.isdir(output_dir):
                matched = [name for name in os.listdir(output_dir)
                           if isinstance(name, str)
                           and name.startswith(f"step_{step_no}_")
                           and name.endswith(".png")]
                matched.sort()
                screenshots_for_step.extend(matched)
        except Exception:
            pass
        # 去重并保持顺序，过滤空值
        deduped: List[str] = []
        seen = set()
        for _u in screenshots_for_step:
            try:
                _s = str(_u).strip()
            except Exception:
                _s = ""
            if not _s:
                continue
            if _s not in seen:
                seen.add(_s)
                deduped.append(_s)
        screenshots_seq.append(deduped)
        # 汇总 image_meta
        image_meta_seq.append(info.get("image_meta") if isinstance(info, dict) else None)
    append_multi_turn_sample(
        output_dir=output_dir,
        messages=messages,
        assistant_response=final_assistant,
        meta={
            "result": result_value,
            "low_level_actions": low_level_actions_seq,
            "pyautogui_actions": pyautogui_actions_seq,
            "screenshots": screenshots_seq,
            "image_meta": image_meta_seq,
        },
        filename=filename,
        steps=steps_payload,
    )
    return True



