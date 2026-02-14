# agent_tools.py

"""
Tool Registry and Definitions for the Desktop Environment Agent.

This module centralizes the definitions of all tools available to the agent.
It provides a flexible way to select and configure tools for different models
and tasks.
"""
from typing import List, Dict, Any, Optional

# --- Tool Definitions ---

# Each tool is defined as a dictionary that matches the API specification.

BROWSE_TOOL = {
    "type": "function",
    "function": {
        "name": "browse_url",
        "description": "Opens a new browser window and navigates to the specified URL. Use this for opening web pages.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to open, e.g., 'https://www.google.com'."
                }
            },
            "required": ["url"]
        }
    }
}

EXECUTE_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Executes a snippet of Python code in the environment. Use it for calculations, string manipulation, or logic that doesn't require direct GUI interaction. The code is sandboxed and cannot interact with the GUI directly.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute. For example, 'print(1+1)'."
                }
            },
            "required": ["code"]
        }
    }
}

EXECUTE_SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_shell",
        "description": "Executes a shell (bash) command in the environment's terminal. Use it for file system operations (ls, cd, mkdir), checking software versions, or running command-line tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute. For example, 'ls -l'."
                }
            },
            "required": ["command"]
        }
    }
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information on the internet",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords"}
            },
            "required": ["query"]
        }
    }
}

VISIT_TOOL = {
    "type": "function",
    "function": {
        "name": "visit",
        "description": "Retrieves the full text content of web pages or PDFs from a list of URLs.",
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "An array of URLs to visit and extract content from."
                }
            },
            "required": ["urls"]
        }
    }
}

# ⭐ 新增: 检查截图工具(用于评估)
CHECK_SCREENSHOT_TOOL = {
    "type": "function",
    "function": {
        "name": "check_screenshot",
        "description": "View specific screenshot of one step from the trajectory to check if certain conditions are met. Use this tool to examine screenshot at particular step (e.g., step_1, step_7, step_10) to verify task progress or completion.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of screenshot names to view, formatted as 'step_N' where N is the step number (e.g., ['step_7']). Only one step is allowed.",
                    "minItems": 1,
                    "maxItems": 1
                }
            },
            "required": ["image_names"]
        }
    }
}

# # ⭐ 新增: 检查截图工具(用于评估)
# CHECK_SCREENSHOT_TOOL = {
#     "type": "function",
#     "function": {
#         "name": "check_screenshot",
#         "description": "View specific screenshots from the trajectory to check if certain conditions are met. Use this tool to examine screenshots at particular steps (e.g., step_1, step_7, step_10) to verify task progress or completion.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "image_names": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "description": "List of screenshot names to view, formatted as 'step_N' where N is the step number (e.g., ['step_1', 'step_7', 'step_10']).",
#                     "minItems": 1,
#                     "maxItems": 5
#                 }
#             },
#             "required": ["image_names"]
#         }
#     }
# }

# --- Dynamic Tool Functions ---

# def get_computer_tool(model_name: str, platform: str = "Ubuntu") -> Dict[str, Any]:
#     """
#     Returns the correct 'computer' tool definition based on the model name and platform.
#     This encapsulates the model-specific logic.
#     """
#     # Default values for display size
#     display_width, display_height = 1280, 720

#     if model_name == "claude-3-5-sonnet-20241022":
#         return {'name': 'computer', 'type': 'computer_20241022', 'display_width_px': display_width, 'display_height_px': display_height, 'display_number': 1}
    
#     # Add more model versions here if needed
#     # elif model_name == "some-other-model":
#     #     return {'name': 'computer', 'type': 'some-other-type', ...}

#     # Fallback to the latest known version for Friday models
#     elif "claude-3.7" in model_name or "claude-4" in model_name:
#          return {'name': 'computer', 'type': 'computer_20250124', 'display_width_px': display_width, 'display_height_px': display_height, 'display_number': 1}

#     # A sensible default if no specific model matches
#     else:
#         # Assuming computer_20250124 is the most recent/default
#         return {'name': 'computer', 'type': 'computer_20250124', 'display_width_px': display_width, 'display_height_px': display_height, 'display_number': 1}


# --- Tool Registry and Selection ---

# A registry mapping tool names to their definitions for easy lookup.
TOOL_REGISTRY = {
    "browse": BROWSE_TOOL,
    "execute_python": EXECUTE_PYTHON_TOOL,
    "execute_shell": EXECUTE_SHELL_TOOL,
    "search": SEARCH_TOOL,
    "visit": VISIT_TOOL,
    "check_screenshot": CHECK_SCREENSHOT_TOOL,  # ⭐ 添加到注册表
    # Note: 'computer' is handled separately as it's dynamic.
}

def get_tool_definitions(
    tool_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Assembles a list of tool definitions based on the provided names.

    Args:
        tool_names: A list of strings with the names of the tools to include.
        model_name: The name of the LLM, for selecting the correct tool version.
        platform: The operating system platform.

    Returns:
        A list of tool definition dictionaries for the API call.
    """
    definitions = []
    for name in tool_names:
        if name in TOOL_REGISTRY:
            definitions.append(TOOL_REGISTRY[name])
        else:
            # You can choose to raise an error or just log a warning
            print(f"Warning: Tool '{name}' not found in registry and is not a dynamic tool. Skipping.")
            # raise ValueError(f"Tool '{name}' not found in registry.")
    return definitions