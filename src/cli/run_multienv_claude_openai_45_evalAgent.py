"""Script to run end-to-end evaluation on the benchmark.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import datetime
import json
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import signal
import time
import traceback
from typing import List
from multiprocessing import Process, Manager, current_process
import contextvars
import lib_run_single
from lib_results_logger import log_task_error
from src.desktop_env.desktop_env import DesktopEnv
from ..agent.anthropic import AnthropicAgent45,AnthropicAgent45Eval
from src.desktop_env.custom_exception import ScreenshotIsNoneException, DockerStartException, SandboxConnectionError, AccessibilityTreeIsNoneException, TerminalOutputIsNoneException, FileTransferException, CommandExecutionException
from show_result_custom import print_result
from utils.metrics_monitor import aggregate_passk_results, start_metrics_reporter, stop_metrics_reporter
import shutil
import concurrent.futures
from src.desktop_env.controllers.python_sandbox import SandboxPythonController
from ..env.llm_eval import LLMEvaluator
from src.env import create_env

# Global variables for signal handling
active_environments = []
processes = []
is_terminating = False

# .env
from dotenv import load_dotenv
load_dotenv()
 
#  Logger Configs {{{ #
def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default="ANON")
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--action_space", type=str, default="claude_computer_use", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument(
        "--provider_name", type=str, default="docker", choices=["aws", "virtualbox", "vmware", "docker", "azure", "sandbox", 'sandbox-modelEval'], help="Provider name"
    )
    parser.add_argument(
        "--client_password", type=str, default="", help="Client password"
    )
    parser.add_argument(
        "--screen_width", type=int, default=1920, help="Screen width"
    )
    parser.add_argument(
        "--screen_height", type=int, default=1080, help="Screen height"
    )
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=100)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument(
        "--test_config_base_dir", type=str, default="ANON"
    parser.add_argument(
        "--only_n_most_recent_images", type=int, default=10
    )
    parser.add_argument(
        "--only_n_most_recent_images_eval", type=int, default=10
    )
    parser.add_argument(
        "--log_image_base64", type=bool, default=False
    )

    # lm config
    parser.add_argument("--model", type=str, default="aws.claude-sonnet-4.5-openai")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--base_url", type=str, default="ANON")
    parser.add_argument("--api_key", type=str, default="ANON")

    # thinking mode config
    parser.add_argument("--no-thinking", action="store_true", 
                       help="Disable thinking mode (no scratchpad)")
    parser.add_argument("--use-isp", action="store_true", 
                       help="Use interleaved scratchpad (ISP) mode")

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_one.json"
    )
    parser.add_argument(
        "--specific_task_id", type=str, default=None, 
        help="Run only a specific task ID (overrides domain filtering)"
    )

    # logging related
    parser.add_argument(
        "--replay", action="store_true", help="Run in replay mode"
    )
    parser.add_argument("--replay_dir", type=str, default="./results")
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")
    parser.add_argument("--cache_result_dir", type=str, default=None)
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")

    # periodic metrics
    parser.add_argument("--metrics_interval_min", type=int, default=30, help="定期(分钟)统计当前已完成指标；<=0 禁用")
    # pass@k inference config
    parser.add_argument(
        "--pass_k",
        type=int,
        default=None,
        help="为每个样本运行多次尝试以计算 pass@k；默认 None 表示单次尝试"
    )
    # aws config
    parser.add_argument(
        "--region", type=str, default="us-east-1", help="AWS region for the VM"
    )

    # api serving config
    parser.add_argument(
        "--customize_inference_ip", type=str, default="", help="api serving ip"
    )
    
    # docker lifecycle control
    parser.add_argument(
        "--keep_docker",
        action="store_true",
        help="If set, keep Docker environment after finishing; default is to close it."
    )
    parser.add_argument("--result-threshold", type=float, default=0.0, help="Minimum result value to be considered as a valid completion.")

    # env
    
    parser.add_argument("--env_url", type=str, default=None)
    parser.add_argument("--env_port", type=int, default=None)
    parser.add_argument("--env_manager_port", type=int, default=None)


    # LLM evaluation
    parser.add_argument("--env_type", type=str, default=None)
    parser.add_argument("--use_llm_evaluator", action="store_true", default=False)
    parser.add_argument("--use_dual_model_voting", action="store_true", default=False)
    parser.add_argument("--test_task_llm_eval", action="store_true", default=False)
    parser.add_argument("--api_type", type=str, default=None)
    parser.add_argument("--api_model", type=str, default=None)
    parser.add_argument("--api_base_url", type=str, default=None)
    # parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--eval_prompt_file", type=str, default=None)
    parser.add_argument("--eval_prompt_file2", type=str, default=None)
    parser.add_argument("--eval_prompt_dir", type=str, default=None)
    parser.add_argument("--llm_eval_temperature", type=float, default=0.6)
    parser.add_argument("--llm_eval_voting_type", type=str, default=None)
    parser.add_argument("--llm_eval_voting_num", type=int, default=1)
    parser.add_argument("--save_name", type=str, default='eval')
    parser.add_argument("--save_path", type=str, default='./results')

    # ⭐ 新增: CUA 评估检查参数
    parser.add_argument(
        "--check_cua_eval",
        action="store_true",
        default=True,
        help="Check CUA evaluation result (cua_eval_log.json) to determine if task is complete. Default: True"
    )
    parser.add_argument(
        "--no_check_cua_eval",
        dest="check_cua_eval",
        action="store_false",
        help="Disable CUA evaluation check"
    )

    
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_path, 'eval', args.save_name)
    os.makedirs(args.save_dir, exist_ok=True)

    return args

args = config()  # Get command line arguments first

# Validate that model is specified to prevent accidental usage with empty model
if not args.model or args.model.strip() == "":
    print("ERROR: Model must be specified. Use --model <model_name>")
    print("Example: --model claude-sonnet-4-5-20250929")
    sys.exit(1)

# Validate model support before proceeding
from ..agent.anthropic.utils import validate_model_support

# Pass same temperature/top_p and thinking parameters as will be used by the agent
validation_kwargs = {}
if args.temperature is not None:
    validation_kwargs['temperature'] = args.temperature
if args.top_p is not None:
    validation_kwargs['top_p'] = args.top_p
validation_kwargs['no_thinking'] = args.no_thinking
validation_kwargs['use_isp'] = args.use_isp

# if not validate_model_support(args.model, **validation_kwargs):
#     print(f"\n💥 Model '{args.model}' api sample failed")
#     sys.exit(1)

# Validate thinking mode options are mutually exclusive
if args.no_thinking and args.use_isp:
    print("ERROR: --no-thinking and --use-isp are mutually exclusive")
    print("Choose one of:")
    print("  (default): Regular scratchpad mode")
    print("  --no-thinking: Disable thinking/scratchpad")
    print("  --use-isp: Use interleaved scratchpad (ISP)")
    sys.exit(1)

# Task context for log records
current_task_id_var = contextvars.ContextVar("current_task_id", default="-")

class TaskIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.task_id = current_task_id_var.get()
        except Exception:
            record.task_id = "-"
        return True

class EvalIOFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # 匹配 [EVAL_IO] 标记的日志
        if "[EVAL_IO]" in record.getMessage() or "[GET_FILE]" in record.getMessage():
            return True
        
        # 匹配 evaluator 相关的日志
        evaluator_logger_patterns = [
            "desktopenv.metrics",
            "desktopenv.getters", 
            "desktopenv.metric",
            "desktopenv.getter",
            "desktop_env.evaluators.metrics",  # 包含使用__name__的模块
            "desktop_env.evaluators.getters"   # 包含使用__name__的模块
        ]
        
        for pattern in evaluator_logger_patterns:
            if pattern in record.name:
                return True
                
        return False

root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, args.log_level.upper()))
logger = logging.getLogger("desktopenv.experiment")


def setup_logger(action_space, use_model, observation_type, result_dir):
    """
    设置日志记录器,日志文件名包含时间戳以区分不同次运行
    
    Args:
        action_space: 动作空间
        use_model: 使用的模型
        observation_type: 观察类型
        result_dir: 结果目录
    """
    log_path = os.path.join(result_dir, action_space, observation_type, use_model)
    os.makedirs(log_path, exist_ok=True)

    # ⭐ 生成时间戳后缀
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ⭐ 创建带时间戳的日志文件名
    file_handler = logging.FileHandler(
        os.path.join(log_path, f"normal_{timestamp}.log"), 
        encoding="utf-8"
    )
    debug_handler = logging.FileHandler(
        os.path.join(log_path, f"debug_{timestamp}.log"), 
        encoding="utf-8"
    )
    error_handler = logging.FileHandler(
        os.path.join(log_path, f"error_{timestamp}.log"), 
        encoding="utf-8"
    )
    warning_handler = logging.FileHandler(
        os.path.join(log_path, f"warning_{timestamp}.log"), 
        encoding="utf-8"
    )
    eval_io_handler = logging.FileHandler(
        os.path.join(log_path, f"evaluation_{timestamp}.log"), 
        encoding="utf-8"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    error_handler.setLevel(logging.ERROR)
    warning_handler.setLevel(logging.WARNING)
    eval_io_handler.setLevel(logging.INFO)
    stdout_handler.setLevel(getattr(logging, args.log_level.upper()))

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(module)s:%(lineno)d-%(processName)s [task=%(task_id)s] %(message)s"
    )
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    warning_handler.setFormatter(formatter)
    eval_io_handler.setFormatter(formatter)

    # Always attach task id filter so every record has task_id
    task_filter = TaskIdFilter()
    eval_io_filter = EvalIOFilter()

    file_handler.addFilter(task_filter)
    debug_handler.addFilter(task_filter)
    stdout_handler.addFilter(task_filter)
    error_handler.addFilter(task_filter)
    warning_handler.addFilter(task_filter)
    eval_io_handler.addFilter(task_filter)
    eval_io_handler.addFilter(eval_io_filter)

    # 创建evaluator过滤器,用于排除evaluator相关日志避免重复记录
    evaluator_filter = EvalIOFilter()
    file_handler.addFilter(lambda record: "desktopagent" not in record.name and not evaluator_filter.filter(record))
    debug_handler.addFilter(lambda record: "desktopagent" not in record.name)
    stdout_handler.addFilter(lambda record: "desktopagent" in record.name or "desktopenv" in record.name)
    error_handler.addFilter(lambda record: "desktopagent" in record.name or "desktopenv" in record.name)
    warning_handler.addFilter(lambda record: "desktopagent" in record.name or "desktopenv" in record.name)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(warning_handler)
    root_logger.addHandler(eval_io_handler)
    
    # ⭐ 记录日志文件路径
    logger.info(f"Log files created with timestamp: {timestamp}")
    logger.info(f"  - Normal log: normal_{timestamp}.log")
    logger.info(f"  - Debug log: debug_{timestamp}.log")
    logger.info(f"  - Error log: error_{timestamp}.log")
    logger.info(f"  - Warning log: warning_{timestamp}.log")
    logger.info(f"  - Evaluation log: evaluation_{timestamp}.log")

def is_cua_eval_successful(example_result_dir: str) -> bool:
    """
    检查 CUA 评估结果是否成功解析
    
    Args:
        example_result_dir: 示例结果目录路径
        
    Returns:
        bool: 如果评估结果成功解析返回 True,否则返回 False
        
    Note:
        只检查评估结果是否成功解析,不关心任务是否成功完成。
        评估解析成功的条件:
        1. cua_eval_log.json 文件存在
        2. 文件是有效的 JSON 格式
        3. parse_method 不是 "failed" 或 "empty"
    """
    cua_eval_log_path = os.path.join(example_result_dir, "cua_eval_log.json")
    
    # 如果文件不存在,认为评估未完成
    if not os.path.exists(cua_eval_log_path):
        logger.debug(f"CUA eval log not found: {cua_eval_log_path}")
        return False
    
    try:
        with open(cua_eval_log_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        # 检查 evaluation_result 字段
        evaluation_result = eval_data.get("evaluation_result", {})
        parse_method = evaluation_result.get("parse_method", "")
        success = evaluation_result.get("success", False)
        confidence = evaluation_result.get("confidence", "UNKNOWN")
        
        # ⭐ 只检查 parse_method,不检查 success
        # parse_method 为 "failed" 或 "empty" 表示解析失败
        if parse_method in ["failed", "empty"]:
            logger.warning(
                f"CUA evaluation parse failed for {example_result_dir}: "
                f"parse_method={parse_method}"
            )
            return False
        
        # ⭐ parse_method 有效(full_format 或 partial_format)就认为解析成功
        # 不管任务本身是成功还是失败
        logger.debug(
            f"CUA evaluation parse successful for {example_result_dir}: "
            f"parse_method={parse_method}, task_success={success}, confidence={confidence}"
        )
        return True
            
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {cua_eval_log_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error reading {cua_eval_log_path}: {e}")
        return False

def is_attempt_completed_with_cua_eval(attempt_dir_path: str, result_threshold: float, check_cua_eval: bool = True) -> bool:
    """
    检查单次尝试是否有效且完成(包括 CUA 评估)
    
    Args:
        attempt_dir_path: 尝试目录路径
        result_threshold: 结果阈值
        check_cua_eval: 是否检查 CUA 评估结果
        
    Returns:
        bool: 如果完成返回 True,否则返回 False
    """
    # 首先检查基本完成条件
    if not is_attempt_completed(attempt_dir_path, result_threshold):
        return False
    
    # 如果需要检查 CUA 评估
    if check_cua_eval:
        return is_cua_eval_successful(attempt_dir_path)
    
    return True


def log_task_container_mapping(env, domain: str, example_id: str):
    """在任务开始时打印对应环境容器/端口等标识到 error 日志。"""
    docker_container_id = None
    try:
        if getattr(env, "provider_name", "") == "docker" and hasattr(env, "provider") and getattr(env.provider, "container", None):
            docker_container_id = getattr(env.provider.container, "short_id", None) or getattr(env.provider.container, "id", None)
    except Exception:
        docker_container_id = None

    logger.warning(
        (
            f"[TaskStart] {current_process().name} PID={os.getpid()} "
            f"domain={domain} example={example_id} provider={getattr(env, 'provider_name', 'unknown')} "
            f"vm_ip={getattr(env, 'vm_ip', 'unknown')} "
            f"ports(server={getattr(env, 'server_port', 'N/A')}, chrome={getattr(env, 'chromium_port', 'N/A')}, "
            f"vnc={getattr(env, 'vnc_port', 'N/A')}, vlc={getattr(env, 'vlc_port', 'N/A')}) "
            + (f"container={docker_container_id}" if docker_container_id else "")
        )
    )


def is_jsonl_all_lines_valid(file_path: str) -> bool:
    """严格 JSONL 校验：任一非空行无法被 json.loads 解析则判为无效；至少需要一行合法 JSON。"""
    import json
    if not os.path.exists(file_path):
        return False
    non_empty_seen = False
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                non_empty_seen = True
                try:
                    json.loads(s)
                except Exception:
                    logger.warning(f"Invalid JSONL at {file_path}:{idx}")
                    return False
    except Exception:
        logger.warning(f"Invalid JSONL file: {file_path}")
        return False
    return non_empty_seen


def is_attempt_completed(attempt_dir_path: str, result_threshold: float) -> bool:
    """
    Checks if a single attempt is valid and complete.
    - result.txt exists, is a valid float, and is >= threshold.
    - traj.jsonl exists and is a valid JSONL file.
    """
    if not os.path.isdir(attempt_dir_path):
        return False

    # Check result.txt
    result_file_path = os.path.join(attempt_dir_path, "result.txt")
    if not os.path.exists(result_file_path):
        return False

    try:
        with open(result_file_path, "r", encoding="utf-8") as f:
            result_value = float(f.read().strip())
        if result_value < result_threshold:
            return False
    except (ValueError, IOError):
        return False  # Not a valid float or unreadable

    # Check traj.jsonl
    if not is_jsonl_all_lines_valid(os.path.join(attempt_dir_path, "traj.jsonl")):
        return False

    return True


def maybe_aggregate_passk_for_example(args: argparse.Namespace, domain: str, example_id: str) -> None:
    """当 pass_k>1 且该 example 的所有 attempts 均已生成各自 result.txt，且父目录尚无聚合结果时，立即对该 example 聚合一次。"""
    try:
        if getattr(args, 'pass_k', None) and isinstance(args.pass_k, int) and args.pass_k > 1:
            example_result_dir_parent = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                args.model,
                domain,
                example_id,
            )
            parent_result_path = os.path.join(example_result_dir_parent, "result.txt")
            if os.path.exists(parent_result_path):
                return
            expected_names = [f"attempt_{i:02d}" for i in range(args.pass_k)]
            for name in expected_names:
                attempt_dir = os.path.join(example_result_dir_parent, name)
                if not os.path.isdir(attempt_dir):
                    return
                if not os.path.exists(os.path.join(attempt_dir, "result.txt")):
                    return
            single_meta = {domain: [example_id]}
            aggregate_passk_results(args, single_meta)
    except Exception as e:
        logger.error(f"单例 Pass@k 聚合失败 {domain}/{example_id}: {e}", exc_info=True)


def _remove_dir_worker(path):
    try:
        logger.info(f"Removing directory: {path}")
        shutil.rmtree(path)
    except FileNotFoundError:
        pass  # It's okay if it's already gone
    except OSError as e:
        logger.error(f"Failed to remove directory {path}: {e}")

def _recreate_dir_worker(path):
    try:
        logger.info(f"Clearing directory by removing and recreating: {path}")
        shutil.rmtree(path)
        os.makedirs(path)
    except FileNotFoundError:
        try:
            os.makedirs(path)
        except OSError as e:
            logger.error(f"Failed to recreate directory {path} after FileNotFoundError: {e}")
    except OSError as e:
        logger.error(f"Failed to clear directory {path}: {e}")


def distribute_tasks(test_all_meta: dict) -> List[tuple]:
    """Distribute tasks evenly across environments."""
    # Flatten the tasks into a single list
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    
    return all_tasks


def process_signal_handler(signum, frame, env_idx):
    """Signal handler for child processes to gracefully shut down their environments."""
    logger.info(f"Process {env_idx + 1} received signal {signum}. Shutting down...")
    
    # Get the active_environments from the caller's frame
    local_vars = frame.f_locals
    active_environments = local_vars.get('active_environments', [])
    
    # Close environment in the current process context
    for env in active_environments:
        if env is not None:
            try:
                logger.info(f"Process {env_idx + 1} closing environment...")
                env.close()
                logger.info(f"Process {env_idx + 1} environment closed successfully")
            except Exception as e:
                logger.error(f"Process {env_idx + 1} error closing environment: {e}")
    
    logger.info(f"Process {env_idx + 1} shutdown complete. Exiting.")
    sys.exit(0)


def run_env_tasks(task_queue: 'Queue', args: argparse.Namespace, shared_scores: list, shared_completed, total_tasks: int,i: int):
    global active_environments
    env = None
    try:

        if args.use_llm_evaluator:
            logging.debug("Initializing LLM Evaluator...")
            api_urls = args.api_base_url.split(',')
            api_url = api_urls[i % len(api_urls)]
            llm_evluator = LLMEvaluator(
                api_type=args.api_type,
                model=args.api_model,
                base_url=api_url,
                api_key=args.api_key,
                prompt_file=args.eval_prompt_file,
                prompt_dir=args.eval_prompt_dir,
                temperature=args.llm_eval_temperature,
                voting_type=args.llm_eval_voting_type,
                voting_num=args.llm_eval_voting_num,
                use_dual_model_voting=args.use_dual_model_voting
            )
        else:
            llm_evluator = None

        REGION = args.region
        screen_size = (args.screen_width, args.screen_height)
        
        def create_env_local():
            if args.env_type == 'osworld-claude':
                env, env_config = create_env(args, env_idx=i)
                return env
            elif args.env_type in ['sandbox', 'sandbox-modelEval', 'sandbox-CUAEvalAgent']:
                return DesktopEnv(
                    path_to_vm=args.path_to_vm,
                    action_space=args.action_space,
                    provider_name=args.provider_name,
                    region=REGION,
                    snapshot_name='ANON',
                    screen_size=screen_size,
                    headless=args.headless,
                    os_type="Ubuntu",
                    require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
                    # enable_proxy=True,
                    enable_proxy=False,
                    client_password=args.client_password
                )

        env = create_env_local()
        active_environments.append(env)
        
        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            
            attempt_idx = None
            try:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    domain, example_id, attempt_idx = item
                else:
                    domain, example_id = item[0], item[1]
            except Exception:
                domain, example_id = item[0], item[1]

            _task_token = None
            try:
                task_id_value = f"{domain}/{example_id}"
                if isinstance(attempt_idx, int):
                    task_id_value = f"{task_id_value}/attempt_{attempt_idx:02d}"
                _task_token = current_task_id_var.set(task_id_value)
            except Exception:
                _task_token = None

            try:
                print("===args.test_config_base_dir: ",args.test_config_base_dir)
                config_file = os.path.join(
                    args.test_config_base_dir, f"ANONPATH/evaluation_examples_Verified/examples/{domain}/{example_id}.json"
                )
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)
                
                logger.info(f"[{current_process().name}][Domain]: {domain}")
                logger.info(f"[{current_process().name}][Example ID]: {example_id}")
                logger.info(f"[{current_process().name}][Instruction]: {example['instruction']}")
                
                replay_result_dir = os.path.join(
                    args.replay_dir,
                    args.action_space,
                    args.observation_type,
                    args.model,
                    domain,
                    example_id,
                )

                example_result_dir_parent = os.path.join(
                    args.result_dir,
                    args.action_space,
                    args.observation_type,
                    args.model,
                    domain,
                    example_id,
                )
                
                if getattr(args, 'pass_k', None) and isinstance(attempt_idx, int):
                    attempt_dir_name = f"attempt_{attempt_idx:02d}"
                    example_result_dir = os.path.join(example_result_dir_parent, attempt_dir_name)
                else:
                    example_result_dir = example_result_dir_parent
                os.makedirs(example_result_dir, exist_ok=True)
                
                # update the save_dir to the example_result_dir
                if env and hasattr(env, 'controller') and isinstance(env.controller, SandboxPythonController):
                    evaluation_meta_path = os.path.join("evaluation_meta", domain, example_id)
                    os.makedirs(evaluation_meta_path, exist_ok=True)
                    env.controller.sandbox.extra_meta_data["evaluation_meta_path"] = evaluation_meta_path
                
                try:
                    # ⭐ 使用新的检查函数,包含 CUA 评估检查
                    check_cua_eval = getattr(args, 'check_cua_eval', True)  # 默认开启检查
                    if getattr(args, 'pass_k', None) and isinstance(attempt_idx, int):
                        if is_attempt_completed_with_cua_eval(example_result_dir, args.result_threshold, check_cua_eval):
                            shared_completed.append((domain, example_id, attempt_idx))
                            logger.info(f"[SkipAttempt] 已完成(包括CUA评估),跳过执行: {domain}/{example_id}#{attempt_idx+1}")
                            continue
                    else:
                        # 非 pass@k 模式也检查 CUA 评估
                        if is_attempt_completed_with_cua_eval(example_result_dir, args.result_threshold, check_cua_eval):
                            shared_completed.append((domain, example_id))
                            logger.info(f"[SkipTask] 已完成(包括CUA评估),跳过执行: {domain}/{example_id}")
                            continue
                except Exception:
                    pass

                def create_agent():
                    return AnthropicAgent45(
                        env=env,
                        model=args.model,
                        api_url=args.base_url,
                        api_key=args.api_key,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        action_space=args.action_space,
                        observation_type=args.observation_type,
                        max_trajectory_length=args.max_trajectory_length,
                        provider_name=args.provider_name,
                        # screen_width=args.screen_width,
                        # screen_height=args.screen_height,
                        screen_size=(args.screen_width, args.screen_height),
                        no_thinking=getattr(args, 'no_thinking', False),
                        use_isp=getattr(args, 'use_isp', False),
                        only_n_most_recent_images=args.only_n_most_recent_images
                    )
                def create_agent_eval():
                    return AnthropicAgent45Eval(
                        model=args.model,
                        api_url=args.base_url,
                        api_key=args.api_key,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        action_space=args.action_space,
                        observation_type=args.observation_type,
                        max_trajectory_length=args.max_trajectory_length,
                        provider_name=args.provider_name,
                        # screen_width=args.screen_width,
                        # screen_height=args.screen_height,
                        screen_size=(args.screen_width, args.screen_height),
                        no_thinking=getattr(args, 'no_thinking', False),
                        use_isp=getattr(args, 'use_isp', False),
                        only_n_most_recent_images=args.only_n_most_recent_images,
                        only_n_most_recent_images_eval=args.only_n_most_recent_images_eval
                    )

                agent = create_agent()
                eval_agent = create_agent_eval()

                retry_times = 3
                while retry_times > 0:
                    try:
                        log_task_container_mapping(env, domain, example_id)

                        if args.replay:
                            replay_success, replay_validation_success, eval_agent_success = lib_run_single.run_single_example_claude_CUAEvalAgent_replay(
                                agent, 
                                env, 
                                example, 
                                args.max_steps,
                                example["instruction"],
                                args,
                                replay_result_dir,
                                example_result_dir,
                                shared_scores, 
                                llm_evluator = llm_evluator,
                                eval_agent = eval_agent
                            )

                            # if replay_success and replay_validation_success:
                            if replay_success and replay_validation_success and eval_agent_success:
                                env.close()
                                break
                            else:
                                if env is not None:
                                    try:
                                        env.close()
                                        if env in active_environments:
                                            active_environments.remove(env)
                                    except Exception as close_e:
                                        logger.error(f"Error closing env before retry: {close_e}")
                                env = create_env_local()
                                if env is not None:
                                    active_environments.append(env)
                                agent = create_agent()
                                eval_agent = create_agent_eval()
                                shutil.rmtree(example_result_dir)
                                os.makedirs(example_result_dir, exist_ok=True)
                                lib_run_single.run_single_example_claude_CUAEvalAgent(
                                    agent,
                                    env,
                                    example,
                                    args.max_steps,
                                    example["instruction"],
                                    args,
                                    example_result_dir,
                                    shared_scores,
                                    llm_evluator = llm_evluator,
                                    eval_agent = eval_agent
                                )
                                env.close()
                                break
                        else:
                            lib_run_single.run_single_example_claude_CUAEvalAgent(
                                agent,
                                env,
                                example,
                                args.max_steps,
                                example["instruction"],
                                args,
                                example_result_dir,
                                shared_scores,
                                llm_evluator = llm_evluator,
                                eval_agent = eval_agent
                            )
                            env.close()
                            break
                            
                    except (ScreenshotIsNoneException, DockerStartException, SandboxConnectionError, AccessibilityTreeIsNoneException, TerminalOutputIsNoneException, FileTransferException, CommandExecutionException) as e:
                        import traceback
                        retry_times -= 1
                        logger.error(f"Handle Custom Exception: {type(e).__name__} in {current_process().name} {domain}/{example_id}: {e}")
                        
                        # try:
                        #     env.controller.end_recording(
                        #         os.path.join(example_result_dir, "recording.mp4")
                        #     )
                        # except Exception as rec_e:
                        #     logger.error(f"Failed to end recording: {rec_e}")

                        if retry_times > 0:
                            for file in os.listdir(example_result_dir):
                                try:
                                    os.remove(os.path.join(example_result_dir, file))
                                except Exception:
                                    pass
                            
                            try:
                                if env is not None:
                                    try:
                                        env.close()
                                        if env in active_environments:
                                            active_environments.remove(env)
                                    except Exception as close_e:
                                        logger.error(f"Error closing env before retry: {close_e}")
                                env = create_env_local()
                                if env is not None:
                                    active_environments.append(env)
                                agent = create_agent()
                                eval_agent = create_agent_eval()
                                logger.info(f"{current_process().name} {domain}/{example_id} retry_times={retry_times}, environment rebuilt and retrying after backoff")
                                time.sleep(60)
                            except Exception as reinit_e:
                                logger.error(f": {reinit_e}")
                                retry_times = 0
                        else:
                            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                                f.write(
                                    json.dumps(
                                        {"Error": f"{domain}/{example_id} - {type(e).__name__}: {str(e)}"}
                                    )
                                )
                                f.write(f"\nException: {traceback.format_exc()}")
                                f.write("\n")
                    except Exception as e:
                        import traceback
                        retry_times = 0
                        logger.error(f"Exception in {current_process().name} {domain}/{example_id}: {e}")
                        logger.error(traceback.format_exc())

                        # Log error to results.json
                        try:
                            example = {"id": example_id}  # Create minimal example dict for error logging
                            log_task_error(example, str(e), example_result_dir, args)
                        except Exception as log_e:
                            logger.error(f"Failed to log error to results.json: {log_e}")

                        # try:
                        #     env.controller.end_recording(
                        #         os.path.join(example_result_dir, "recording.mp4")
                        #     )
                        # except Exception as rec_e:
                        #     logger.error(f"Failed to end recording: {rec_e}")
                        with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                            f.write(
                                json.dumps(
                                    {"Error": f"{domain}/{example_id} - {e}"}
                                )
                            )
                            f.write(f"\nException: {traceback.format_exc()}")
                            f.write("\n")

                try:
                    if getattr(args, 'pass_k', None) and isinstance(attempt_idx, int):
                        shared_completed.append((domain, example_id, attempt_idx))
                    else:
                        shared_completed.append((domain, example_id))
                    completed_num = len(shared_completed)
                    postfix = f"#{attempt_idx+1}" if isinstance(attempt_idx, int) else ""
                    logger.info(f"[Progress] {completed_num}/{total_tasks} 已完成。最近完成: {domain}/{example_id}{postfix}")
                    
                    maybe_aggregate_passk_for_example(args, domain, example_id)
                except Exception as progress_e:
                    logger.error(f"更新进度失败: {progress_e}")
            except Exception as e:
                logger.error(f"Task-level error in {current_process().name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                try:
                    if getattr(args, 'pass_k', None) and isinstance(attempt_idx, int):
                        shared_completed.append((domain, example_id, attempt_idx))
                    else:
                        shared_completed.append((domain, example_id))
                    completed_num = len(shared_completed)
                    postfix = f"#{attempt_idx+1}" if isinstance(attempt_idx, int) else ""
                    logger.info(f"[Progress] {completed_num}/{total_tasks} 已完成。最近完成(异常): {domain}/{example_id}{postfix}")
                except Exception as progress_e:
                    logger.error(f"更新进度失败: {progress_e}")
            finally:
                try:
                    if _task_token is not None:
                        current_task_id_var.reset(_task_token)
                    else:
                        current_task_id_var.set("-")
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"Process-level error in {current_process().name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"{current_process().name} cleaning up environment...")
        try:
            if env:
                if args.provider_name == "docker" and getattr(args, "keep_docker", False):
                    logger.info(f"{current_process().name} keep_docker=True; skip closing environment to preserve Docker state")
                else:
                    env.close()
                    logger.info(f"{current_process().name} environment closed successfully")
        except Exception as e:
            logger.error(f"{current_process().name} error during environment cleanup: {e}")


def signal_handler(signum, frame):
    """Handle termination signals (SIGINT, SIGTERM) to gracefully shutdown environments."""
    global is_terminating, active_environments, processes
    
    # Avoid duplicate handling
    if is_terminating:
        return
    
    is_terminating = True
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    
    # Close all registered environments in the main process
    for env in active_environments:
        try:
            if args.provider_name == "docker" and getattr(args, "keep_docker", False):
                logger.info(f"keep_docker=True; skip closing environment to preserve Docker state")
            else:
                logger.info(f"Closing environment...")
                env.close()
                logger.info(f"Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
    
    # Send termination signal to all child processes first
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Sending termination signal to process {p.name}...")
                p.terminate()
            except Exception as e:
                logger.error(f"Error sending termination signal to process: {e}")
    
    # Allow a short time for processes to handle their own cleanup
    time.sleep(1)
    
    # Forcefully terminate any processes that didn't exit
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Forcefully terminating process {p.name}...")
                import signal as sig
                os.kill(p.pid, sig.SIGKILL)
            except Exception as e:
                logger.error(f"Error forcefully terminating process: {e}")
    
    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    global processes
    logger.info("Args: %s", args)
    base_tasks = distribute_tasks(test_all_meta)
    # Expand by attempts when pass_k enabled
    all_tasks = []
    if getattr(args, 'pass_k', None) and isinstance(args.pass_k, int) and args.pass_k and args.pass_k > 1:
        # 仅入队未完成 attempts
        for (domain, example_id) in base_tasks:
            example_result_dir_parent = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                args.model,
                domain,
                example_id,
            )
            for attempt_idx in range(args.pass_k):
                ad_path = os.path.join(example_result_dir_parent, f"attempt_{attempt_idx:02d}")
                
                if is_attempt_completed(ad_path, args.result_threshold):
                    continue
                
                all_tasks.append((domain, example_id, attempt_idx))
    else:
        all_tasks = base_tasks
    logger.info(f"Total tasks: {len(all_tasks)} (examples={len(base_tasks)}; attempts per example={args.pass_k if getattr(args, 'pass_k', None) else 1})")
    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        shared_completed = manager.list()
        total_tasks = len(all_tasks)

        for item in all_tasks:
            task_queue.put(item)
        
        num_envs = args.num_envs
        processes = []

        # 启动周期指标线程
        metrics_handle = start_metrics_reporter(args, test_all_meta, shared_completed, total_tasks)

        for i in range(num_envs):
            p = Process(
                target=run_env_tasks,
                args=(task_queue, args, shared_scores, shared_completed, total_tasks,i),
                name=f"EnvProcess-{i+1}"
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info(f"Started process {p.name} with PID {p.pid}")
        try:
            while True:
                alive_count = 0
                for idx, p in enumerate(processes):
                    if not p.is_alive():
                        if not task_queue.empty():
                            logger.error(f"Process {p.name} died, restarting...")
                            new_p = Process(
                                target=run_env_tasks,
                                args=(task_queue, args, shared_scores, shared_completed, total_tasks,i),
                                name=f"EnvProcess-Restart-{idx+1}"
                            )
                            new_p.daemon = True
                            new_p.start()
                            processes[idx] = new_p
                            logger.info(f"Restarted process {new_p.name} with PID {new_p.pid}")
                        else:
                            logger.info(f"Process {p.name} done, Task Queue is Empty")
                    else:
                        # logger.info(f"Process {p.name} is alive")
                        alive_count += 1
                        
                if task_queue.empty():
                    completed_num = len(shared_completed)
                    if completed_num >= total_tasks:
                        logger.info("All tasks finished")
                        break

                if alive_count == 0:
                    logger.error("All processes died, exiting.")
                    break
                time.sleep(5)
            
            # Graceful shutdown with timeout
            logger.info("All tasks have been processed. Waiting for worker processes to terminate...")
            for p in processes:
                p.join(timeout=60) # Wait for 60 seconds

            # Force terminate any processes that are still alive
            for p in processes:
                if p.is_alive():
                    logger.warning(f"Process {p.name} did not terminate gracefully. Forcing termination.")
                    p.terminate()
                    p.join(timeout=5) # Wait a bit for termination to complete
                    if p.is_alive():
                        logger.error(f"Could not terminate process {p.name}. It might be stuck.")
                        os.kill(p.pid, signal.SIGKILL)

        except KeyboardInterrupt:
            logger.info("Main process received KeyboardInterrupt. Initiating graceful shutdown...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while waiting for processes: {e}", exc_info=True)
            for p in processes:
                if p.is_alive():
                    try:
                        logger.info(f"Terminating process {p.name} due to error...")
                        p.terminate()
                    except Exception as term_e:
                        logger.error(f"Error terminating process {p.name}: {term_e}")
            raise
        finally:
            # 停止周期指标线程
            stop_metrics_reporter(metrics_handle)

        scores = list(shared_scores)
    # logger.info(f"Average score: {sum(scores) / len(scores) if scores else 0}")
    sum_scores = sum(scores)
    len_scores = len(scores)
    logger.info(f"Average score: {sum_scores}/{len_scores}={sum_scores / len_scores if scores else 0}")

def find_unknown_confidence_simple(root_folder):
    target_filename = "cuaEvalAgent_result.txt"
    
    if not os.path.exists(root_folder):
        logger.info(f"错误: 找不到路径 '{root_folder}'")
        return

    logger.info(f"开始在 {root_folder} 下扫描...\n")

    all_list = []
    pass_list = []
    path_list = []

    domains = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])
    
    for domain in domains:
        domain_path = os.path.join(root_folder, domain)
        tasks = [t for t in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, t))]
        
        for task in tasks:
            dirpath = os.path.join(domain_path, task)

            # if 'runtime.log' in filenames:
            file_path = os.path.join(dirpath, target_filename)
            all_list.append(os.path.abspath(dirpath))
            if os.path.exists(file_path) and os.path.exists(os.path.join(dirpath, "result.txt")):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    if "UNKNOWN" not in content:
                        pass_list.append(os.path.abspath(dirpath))
                        continue

            logger.info(os.path.abspath(dirpath))
            path_list.append(os.path.abspath(dirpath))
    
    logger.info(f'所有文件夹数量: {len(all_list)}')
    logger.info(f'成功验证数量: {len(pass_list)}')
    logger.info(f'未成功验证数量: {len(path_list)}')

    for path in path_list:
        if os.path.exists(path):
            # rmtree = remove directory tree (递归删除文件夹及其中所有内容)
            shutil.rmtree(path)
            logger.info(f"已删除: {path}")
        else:
            logger.info("文件夹不存在")


    # random.seed(2026)
    # random.shuffle(pass_list)
    # delete_wrong_count = 0
    # for dirpath in pass_list:
    #     result_path = os.path.join(dirpath, "result.txt")
    #     if os.path.exists(result_path):
    #         with open(result_path, 'r', encoding='utf-8') as f:
    #             result = float(f.read().strip())
            
    #             if result < 0.5:
    #                 shutil.rmtree(dirpath)
    #                 logger.info(f"错误轨迹已删除: {dirpath}")
    #                 delete_wrong_count += 1

    #                 if delete_wrong_count == 10:
    #                     break

def get_unfinished(
    action_space, use_model, observation_type, result_dir, total_file_json, result_threshold=0.0, check_cua_eval=True
):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    find_unknown_confidence_simple(target_dir)

    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    dirs_to_delete = []
    dirs_to_recreate = []

    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    entries = os.listdir(example_path)
                    # Detect pass@k by presence of attempt_* subdirectories
                    attempt_subdirs = [d for d in entries if d.startswith("attempt_") and os.path.isdir(os.path.join(example_path, d))]
                    if attempt_subdirs:
                        # 若父目录已有聚合结果，则禁用任何删除操作并直接标记完成
                        parent_has_aggregate = ("result.txt" in entries) or ("passk_meta.json" in entries)
                        
                        is_all_attempt_subdirs_valid = True
                        for ad in attempt_subdirs:
                            ad_path = os.path.join(example_path, ad)
                            # if not is_attempt_completed(ad_path, result_threshold):
                            #     logger.warning(f"Invalid attempt subdirectory: {ad_path}")
                            #     is_all_attempt_subdirs_valid = False
                            #     if not parent_has_aggregate:
                            #         dirs_to_delete.append(ad_path)
                            # ⭐ 使用新的检查函数,包含 CUA 评估检查
                            if not is_attempt_completed_with_cua_eval(ad_path, result_threshold, check_cua_eval):
                                logger.warning(f"Invalid attempt subdirectory (or CUA eval failed): {ad_path}")
                                is_all_attempt_subdirs_valid = False
                                if not parent_has_aggregate:
                                    dirs_to_delete.append(ad_path)

                        # Pass@k mode specific checks
                        is_pass_k_mode = getattr(args, 'pass_k', None) and isinstance(args.pass_k, int) and args.pass_k > 1
                        all_attempts_completed = is_pass_k_mode and is_all_attempt_subdirs_valid and (len(attempt_subdirs) == args.pass_k)

                        # State Correction: If state is inconsistent, fix it.
                        if parent_has_aggregate and not all_attempts_completed:
                            logger.warning(f"Inconsistent state for {example_path}: Found stale aggregate result but attempts are incomplete or invalid. Clearing stale aggregate.")
                            parent_result_path = os.path.join(example_path, "result.txt")
                            parent_meta_path = os.path.join(example_path, "passk_meta.json")
                            try:
                                if os.path.exists(parent_result_path): os.remove(parent_result_path)
                                if os.path.exists(parent_meta_path): os.remove(parent_meta_path)
                            except OSError as e:
                                logger.error(f"Failed to remove stale aggregate files for {example_path}: {e}")
                            parent_has_aggregate = False # Reflect the change

                        # Decision: Mark as finished or leave for re-run.
                        if all_attempts_completed:
                            finished[domain].append(example_id)
                            if not parent_has_aggregate:
                                # All attempts are done, but aggregate is missing. Let's create it.
                                logger.info(f"Attempts for {example_path} are complete. Generating aggregate result now.")
                                try:
                                    maybe_aggregate_passk_for_example(args, domain, example_id)
                                except Exception as e:
                                    logger.error(f"On-the-fly aggregation for {example_path} failed: {e}")
                    else:
                        # Non pass@k: keep original behavior 
                        result_ok = False
                        if "result.txt" in entries:
                            try:
                                with open(os.path.join(example_path, "result.txt"), "r") as f:
                                    result_value = float(f.read().strip())
                                if result_value >= result_threshold:
                                    result_ok = True
                                else:
                                    logger.warning(f"Invalid result value: {result_value} in {example_path}")
                            except (ValueError, IOError):
                                pass

                        # ⭐ 检查 traj.jsonl 和 CUA 评估结果
                        traj_valid = is_jsonl_all_lines_valid(os.path.join(example_path, "traj.jsonl")) or os.path.exists(os.path.join(example_path, "replay.jsonl"))
                        cua_eval_ok = is_cua_eval_successful(example_path) if check_cua_eval else True

                        # if (not result_ok) or (not is_jsonl_all_lines_valid(os.path.join(example_path, "traj.jsonl"))):
                        #     # empty all files under example_id
                        #     dirs_to_recreate.append(example_path)
                        # else:
                        #     finished[domain].append(example_id)
                        if (not result_ok) or (not traj_valid) or (not cua_eval_ok):
                            # ⭐ 如果 CUA 评估失败,记录原因
                            if not cua_eval_ok:
                                logger.warning(f"CUA evaluation failed for {example_path}, will re-run")
                            # empty all files under example_id
                            dirs_to_recreate.append(example_path)
                        else:
                            finished[domain].append(example_id)

    if dirs_to_delete or dirs_to_recreate:
        # Cap workers to prevent disk thrashing.
        max_workers = min(16, (os.cpu_count() or 1) * 2)
        logger.info(f"Starting parallel cleanup of {len(dirs_to_delete)} directories to delete and {len(dirs_to_recreate)} to recreate using up to {max_workers} threads.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            remove_futures = {executor.submit(_remove_dir_worker, path) for path in dirs_to_delete}
            recreate_futures = {executor.submit(_recreate_dir_worker, path) for path in dirs_to_recreate}

            for future in concurrent.futures.as_completed(remove_futures | recreate_futures):
                try:
                    future.result()  # retrieve result to propagate exceptions
                except Exception as exc:
                    logger.error(f'A cleanup task generated an exception: {exc}')
        logger.info("Parallel cleanup finished.")

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        # empty all files under example_id
                        try:
                            all_result.append(
                                float(
                                    open(
                                        os.path.join(example_path, "result.txt"), "r"
                                    ).read()
                                )
                            )
                        except:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == "__main__":
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Register signal handlers for graceful termination
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal
    
    try:
        args = config()
        
        # save args to json in result_dir/action_space/observation_type/model/args.json
        # args.result_dir = f"./results_log/{args.result_dir}"
        path_to_args = os.path.join(
            args.result_dir,
            args.action_space,
            args.observation_type,
            args.model,
            "args.json",
        )
        os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
        with open(path_to_args, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4)

        with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
            test_all_meta = json.load(f)

        # if args.domain != "all":
        # Filter for specific task ID if provided
        if args.specific_task_id:
            logger.info(f"Filtering for specific task ID: {args.specific_task_id}")
            filtered_meta = {}
            task_found = False

            for domain, task_ids in test_all_meta.items():
                for task_id in task_ids:
                    if task_id == args.specific_task_id:
                        filtered_meta[domain] = [task_id]
                        task_found = True
                        logger.info(f"Found task {args.specific_task_id} in domain: {domain}")
                        break
                if task_found:
                    break

            if not task_found:
                logger.error(f"Task ID {args.specific_task_id} not found in test file!")
                sys.exit(1)

            test_all_meta = filtered_meta
        elif args.domain != "all":
            test_all_meta = {args.domain: test_all_meta[args.domain]}

        setup_logger(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir
        )

        test_file_list = get_unfinished(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            test_all_meta,
            args.result_threshold,
            check_cua_eval=args.check_cua_eval,  # 新增参数
        )
        left_info = ""
        for domain in test_file_list:
            left_info += f"{domain}: {len(test_file_list[domain])}\n"
        logger.info(f"Left tasks (after CUA eval check):\n{left_info}")

        get_result(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            test_all_meta,
        )
        test(args, test_file_list)

        # 若启用 pass@k，则在所有尝试完成后聚合结果（对全量清单聚合，而非仅未完成子集）
        try:
            if getattr(args, 'pass_k', None) and isinstance(args.pass_k, int) and args.pass_k > 1:
                aggregate_passk_results(args, test_all_meta)
        except Exception as e:
            logger.error(f"Pass@k 聚合失败: {e}")

        eval_result_path = os.path.join(args.result_dir, args.action_space, args.observation_type, args.model)
        print_result(eval_result_path, logger=logger)
    except KeyboardInterrupt:
        logger.info("Main process received KeyboardInterrupt.")
        # Signal handler will take care of cleanup
    except Exception as e:
        logger.error(f"Unexpected error in main process: {e}", exc_info=True)
        # Also trigger cleanup for unhandled exceptions
        signal_handler(signal.SIGTERM, None)
    finally:
        # Final cleanup in case any environments or processes remain
        logger.info("Main process final cleanup...")
        for env in active_environments:
            if env is not None:
                try:
                    if args.provider_name == "docker" and getattr(args, "keep_docker", False):
                        logger.info(f"keep_docker=True; skip closing environment in final cleanup to preserve Docker state")
                    else:
                        logger.info(f"Closing environment in final cleanup...")
                        env.close()
                        logger.info(f"Environment closed successfully in final cleanup")
                except Exception as e:
                    logger.error(f"Error during final environment cleanup: {e}")
        
        # First try gentle termination
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    logger.info(f"Terminating process {p.name}...")
                    p.terminate()
                except Exception as e:
                    logger.error(f"Error terminating process: {e}")
        
        # Wait a moment for processes to terminate
        time.sleep(1)
        
        # Then force kill if needed
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    logger.info(f"Force killing process {p.name}...")
                    os.kill(p.pid, signal.SIGKILL)
                    logger.info(f"Process {p.name} force killed")
                except Exception as e:
                    logger.error(f"Error force killing process: {e}")