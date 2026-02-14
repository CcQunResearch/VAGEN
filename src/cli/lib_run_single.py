import datetime
import json
import logging
import os
import time
import copy
# import shutil
from wrapt_timeout_decorator import *
# from desktop_env.desktop_env import DesktopEnv
from src.desktop_env.desktop_env import DesktopEnv
from utils.multiturn_exporter import export_from_collected_if_success
from lib_results_logger import log_task_completion

logger = logging.getLogger("desktopenv.experiment")

def evaluate(task_config=None, trajectory=None,llm_evluator=None,save_dir=None):
    actions = trajectory["actions"]
    if task_config["evaluator"]["func"] == "infeasible":
        if len(actions) > 0 and actions[-1] == "FAIL":
            # return {"reward": 1, "note": "infeasible task"}
            return 1
        else:
            # return {"reward": 0, "note": "infeasible task"}
            return 0
    else:
        if len(actions) > 0 and actions[-1] == "FAIL":
            # return {"reward": 0, "note": "FAIL action"}
            return 0

    # llm eval
    print("===task_config: ",task_config)
    if task_config is not None and task_config["evaluator"]["func"] == "LLM":
        try:
            return llm_evluator.evaluate_task(task_config, trajectory,save_dir=save_dir)
        except Exception as e:
            print("Evaluation failed, return -1.")
            print(e)
            logger.info(f"===error: Evaluation failed, return -1. e: {e}")

            # return {"reward": -1}
            return -1

def save_detailed_evaluation_result(result: dict, output_file: str):
    """保存详细评估结果到文本文件
    
    支持的模式：
    1. 单模型单次评估
    2. 单模型多次投票
    3. 双模型投票
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("LLM EVALUATION RESULT\n")
        f.write("="*60 + "\n\n")
        
        # ===== 检测评估模式 =====
        evaluation_mode = result.get('evaluation_mode', 'unknown')
        is_dual_model = 'model1' in result and 'model2' in result
        
        # ===== 元数据 =====
        if 'metadata' in result:
            metadata = result['metadata']
            f.write(f"Evaluation Time: {metadata.get('timestamp', 'N/A')}\n")
            f.write(f"Instruction: {metadata.get('instruction', 'N/A')}\n")
            f.write(f"Number of Screenshots: {metadata.get('num_screenshots', 'N/A')}\n")
            f.write("\n")
        
        # ===== 模型信息 =====
        if 'model_info' in result:
            model_info = result['model_info']
            f.write("Model Configuration:\n")
            f.write(f"  Model: {model_info.get('name', 'N/A')}\n")
            f.write(f"  Parse Mode: {model_info.get('parse_mode', 'N/A')}\n")
            f.write(f"  Temperature: {model_info.get('temperature', 'N/A')}\n")
            if 'voting_num' in model_info:
                f.write(f"  Voting Number: {model_info.get('voting_num')}\n")
            f.write("\n")
        
        # ===== 最终结果 =====
        f.write("="*60 + "\n")
        f.write("FINAL RESULT\n")
        f.write("="*60 + "\n")
        f.write(f"Evaluation Mode: {evaluation_mode}\n")
        f.write(f"Final Reward: {result.get('reward', 'N/A')}\n")
        
        # ===== 模式1：双模型投票 =====
        if is_dual_model:
            f.write(f"Voting Type: {result.get('voting_type', 'N/A')}\n")
            f.write(f"All Votes: {result.get('all_rewards', [])}\n")
            f.write("\n")
            
            # 模型1结果
            if 'model1' in result:
                model1 = result['model1']
                f.write("="*60 + "\n")
                f.write(f"MODEL 1: {model1.get('name', 'N/A')}\n")
                f.write("="*60 + "\n")
                f.write(f"Parse Mode: {model1.get('parse_mode', 'N/A')}\n")
                f.write(f"Rewards: {model1.get('rewards', [])}\n")
                f.write("\n")
                
                analyses = model1.get('analyses', [])
                for i, analysis in enumerate(analyses, 1):
                    f.write(f"--- Sample {i} ---\n")
                    f.write(f"{analysis}\n\n")
                
                detailed_results = model1.get('detailed_results', [])
                if detailed_results:
                    f.write("Detailed Results:\n")
                    for i, detail in enumerate(detailed_results, 1):
                        f.write(f"\nSample {i} Details:\n")
                        f.write(json.dumps(detail, indent=2, ensure_ascii=False))
                        f.write("\n")
                f.write("\n")
            
            # 模型2结果
            if 'model2' in result:
                model2 = result['model2']
                f.write("="*60 + "\n")
                f.write(f"MODEL 2: {model2.get('name', 'N/A')}\n")
                f.write("="*60 + "\n")
                f.write(f"Parse Mode: {model2.get('parse_mode', 'N/A')}\n")
                f.write(f"Rewards: {model2.get('rewards', [])}\n")
                f.write("\n")
                
                analyses = model2.get('analyses', [])
                for i, analysis in enumerate(analyses, 1):
                    f.write(f"--- Sample {i} ---\n")
                    f.write(f"{analysis}\n\n")
                
                detailed_results = model2.get('detailed_results', [])
                if detailed_results:
                    f.write("Detailed Results:\n")
                    for i, detail in enumerate(detailed_results, 1):
                        f.write(f"\nSample {i} Details:\n")
                        f.write(json.dumps(detail, indent=2, ensure_ascii=False))
                        f.write("\n")
                f.write("\n")
            
            # 投票详情
            f.write("="*60 + "\n")
            f.write("VOTING DETAILS\n")
            f.write("="*60 + "\n")
            f.write(f"Model 1 Votes: {result.get('model1', {}).get('rewards', [])}\n")
            f.write(f"Model 2 Votes: {result.get('model2', {}).get('rewards', [])}\n")
            f.write(f"Combined Votes: {result.get('all_rewards', [])}\n")
            f.write(f"Voting Strategy: {result.get('voting_type', 'N/A')}\n")
            f.write(f"Final Reward: {result.get('reward', 'N/A')}\n")
            f.write("="*60 + "\n")
        
        # ===== 模式2：单模型投票 =====
        elif evaluation_mode == "single_model_voting":
            f.write(f"Voting Type: {result.get('voting_type', 'N/A')}\n")
            f.write(f"Voting Rewards: {result.get('voting_rewards', [])}\n")
            f.write("\n")
            
            # ⭐⭐⭐ 保存所有样本的详细分析
            llm_outputs = result.get('llm_outputs', [])
            if isinstance(llm_outputs, list):
                f.write("="*60 + "\n")
                f.write("DETAILED ANALYSES\n")
                f.write("="*60 + "\n\n")
                
                for i, analysis in enumerate(llm_outputs, 1):
                    f.write(f"{'='*60}\n")
                    f.write(f"Sample {i}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"{analysis}\n\n")
            
            # ⭐ 保存详细结果（如果有）
            detailed_results = result.get('detailed_results', [])
            if detailed_results:
                f.write("="*60 + "\n")
                f.write("STRUCTURED RESULTS\n")
                f.write("="*60 + "\n\n")
                
                for i, detail in enumerate(detailed_results, 1):
                    f.write(f"Sample {i} Structured Data:\n")
                    f.write(json.dumps(detail, indent=2, ensure_ascii=False))
                    f.write("\n\n")
        
        # ===== 模式3：单模型单次评估 =====
        elif evaluation_mode == "single_model_single":
            f.write("\n")
            
            # ⭐⭐⭐ 保存完整的 LLM 分析
            llm_output = result.get('llm_output', '')
            if llm_output:
                f.write("="*60 + "\n")
                f.write("LLM ANALYSIS\n")
                f.write("="*60 + "\n")
                f.write(f"{llm_output}\n\n")
            
            # ⭐ 保存详细结果（如果有）
            detailed_result = result.get('detailed_result')
            if detailed_result:
                f.write("="*60 + "\n")
                f.write("STRUCTURED RESULT\n")
                f.write("="*60 + "\n")
                f.write(json.dumps(detailed_result, indent=2, ensure_ascii=False))
                f.write("\n\n")
        
        # ===== 模式4：错误情况 =====
        elif evaluation_mode == "error":
            f.write("\n")
            f.write("="*60 + "\n")
            f.write("ERROR DETAILS\n")
            f.write("="*60 + "\n")
            f.write(f"Error: {result.get('error', 'Unknown error')}\n")
        
        # ===== 未知模式（兼容旧代码） =====
        else:
            f.write("\n")
            f.write("="*60 + "\n")
            f.write("RAW RESULT\n")
            f.write("="*60 + "\n")
            f.write(json.dumps(result, indent=2, ensure_ascii=False))
            f.write("\n")
    
    logger.info(f"详细评估结果已保存到: {output_file}")


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    try:
        agent.reset(runtime_logger)
    except Exception as e:
        agent.reset()

    env.reset(task_config=example)
    
    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        for action in actions:
            # Capture the timestamp before executing the action
            # action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    # Log task completion to results.json
    log_task_completion(example, result, example_result_dir, args)
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))

def run_single_example_claude_CUAEvalAgent(agent, env: DesktopEnv, example, max_steps, instruction, args, example_result_dir, scores, llm_evluator, eval_agent, max_round=40):
    with open(os.path.join(example_result_dir, "task_config.json"), "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)

    runtime_logger = setup_logger(example, example_result_dir)
    try:
        agent.reset(runtime_logger)
    except Exception as e:
        agent.reset()

    env.reset(task_config=example)
    # if env.provider_name != "sandbox":
    # env.controller.start_recording()
    
    time.sleep(10) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    with open(os.path.join(example_result_dir, f"step_0_{action_timestamp}.png"),
            "wb") as _f:
        _f.write(obs['screenshot'])
        obs["screenshot_name"] = f"step_0_{action_timestamp}.png"
    done = False
    step_idx = 0
    example_id = os.path.basename(example_result_dir)
    per_step_info_dicts = []
    traj_for_eval = {"screenshots": [], "actions": []}
    while not done and step_idx < max_steps:
        obs["example_id"] = example_id
        response, actions, info_dict = agent.predict(
            instruction,
            obs
        )
        per_step_info_dicts.append(info_dict)
        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
                obs["screenshot_name"] = f"step_{step_idx + 1}_{action_timestamp}.png"
            traj_for_eval["screenshots"].append(obs['screenshot'])
            traj_for_eval["actions"].append(action)
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    if args.env_type == 'osworld-claude':
        result = env.evaluate(task_config=example, trajectory=traj_for_eval)
        reward_value = result
    elif args.env_type == 'sandbox':
        result = env.evaluate()
        reward_value = result
    elif args.env_type == 'sandbox-modelEval':
        result = evaluate(
            task_config=example, 
            trajectory=traj_for_eval,
            llm_evluator=llm_evluator,
            save_dir=example_result_dir
        )
    elif args.env_type == 'sandbox-CUAEvalAgent':
        result = env.evaluate()
        reward_value = result
        ######cuaEvalAgent######
        cuaEvalAgent_result = eval_agent.eval_task(
            env = env,
            task_config=example, 
            trajectory=traj_for_eval,
            save_dir=example_result_dir,
            max_round=max_round
        )
        ######cuaEvalAgent######
         
        
    logger.info("reward_value: %.2f", reward_value)
    
    # 保存简单的 reward 值到 result.txt
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{reward_value}\n")
    with open(os.path.join(example_result_dir, "cuaEvalAgent_result.json"), "w", encoding="utf-8") as f:
        json.dump(cuaEvalAgent_result, f, indent=2, ensure_ascii=False)
    with open(os.path.join(example_result_dir, "cuaEvalAgent_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{cuaEvalAgent_result}\n")
    
    scores.append(reward_value)
    
    try:
        export_from_collected_if_success(
            output_dir=example_result_dir,
            info_dicts=per_step_info_dicts,
            result_value=reward_value,
            success_threshold=-2.0,
            filename="multi_turn.jsonl",
        )
    except Exception as _e:
        logger.error(f"session export failed: {_e}")


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, "runtime.log")))
    return runtime_logger

def run_single_example_claude_CUAEvalAgent_replay(
    agent, 
    env: DesktopEnv, 
    example, 
    max_steps, 
    instruction, 
    args,
    replay_result_dir,
    example_result_dir,
    scores, 
    llm_evluator, 
    eval_agent, 
    max_round=35,
    domain=None,
    example_id=None
):  
    replay_success, replay_validation_success = False, False

    result_path = os.path.join(replay_result_dir, "result.txt")
    if not os.path.exists(result_path):
        logger.error(f"结果文件不存在: {result_path}")
        return replay_success, replay_validation_success, False
    
    traj_file = os.path.join(replay_result_dir, "traj.jsonl")
    if not os.path.exists(traj_file):
        logger.error(f"轨迹文件不存在: {traj_file}")
        return replay_success, replay_validation_success, False
    
    logger.info("=" * 50)
    logger.info(f"开始回放任务: {os.path.basename(replay_result_dir)}")
    logger.info(f"任务配置: {example.get('id', 'N/A')}")
    logger.info(f"指令: {instruction}")
    logger.info(f"轨迹文件: {traj_file}")
    logger.info("=" * 50)

    logger.info("=" * 50)
    logger.info("阶段1：初始化环境")
    logger.info("=" * 50)
    
    env.reset(task_config=example)
    time.sleep(10)

    logger.info("=" * 50)
    logger.info("阶段2：开始回放轨迹")
    logger.info("=" * 50)
    
    replay_success = False
    try:
        replay_success = replay_trajectory_from_jsonl(
            env=env,
            replay_result_dir=replay_result_dir,
            example_result_dir=example_result_dir,
            task_config=example,
            args=args,
            verify_screenshots=True
        )
    except Exception as e:
        logger.error(f"轨迹回放失败: {e}", exc_info=True)
        replay_success = False
    
    if not replay_success:
        logger.error("轨迹回放失败，无法进行评测")
        with open(os.path.join(example_result_dir, "replay_failed.txt"), "w") as f:
            f.write("Replay failed\n")
        return replay_success, replay_validation_success, False

    logger.info("=" * 50)
    logger.info("阶段2.5：回放后评估验证")
    logger.info("=" * 50)
    
    try:
        # 读取原始评测结果
        original_result_file = os.path.join(replay_result_dir, "result.txt")
        original_result = None
        
        if os.path.exists(original_result_file):
            with open(original_result_file, "r", encoding="utf-8") as f:
                original_result = float(f.read().strip())
        else:
            return replay_success, replay_validation_success, False
            logger.warning(f"原始评测结果文件不存在: {original_result_file}")
        
        # 评估回放后的状态
        logger.info("评估回放后的状态...")
        replay_result = None
        
        # if args.env_type == 'osworld-claude':
        #     traj_for_eval = build_eval_trajectory_from_jsonl(example_result_dir)
        #     replay_result = env.evaluate(task_config=example, trajectory=traj_for_eval)
        # elif args.env_type in ['sandbox', 'sandbox-modelEval', 'sandbox-CUAEvalAgent']:
        #     replay_result = env.evaluate()
        replay_result = env.evaluate()
        
        logger.info(f"回放后评测结果: {replay_result}")
        
        # 保存回放后的评测结果
        replay_result_file = os.path.join(example_result_dir, "result.txt")
        with open(replay_result_file, "w", encoding="utf-8") as f:
            f.write(f"{replay_result}\n")
        logger.info(f"保存回放评测结果到: {replay_result_file}")
        
        # ⭐ 比较结果一致性
        if original_result is not None:
            # 允许小的浮点误差
            result_match = bool(abs(original_result - replay_result) < 0.1)
            
            validation_report = {
                "original_result": original_result,
                "replay_result": replay_result,
                "results_match": result_match,
                "difference": abs(original_result - replay_result),
                "validation_timestamp": datetime.datetime.now().isoformat()
            }
            
            # 保存验证报告
            validation_report_file = os.path.join(example_result_dir, "validation_report.json")
            with open(validation_report_file, "w", encoding="utf-8") as f:
                json.dump(validation_report, f, indent=2, ensure_ascii=False)
            
            if result_match:
                logger.info("✓ 回放验证成功：评测结果一致")
                replay_validation_success = True
            else:
                logger.error(
                    f"✗ 回放验证失败：评测结果不一致\n"
                    f"  原始结果: {original_result}\n"
                    f"  回放结果: {replay_result}\n"
                    f"  差异: {abs(original_result - replay_result)}"
                )
                
                # 标记回放失败
                with open(os.path.join(example_result_dir, "replay_validation_failed.txt"), "w") as f:
                    f.write(f"Replay validation failed\n")
                    f.write(f"Original result: {original_result}\n")
                    f.write(f"Replay result: {replay_result}\n")
                    f.write(f"Difference: {abs(original_result - replay_result)}\n")
        else:
            logger.warning("无法进行结果比较：原始评测结果不可用")
            # 如果没有原始结果，只要回放评估成功就认为验证通过
            return replay_success, replay_validation_success, False
    
    except Exception as e:
        logger.error(f"回放后评估失败: {e}", exc_info=True)
        # 保存评估失败信息
        with open(os.path.join(example_result_dir, "replay_eval_failed.txt"), "w") as f:
            f.write(f"Replay evaluation failed: {e}\n")
    
    # ⭐ 如果回放验证失败，可以选择是否继续CUA评测
    if not replay_validation_success:
        logger.warning("回放验证失败，但将继续进行CUA评测")
        return replay_success, replay_validation_success, False
    
    logger.info("=" * 50)
    logger.info("阶段3：开始执行CUA评测")
    logger.info("=" * 50)
    
    reward_value = 0
    try:
        if args.env_type == 'sandbox-CUAEvalAgent':
            # 使用回放后的评测结果作为基础评测
            reward_value = replay_result
            logger.info(f"使用回放评测结果作为基础评测: {reward_value}")
            
            # ⭐ CUA评测代理评测（基于回放后的状态）
            logger.info("开始CUA评测代理评测...")
            traj_for_eval = build_eval_trajectory_from_jsonl(replay_result_dir)
            cuaEvalAgent_result = eval_agent.eval_task(
                env=env,
                task_config=example,
                trajectory=traj_for_eval,
                save_dir=example_result_dir,
                max_round=max_round
            )
            logger.info(f"CUA评测完成，结果: {cuaEvalAgent_result}")
            
            # 保存CUA评测结果
            with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
                f.write(f"{reward_value}\n")
            with open(os.path.join(example_result_dir, "cuaEvalAgent_result.json"), "w", encoding="utf-8") as f:
                json.dump(cuaEvalAgent_result, f, indent=2, ensure_ascii=False)
            with open(os.path.join(example_result_dir, "cuaEvalAgent_result.txt"), "w", encoding="utf-8") as f:
                f.write(f"{cuaEvalAgent_result}\n")
            
            scores.append(reward_value)
        elif args.env_type == 'osworld-claude':
            traj_for_eval = build_eval_trajectory_from_jsonl(replay_result_dir)
            result = env.evaluate(task_config=example, trajectory=traj_for_eval)
            reward_value = result
        elif args.env_type == 'sandbox':
            result = env.evaluate()
            reward_value = result
        elif args.env_type == 'sandbox-modelEval':
            traj_for_eval = build_eval_trajectory_from_jsonl(replay_result_dir)
            result = evaluate(
                task_config=example,
                trajectory=traj_for_eval,
                llm_evluator=llm_evluator,
                save_dir=example_result_dir
            )
            reward_value = result
        
        logger.info(f"最终评测 reward_value: {reward_value}")
        return replay_success, replay_validation_success, True 
        
    except Exception as e:
        logger.error(f"评测失败: {e}", exc_info=True)
        return replay_success, replay_validation_success, False 
    
    

# ==================== 核心回放函数 ====================


def replay_trajectory_from_jsonl(
    env: DesktopEnv,
    replay_result_dir: str,
    example_result_dir: str,
    task_config: dict,
    args,
    verify_screenshots: bool = True
    # skip_env_reset: bool = False
) -> tuple[bool, str]:

    traj_file = os.path.join(replay_result_dir, "traj.jsonl")
    comparison_dir = os.path.join(example_result_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)

    # 读取轨迹数据
    logger.info(f"读取轨迹文件: {traj_file}")
    raw_trajectory_steps = []
    trajectory_steps = []
    try:
        with open(traj_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_trajectory_steps.append(line)
                    step_data = json.loads(line)
                    trajectory_steps.append(step_data)
                except json.JSONDecodeError as e:
                    logger.error(f"解析第 {line_num} 行失败: {e}")
                    logger.error(f"问题行内容: {line[:200]}...")
                    continue
    except Exception as e:
        logger.error(f"读取轨迹文件失败: {e}")
        return False
    
    # if not trajectory_steps:
    #     logger.warning("轨迹为空，无需回放")
    #     return True
    
    # ⭐ 排除最后一步
    total_steps = len(trajectory_steps)
    skip_last = False
    if '"action_type": "DONE"' in raw_trajectory_steps[-1]:
        skip_last = True
        steps_to_replay = trajectory_steps[:-1]
    else:
        steps_to_replay = trajectory_steps
    
    # if not steps_to_replay:
    #     logger.info("只有一步轨迹，无需回放")
    #     return True
    
    logger.info(f"成功读取 {total_steps} 个步骤，将回放 {len(steps_to_replay)} 步")
    
    
    # 保存初始状态截图
    try:
        # initial_obs = env._get_obs()
        # initial_screenshot_path = os.path.join(example_result_dir, "replay_step_0.png")
        # with open(initial_screenshot_path, "wb") as f:
        #     f.write(initial_obs['screenshot'])
        # logger.info(f"保存初始状态截图: replay_step_0.png")

        initial_obs = env._get_obs()
        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        with open(os.path.join(example_result_dir, f"step_0_{action_timestamp}.png"), "wb") as _f:
            _f.write(initial_obs['screenshot'])
        
        # # 如果存在原始的 step_0 截图，进行对比
        # original_step_0 = None
        # for step_data in trajectory_steps:
        #     if step_data.get("step_num") == 0:
        #         original_step_0 = step_data.get("screenshot_file")
        #         break
        
        # 如果没有 step_0，尝试查找 step_0_*.png 文件
        original_step_0 = None
        if not original_step_0:
            import glob
            step_0_files = glob.glob(os.path.join(replay_result_dir, "step_0_*.png"))
            if step_0_files:
                original_step_0 = os.path.basename(step_0_files[0])
        
        if original_step_0:
            original_step_0_path = os.path.join(replay_result_dir, original_step_0)
            if os.path.exists(original_step_0_path):
                is_consistent, similarity = compare_screenshots_with_score(
                    initial_obs['screenshot'],
                    original_step_0_path
                )
                logger.info(f"初始状态截图一致性: {is_consistent} (相似度: {similarity:.2%})")
                if not is_consistent:
                    logger.warning("初始状态与原始轨迹不一致，回放可能不准确")
            
    except Exception as e:
        logger.error(f"保存初始状态截图失败: {e}")
    
    # 4. 创建回放日志文件
    replay_log_file = os.path.join(example_result_dir, "replay.jsonl")
    replay_report = {
        "total_steps": total_steps,
        "steps_to_replay": len(steps_to_replay),  # ⭐ 新增：实际回放的步数
        "last_step_skipped": skip_last,                 # ⭐ 新增：标记跳过了最后一步
        "executed_steps": 0,
        "failed_steps": 0,
        "consistent_steps": 0,
        "execution_failures": [],
        "divergent_steps": [],
        "replay_timestamp": datetime.datetime.now().isoformat()
        # "skip_env_reset": skip_env_reset
    }
    
    # 5. 逐步回放动作（排除最后一步）
    logger.info(f"开始回放前 {len(steps_to_replay)} 个步骤...")
    
    last_replay_step_is_consistent = False
    for i, step_data in enumerate(steps_to_replay):
        step_num = step_data.get("step_num", i + 1)
        action_data = step_data.get("action", {})
        original_screenshot_file = step_data.get("screenshot_file")
        
        # ⭐ 提取 command 字段
        command = action_data.get("command")
        if not command:
            logger.warning(f"步骤 {step_num} 缺少 command 字段，跳过")
            replay_report["failed_steps"] += 1
            replay_report["execution_failures"].append({
                "step_num": step_num,
                "error_type": "MISSING_COMMAND",
                "error_message": "Command field is missing",
                "action_data": action_data
            })
            continue
        
        logger.info(f"回放步骤 {step_num}/{len(steps_to_replay)} (总共 {total_steps} 步)")
        logger.debug(f"  执行命令: {command.strip()}")
        
        try:
            # 执行命令
            obs, reward, done, info = env.step(command, args.sleep_after_execution)
            
            # 保存回放截图
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            replay_screenshot_filename = f"step_{step_num}_{action_timestamp}.png"
            replay_screenshot_path = os.path.join(example_result_dir, replay_screenshot_filename)
            with open(replay_screenshot_path, "wb") as f:
                f.write(obs['screenshot'])
            
            # 验证截图一致性
            is_consistent = False
            similarity_score = 0.0
            if verify_screenshots and original_screenshot_file:
                original_screenshot_path = os.path.join(replay_result_dir, original_screenshot_file)
                if os.path.exists(original_screenshot_path):
                    is_consistent, similarity_score = compare_screenshots_with_score(
                        obs['screenshot'],
                        original_screenshot_path
                    )

                    last_replay_step_is_consistent = is_consistent
                    
                    if not is_consistent:
                        logger.warning(
                            f"步骤 {step_num} 截图不一致 (相似度: {similarity_score:.2%})，回放可能偏离原轨迹"
                        )
                        
                        # ⭐ 记录偏离信息
                        divergence_info = {
                            "step_num": step_num,
                            "similarity": similarity_score,
                            "command": command.strip(),
                            "original_screenshot": original_screenshot_file,
                            "replay_screenshot": replay_screenshot_filename,
                            "comparison_image": f"comparison_step_{step_num}.png"
                        }
                        replay_report["divergent_steps"].append(divergence_info)
                        
                        # ⭐ 生成对比图
                        try:
                            comparison_image_path = create_comparison_image(
                                original_screenshot_path,
                                replay_screenshot_path,
                                comparison_dir,
                                step_num,
                                similarity_score
                            )
                            logger.info(f"生成对比图: {os.path.basename(comparison_image_path)}")
                        except Exception as comp_e:
                            logger.error(f"生成对比图失败: {comp_e}")
                    else:
                        replay_report["consistent_steps"] += 1
                else:
                    logger.warning(f"原始截图不存在: {original_screenshot_path}")
            
            # 记录回放日志
            replay_log_entry = {
                "step_num": step_num,
                "replay_timestamp": datetime.datetime.now().strftime("%Y%m%d@%H%M%S"),
                "command": command.strip(),
                "original_action": action_data,
                "reward": reward,
                "done": done,
                "screenshot_consistent": is_consistent,
                "similarity_score": similarity_score,
                "replay_screenshot_file": replay_screenshot_filename
            }
            
            with open(replay_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(replay_log_entry, ensure_ascii=False))
                f.write("\n")
            
            replay_report["executed_steps"] += 1
            
            if done:
                logger.info(f"步骤 {step_num} 触发done，提前结束回放")
                break
                
        except Exception as e:
            logger.error(f"回放步骤 {step_num} 失败: {e}", exc_info=True)
            replay_report["failed_steps"] += 1
            
            # ⭐ 记录执行失败信息
            failure_info = {
                "step_num": step_num,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "command": command.strip(),
                "traceback": traceback.format_exc()
            }
            replay_report["execution_failures"].append(failure_info)
            
            # 记录失败信息到日志文件
            with open(replay_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step_num": step_num,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "command": command.strip()
                }, ensure_ascii=False))
                f.write("\n")
            
            last_replay_step_is_consistent = False
    
    if skip_last:
        # ⭐ 记录最后一步信息（未回放）
        last_step = trajectory_steps[-1]
        last_step_info = {
            "step_num": last_step.get("step_num"),
            "action": last_step.get("action"),
            "screenshot_file": last_step.get("screenshot_file"),
            "note": "This step was not replayed (final step)"
        }
        
        with open(replay_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "last_step_skipped": last_step_info
            }, ensure_ascii=False))
            f.write("\n")

        # ⭐ 添加最后一步信息到报告
        replay_report["last_step_info"] = last_step_info
    
    # 6. 保存回放报告
    replay_report["success_rate"] = (
        replay_report["consistent_steps"] / replay_report["steps_to_replay"]
        if replay_report["steps_to_replay"] > 0 else 0
    )
    
    report_path = os.path.join(example_result_dir, "replay_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(replay_report, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 50)
    logger.info("回放完成统计:")
    logger.info(f"  总步数: {replay_report['total_steps']}")
    logger.info(f"  回放步数: {replay_report['steps_to_replay']} (跳过最后一步)")
    logger.info(f"  执行: {replay_report['executed_steps']}")
    logger.info(f"  失败: {replay_report['failed_steps']}")
    logger.info(f"  一致: {replay_report['consistent_steps']}")
    logger.info(f"  偏离: {len(replay_report['divergent_steps'])}")
    logger.info(f"  成功率: {replay_report['success_rate']:.2%}")
    logger.info("=" * 50)
    
    # # 判断回放是否成功（可以根据需要调整标准）
    # success = replay_report["failed_steps"] == 0

    # ⭐⭐ 核心判定逻辑 ⭐⭐
    # 条件1: 成功率 > 90%
    cond_high_rate = replay_report["success_rate"] > 0.9
    # 条件2: 容错步数 < 2 (允许错1步，错2步不行)
    # 例如：总5步，一致4步 -> 5-4=1 < 2 (True)
    cond_low_error_count = (replay_report['steps_to_replay'] - replay_report['consistent_steps']) < 2
    # 组合条件: (高成功率 OR 低错误数) AND 最后一步一致
    is_replay_success = (cond_high_rate or cond_low_error_count) and last_replay_step_is_consistent
    
    return is_replay_success


def create_comparison_image(
    original_path: str,
    replay_path: str,
    output_dir: str,
    step_num: int,
    similarity: float
) -> str:
    """
    创建原图和回放图的上下对比图
    
    Args:
        original_path: 原始截图路径
        replay_path: 回放截图路径
        output_dir: 输出目录
        step_num: 步骤编号
        similarity: 相似度分数
        
    Returns:
        str: 对比图路径
    """
    from PIL import Image, ImageDraw, ImageFont
    import io
    
    # 加载图像
    original_img = Image.open(original_path)
    replay_img = Image.open(replay_path)
    
    # 确保两张图尺寸一致
    if original_img.size != replay_img.size:
        logger.warning(f"图像尺寸不一致: {original_img.size} vs {replay_img.size}")
        # 调整回放图尺寸以匹配原图
        replay_img = replay_img.resize(original_img.size, Image.Resampling.LANCZOS)
    
    width, height = original_img.size
    
    # 创建标签高度
    label_height = 40
    
    # 创建新图像（上下拼接，加上标签）
    comparison_img = Image.new('RGB', (width, height * 2 + label_height * 2), color='white')
    
    # 绘制标签
    draw = ImageDraw.Draw(comparison_img)
    
    # 尝试使用系统字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # 绘制原图标签
    original_label = f"Original (Step {step_num})"
    draw.rectangle([(0, 0), (width, label_height)], fill='#2196F3')
    draw.text((10, 10), original_label, fill='white', font=font)
    
    # 粘贴原图
    comparison_img.paste(original_img, (0, label_height))
    
    # 绘制回放图标签
    replay_label = f"Replay (Similarity: {similarity:.2%})"
    label_color = '#4CAF50' if similarity >= 0.95 else '#FF9800' if similarity >= 0.85 else '#F44336'
    draw.rectangle([(0, height + label_height), (width, height + label_height * 2)], fill=label_color)
    draw.text((10, height + label_height + 10), replay_label, fill='white', font=font)
    
    # 粘贴回放图
    comparison_img.paste(replay_img, (0, height + label_height * 2))
    
    # 保存对比图
    output_path = os.path.join(output_dir, f"comparison_step_{step_num}.png")
    comparison_img.save(output_path, 'PNG')
    
    return output_path


def compare_screenshots_with_score(
    current_screenshot_bytes: bytes,
    reference_path: str,
    threshold: float = 0.95
) -> tuple[bool, float]:
    """
    比较两张截图的相似度，返回是否一致和相似度分数
    
    Args:
        current_screenshot_bytes: 当前截图字节
        reference_path: 参考截图路径
        threshold: 相似度阈值
        
    Returns:
        (is_consistent, similarity_score)
    """
    try:
        from PIL import Image, ImageChops, ImageStat
        import io
        
        img1 = Image.open(io.BytesIO(current_screenshot_bytes))
        img2 = Image.open(reference_path)
        
        if img1.size != img2.size:
            logger.warning(f"截图尺寸不一致: {img1.size} vs {img2.size}")
            return False, 0.0
        
        diff = ImageChops.difference(img1, img2)
        stat = ImageStat.Stat(diff)
        avg_diff = sum(stat.mean) / len(stat.mean)
        similarity = 1 - (avg_diff / 255)
        
        return similarity >= threshold, similarity
    except Exception as e:
        logger.error(f"截图比较失败: {e}")
        return False, 0.0

def compare_screenshots_with_score(
    current_screenshot_bytes: bytes,
    reference_path: str,
    threshold: float = 0.95
) -> tuple[bool, float]:
    """
    比较两张截图的相似度，返回是否一致和相似度分数
    
    Args:
        current_screenshot_bytes: 当前截图字节
        reference_path: 参考截图路径
        threshold: 相似度阈值
        
    Returns:
        (is_consistent, similarity_score)
    """
    try:
        from PIL import Image, ImageChops, ImageStat
        import io
        
        img1 = Image.open(io.BytesIO(current_screenshot_bytes))
        img2 = Image.open(reference_path)
        
        if img1.size != img2.size:
            logger.warning(f"截图尺寸不一致: {img1.size} vs {img2.size}")
            return False, 0.0
        
        diff = ImageChops.difference(img1, img2)
        stat = ImageStat.Stat(diff)
        avg_diff = sum(stat.mean) / len(stat.mean)
        similarity = 1 - (avg_diff / 255)
        
        return similarity >= threshold, similarity
    except Exception as e:
        logger.error(f"截图比较失败: {e}")
        return False, 0.0


def build_eval_trajectory_from_jsonl(example_result_dir: str) -> dict:
    """
    从 traj.jsonl 文件构建评测所需的轨迹格式
    
    Args:
        example_result_dir: 示例结果目录
        
    Returns:
        dict: 评测用的轨迹数据 {"screenshots": [...], "actions": [...]}
    """
    traj_file = os.path.join(example_result_dir, "traj.jsonl")
    
    if not os.path.exists(traj_file):
        logger.error(f"轨迹文件不存在: {traj_file}")
        return {"screenshots": [], "actions": []}
    
    traj_for_eval = {"screenshots": [], "actions": []}
    
    try:
        with open(traj_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    step_data = json.loads(line)
                    
                    # 加载截图
                    screenshot_file = step_data.get("screenshot_file")
                    if screenshot_file:
                        screenshot_path = os.path.join(example_result_dir, screenshot_file)
                        if os.path.exists(screenshot_path):
                            with open(screenshot_path, "rb") as sf:
                                screenshot_bytes = sf.read()
                                traj_for_eval["screenshots"].append(screenshot_bytes)
                        else:
                            logger.warning(f"截图文件不存在: {screenshot_path}")
                    
                    # 添加动作
                    action_data = step_data.get("action")
                    if action_data:
                        traj_for_eval["actions"].append(action_data)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"解析行失败: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"构建评测轨迹失败: {e}")
    
    logger.info(f"构建评测轨迹: {len(traj_for_eval['screenshots'])} 张截图, {len(traj_for_eval['actions'])} 个动作")
    
    return traj_for_eval


def find_trajectory_dir(base_dir: str, domain: str, example_id: str) -> str:
    """
    递归搜索轨迹目录
    
    Args:
        base_dir: 基础目录（如：claude-sandbox-newVM-45-fixac-1280x720）
        domain: 域名（如：chrome）
        example_id: 示例ID（如：fc6d8143-9452-4171-9459-7f515143419a）
        
    Returns:
        str: 轨迹目录的完整路径，如果找不到返回None
    """
    logger.debug(f"搜索轨迹目录: base_dir={base_dir}, domain={domain}, example_id={example_id}")
    
    # 可能的路径模式
    search_patterns = [
        # 模式1: base_dir/action_space/observation_type/model/domain/example_id
        os.path.join(base_dir, "*", "*", "*", domain, example_id),
        # 模式2: base_dir/domain/example_id
        os.path.join(base_dir, domain, example_id),
        # 模式3: base_dir/*/domain/example_id
        os.path.join(base_dir, "*", domain, example_id),
        # 模式4: base_dir/*/*/domain/example_id
        os.path.join(base_dir, "*", "*", domain, example_id),
    ]
    
    import glob
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        for match in matches:
            if os.path.isdir(match):
                # 检查是否包含 traj.jsonl
                traj_file = os.path.join(match, "traj.jsonl")
                if os.path.exists(traj_file):
                    logger.info(f"找到轨迹目录: {match}")
                    return match
    
    # 如果上述模式都找不到，进行更深层的递归搜索
    logger.debug(f"使用递归搜索: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        # 检查当前目录是否匹配
        if os.path.basename(root) == example_id:
            parent_dir = os.path.basename(os.path.dirname(root))
            if parent_dir == domain:
                traj_file = os.path.join(root, "traj.jsonl")
                if os.path.exists(traj_file):
                    logger.info(f"递归找到轨迹目录: {root}")
                    return root
    
    logger.error(f"未找到轨迹目录: domain={domain}, example_id={example_id}")
    return None


def find_all_trajectory_dirs_in_base(base_dir: str) -> dict:
    """
    在基础目录中查找所有轨迹目录
    
    Args:
        base_dir: 基础目录
        
    Returns:
        dict: {domain: [example_id, ...]}
    """
    logger.info(f"扫描基础目录: {base_dir}")
    
    trajectory_map = {}
    
    for root, dirs, files in os.walk(base_dir):
        if "traj.jsonl" in files:
            # 找到一个轨迹目录
            example_id = os.path.basename(root)
            parent_dir = os.path.dirname(root)
            domain = os.path.basename(parent_dir)
            
            if domain not in trajectory_map:
                trajectory_map[domain] = []
            
            if example_id not in trajectory_map[domain]:
                trajectory_map[domain].append(example_id)
                logger.debug(f"发现轨迹: {domain}/{example_id}")
    
    # 统计
    total_count = sum(len(examples) for examples in trajectory_map.values())
    logger.info(f"扫描完成，共找到 {total_count} 个轨迹，分布在 {len(trajectory_map)} 个域中")
    for domain, examples in trajectory_map.items():
        logger.info(f"  {domain}: {len(examples)} 个轨迹")
    
    return trajectory_map