from __future__ import annotations
import os
import json
import datetime
import logging
import threading
from typing import List, Dict, Any


logger = logging.getLogger("desktopenv.experiment")


def compute_single_success_scores(base_dir: str) -> List[float]:
    """遍历 base_dir 下已完成样本，返回其分数列表（读取 result.txt）。"""
    scores: List[float] = []
    try:
        if not os.path.exists(base_dir):
            return scores
        for domain in os.listdir(base_dir):
            domain_path = os.path.join(base_dir, domain)
            if not os.path.isdir(domain_path):
                continue
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if not os.path.isdir(example_path):
                    continue
                result_file = os.path.join(example_path, "result.txt")
                if os.path.exists(result_file):
                    try:
                        with open(result_file, "r", encoding="utf-8") as f:
                            txt = f.read().strip()
                            scores.append(float(txt) if txt else 0.0)
                    except Exception:
                        scores.append(0.0)
    except Exception as e:
        logger.error(f"扫描完成样本失败: {e}")
    return scores


def compute_passk_aggregate(base_dir: str, test_all_meta: dict, k: int) -> Dict[str, Any]:
    """纯计算 pass@k 聚合，不落盘。见返回字段定义。"""
    total_examples = 0
    attempted_examples = 0
    attempted_passed = 0
    details: Dict[str, Any] = {}
    for domain, example_ids in test_all_meta.items():
        for example_id in example_ids:
            total_examples += 1
            example_dir = os.path.join(base_dir, domain, example_id)
            attempt_results: List[float] = []
            for attempt_idx in range(k):
                attempt_dir = os.path.join(example_dir, f"attempt_{attempt_idx:02d}")
                result_file = os.path.join(attempt_dir, "result.txt")
                if os.path.exists(result_file):
                    try:
                        with open(result_file, "r", encoding="utf-8") as f:
                            txt = f.read().strip()
                            attempt_results.append(float(txt) if txt else 0.0)
                    except Exception:
                        attempt_results.append(0.0)
            if attempt_results:
                attempted_examples += 1
                passed = 1 if any(s >= 0.5 for s in attempt_results) else 0
                attempted_passed += passed
                details[f"{domain}/{example_id}"] = {"attempted": len(attempt_results), "passed": passed}
            else:
                details[f"{domain}/{example_id}"] = {"attempted": 0, "passed": 0}
    attempted_rate = (attempted_passed / attempted_examples) if attempted_examples > 0 else None
    overall_rate = (attempted_passed / total_examples) if total_examples > 0 else 0.0
    return {
        "total_examples": total_examples,
        "attempted_examples": attempted_examples,
        "attempted_passed": attempted_passed,
        "attempted_pass_rate": attempted_rate,
        "overall_pass_rate": overall_rate,
        "examples": details,
    }


def aggregate_passk_results(args, test_all_meta: dict, write_files: bool = True) -> dict:
    """聚合 pass@k 结果（可选是否落盘），与原脚本保持兼容签名。"""
    k = getattr(args, 'pass_k', None)
    if not (isinstance(k, int) and k > 1):
        return {"enabled": False}

    base_dir = os.path.join(args.result_dir, args.action_space, args.observation_type, args.model)
    total_examples = 0
    total_pass = 0
    per_example = {}

    for domain, example_ids in test_all_meta.items():
        for example_id in example_ids:
            example_dir = os.path.join(base_dir, domain, example_id)
            attempt_results: List[float] = []
            for attempt_idx in range(k):
                attempt_dir = os.path.join(example_dir, f"attempt_{attempt_idx:02d}")
                result_file = os.path.join(attempt_dir, "result.txt")
                if os.path.exists(result_file):
                    try:
                        with open(result_file, "r", encoding="utf-8") as f:
                            txt = f.read().strip()
                            score = float(txt) if txt else 0.0
                        attempt_results.append(score)
                    except Exception:
                        # 若读取失败，将该 attempt 视为无效（不计入）
                        pass

            if len(attempt_results) == k:    
                total_examples += 1
                example_pass = max(attempt_results) if attempt_results else 0.0
                total_pass += example_pass

            if write_files:
                try:
                    os.makedirs(example_dir, exist_ok=True)
                    # 仅当恰好有 k 次有效 attempt（即每个 attempt 子目录均有 result.txt）时，才写顶层 result.txt
                    if len(attempt_results) == k:
                        with open(os.path.join(example_dir, "result.txt"), "w", encoding="utf-8") as f:
                            f.write(f"{example_pass}\n")
                except Exception as e:
                    logger.error(f"写入聚合结果失败: {domain}/{example_id}: {e}")

            meta = {
                "domain": domain,
                "example_id": example_id,
                "pass_k": k,
                "attempt_results": attempt_results,
                "aggregated_pass": example_pass,
                "aggregated_time": datetime.datetime.now().isoformat()
            }
            if write_files:
                try:
                    with open(os.path.join(example_dir, "passk_meta.json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"写入 passk_meta 失败: {domain}/{example_id}: {e}")
            per_example[f"{domain}/{example_id}"] = meta

    pass_rate = (total_pass / total_examples) if total_examples > 0 else 0.0
    summary = {
        "enabled": True,
        "pass_k": k,
        "total_examples": total_examples,
        "passed_examples": int(total_pass),
        "pass_at_k": pass_rate,
        "aggregated_time": datetime.datetime.now().isoformat(),
    }

    if write_files:
        try:
            with open(os.path.join(base_dir, "passk_summary.json"), "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "examples": per_example}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"写入 passk_summary 失败: {e}")

    logger.info(f"Pass@{k} 成功率: {pass_rate * 100:.2f}% ({int(total_pass)}/{total_examples})")
    return {"summary": summary, "examples": per_example}


def periodic_metrics_reporter(args, test_all_meta: dict, shared_completed, total_tasks: int, stop_event: threading.Event):
    """后台线程：每隔 args.metrics_interval_min 分钟统计一次当前指标并记录到日志与 JSONL。"""
    try:
        interval_sec = max(1, int(args.metrics_interval_min) * 60) if getattr(args, "metrics_interval_min", 0) else 0
    except Exception:
        interval_sec = 0
    if interval_sec <= 0:
        return

    base_dir = os.path.join(args.result_dir, args.action_space, args.observation_type, args.model)
    history_path = os.path.join(base_dir, "periodic_metrics_history.jsonl")

    while not stop_event.is_set():
        try:
            now_iso = datetime.datetime.now().isoformat()
            completed = len(shared_completed) if shared_completed is not None else None
            k = getattr(args, 'pass_k', None)
            report = {
                "time": now_iso,
                "completed": completed,
                "total_tasks": total_tasks,
            }
            if isinstance(k, int) and k > 1:
                comp = compute_passk_aggregate(base_dir, test_all_meta, k)
                report.update({
                    "mode": f"pass@{k}",
                    "examples_attempted": comp["attempted_examples"],
                    "examples_passed": comp["attempted_passed"],
                    "pass_rate_so_far": comp["attempted_pass_rate"],
                    "overall_pass_rate": comp["overall_pass_rate"],
                })
                rate_str = f"{comp['attempted_pass_rate']*100:.2f}%" if comp["attempted_pass_rate"] is not None else "N/A"
                readable = (
                    f"[PeriodicMetrics] 完成任务: {completed}/{total_tasks}; 已尝试样本: {comp['attempted_examples']}; pass@{k} 通过率(已尝试): {rate_str}"
                )
            else:
                scores = compute_single_success_scores(base_dir)
                avg = (sum(scores) / len(scores)) if scores else None
                report.update({
                    "mode": "single",
                    "examples_finished": len(scores),
                    "success_rate_finished": avg,
                })
                avg_str = f"{avg*100:.2f}%" if avg is not None else "N/A"
                readable = (
                    f"[PeriodicMetrics] 完成任务: {completed}/{total_tasks}; 已完成样本: {len(scores)}; 成功率(已完成): {avg_str}"
                )
            logger.info(readable)
            try:
                os.makedirs(base_dir, exist_ok=True)
                with open(history_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(report, ensure_ascii=False) + "\n")
            except Exception as io_e:
                logger.error(f"写入周期指标失败: {io_e}")
        except Exception as e:
            logger.error(f"周期指标统计失败: {e}")
        if stop_event.wait(interval_sec):
            break


def start_metrics_reporter(args, test_all_meta: dict, shared_completed, total_tasks: int):
    """封装启动定时监控线程，返回句柄(dict)或 None。"""
    handle = None
    try:
        if getattr(args, 'metrics_interval_min', 0) and int(args.metrics_interval_min) > 0:
            stop_event = threading.Event()
            reporter_thread = threading.Thread(
                target=periodic_metrics_reporter,
                args=(args, test_all_meta, shared_completed, total_tasks, stop_event),
                name="PeriodicMetricsReporter",
                daemon=False,
            )
            reporter_thread.start()
            logger.info(f"已启动周期指标统计线程，间隔 {args.metrics_interval_min} 分钟")
            handle = {"thread": reporter_thread, "stop_event": stop_event}
    except Exception as e:
        logger.error(f"启动周期指标线程失败: {e}")
    return handle


def stop_metrics_reporter(handle, timeout: int = 5):
    """封装停止定时监控线程。"""
    try:
        if not handle:
            return
        stop_event = handle.get("stop_event") if isinstance(handle, dict) else getattr(handle, "stop_event", None)
        reporter_thread = handle.get("thread") if isinstance(handle, dict) else getattr(handle, "thread", None)
        if stop_event is not None:
            stop_event.set()
        if reporter_thread is not None:
            reporter_thread.join(timeout=timeout)
    except Exception as e:
        logger.error(f"停止周期指标线程失败: {e}")


