import os
import json
import logging

def print_result(target_dir, logger: logging=None):

    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None
    
    all_result = []
    domain_result = {}
    domain_result_detail = {}
    all_result_for_analysis = {}

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)

                # if "succeed_" in  example_id or "done_" in example_id:
                #     new_example_id = example_id.split("_")[1]
                #     new_example_path = os.path.join(domain_path, new_example_id)
                #     os.rename(example_path, new_example_path)
                #     example_path = new_example_path
                #     example_id = new_example_id
                
                if domain not in domain_result:
                    domain_result[domain] = []
                if domain not in all_result_for_analysis:
                    all_result_for_analysis[domain] = {}
                if domain not in domain_result_detail:
                    domain_result_detail[domain] = {}
                
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        # 读取并解析结果
                        try:
                            with open(os.path.join(example_path, "result.txt"), "r") as f:
                                result = f.read().strip()
                            
                            # 尝试多种解析方法
                            try:
                                # 直接转换为浮点数
                                parsed_result = float(result)
                            except ValueError:
                                try:
                                    # 尝试解析为字典并提取reward值
                                    import ast
                                    result_dict = ast.literal_eval(result)
                                    if isinstance(result_dict, dict):
                                        parsed_result = float(result_dict.get('reward', result_dict.get('score', 0.0)))
                                    else:
                                        parsed_result = float(result_dict)
                                except (ValueError, SyntaxError):
                                    # 如果都失败了，尝试布尔值转换
                                    try:
                                        parsed_result = float(bool(result))
                                    except:
                                        parsed_result = 0.0
                            
                            domain_result[domain].append(parsed_result)
                            
                        except Exception as e:
                            print(f"Error processing result for {example_id}: {e}")
                            domain_result[domain].append(0.0)

                        domain_result_detail[domain][example_id] = domain_result[domain][-1]
                        
                        # 对于 all_result 使用相同的逻辑
                        all_result.append(domain_result[domain][-1])

                    # else:
                    #     domain_result[domain].append(0.0)
                    #     domain_result_detail[domain][example_id] = 0.0
                        # all_result_for_analysis[domain][example_id] = 0.0
                        # all_result.append(0.0)

                    # result_reason = "未知原因"
                    # if "result_reason.txt" in os.listdir(example_path):
                    #     result_reason = open(os.path.join(example_path, "result_reason.txt"), "r").read()
                    # domain_result_detail[domain][example_id] = f"{domain_result_detail[domain][example_id]} {result_reason}"

                    # if result_reason not in ["评测通过", "评测未通过", "最大步数限制内未完成"]:
                    #     reason = []
                    #     with open(os.path.join(example_path, "traj.jsonl"), "r") as traj_file:
                    #         for line in traj_file:
                    #             if line.lstrip().startswith("{"):
                    #                 continue
                    #             reason.append(line)
                    #             reason.extend(traj_file.readlines())
                    #     result_reason = result_reason + "\n" + "\n".join(reason)  
                    # else:
                    #     all_result_for_analysis[domain].pop(example_id)
                    #     continue    
                    reason = []
                    traj_jsonl_file = os.path.join(example_path, "traj.jsonl")
                    last_step_line = ""
                    if os.path.exists(traj_jsonl_file):
                        with open(traj_jsonl_file, "r") as traj_file:
                            for line in traj_file:
                                if line.lstrip().startswith("{"):
                                    last_step_line = line
                                    continue
                                reason.append(line)
                                reason.extend(traj_file.readlines())
                        result_reason = "\n".join(reason)  
                        if "Too Many Requests" in result_reason:
                            result_reason = "Too Many Requests"
                    else:
                        result_reason = "traj.jsonl not exists"
                    # if '"action": "DONE",' in last_step_line and "_succeed_" not in example_path and "_done_" not in example_path :
                    #         new_example_path = os.path.join(domain_path, f"_done_{example_id}")
                    #         os.rename(example_path, new_example_path)
                    #         example_path = new_example_path

                    # all_result_for_analysis[domain][example_id] = f"{all_result_for_analysis[domain][example_id]} {result_reason}"
                    if result_reason:
                        all_result_for_analysis[domain][example_id] = f"{result_reason}"
                    
    print_eval_error_info(target_dir, all_result, all_result_for_analysis)
    print_eval_result_detail(target_dir, all_result, domain_result_detail)
    print_eval_result(target_dir, all_result, domain_result, logger)

    

def print_eval_result(target_dir, all_result, domain_result, logger: logging=None):
    with open(os.path.join(target_dir, "eval_result.txt"), "w", encoding="utf-8") as eval_result:
        if len(all_result) == 0:
            eval_result.write("New experiment, no result yet.")
            if logger:
                logger.info("New experiment, no result yet.")
            else:
                print("New experiment, no result yet.")
            return
        
        for domain in domain_result:
            domain_sum = sum(domain_result[domain])
            domain_len = len(domain_result[domain])
            if domain_len == 0:
                domain_len = 1
            domain_success_rate = domain_sum / domain_len * 100
            result_text = f"Domain: {domain:<20} Runned: {domain_len:<4} Success Rate: {domain_sum} / {domain_len} = {domain_success_rate:.2f}%"
            eval_result.write(f"{result_text}\n")
            if logger:
                logger.info(result_text)
            else:
                print(result_text)

        eval_result.write("\n\n\n")
        office_sum = sum(
            domain_result["libreoffice_calc"] if "libreoffice_calc" in domain_result else [] + \
            domain_result["libreoffice_impress"] if "libreoffice_impress" in domain_result else [] + \
            domain_result["libreoffice_writer"] if "libreoffice_impress" in domain_result else []
        )
        office_len = len(
            domain_result["libreoffice_calc"] if "libreoffice_calc" in domain_result else [] + \
            domain_result["libreoffice_impress"] if "libreoffice_calc" in domain_result else [] + \
            domain_result["libreoffice_writer"] if "libreoffice_calc" in domain_result else []
        )
        if office_len == 0:
            office_len = 1
        office_success_rate = office_sum / office_len * 100
        result_text = f"Office Success Rate: {office_success_rate:.2f}%"
        eval_result.write(f"{result_text}\n")
        if logger:
            logger.info(result_text)
        else:
            print(result_text)

        daily_sum = sum(
            domain_result["vlc"] if "vlc" in domain_result else [] + \
            domain_result["thunderbird"] if "thunderbird" in domain_result else [] + \
            domain_result["chrome"] if "chrome" in domain_result else 0
        )
        daily_len = len(
            domain_result["vlc"] if "vlc" in domain_result else [] + \
            domain_result["thunderbird"] if "thunderbird" in domain_result else [] + \
            domain_result["chrome"] if "chrome" in domain_result else []
        )
        if daily_len == 0:
            daily_len = 1
        daily_success_rate = daily_sum / daily_len * 100
        result_text = f"Daily Success Rate: {daily_success_rate:.2f}%"
        eval_result.write(f"{result_text}\n")
        if logger:
            logger.info(result_text)
        else:
            print(result_text)

        professional_sum = sum(
            domain_result["gimp"] if "gimp" in domain_result else [] + \
            domain_result["vs_code"] if "vs_code" in domain_result else []
        )
        professional_len = len(
            domain_result["gimp"] if "gimp" in domain_result else [] + \
            domain_result["vs_code"] if "vs_code" in domain_result else []
        )
        if professional_len == 0:
            professional_len = 1
        professional_success_rate = professional_sum / professional_len * 100
        result_text = f"Professional Success Rate: {professional_success_rate:.2f}%"
        eval_result.write(f"{result_text}\n")
        if logger:
            logger.info(result_text)
        else:
            print(result_text)

        eval_result.write("\n\n")
        all_result_sum = sum(all_result)
        all_result_len = len(all_result)
        all_result_success_rate = all_result_sum / all_result_len * 100
        result_text = f"Runned: {all_result_len}, Current Success Rate: {all_result_sum} / {all_result_len} = {all_result_success_rate:.2f}%"
        eval_result.write(f"{result_text}\n")
        eval_result.write(f"{all_result_success_rate * 0.01}")
        if logger:
            logger.info(result_text)
        else:
            print(result_text)


def print_eval_result_detail(target_dir, all_result, domain_result_detail):
    with open(os.path.join(target_dir, "eval_result_detail.json"), "w", encoding="utf-8") as eval_result_detail:
        if len(all_result) == 0:
            eval_result_detail.write("{}")
            print("New experiment, no result yet.")
            return
        
        json.dump(domain_result_detail, eval_result_detail, indent=2, ensure_ascii=False)


def print_eval_error_info(target_dir, all_result, all_result_for_analysis):
    with open(os.path.join(target_dir, "eval_error_info.txt"), "w", encoding="utf-8") as eval_error_info:
        if len(all_result) == 0:
            eval_error_info.write("no error yet")
            print("New experiment, no result yet.")
            return
        
        for domain in all_result_for_analysis:
            eval_error_info.write(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {domain} start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n\n")
            for example_id in all_result_for_analysis[domain]:
                eval_error_info.write(f"{domain}/{example_id}    ")
                reason = all_result_for_analysis[domain][example_id]
                reason_arr = reason.split("\n")
                for line in reason_arr:
                    if not line.strip():
                        continue
                    eval_error_info.write(f"{line}\n")
                eval_error_info.write("\n\n")
            eval_error_info.write(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {domain} end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n\n\n\n\n\n\n\n\n\n\n")




if __name__ == '__main__':

    result_dir = "ANON"
    use_model = "ANON"

    action_space = "claude_computer_use"
    observation_type = "screenshot"

    target_dir = os.path.join("./results_log/", result_dir, action_space, observation_type, use_model)
    print_result(target_dir)
    
