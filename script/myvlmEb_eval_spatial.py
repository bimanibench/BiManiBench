import numpy as np
import subprocess
import importlib
import argparse
import yaml
import os
from pathlib import Path
import sys
import cv2
import json

from vlmplanner_dualarm_spatial import VLMPlanner
# from systemPrompt import system_prompt

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")

from envs import CONFIGS_PATH
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def write_print(content,file_path,writeIt):  
    with open(file_path, 'a', encoding="utf-8") as f:
        print(f"step {writeIt}: ***************",file=f)
        print(content,file=f)

def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                save_dir,
                test_num=50,
                video_size=None,
                instruction_type=None,
                policy_conda_env=None):

    TASK_ENV.total_score = 0
    TASK_ENV.test_num = 0
    
    now_seed = st_seed

    args["eval_mode"] = True

    model_name = model
    print("model_name:", model_name)

    # actionChunkNum = 6
    task_description = ""
    task_template_example = ""
    # action_limit = 10
    try:
        with open(f'./description/task_instruction/{task_name}.json','r',encoding='utf8') as fp:
            json_data = json.load(fp)
            task_description = json_data.get("full_description", "")
            task_template_example = json_data.get("template_example_spatial",{})
            # action_limit = json_data.get("action_limit",10)
            # actionChunkNum = json_data.get("action_chunk_num",6)
    except FileNotFoundError:
        print(f"Task description file for {task_name} not found. Using default description.")
    
    
    prompt_out_path = "prompt_output.txt"
    if os.path.exists(prompt_out_path):
        os.remove(prompt_out_path) 

    planner = VLMPlanner(model_name,"remote_model", obs_key='head_rgb',
                                                        chat_history=True, language_only=False, 
                                                        use_feedback=True, tp=1,prompt_out_path=prompt_out_path,user_instruction=task_description)

    score_record_path = f"{save_dir}/score_record.txt"
    # failed_record_path = f"{save_dir}/failed_record.txt"
    modelname_record_path = f"{save_dir}/model_name.txt"
    # record the model name
    with open(modelname_record_path, 'a', encoding='utf-8') as f:
        f.write(f"Model used: {model_name}\n")
    round_num = 2 
    for round in range(round_num):
        now_episode = 0
        while now_episode < test_num:
            planner.reset()
            planner.set_task(task_name)
            planner.set_template_example(json.dumps(task_template_example,indent=4))
            args["eval_obsimg_save_dir"] = save_dir / "obsimg" /f"round{round}" / f"episode{now_episode}"
            try:
                TASK_ENV.setup_demo(now_ep_num=now_episode, seed=now_seed, is_test=True, **args)
            except Exception as e:
                print(f"Error setting up environment for round{round}, episode {now_episode}: {e}")
                now_seed += 1
                continue
            TASK_ENV.set_instruction(instruction=task_description)  # set language instruction
            ####!!!!!!!!!!!!!!!!!!!!!!!!
            # if(round==0 or now_episode<2):
            #     print(f"skip: round {round}, nowepisode {now_episode}")
            #     TASK_ENV.test_num += 1
            #     now_episode += 1
            #     now_seed += 1
            #     TASK_ENV.close()
            #     continue
            ####!!!!!!!!!!!!!!!!!!!!!!!!
            now_action_cnt = 0
            output_limit = 3
            # while now_action_cnt < action_limit:
            while now_action_cnt < output_limit :
                observation = TASK_ENV.get_obs()
                img_bgr_third = cv2.cvtColor(observation["third_view_rgb"], cv2.COLOR_RGB2BGR)  
                img_bgr = cv2.cvtColor(observation["observation"]["head_camera"]["rgb"], cv2.COLOR_RGB2BGR)

                if not os.path.exists(f"{TASK_ENV.eval_obsimg_path}/"):
                        os.makedirs(f"{TASK_ENV.eval_obsimg_path}/")
                filename = f"{TASK_ENV.eval_obsimg_path}/obsimg_{now_action_cnt:04d}.jpg"
                filename_third = f"{TASK_ENV.eval_obsimg_path}/obsimg_third_{now_action_cnt:04d}.jpg"
                cv2.imwrite(filename_third, img_bgr_third)  
                cv2.imwrite(filename, img_bgr)   
                print(f"Saved observation image to {filename}")
                filename_vlmout = f"{TASK_ENV.eval_obsimg_path}/vlmout.txt"
                planner.set_prompt_output_path(f"{TASK_ENV.eval_obsimg_path}/prompt_output.txt")
                resultsList,out = planner.act(observation=filename,assistant_obs=[] , assistant_info=None)
                print("out:")
                print(out)
                write_print(out,filename_vlmout,now_action_cnt)
                eval_res = TASK_ENV.evaluate_spatial(resultsList)

                if isinstance(eval_res,int) or isinstance(eval_res,float):
                    print(f"round {round}, episode {now_episode}, score: {eval_res}\n")
                    with open(score_record_path, 'a', encoding='utf-8') as f:
                        f.write(f"Task {task_name} at round {round}, episode {now_episode}, score: {eval_res}\n")
                    TASK_ENV.total_score += eval_res
                    break
                else:
                    now_action_cnt += 1
            if now_action_cnt >= output_limit:
                print(f"round {round}, episode {now_episode}, score: 0\n")
                with open(score_record_path, 'a', encoding='utf-8') as f:
                    f.write(f"Task {task_name} at round {round}, episode {now_episode}, score: 0\n")
            TASK_ENV.test_num += 1
            now_episode += 1
            now_seed += 1
            TASK_ENV.close()
    print(f"Task {task_name} spatial evaluation completed. Average score: {TASK_ENV.total_score/100}")
    with open(score_record_path, 'a', encoding='utf-8') as f:
        f.write(f"Task {task_name} spatial evaluation completed. Average score: {TASK_ENV.total_score/100}\n")
    return

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]

def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # task_name = "place_object_scale"
    task_name = usr_args["task_name"]
    policy_name = usr_args["policy_name"]
    TASK_ENV = class_decorator(task_name)
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    model_name = usr_args["model_name"]
    seed = usr_args["seed"]
    st_seed = 100000 * (1 + seed)
    instruction_type = usr_args["instruction_type"]
    
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    save_dir = Path(f"eval_result/spatial/{task_name}/{policy_name}/{task_config}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    eval_policy(task_name,TASK_ENV, args,save_dir=save_dir,model=model_name,st_seed=st_seed, video_size=video_size)

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config

if __name__ == "__main__":
    main(usr_args=parse_args_and_config())
    