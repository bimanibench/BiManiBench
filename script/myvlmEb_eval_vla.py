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
import numpy as np

from vlmplanner_vla import VLMPlanner
from systemPrompt import system_prompt

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
        # writeIt_out+=1
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

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_seed = st_seed

    args["eval_mode"] = True

    model_name = model
    print("model_name:", model_name)

    actionChunkNum = 6
    task_description = ""
    task_template_example = ""
    action_limit = 10
    try:
        with open(f'./description/task_instruction/{task_name}.json','r',encoding='utf8') as fp:
            json_data = json.load(fp)
            task_description = json_data.get("full_description", "")
            task_template_example = json_data.get("template_example_vla",{})
            action_limit = json_data.get("action_limit_vla",8)
            actionChunkNum = json_data.get("action_chunk_num_vla",6)
    except FileNotFoundError:
        print(f"Task description file for {task_name} not found. Using default description.")
    
    
    prompt_out_path = "prompt_output.txt"
    if os.path.exists(prompt_out_path):
        os.remove(prompt_out_path) 

    planner = VLMPlanner(model_name,"remote_model",system_prompt.format(task_name,task_description), obs_key='head_rgb',
                                                        chat_history=True, language_only=False, 
                                                        use_feedback=True, tp=1,prompt_out_path=prompt_out_path,user_instruction=task_description)

    success_record_path = f"{save_dir}/success_record.txt"
    failed_record_path = f"{save_dir}/failed_record.txt"
    modelname_record_path = f"{save_dir}/model_name.txt"
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
            if TASK_ENV.eval_video_path is not None:
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        video_size,
                        "-framerate",
                        "10",
                        "-i",
                        "-",
                        "-pix_fmt",
                        "yuv420p",
                        "-vcodec",
                        "libx264",
                        "-crf",
                        "23",
                        f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)
            ###!!!!!!!!!!!!!!!!!!!!!!!!
            # if(round==0 and now_episode<17):
            #     print(f"skip: round {round}, nowepisode {now_episode}")
            #     TASK_ENV.test_num += 1
            #     now_episode += 1
            #     now_seed += 1
            #     TASK_ENV.close()
            #     continue
            ###!!!!!!!!!!!!!!!!!!!!!!!!
            now_action_cnt = 0
            
            action_count_vla = 0
            while (now_action_cnt < action_limit or action_count_vla <=12) and now_action_cnt < np.ceil(1.5*action_limit):
                observation = TASK_ENV.get_obs()
                img_bgr_third = cv2.cvtColor(observation["third_view_rgb"], cv2.COLOR_RGB2BGR)  
                img_bgr = cv2.cvtColor(observation["observation"]["head_camera"]["rgb"], cv2.COLOR_RGB2BGR)
                img_bgr_left = cv2.cvtColor(observation["observation"]["left_camera"]["rgb"], cv2.COLOR_RGB2BGR)
                img_bgr_right = cv2.cvtColor(observation["observation"]["right_camera"]["rgb"], cv2.COLOR_RGB2BGR)
                divider = np.ones((img_bgr_left.shape[0], 5, 3), dtype=np.uint8) * 255
                combined = np.hstack([img_bgr_left, divider, img_bgr_right])

                if not os.path.exists(f"{TASK_ENV.eval_obsimg_path}/"):
                    os.makedirs(f"{TASK_ENV.eval_obsimg_path}/")
                filename = f"{TASK_ENV.eval_obsimg_path}/obsimg_{now_action_cnt:04d}.jpg"
                filename_leftright = f"{TASK_ENV.eval_obsimg_path}/obsimg_leftright_{now_action_cnt:04d}.jpg"
                filename_third = f"{TASK_ENV.eval_obsimg_path}/obsimg_third_{now_action_cnt:04d}.jpg"
                cv2.imwrite(filename_third, img_bgr_third)  
                cv2.imwrite(filename_leftright, combined)  
                cv2.imwrite(filename, img_bgr)  
                print(f"Saved observation image to {filename}")
                filename_vlmout = f"{TASK_ENV.eval_obsimg_path}/vlmout.txt"
                planner.set_prompt_output_path(f"{TASK_ENV.eval_obsimg_path}/prompt_output.txt")
                actionList,out = planner.act(observation=filename,assistant_obs=None, assistant_info=TASK_ENV.get_assistantInfo_vla(),actionChunkNum=actionChunkNum)
                print("actionOut:")
                print(out)
                write_print(out,filename_vlmout,now_action_cnt)
                action_feedback = ""
                if actionList!={}:
                    if isinstance(actionList, dict):
                        res = TASK_ENV.take_action_by_dict_vla(actionList)
                        if(res!=True):
                            action_feedback += f"Action failed: {res}\n"
                            with open(failed_record_path, 'a', encoding='utf-8') as f:
                                        f.write(f"Task {task_name} failed at round {round}, episode {now_episode}, action count {now_action_cnt}, action it:{actionit}: Action failed: {res}\n")
                            break
                        else:
                            action_feedback += "Action succeeded.\n"
                        if not os.path.exists(f"{TASK_ENV.eval_obsimg_path}/obsimg_{now_action_cnt:04d}-{now_action_cnt+1:04d}/"):
                                os.makedirs(f"{TASK_ENV.eval_obsimg_path}/obsimg_{now_action_cnt:04d}-{now_action_cnt+1:04d}/")
                        filename_step = f"{TASK_ENV.eval_obsimg_path}/obsimg_{now_action_cnt:04d}-{now_action_cnt+1:04d}/obsimg_{actionit:02d}.jpg"
                        filename_step_third = f"{TASK_ENV.eval_obsimg_path}/obsimg_{now_action_cnt:04d}-{now_action_cnt+1:04d}/obsimg_{actionit:02d}_third.jpg"
                        observation = TASK_ENV.get_obs()
                        img_bgr_step = cv2.cvtColor(observation["observation"]["head_camera"]["rgb"], cv2.COLOR_RGB2BGR)
                        img_bgr_third = cv2.cvtColor(observation["third_view_rgb"], cv2.COLOR_RGB2BGR)  
                        cv2.imwrite(filename_step, img_bgr_step)  
                        cv2.imwrite(filename_step_third, img_bgr_third)  
                    else:
                        actionit = 0
                        for action in actionList:
                                res = TASK_ENV.take_action_by_dict_vla(action)
                                if(res!=True):
                                    action_feedback += f"Action failed: {res}\n"
                                    with open(failed_record_path, 'a', encoding='utf-8') as f:
                                        f.write(f"Task {task_name} failed at round {round}, episode {now_episode}, action count {now_action_cnt}, action it:{actionit}: Action failed: {res}\n")
                                    break
                                else:
                                    action_feedback += "Action succeeded.\n"
                                if not os.path.exists(f"{TASK_ENV.eval_obsimg_path}/obsimg_{now_action_cnt:04d}-{now_action_cnt+1:04d}/"):
                                    os.makedirs(f"{TASK_ENV.eval_obsimg_path}/obsimg_{now_action_cnt:04d}-{now_action_cnt+1:04d}/")
                                filename_step = f"{TASK_ENV.eval_obsimg_path}/obsimg_{now_action_cnt:04d}-{now_action_cnt+1:04d}/obsimg_{actionit:02d}.jpg"
                                filename_step_third = f"{TASK_ENV.eval_obsimg_path}/obsimg_{now_action_cnt:04d}-{now_action_cnt+1:04d}/obsimg_{actionit:02d}_third.jpg"
                                observation = TASK_ENV.get_obs()
                                img_bgr_step = cv2.cvtColor(observation["observation"]["head_camera"]["rgb"], cv2.COLOR_RGB2BGR)
                                img_bgr_third = cv2.cvtColor(observation["third_view_rgb"], cv2.COLOR_RGB2BGR)  
                                cv2.imwrite(filename_step, img_bgr_step)  
                                cv2.imwrite(filename_step_third, img_bgr_third)  
                                actionit += 1
                        action_count_vla += actionit
                else:
                    action_feedback += "Action is empty, no action taken. If task is not finished, check the task is finished or not according to the image.\n"
                    print("Action is empty, no action taken.")
                if TASK_ENV.check_success():
                    succ = True
                    observation = TASK_ENV.get_obs()
                    img_bgr = cv2.cvtColor(observation["observation"]["head_camera"]["rgb"], cv2.COLOR_RGB2BGR)
                    img_bgr_third = cv2.cvtColor(observation["third_view_rgb"], cv2.COLOR_RGB2BGR)  
                    filename = f"{TASK_ENV.eval_obsimg_path}/obsimg_{now_action_cnt+1:04d}.jpg"
                    filename_third = f"{TASK_ENV.eval_obsimg_path}/obsimg_third_{now_action_cnt+1:04d}.jpg"
                    cv2.imwrite(filename, img_bgr)  
                    cv2.imwrite(filename_third, img_bgr_third)  
                    print(f"Task success, Saved observation image to {filename}")
                    with open(success_record_path, 'a', encoding='utf-8') as f:
                        f.write(f"Task {task_name} success at round {round}, episode {now_episode}, action count {now_action_cnt}\n")
                    TASK_ENV.suc += 1
                    break

                planner.update_info({
                    "action_list": actionList,
                    "action_feedback": action_feedback
                })
                now_action_cnt += 1

            TASK_ENV.test_num += 1
            now_episode += 1
            now_seed += 1
            if TASK_ENV.eval_video_path is not None:
                TASK_ENV._del_eval_video_ffmpeg()
            TASK_ENV.close()
    print(f"Task {task_name} evaluation completed. Total successes: {TASK_ENV.suc}/{test_num*round_num}")
    with open(success_record_path, 'a', encoding='utf-8') as f:
        f.write(f"Task {task_name} evaluation completed. Total successes: {TASK_ENV.suc}/{test_num*round_num}\n")
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

    save_dir = Path(f"eval_result/low_level/{task_name}/{policy_name}/{task_config}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = Path(f"eval_result/low_level/{task_name}/{policy_name}/{task_config}/{current_time}/video")
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    eval_policy(task_name,TASK_ENV, args,save_dir=save_dir,model=model_name ,st_seed=st_seed, video_size=video_size)

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
    