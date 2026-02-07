import argparse
import subprocess
import os
import sys

TASK_CONFIG_MAP = {
    "handover_mic": "config_arx_x5_0_6",
    "blocks_ranking_size" : "config_arx_x5_0_6",
    "hanging_mug": "config_arx_x5_0_6",

    "place_cans_plasticbox": "config_arx_x5_0_7",
    "place_burger_fries": "config_arx_x5_0_7",
    "stack_blocks_three": "config_arx_x5_0_7",
    "handover_block": "config_arx_x5_0_7",

    "blocks_ranking_rgb": "config_franka_0_6",

    "place_object_basket": "config_franka_0_7",
    "place_bread_skillet": "config_franka_0_7",
    "stack_bowls_three": "config_franka_0_7",

    "blocks_tower": "config_franka_0_6_clean",
    "blocks_cross_shape": "config_franka_0_6_clean",

    "put_bottles_dustbin": "config_piper_0_7",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name to evaluate")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID(s), e.g., 0 or 0,1")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    args = parser.parse_args()

    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))

    config_name = TASK_CONFIG_MAP.get(args.task)
    if not config_name:
        print(f"Error: Task '{args.task}' not found in TASK_CONFIG_MAP.")
        return

    # 3. 构建命令 (注意这里使用 script/myvlmEb_eval.py，因为我们要从根目录启动)
    cmd = [
        "python", "script/myvlmEb_eval.py",
        "--config", "policy/vlmpolicy/deploy_policy.yml",
        "--overrides",
        "--task_name", args.task,
        "--task_config", config_name,
        "--ckpt_setting", config_name,
        "--seed", "0",
        "--policy_name", "vlmpolicy",
        "--model_name", args.model
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = root_dir + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = root_dir

    print("-" * 50)
    print(f"Project Root: {root_dir}")
    print(f"Task:         {args.task}")
    print(f"Config:       {config_name}")
    print(f"GPU:          {args.gpu}")
    print(f"Model:        {args.model}")
    print("-" * 50)

    try:
        subprocess.run(cmd, env=env, cwd=root_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the evaluation: {e}")
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")

if __name__ == "__main__":
    main()