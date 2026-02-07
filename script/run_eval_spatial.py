import argparse
import subprocess
import os
import sys

# Mapping logic for spatial evaluation
# User chooses a "setting", and the script determines the internal task_name and config file
SPATIAL_SETTING_MAP = {
    "sparse": {
        "task_name": "blocks_ranking_rgb_spatial",
        "task_config": "config_spatial_clean"
    },
    "cluttered": {
        "task_name": "blocks_ranking_rgb_spatial",
        "task_config": "config_spatial_cluttered"
    },
    "dense": {
        "task_name": "blocks_five_spatial",
        "task_config": "config_spatial_clean"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Spatial Evaluation Wrapper for MLLM")
    
    # 1. User-facing arguments
    parser.add_argument("--setting", type=str, required=True, choices=["sparse", "cluttered", "dense"],
                        help="The spatial evaluation setting (sparse, cluttered, or dense)")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID(s) to use")
    parser.add_argument("--model", type=str, required=True, help="Path to the model or model name")
    # parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation")
    
    args = parser.parse_args()

    # 2. Automatically locate the project root directory
    # Assumes this script is located in project_root/script/
    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))

    # 3. Retrieve internal configuration based on the chosen setting
    eval_params = SPATIAL_SETTING_MAP.get(args.setting)
    task_name = eval_params["task_name"]
    config_name = eval_params["task_config"]
    
    policy_name = "vlmpolicy"

    # 4. Construct the execution command
    # Note: We use the relative path from the project root
    cmd = [
        "python", "script/myvlmEb_eval_spatial.py",
        "--config", f"policy/{policy_name}/deploy_policy.yml",
        "--overrides",
        "--task_name", task_name,
        "--task_config", config_name,
        "--ckpt_setting", config_name,
        "--seed", str(0),
        "--policy_name", policy_name,
        "--model_name", args.model
    ]

    # 5. Prepare environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Add root directory to PYTHONPATH so that 'envs' module can be found
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = root_dir + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = root_dir

    # Log execution details
    print("=" * 60)
    print(f"Spatial Setting: {args.setting}")
    # print(f"Mapped Task:    {task_name}")
    print(f"Mapped Config:  {config_name}")
    print(f"GPU ID:         {args.gpu}")
    print(f"Model Path:     {args.model}")
    print(f"Root Dir:       {root_dir}")
    print("=" * 60)

    # 6. Execute the evaluation script in the project root directory
    try:
        subprocess.run(cmd, env=env, cwd=root_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Evaluation failed with return code {e.returncode}")
    except KeyboardInterrupt:
        print("\n[Info] Evaluation interrupted by user.")

if __name__ == "__main__":
    main()