import argparse
import subprocess
import os
import sys

# Define the mapping between task names and their respective robot configurations
# Note: Only 'place_bread_skillet' requires a '_vla' suffix internally
TASK_CONFIG_MAP = {
    "place_object_scale": {
        "internal_name": "place_object_scale",
        "config": "config_arx_x5_0_7"
    },
    "place_burger_fries": {
        "internal_name": "place_burger_fries",
        "config": "config_arx_x5_0_7"
    },
    "grab_roller": {
        "internal_name": "grab_roller",
        "config": "config_franka_0_7"
    },
    "stack_blocks_two": {
        "internal_name": "stack_blocks_two",
        "config": "config_arx_x5_0_7"
    },
    "place_bread_skillet": {
        "internal_name": "place_bread_skillet_vla", # Special case for internal naming
        "config": "config_franka_0_7"
    },
}

def main():
    parser = argparse.ArgumentParser(description="Low-Level End-Effector Control Evaluation")
    
    # 1. Arguments for users
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_CONFIG_MAP.keys()),
                        help="The task name to evaluate")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID(s) to use")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    # parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()

    # 2. Get the project root directory (auto-detect based on script location)
    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))

    # 3. Retrieve internal task mapping details
    task_info = TASK_CONFIG_MAP[args.task]
    internal_task_name = task_info["internal_name"]
    config_name = task_info["config"]
    
    policy_name = "vlmpolicy"

    # 4. Construct the execution command
    cmd = [
        "python", "script/myvlmEb_eval_vla.py",
        "--config", f"policy/{policy_name}/deploy_policy.yml",
        "--overrides",
        "--task_name", internal_task_name,
        "--task_config", config_name,
        "--ckpt_setting", config_name,
        "--seed", str(0),
        "--policy_name", policy_name,
        "--model_name", args.model
    ]

    # 5. Prepare environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Ensure 'envs' and other modules are importable from the root directory
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = root_dir + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = root_dir

    # Informative output for the user (Avoiding 'VLA' branding)
    print("=" * 70)
    print(" EVALUATION: Low-Level End-Effector Control")
    print("-" * 70)
    print(f" Task Name:      {args.task}")
    print(f" Robot Config:   {config_name}")
    print(f" Model Path:     {args.model}")
    print(f" GPU ID:         {args.gpu}")
    print(f" Root Directory: {root_dir}")
    print("=" * 70)

    # 6. Run the subprocess with root_dir as the working directory
    try:
        subprocess.run(cmd, env=env, cwd=root_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Evaluation failed with return code {e.returncode}")
    except KeyboardInterrupt:
        print("\n[Info] Evaluation stopped by user.")

if __name__ == "__main__":
    main()