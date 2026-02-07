#!/bin/bash

policy_name=vlmpolicy # [TODO] 

# task_name=blocks_ranking_rgb_spatial
task_name=blocks_five_spatial


task_config=demo_spatial
ckpt_setting=demo_spatial
seed=0
gpu_id=3
# [TODO] add parameters here

model_name=claude-sonnet-4-20250514

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

# PYTHONWARNINGS=ignore::UserWarning \
python script/myvlmEb_eval_spatial.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --model_name ${model_name}
