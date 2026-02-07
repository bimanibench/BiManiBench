#!/bin/bash

policy_name=vlmpolicy # [TODO] 
# task_name=place_burger_fries
# task_name=grab_roller
# task_name=place_object_scale
# task_name=stack_blocks_two
task_name=place_bread_skillet_vla

task_config=demo_vla
ckpt_setting=demo_vla
seed=0
gpu_id=3
# [TODO] add parameters here

model_name=gemini-2.0-flash

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

# PYTHONWARNINGS=ignore::UserWarning \
python script/myvlmEb_eval_vla.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --model_name ${model_name}
