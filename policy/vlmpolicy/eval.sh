#!/bin/bash

policy_name=vlmpolicy # [TODO] 
# task_name=place_cans_plasticbox
# task_name=lift_pot
# task_name=blocks_ranking_rgb
# task_name=blocks_ranking_size
task_name=place_burger_fries
# task_name=put_object_cabinet
# task_name=hanging_mug
# task_name=handover_block
# task_name=handover_mic
# task_name=place_object_basket
# task_name=stack_blocks_three
# task_name=place_bread_skillet
# task_name=pick_dual_bottles
# task_name=stack_bowls_three
# task_name=put_bottles_dustbin
# task_name=grab_roller
# task_name=blocks_tower
# task_name=blocks_cross_shape
# task_name=blocks_cross

task_config=demo_randomized
ckpt_setting=demo_randomized
seed=0
gpu_id=2

# model_name=/data3/wuxin/model/Qwen/Qwen3-VL-30B-A3B-Instruct
# model_name=/data/wuxin/vlm/internVL/OpenGVLab/InternVL2_5-8B
# model_name=/data/wuxin/vlm/Qwen/Qwen2.5-VL-7B-Instruct
# model_name=/data/wuxin/vlm/internVL/OpenGVLab/InternVL3-78B-AWQ
# model_name=claude-sonnet-4-20250514
# model_name=/data3/wuxin/model/AIDC-AI/Ovis2-16B
model_name=/data3/wuxin/model/LLM-Research/Llama-4-Scout-17B-16E-Instruct

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

# PYTHONWARNINGS=ignore::UserWarning \
python script/myvlmEb_eval.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}\
    --model_name ${model_name}
