#!/bin/sh
accelerate_config=$1
train_config=$2

accelerate launch --config_file $accelerate_config \
    --main_process_port 27643 train_reward_model.py \
    --train_config $train_config 