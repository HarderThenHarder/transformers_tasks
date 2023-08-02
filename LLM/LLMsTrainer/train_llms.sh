#!/bin/sh
accelerate_config=$1
train_config=$2
model_config=$3

accelerate launch --config_file $accelerate_config --main_process_port 27643 train_llms.py \
    --train_config $train_config \
    --model_config $model_config