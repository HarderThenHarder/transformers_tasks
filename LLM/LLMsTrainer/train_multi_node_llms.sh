#!/bin/sh
accelerate_config=$1
train_config=$2
model_config=$3

USER_ENV=`whoami`

echo "============== Activate Multi Node Training =============="

master_addr=127.0.0.1
master_port=27633
nnodes=1
node_rank=0
num_processes=8

echo 

echo "MASTER IP: "$master_addr 
echo "MASTER PORT: "$master_port
echo "TOTAL NODES: "$nnodes
echo "NUM PROCESSES: "$num_processes
echo "CURRENT NODE RANK: "$node_rank

echo 
echo "[*] Waiting For Other Machines..."
echo 

accelerate launch --config_file $accelerate_config  \
    --main_process_ip $master_addr \
    --main_process_port $master_port \
    --machine_rank $node_rank \
    --num_machines $nnodes \
    --num_processes $num_processes train_llms.py \
    --train_config $train_config \
    --model_config $model_config