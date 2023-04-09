# LoRA Finetune
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train_multi_gpu.py \
    --train_path data/mixed_train_dataset.jsonl \
    --dev_path data/mixed_dev_dataset.jsonl \
    --use_lora True \
    --lora_rank 8 \
    --batch_size 1 \
    --num_train_epochs 2 \
    --save_freq 500 \
    --learning_rate 3e-5 \
    --logging_steps 100 \
    --max_source_seq_len 400 \
    --max_target_seq_len 300 \
    --save_dir checkpoints_parrallel/finetune \
    --img_log_dir "log/fintune_log" \
    --img_log_name "ChatGLM Fine-Tune(parallel)"


# P-Tuning
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train_multi_gpu.py \
#     --train_path data/mixed_train_dataset.jsonl \
#     --dev_path data/mixed_dev_dataset.jsonl \
#     --use_ptuning True \
#     --pre_seq_len 128 \
#     --batch_size 1 \
#     --num_train_epochs 2 \
#     --save_freq 500 \
#     --learning_rate 2e-4 \
#     --logging_steps 100 \
#     --max_source_seq_len 400 \
#     --max_target_seq_len 300 \
#     --save_dir checkpoints_parrallel/ptuning \
#     --img_log_dir "log/fintune_log" \
#     --img_log_name "ChatGLM P-Tuning(parallel)"