# LoRA Finetune
python train.py \
    --train_path data/mixed_train_dataset.jsonl \
    --dev_path data/mixed_dev_dataset.jsonl \
    --use_lora True \
    --lora_rank 8 \
    --batch_size 1 \
    --num_train_epochs 2 \
    --save_freq 1000 \
    --learning_rate 3e-5 \
    --logging_steps 100 \
    --max_source_seq_len 400 \
    --max_target_seq_len 300 \
    --save_dir checkpoints/finetune \
    --img_log_dir "log/fintune_log" \
    --img_log_name "ChatGLM Fine-Tune" \
    --device cuda:0


# P-Tuning
# python train.py \
#     --train_path data/mixed_train_dataset.jsonl \
#     --dev_path data/mixed_dev_dataset.jsonl \
#     --use_ptuning True \
#     --pre_seq_len 128 \
#     --batch_size 1 \
#     --num_train_epochs 2 \
#     --save_freq 200 \
#     --learning_rate 2e-4 \
#     --logging_steps 100 \
#     --max_source_seq_len 400 \
#     --max_target_seq_len 300 \
#     --save_dir checkpoints/ptuning \
#     --img_log_dir "log/fintune_log" \
#     --img_log_name "ChatGLM P-Tuning" \
#     --device cuda:0