python train.py \
    --pretrained_model "uie-base-zh" \
    --save_dir "checkpoints/simple_ner" \
    --train_path "data/simple_ner/train.txt" \
    --dev_path "data/simple_ner/dev.txt" \
    --img_log_dir "logs/simple_ner" \
    --img_log_name "UIE Base" \
    --batch_size 8 \
    --max_seq_len 128 \
    --num_train_epochs 100 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device "cuda:0"