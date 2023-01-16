CUDA_VISIBLE_DEVICES=7 \
python3 train_response_generator.py \
    --dataset_path /mnt/16tb/minyoung/code/photochat/dataset/photochat \
    --generator_model_name gpt2-large \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/gpt2_large_dummy \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 500 \
    --seed 1234 \
    --report_to tensorboard