CUDA_VISIBLE_DEVICES=3 \
python3 run_response_generator.py \
    --task training \
    --seed 1234 \
    --dataset_path /mnt/16tb/minyoung/code/photochat/dataset/photochat \
    --generator_model_name gpt2-large \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/gpt2_large \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --report_to tensorboard