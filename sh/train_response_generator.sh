CUDA_VISIBLE_DEVICES=2 \
python3 train_response_generator.py \
    --generator_model_name gpt2 \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/v1_dummy \
    --save_strategy steps \
    --save_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --report_to tensorboard