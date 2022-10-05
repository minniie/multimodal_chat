CUDA_VISIBLE_DEVICES=5 \
python3 train_response_generator.py \
    --dataset_path /mnt/16tb/minyoung/code/photochat/dataset/dummy \
    --generator_model_name gpt2-large \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/gpt2_large \
    --save_strategy steps \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 1 \
    --report_to tensorboard