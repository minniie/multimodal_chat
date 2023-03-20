CUDA_VISIBLE_DEVICES=3 \
python3 run_response_generator.py \
    --do_train \
    --dataset_path /mnt/16tb/minyoung/code/photochat/dataset/photochat \
    --generator_image_encoder_path google/vit-large-patch32-384 \
    --generator_text_decoder_path microsoft/DialoGPT-medium \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/vit_large_dialogpt_medium \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 4 \
    --save_strategy epoch \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --logging_steps 500 \
    --seed 1234 \
    --report_to tensorboard