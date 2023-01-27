CUDA_VISIBLE_DEVICES=7 \
python3 run_response_generator.py \
    --task training \
    --dataset_path /mnt/16tb/minyoung/code/photochat/dataset/photochat \
    --generator_model_name Salesforce/blip-vqa-base \
    --use_image_as_generator_input \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/blip-base \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy epoch \
    --seed 1234 \
    --report_to tensorboard