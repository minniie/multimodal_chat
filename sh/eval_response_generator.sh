CUDA_VISIBLE_DEVICES=5 \
python3 run_response_generator.py \
    --do_eval \
    --dataset_path /mnt/16tb/minyoung/code/photochat/dataset/photochat \
    --generator_image_encoder_path google/vit-large-patch32-384 \
    --generator_text_decoder_path microsoft/DialoGPT-medium \
    --generator_finetuned_path /mnt/16tb/minyoung/checkpoints/photochat/vit_large_dialogpt_medium/checkpoint-11174 \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/dummy \
    --per_device_eval_batch_size 4 \
    --seed 1234 \
    --report_to tensorboard