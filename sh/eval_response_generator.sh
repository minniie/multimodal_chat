CUDA_VISIBLE_DEVICES=5 \
python3 run_response_generator.py \
    --do_eval \
    --dataset_path /mnt/16tb/minyoung/code/photochat/dataset/photochat \
    --generator_text_decoder_path /mnt/16tb/minyoung/checkpoints/photochat/dialogpt_medium/checkpoint-6500 \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/dummy \
    --per_device_eval_batch_size 4 \
    --seed 1234 \
    --report_to tensorboard