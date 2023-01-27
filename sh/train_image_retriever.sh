CUDA_VISIBLE_DEVICES=3 \
python3 run_image_retriever.py \
    --task training \
    --dataset_path /mnt/16tb/minyoung/code/photochat/dataset/photochat \
    --text_model_name bert-base-uncased \
    --image_model_name google/vit-base-patch16-224 \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/bert_vit_dummy \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_strategy epoch \
    --seed 1234 \
    --report_to tensorboard