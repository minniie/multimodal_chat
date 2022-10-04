CUDA_VISIBLE_DEVICES=7 \
python3 train_image_retriever.py \
    --text_model_name bert-base-uncased \
    --image_model_name google/vit-base-patch16-224 \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/image_retriever_dummy \
    --save_strategy steps \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 1 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 12 \
    --prediction_loss_only True \
    --report_to tensorboard