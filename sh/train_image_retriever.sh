CUDA_VISIBLE_DEVICES=6 \
python3 train_image_retriever.py \
    --text_model_name bert-base-uncased \
    --image_model_name google/vit-base-patch16-224 \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/bert_vit \
    --save_strategy steps \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 16 \
    --prediction_loss_only True \
    --report_to tensorboard