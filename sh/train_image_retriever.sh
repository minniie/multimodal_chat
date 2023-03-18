CUDA_VISIBLE_DEVICES=5 \
python3 run_image_retriever.py \
    --do_train \
    --dataset_path /mnt/16tb/minyoung/code/photochat/dataset/photochat \
    --retriever_image_encoder_path google/vit-base-patch16-224 \
    --retriever_text_encoder_path bert-large-uncased \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/vit_base_bert_large \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_strategy epoch \
    --seed 1234 \
    --report_to tensorboard