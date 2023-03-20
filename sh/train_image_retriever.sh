CUDA_VISIBLE_DEVICES=2 \
python3 run_image_retriever.py \
    --do_train \
    --dataset_path /mnt/16tb/minyoung/code/photochat/dataset/photochat \
    --retriever_image_encoder_path google/vit-large-patch32-384 \
    --retriever_text_encoder_path bert-large-uncased \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/vit_large_bert_large_5678 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-4 \
    --save_strategy epoch \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --logging_steps 100 \
    --seed 5678 \
    --report_to tensorboard