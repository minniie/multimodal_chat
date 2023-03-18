CUDA_VISIBLE_DEVICES=0 \
python3 run_image_retriever.py \
    --do_eval \
    --dataset_path /mnt/16tb/minyoung/code/photochat/dataset/photochat \
    --encoding_path /mnt/16tb/minyoung/checkpoints/photochat/vit_base_bert_base/checkpoint-643/image_encodings.pt \
    --retriever_image_encoder_path google/vit-base-patch16-224 \
    --retriever_text_encoder_path bert-base-uncased \
    --retriever_finetuned_path /mnt/16tb/minyoung/checkpoints/photochat/vit_base_bert_base/checkpoint-643 \
    --output_dir /mnt/16tb/minyoung/checkpoints/photochat/dummy \
    --per_device_eval_batch_size 16 \
    --seed 1234 \
    --report_to tensorboard