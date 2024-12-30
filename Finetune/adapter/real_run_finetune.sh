TYPE=GO
MODEL=CNN
DATA=genome_ar53
for RESO in 10240; do
    for MAX_LEN in 10240; do
        CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port 2011 main.py \
            --model_type $MODEL \
            --label_dir ./data/$DATA \
            --data_dir ./embedding/${TYPE}_embeddings_${RESO}_${MAX_LEN}/$DATA \
            --feat_dim 3072 \
            --output_dim 768 \
            --resolution $RESO \
            --model_max_length $MAX_LEN \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 2 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy steps \
            --logging_steps 100 \
            --num_train_epochs 200 \
            --save_steps 100 \
            --eval_steps 100 \
            --warmup_ratio 0.05 \
            --learning_rate 4e-5 \
            --save_total_limit 10000 \
            --do_train \
            --do_eval \
            --dataloader_num_workers 4 \
            --fp16 \
            --overwrite_output_dir \
            --output_dir ./output/$DATA/$MODEL/${TYPE}_${RESO}_${MAX_LEN} \
            --pj_name Genome_Ocean_Finetune \
            --log_name ${TYPE}_${DATA}_${MODEL}_${RESO}_${MAX_LEN}
    done
done