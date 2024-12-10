# torchrun --nproc_per_node 1 \
#     --nnodes 1 \
#     --node_rank 0 \
#     --master_addr localhost \
#     --master_port 12355 \
#     training.py \
#     --deepspeed ./ds_config/zero3.json \
#     --config_name ./model/model_qwenmoe_8_100M_random \
#     --tokenizer_name zhihan1996/DNABERT-2-117M \
#     --train_file train.txt \
#     --validation_file dev.txt \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 \
#     --adam_beta1 0.9 \
#     --adam_beta2 0.95 \
#     --adam_epsilon 1e-5 \
#     --weight_decay 0.1 \
#     --evaluation_strategy steps \
#     --max_steps 1200 \
#     --logging_steps 1 \
#     --save_steps 1000 \
#     --eval_steps 1000 \
#     --lr_scheduler_type cosine \
#     --warmup_steps 4000 \
#     --learning_rate 1e-4 \
#     --save_total_limit 1000 \
#     --block_size 1024 \
#     --do_train \
#     --do_eval \
#     --bf16 \
#     --save_on_each_node False \
#     --preprocessing_num_workers 1 \
#     --output_dir ./tmp/qwenmoe_8_100M_random

# deepspeed --num_gpus 1 \
#     --master_addr localhost \
#     --master_port 12355 \
#     training.py \
#     --deepspeed ./ds_config/zero3.json \
#     --config_name ./model/model_qwenmoe_8_100M_random \
#     --tokenizer_name zhihan1996/DNABERT-2-117M \
#     --train_file train.txt \
#     --validation_file dev.txt \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 \
#     --adam_beta1 0.9 \
#     --adam_beta2 0.95 \
#     --adam_epsilon 1e-5 \
#     --weight_decay 0.1 \
#     --evaluation_strategy steps \
#     --max_steps 1200 \
#     --overwrite_output_dir \
#     --logging_steps 1 \
#     --save_steps 1000 \
#     --eval_steps 1000 \
#     --lr_scheduler_type cosine \
#     --warmup_steps 4000 \
#     --learning_rate 1e-4 \
#     --save_total_limit 1000 \
#     --block_size 1024 \
#     --do_train \
#     --do_eval \
#     --bf16 \
#     --save_on_each_node False \
#     --preprocessing_num_workers 1 \
#     --output_dir ./tmp/qwenmoe_8_100M_random


 deepspeed  --master_addr localhost --node_rank 1 --include="0.0.0.0:0@localhost:1" -H host.txt --no_ssh \
 training.py \
    --deepspeed ./ds_config/zero3.json \
    --config_name ./model/model_mistral_500M \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file train.txt \
    --validation_file dev.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.1 \
    --evaluation_strategy steps \
    --max_steps 1200 \
    --overwrite_output_dir \
    --logging_steps 1 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --lr_scheduler_type cosine \
    --warmup_steps 4000 \
    --learning_rate 1e-4 \
    --save_total_limit 1000 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --save_on_each_node False \
    --preprocessing_num_workers 1 \
    --output_dir ./tmp/qwenmoe_8_100M_random