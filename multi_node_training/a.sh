torchrun --nproc_per_node=1 --node_rank=0 --master_addr="localhost" --master_port=12355 --nnodes=1  training.py \
    --config_name ./model/model_gpt2 \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file train.txt \
    --validation_file dev.txt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 128 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --find_unused_parameters False \
    --output_dir ./tmp/test-mixtral --remove_unused_columns=False
    



torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_gpt2 \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/train_5m.txt \
    --validation_file /root/data/pre-train/dev.txt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 128 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --find_unused_parameters False \
    --output_dir ./tmp/test-gpt2


torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_mamba \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/dev.txt \
    --validation_file /root/data/pre-train/dev.txt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 20 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 128 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mamba-128



torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/train_5m.txt \
    --validation_file /root/data/pre-train/dev.txt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 128 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mistral





torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_mambamoe \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/train_5m.txt \
    --validation_file /root/data/pre-train/dev.txt \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mambamoe-32





torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_mixtral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/train_5m.txt \
    --validation_file /root/data/pre-train/dev.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mixtral-8-1024


torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/train_5m.txt \
    --validation_file /root/data/pre-train/dev.txt \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mistral-1024






















torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_mamba \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/train.txt \
    --validation_file /root/data/pre-train/dev.txt \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 10000 \
    --eval_steps 20000000000000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 20 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mamba




torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_mixtral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/train.txt \
    --validation_file /root/data/pre-train/dev.txt \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 20000000000000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 20 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mixtral





torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_llama \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/train.txt \
    --validation_file /root/data/pre-train/dev.txt \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 10000 \
    --eval_steps 20000000000000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 20 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-llama



















for dir in $(ls -d */)
do
    echo $dir
    cat $dir/eval_results.json
    echo "===================="
done









# len5k

torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_mixtral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/train_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mixtral-8-1024-alldata


torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/train_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mistral-1024-alldata


torchrun --nproc_per_node=8 --rdzv-endpoint 0.0.0.0:11224 training.py \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/train_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --mlm \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mistral-1024-alldata-mlm-new



torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_mixtral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/train_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --mlm \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mixtral-8-1024-alldata-mlm



torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_mamba \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/train_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mamba-1024-alldata



torchrun --nproc_per_node=8 training.py \
    --config_name ./model/model_mambamoe \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/train_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mambamoe-32-1024-alldata





# debug mamba

torchrun --nproc_per_node=4 --rdzv-endpoint 0.0.0.0:11222 training.py \
    --config_name ./model/model_mamba \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/debug.txt \
    --validation_file /root/data/pre-train/debug.txt \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 10 \
    --eval_steps 10 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 128 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-debug-mamba-128




torchrun --nproc_per_node=4 --rdzv-endpoint 0.0.0.0:11222 training.py \
    --config_name ./model/model_mambamoe \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/debug.txt \
    --validation_file /root/data/pre-train/debug.txt \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 10 \
    --eval_steps 10 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 128 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-debug-mambamoe-128





# debug gemma


CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --rdzv-endpoint 0.0.0.0:11224 training.py \
    --config_name ./model/model_gemma \
    --model_name_or_path ./model/model_gemma \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/debug.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 3e-5 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-debug-1024-alldata




# no freeze
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --rdzv-endpoint 0.0.0.0:11224 training.py \
    --config_name ./model/model_gemma \
    --model_name_or_path ./model/model_gemma \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/train_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy steps \
    --logging_steps 20 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 5e-5 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-gemma-1024-alldata-nofreeze


# freeze
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=1 --rdzv-endpoint 0.0.0.0:11222 training.py \
    --config_name ./model/model_gemma \
    --model_name_or_path ./model/model_gemma \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/train_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy steps \
    --logging_steps 20 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --bf16 \
    --freeze_backbone \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-gemma-1024-alldata



CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=1 --rdzv-endpoint 0.0.0.0:11222 training.py \
    --config_name google/gemma-2b \
    --model_name_or_path google/gemma-2b \
    --tokenizer_name google/gemma-2b \
    --train_file /root/data/pre-train/debug.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 1 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --bf16 \
    --freeze_backbone \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-gemma-1024-alldata




CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --rdzv-endpoint 0.0.0.0:11224 training.py \
    --config_name ./model/model_gemma \
    --model_name_or_path ./tmp/test-gemma-1024-alldata/test \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/train_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy steps \
    --logging_steps 20 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-5 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-gemma-1024-alldata-fromfreeze




# mistral large


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 training.py \
    --config_name ./model/model_mistral_large \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/dev_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-debug




CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 training.py \
    --config_name ./model/model_mistral_large \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/train_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-debug





CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 training.py \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/BGC/train_bgc_2200k.txt \
    --validation_file /root/data/BGC/val_bgc.txt \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mistral-bgc-1024






## debug deepspeed


CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 training.py \
    --deepspeed ds_config.json \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/BGC/val_bgc.txt \
    --validation_file /root/data/BGC/val_bgc.txt \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.05 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mistral-bgc-1024



python training.py \
    --config_name ./model/model_mistral_4B \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --prepared_dataset /root/data/pre-train/len5k/prepared_1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --max_steps 50000 \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_steps 800 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 128 \
    --do_train \
    --do_eval \
    --bf16 \
    --preprocessing_num_workers 16 \
    --overwrite_output_dir \
    --output_dir ./tmp/test-debug



--deepspeed ./ds_config/zero1_fp32.json \
    --config_name ./model/model_mistral_500M \
    --tokenizer_name /pscratch/sd/z/zhihanz/bpe_metagenomics_4096 \
    --prepared_dataset /pscratch/sd/z/zhihanz/data/metegenomics/prepared/prepared_1024_new4096 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.1 \
    --evaluation_strategy steps \
    --max_steps 55000 \
    --logging_steps 1 \
    --save_steps 250 \
    --eval_steps 250000 \
    --lr_scheduler_type cosine \
    --warmup_steps 2000 \
    --learning_rate 4e-4 \
    --save_total_limit 1000 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --save_on_each_node False \
    --preprocessing_num_workers 16 \
    --cache_dir $CACHE_DIR \
    --output_dir /pscratch/sd/z/zhihanz/models/mistral_500M_1024_4M


CUDA_VISIBLE_DEVICES=0,1 deepspeed training.py \
    --deepspeed ds_config/zero1.json \
    --config_name ./model/model_mistral_500M \
    --tokenizer_name /root/data/pre-train/metagenomics/new_4096 \
    --prepared_dataset /root/data/pre-train/metagenomics/500_10k/prepared_1024_new4096 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy steps \
    --max_steps 50000 \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_steps 800 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --preprocessing_num_workers 16 \
    --overwrite_output_dir \
    --output_dir ./tmp/test-debug



deepspeed training.py \
    --deepspeed ds_config_large.json \
    --config_name ./model/model_mistral_large \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --prepared_dataset /pscratch/sd/z/zhihanz/data/len5k/prepared_1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.05 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir /pscratch/sd/z/zhihanz/tmp/debug-4B-mistral




deepspeed --num_gpus=1  training.py \
    --deepspeed ds_config.json \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /pscratch/sd/z/zhihanz/data/len5k/dev_len5k.txt \
    --validation_file /pscratch/sd/z/zhihanz/data/len5k/dev_len5k.txt \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 20 \
    --gradient_accumulation_steps 1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.05 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/debug


python -m torch.distributed.run \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr login05 \
    --master_port 9901 \
    training.py \
    --deepspeed ds_config.json \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /pscratch/sd/z/zhihanz/data/len5k/dev_len5k.txt \
    --validation_file /pscratch/sd/z/zhihanz/data/len5k/dev_len5k.txt \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.05 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir /pscratch/sd/z/zhihanz/tmp/debug





### test metagenomics data

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4  training.py \
    --deepspeed ds_config.json \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/metagenomics/gre_9m.txt \
    --validation_file /root/data/pre-train/metagenomics/gre_15k.txt \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-mistral-metagenomics-1024



python pre_tokenize_data.py \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/train_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --block_size 4096 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 32 \
    --output_dir /root/data/pre-train/len5k/prepared_4096


python pre_tokenize_data.py \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/len5k/train_len5k.txt \
    --validation_file /root/data/pre-train/len5k/dev_len5k.txt \
    --block_size 4096 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 32 \
    --output_dir /root/data/pre-train/len5k/prepared_4096





python pre_tokenize_data.py \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --train_file /root/data/pre-train/metagenomics/gre_15m.txt \
    --validation_file /root/data/pre-train/metagenomics/gre_15k.txt \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 32 \
    --output_dir /root/data/pre-train/metagenomics/gre_15m_gre_15k_1024




CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 training.py \
    --config_name ./model/model_mistral \
    --tokenizer_name zhihan1996/DNABERT-2-117M \
    --prepared_dataset /root/data/pre-train/len5k/prepared_1024 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --warmup_ratio 0.05 \
    --learning_rate 4e-4 \
    --save_total_limit 10 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/test-debug




for split in aa ab ac ad ae
do
    python pre_tokenize_data.py \
        --config_name ./model/model_mistral \
        --tokenizer_name /root/data/pre-train/metagenomics/new_4096 \
        --train_file /root/data/pre-train/metagenomics/500_10k/train_${split}.txt \
        --validation_file /root/data/pre-train/metagenomics/500_10k/dev.txt \
        --block_size 1024 \
        --do_train \
        --do_eval \
        --overwrite_output_dir \
        --preprocessing_num_workers 64 \
        --output_dir /root/data/pre-train/metagenomics/500_10k/prepared_1024_new4096_${split} 
done



python training.py \
    --deepspeed ./ds_config/zero1_bf16_uni.json \
    --config_name ./model/model_mistral_4B \
    --tokenizer_name /pscratch/sd/z/zhihanz/bpe_metagenomics_4096 \
    --prepared_dataset /pscratch/sd/z/zhihanz/data/metegenomics/prepared/prepared_1024_new4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.1 \
    --evaluation_strategy steps \
    --max_steps 65000 \
    --logging_steps 1 \
    --save_steps 5 \
    --eval_steps 250000 \
    --lr_scheduler_type cosine \
    --warmup_steps 2000 \
    --learning_rate 4e-4 \
    --save_total_limit 1000 \
    --block_size 32 \
    --do_train \
    --do_eval \
    --bf16 \
    --save_on_each_node False \
    --preprocessing_num_workers 16 \
    --cache_dir /pscratch/sd/z/zhihanz/hf_cache \
    --output_dir /pscratch/sd/z/zhihanz/models/mistral_4B_1024_new


python compute_embedding.py \
    --model_path /root/DNABERT_3/models/4B \
    --data_path /root/MOE_DNA/example_data/debug.txt \
    --output_path /root/MOE_DNA/example_data/debug_embedding.npy \
    --model_max_length 1024 \
    --batch_size 8 


python pre_tokenize_data.py \
    --config_name ./model/model_mistral \
    --tokenizer_name /root/data/pre-train/metagenomics/new_4096 \
    --train_file /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max20000/train.txt \
    --validation_file /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max20000/dev.txt \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 64 \
    --output_dir /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max20000/prepared_1024



python pre_tokenize_data.py \
    --config_name ./model/model_mistral \
    --tokenizer_name /root/data/pre-train/metagenomics/new_4096 \
    --train_file /root/data/pre-train/BGC/allbgcs_len10k_train.txt \
    --validation_file /root/data/pre-train/BGC/allbgcs_len10k_validation.txt \
    --block_size 10240 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 64 \
    --output_dir /root/data/pre-train/BGC/prepared_10240

python pre_tokenize_data.py \
    --config_name ./model/model_mistral \
    --tokenizer_name /root/data/pre-train/metagenomics/new_4096 \
    --train_file /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max100000/train.txt \
    --validation_file /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max100000/dev.txt \
    --block_size 20480 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 64 \
    --output_dir /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max100000/prepared_20480




python pre_tokenize_data.py \
    --config_name ./model/model_mistral \
    --tokenizer_name /root/data/pre-train/metagenomics/new_4096 \
    --train_file /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max100000/train.txt \
    --validation_file /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max100000/dev.txt \
    --block_size 20480 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 64 \
    --output_dir /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max100000/prepared_20480


python pre_tokenize_data.py \
    --config_name ./model/model_mistral \
    --tokenizer_name /root/data/pre-train/metagenomics/new_4096 \
    --train_file /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max50000/train.txt \
    --validation_file /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max50000/dev.txt \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 64 \
    --output_dir /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max50000/prepared_1024


python pre_tokenize_data.py \
    --config_name ./model/model_mistral \
    --tokenizer_name /root/data/pre-train/metagenomics/new_4096 \
    --train_file /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max50000/train.txt \
    --validation_file /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max50000/dev.txt \
    --block_size 10240 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --preprocessing_num_workers 64 \
    --output_dir /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max50000/prepared_10240


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

m1 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B",
    device_map="cpu",
    trust_remote_code=True
).eval()

m2 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-1_8B",
    device_map="cpu",
    trust_remote_code=True
).eval()

m3 = AutoModelForCausalLM.from_pretrained(
    "/root/weiminwu/dnabert-3/llm2vec/model/meta-100M",
    device_map="cpu",
    trust_remote_code=True
).eval()

m4 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-1.8B",
    device_map="cpu",
    trust_remote_code=True
).eval()


CUDA_VISIBLE_DEVICES=2 torchrun \
    --nproc_per_node 1 \
    --master-port=17673 \
    training.py \
    --config_name ./model/model_qwenmoe_8_100M_random \
    --tokenizer_name /root/data/pre-train/metagenomics/new_4096/ \
    --model_name_or_path /root/MOE_DNA/trained_model/qwenmoe_8_100M_random \
    --prepared_dataset /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max50000/prepared_1024 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.1 \
    --evaluation_strategy steps \
    --max_steps 120000 \
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
    --preprocessing_num_workers 16 \
    --output_dir ./tmp/debug-qwenmoe-8-100M-random


CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 training.py \
    --config_name ./model/model_qwenmoe_8_100M \
    --tokenizer_name /root/data/pre-train/metagenomics/new_4096/ \
    --prepared_dataset /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max50000/prepared_1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.1 \
    --evaluation_strategy steps \
    --max_steps 55000 \
    --logging_steps 1 \
    --save_steps 250 \
    --eval_steps 250000 \
    --lr_scheduler_type cosine \
    --warmup_steps 2000 \
    --learning_rate 4e-4 \
    --save_total_limit 1000 \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --save_on_each_node False \
    --preprocessing_num_workers 16 \
    --overwrite_output_dir \
    --output_dir ./tmp/mixtral_700M_1024



CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 training.py \
    --config_name ./model/model_mistral_4B \
    --tokenizer_name /root/data/pre-train/metagenomics/new_4096/ \
    --prepared_dataset /root/data/pre-train/metagenomics/processed/hmp_meta_min500_max50000/prepared_1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.1 \
    --evaluation_strategy steps \
    --max_steps 55000 \
    --logging_steps 1 \
    --save_steps 250 \
    --eval_steps 250000 \
    --lr_scheduler_type cosine \
    --warmup_steps 2000 \
    --learning_rate 4e-4 \
    --save_total_limit 1000 \
    --load_in_8bit \
    --low_cpu_mem_usage \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --bf16 \
    --save_on_each_node False \
    --preprocessing_num_workers 16 \
    --overwrite_output_dir \
    --output_dir ./tmp/mixtral_700M_1024