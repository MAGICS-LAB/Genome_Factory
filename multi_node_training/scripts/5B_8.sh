#!/bin/bash
#SBATCH -J mistral-5B             # name
#SBATCH -q regular
#SBATCH --mail-user=zzh760998379@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -A m4479_g             # allocation
#SBATCH -t 24:00:00          # time
#SBATCH -N 8                 # nodes
#SBATCH -G 32                 # gpus in each node
#SBATCH -C gpu               # gpu type
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10


export NUM_NODES=8
export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export WANDB_API_KEY=95f30abbd09d1c519dcc212b1e38b765ae007a48
export CACHE_DIR=/pscratch/sd/z/zhihanz/hf_cache

# rm -rf $CACHE_DIR
# mkdir -p $CACHE_DIR

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT node_rank=$SLURM_PROCID world_size=$NUM_NODES gpus_per_node=$GPUS_PER_NODE"
# printenv


module load python
conda activate py311

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    training.py \
    --deepspeed ./ds_config/zero1_bf16.json \
    --config_name ./model/model_mistral_5B \
    --tokenizer_name /pscratch/sd/z/zhihanz/bpe_metagenomics_4096 \
    --prepared_dataset /pscratch/sd/z/zhihanz/data/metegenomics/prepared/prepared_1024_new4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
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
    --output_dir /pscratch/sd/z/zhihanz/models/mistral_5B_1024'


