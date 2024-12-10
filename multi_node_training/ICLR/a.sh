for data in  real_go
do
    CUDA_VISIBLE_DEVICES=1 python train.py \
        --model_name_or_path kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16  \
        --data_path  /root/MOE_DNA/ICLR/classification/species/$data \
        --kmer -1 \
        --run_name debug_caduceus_$data \
        --model_max_length 2048 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 3e-4 \
        --num_train_epochs 20 \
        --fp16 \
        --save_steps 200 \
        --output_dir output/debug_caduceus/$data \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_ratio 0.05 \
        --logging_steps 10 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False
done

for data in  real_reorder real_evo real_genslm real_go
do
    CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=12313 train.py \
        --model_name_or_path zhihan1996/DNABERT-2-117M \
        --data_path  /root/MOE_DNA/ICLR/classification/pairwise/$data \
        --kmer -1 \
        --run_name debug_nt_$data \
        --model_max_length 512 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 3e-5 \
        --num_train_epochs 20 \
        --fp16 \
        --save_steps 200 \
        --output_dir output/debug_nt \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_ratio 0.05 \
        --logging_steps 100000 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False
done

for data in   go_real
do
    CUDA_VISIBLE_DEVICES=1 python train.py \
        --model_name_or_path zhihan1996/DNABERT-2-117M \
        --data_path  /root/MOE_DNA/ICLR/classification/species/$data \
        --kmer -1 \
        --run_name debug_dnabert2_$data \
        --model_max_length 512 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 3e-5 \
        --num_train_epochs 20 \
        --fp16 \
        --save_steps 200 \
        --output_dir output/debug_dnabert2_1 \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_ratio 0.05 \
        --logging_steps 100000 \
        --overwrite_output_dir True \
        --greater_is_better True \
        --metric_for_best_model f1 \
        --log_level info \
        --find_unused_parameters False
done



for data in   go_real 
do
    CUDA_VISIBLE_DEVICES=3 python train.py \
        --model_name_or_path LongSafari/hyenadna-small-32k-seqlen-hf \
        --data_path  /root/MOE_DNA/ICLR/classification/species_1_500/$data \
        --kmer -1 \
        --run_name ddddd$data \
        --model_max_length 2000 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 3e-5 \
        --num_train_epochs 20 \
        --fp16 \
        --save_steps 200 \
        --output_dir output/debugddd \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_ratio 0.05 \
        --logging_steps 100000 \
        --overwrite_output_dir True \
        --greater_is_better True \
        --metric_for_best_model f1 \
        --log_level info \
        --find_unused_parameters False
done




for data in real_reorder
do
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=12314 train.py \
        --model_name_or_path InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
        --data_path  /root/MOE_DNA/ICLR/classification/species/$data \
        --kmer -1 \
        --run_name debug_nt_$data \
        --model_max_length 512 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 3e-5 \
        --num_train_epochs 20 \
        --fp16 \
        --save_steps 200 \
        --output_dir output/debug_nt_$data \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_ratio 0.05 \
        --logging_steps 100000 \
        --overwrite_output_dir True \
        --greater_is_better True \
        --metric_for_best_model f1 \
        --log_level info \
        --find_unused_parameters False
done



for data in  real_rand_reorder
do
    CUDA_VISIBLE_DEVICES=5 python train.py \
        --regression \
        --model_name_or_path zhihan1996/DNABERT-2-117M \
        --data_path  /root/MOE_DNA/ICLR/regression/$data \
        --kmer -1 \
        --run_name debug_dnabert2_$data \
        --model_max_length 512 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 256 \
        --gradient_accumulation_steps 1 \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --fp16 \
        --save_steps 500 \
        --output_dir output/dnabert2_regression \
        --evaluation_strategy steps \
        --eval_steps 500 \
        --warmup_ratio 0.05 \
        --logging_steps 1 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False
done



for data in  ref_1/no_aug
do
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=12342 train.py \
        --model_name_or_path zhihan1996/DNABERT-2-117M \
        --data_path  /root/MOE_DNA/ICLR/augmentation/$data \
        --kmer -1 \
        --run_name debug_dnabert2_$data \
        --model_max_length 512 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate 3e-5 \
        --num_train_epochs 40 \
        --fp16 \
        --save_steps 200 \
        --output_dir output/debug_aug_dnabert2 \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_ratio 0.05 \
        --logging_steps 100 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False
done


for data in  ref_1/aug
do
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=1234 train.py \
        --model_name_or_path zhihan1996/DNABERT-2-117M \
        --data_path  /root/MOE_DNA/ICLR/augmentation/$data \
        --kmer -1 \
        --run_name debug_dnabert2_$data \
        --model_max_length 1024 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate 3e-5 \
        --num_train_epochs 40 \
        --fp16 \
        --save_steps 200 \
        --output_dir output/debug_aug_dnabert2 \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_ratio 0.05 \
        --logging_steps 100 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False
done


CUDA_VISIBLE_DEVICES=5 python generate_go.py \
    --temperature 0.7 \
    --presence_penalty 0.0 \
    --frequency_penalty 0.0 \
    --repetition_penalty 1.0 \
    --num_generation_from_each_prompt 1 \
    --min_length 700 \
    --max_length 1000 \
    --data_dir /root/MOE_DNA/ICLR/augmentation/ref_1/no_aug/train.csv


for temp in 0.5 0.7 1 
    do
    for presence_penalty in 0.0 0.5
        do
        for frequency_penalty in 0.0 0.5
            do
            for repetition_penalty in 1.0 1.5
                do

                    CUDA_VISIBLE_DEVICES=4 python generate_go.py \
                        --temperature $temp \
                        --presence_penalty $presence_penalty \
                        --frequency_penalty $frequency_penalty \
                        --repetition_penalty $repetition_penalty \

                done
            done
        done
    done
done


CUDA_VISIBLE_DEVICES=5 python generate_go.py \
    --temperature 1.0 \
    --presence_penalty 0.0 \
    --frequency_penalty 0.0 \
    --repetition_penalty 1.0 \
    --data_dir /root/MOE_DNA/ICLR/augmentation/ref_1/no_aug/train.csv

for temp in 0.3 0.5 1 
    do
    for presence_penalty in 0.0 0.5
        do
        for frequency_penalty in 0.0 0.5
            do
            for repetition_penalty in 1.0 1.5
                do

                    CUDA_VISIBLE_DEVICES=4 python generate_go.py \
                        --num_generation_from_each_prompt 1 \
                        --temperature $temp \
                        --presence_penalty $presence_penalty \
                        --frequency_penalty $frequency_penalty \
                        --repetition_penalty $repetition_penalty \

                done
            done
        done
    done
done



for temp in 0.3 0.7 1 
do
    for presence_penalty in 0.5 
    do
    for frequency_penalty in 0.5 
        do
        for repetition_penalty in 1.0 
            do

                CUDA_VISIBLE_DEVICES=4 python generate_go.py \
                    --num_generation_from_each_prompt 1 \
                    --data_dir /root/MOE_DNA/ICLR/data/coding_non_coding.csv \
                    --output_dir /root/MOE_DNA/ICLR/generated/coding_non_coding \
                    --max_output_bp 2000 \
                    --temperature $temp \
                    --presence_penalty $presence_penalty \
                    --frequency_penalty $frequency_penalty \
                    --repetition_penalty $repetition_penalty \

            done
        done
    done
done



export temp=0.3
CUDA_VISIBLE_DEVICES=0 python run_evo.py --temperature $temp 


export temp=0.5
CUDA_VISIBLE_DEVICES=1 python run_evo.py --temperature $temp

export temp=0.7
CUDA_VISIBLE_DEVICES=2 python run_evo.py --temperature $temp


export temp=1
CUDA_VISIBLE_DEVICES=5 python run_genslm.py --temperature $temp --model-name genslm_250M_patric


export temp=1
CUDA_VISIBLE_DEVICES=0 python run_evo.py \
            --temperature $temp \
            --data_dir /root/MOE_DNA/ICLR/data/coding_non_coding.csv \
            --output_dir /root/MOE_DNA/ICLR/generated/coding_non_coding



for temp in 1 
do
    for presence_penalty in 0.5 
    do
    for frequency_penalty in 0.5 
        do
        for repetition_penalty in 1.0 
            do

                CUDA_VISIBLE_DEVICES=3 python generate_go.py \
                    --num_generation_from_each_prompt 1 \
                    --model_size 4B \
                    --data_dir /root/MOE_DNA/ICLR/data/cai/extracted_random_contigs.csv \
                    --output_dir /root/MOE_DNA/ICLR/generated/cai \
                    --max_output_bp 2100 \
                    --temperature $temp \
                    --presence_penalty $presence_penalty \
                    --frequency_penalty $frequency_penalty \
                    --repetition_penalty $repetition_penalty \

            done
        done
    done
done



for temp in 1 
do
    for presence_penalty in 0.5 
    do
    for frequency_penalty in 0.5 
        do
        for repetition_penalty in 1.0 
            do

                CUDA_VISIBLE_DEVICES=3 python generate_go.py \
                    --num_generation_from_each_prompt 1 \
                    --model_size 4B \
                    --data_dir /root/MOE_DNA/ICLR/data/microbes_cds_filtered.csv \
                    --output_dir /root/MOE_DNA/ICLR/generated/cai \
                    --max_prompt_bp 2100 \
                    --max_output_bp 2100 \
                    --temperature $temp \
                    --min_length 600 \
                    --max_length 1000 \
                    --presence_penalty $presence_penalty \
                    --frequency_penalty $frequency_penalty \
                    --repetition_penalty $repetition_penalty \

            done
        done
    done
done




export prompt=500
CUDA_VISIBLE_DEVICES=0 python generate_go.py \
    --num_generation_from_each_prompt 1 \
    --model_size 4B \
    --data_dir /root/data/cami2/marine_plant_30_known.tsv \
    --output_dir /root/MOE_DNA/ICLR/generated/known/$prompt/ \
    --max_prompt_bp $prompt \
    --max_output_bp -1 \
    --temperature 1.0 \
    --min_length 2000 \
    --max_length 2000 \
    --presence_penalty 0.5 \
    --frequency_penalty 0.5 \
    --repetition_penalty 1.0 


export prompt=2000
CUDA_VISIBLE_DEVICES=3 python generate_go.py \
    --num_generation_from_each_prompt 1 \
    --model_size 4B \
    --data_dir /root/MOE_DNA/ICLR/data/coding_non_coding.csv \
    --output_dir /root/MOE_DNA/ICLR/generated/coding_non_coding \
    --max_prompt_bp $prompt \
    --max_output_bp -1 \
    --temperature 1.0 \
    --min_length 600 \
    --max_length 1000 \
    --presence_penalty 0.5 \
    --frequency_penalty 0.5 \
    --repetition_penalty 1.0 


export prompt=2000
for  model_size in 500M 100M
do
    CUDA_VISIBLE_DEVICES=3 python generate_go.py \
        --num_generation_from_each_prompt 1 \
        --model_size ${model_size} \
        --data_dir /root/data/cami2/marine_plant_20_unknown.tsv \
        --output_dir /root/MOE_DNA/ICLR/generated/unknown/${prompt}_${model_size}/ \
        --max_prompt_bp $prompt \
        --max_output_bp 2000 \
        --temperature 1.0 \
        --min_length 600 \
        --max_length 1000 \
        --presence_penalty 0.5 \
        --frequency_penalty 0.5 \
        --repetition_penalty 1.0 
done


export temp=1
CUDA_VISIBLE_DEVICES=3 python run_evo.py \
            --temperature $temp \
            --data_dir /root/MOE_DNA/ICLR/data/microbes_cds_filtered.csv \
            --output_dir /root/MOE_DNA/ICLR/generated/cai \
            --start_idx 0 \
            --end_idx 233


export temp=1
CUDA_VISIBLE_DEVICES=4 python run_evo.py \
            --temperature $temp \
            --data_dir /root/MOE_DNA/ICLR/data/microbes_cds_filtered.csv \
            --output_dir /root/MOE_DNA/ICLR/generated/cai \
            --start_idx 233 \
            --end_idx 466



export temp=1
CUDA_VISIBLE_DEVICES=5 python run_evo.py \
            --temperature $temp \
            --data_dir /root/MOE_DNA/ICLR/data/microbes_cds_filtered.csv \
            --output_dir /root/MOE_DNA/ICLR/generated/cai \
            --start_idx 466 \
            --end_idx 700


export temp=1
CUDA_VISIBLE_DEVICES=3 python run_evo.py \
            --temperature $temp \
            --data_dir /root/data/cami2/marine_plant_30_known.tsv \
            --output_dir /root/MOE_DNA/ICLR/generated/known \
            --start_idx 0 \
            --end_idx 1500


export temp=1
CUDA_VISIBLE_DEVICES=4 python run_evo.py \
            --temperature $temp \
            --data_dir /root/data/cami2/marine_plant_30_known.tsv \
            --output_dir /root/MOE_DNA/ICLR/generated/known \
            --start_idx 1500 \
            --end_idx 3000


export prompt=1000
export model_size=4B
CUDA_VISIBLE_DEVICES=2 python generate_go.py \
    --num_generation_from_each_prompt 1 \
    --model_size ${model_size} \
    --data_dir /root/data/cami2/marine_plant_30_known.tsv \
    --output_dir /root/MOE_DNA/ICLR/generated/known/${prompt}_${model_size}/ \
    --max_prompt_bp $prompt \
    --max_output_bp 2048 \
    --temperature 1.0 \
    --min_length 600 \
    --max_length 600 \
    --presence_penalty 0.5 \
    --frequency_penalty 0.5 \
    --repetition_penalty 1.0 





python display_distribution.py \
        -i ./CODE_NON_CODE/orf_stick/ \
        -o ./CODE_NON_CODE/orf_stick_pdf_new/ \
        -f best_orf_length \
        -s '[["ground truth", [1, 501]], ["genlsm", [501, 1001]], ["we", [1001, 1501]], ["evo", [1501, 2001]]]'


