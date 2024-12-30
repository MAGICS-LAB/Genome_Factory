# Finetuning with Model Adapter (Phylogenetic Distance Prediction)

## Dataset Set

The datasets include two parts for train/val/test
### (i) One '.txt' file, each line is a DNA sequence. 
  Load the file using

  #### code
  with open('./data/genome_ar53/train.txt', "r") as f:
      dna_sequences = f.read().splitlines()

### (ii) One '.csv' file, the shape is N * N (N is the number of sequences in the '.txt' file)
  Each element denote the true phylogenetic distance between every two dna sequences
  Load the file using

  #### code
  labels = pd.read_csv('./data/genome_ar53/train/label.csv', index_col=None, header=None)
  
## Embedding Generation
You need to generate the embedding for the '.txt' file before training the model.
The code is './data/generate_embedding.py'

To run the code, you can follow these steps:

TYPE=GO
MODEL_PATH=./mistral_4B_meta55k_hmp4k_seq10240/checkpoint-1600
for RESO in 10240 320 160 80; do
    for MAX_LEN in 10240; do
        for DATA in genome_ar53; do
            for NAME in train val test; do
                python generate_embedding.py \
                  --type $TYPE \
                  --model_tokenizer_path $MODEL_PATH \
                  --data_path ./data/$DATA/${NAME}.txt \
                  --model_max_length $MAX_LEN \
                  --resolution $RESO \
                  --output_path ./embedding/${TYPE}_embeddings_${RESO}_${MAX_LEN}/$DATA/$NAME
            done
        done
    done
done

BPE/TNF embeddings

MODEL_PATH=./mistral_4B_meta55k_hmp4k_seq10240/checkpoint-1600
MAX_LEN=10240
for TYPE in BPE TNF; do
    for DATA in genome_ar53; do
        for NAME in train val test; do
            python generate_embedding.py \
              --type $TYPE \
              --model_tokenizer_path $MODEL_PATH \
              --data_path ./data/$DATA/${NAME}.txt \
              --model_max_length $MAX_LEN \
              --output_path ./embedding/${TYPE}_embeddings/$DATA/$NAME
        done
    done
done

## Adapter Model Finetuning
Use the main.py to train the adapter model.
Then rename the best checkpoints as 'best' in all the saved checkpoints. (Do this be checking the validating loss in wandb)

TYPE=GO
MODEL=CNN
DATA=genome_ar53
for RESO in 10240 320 160 80; do
    for MAX_LEN in 10240; do
        CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port 2011 main.py \
            --model_type $MODEL \
            --label_dir ./data/$DATA \
            --data_dir ./embedding/${TYPE}_embeddings_${RESO}_${MAX_LEN}/$DATA \
            --feat_dim 3072 \
            --output_dim 768 \
            --resolution $RESO \
            --model_max_length $MAX_LEN \
            --per_device_train_batch_size 256 \
            --per_device_eval_batch_size 256 \
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

Use BPE/TNF embeddings as the input data of adapter model

MODEL=CNN
DATA=genome_ar53
MAX_LEN=10240
for TYPE in BPE TNF; do
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port 2011 main.py \
        --model_type $MODEL \
        --label_dir ./data/$DATA \
        --data_dir ./embedding/${TYPE}_embeddings/$DATA \
        --feat_dim 3072 \
        --output_dim 768 \
        --resolution $MAX_LEN \
        --model_max_length $MAX_LEN \
        --per_device_train_batch_size 256 \
        --per_device_eval_batch_size 256 \
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
        --output_dir ./output/$DATA/$MODEL/${TYPE} \
        --pj_name Genome_Ocean_Finetune \
        --log_name ${TYPE}_${MODEL}
done

## Model Evaluation
Test the model performance with test_corr.py file

TYPE=GO
DATA_TRAIN=genome_ar53
DATA_TEST=genome_ar53
for RESO in 10240; do
    for MAX_LEN in 10240; do
        for MODEL in CNN; do
            python test_corr.py \
                --backbone_type $TYPE \
                --model_type $MODEL \
                --model_path ./output/$DATA_TRAIN/$MODEL/${TYPE}_${RESO}_${MAX_LEN}/best \
                --data_path ./embedding/${TYPE}_embeddings_${RESO}_${MAX_LEN}/$DATA_TEST \
                --label_path ./data/$DATA_TEST \
                --data_type_train $DATA_TRAIN \
                --data_type_test $DATA_TEST \
                --batch_size 10 \
                --resolution $RESO \
                --max_length $MAX_LEN \
                --feat_dim 3072 \
                --output_dim 768 \
                --output_path ./test_results 
        done
    done
done

Test the performance when using BPE/TNF embeddings as the input data of adapter model

DATA_TRAIN=genome_ar53
DATA_TEST=genome_ar53
MAX_LEN=10240
for TYPE in BPE TNF; do
    for MODEL in CNN; do
        python test_corr.py \
            --backbone_type $TYPE \
            --model_type $MODEL \
            --model_path ./output/$DATA_TRAIN/$MODEL/${TYPE}/best \
            --data_path ./embedding/${TYPE}_embeddings/$DATA_TEST \
            --label_path ./data/$DATA_TEST \
            --data_type_train $DATA_TRAIN \
            --data_type_test $DATA_TEST \
            --batch_size 10 \
            --resolution $MAX_LEN \
            --max_length $MAX_LEN \
            --feat_dim 3072 \
            --output_dim 768 \
            --output_path ./test_results 
    done
done

The following are baselines without the adapter model

TYPE=GO
DATA_TRAIN=genome_ar53
DATA_TEST=genome_ar53
for RESO in 10240; do
    for MAX_LEN in 10240; do
            python test_corr.py \
                --backbone_type $TYPE \
                --data_path ./embedding/${TYPE}_embeddings_${RESO}_${MAX_LEN}/$DATA_TEST \
                --label_path ./data/$DATA_TEST \
                --data_type_test $DATA_TEST \
                --batch_size 10 \
                --resolution $RESO \
                --max_length $MAX_LEN \
                --output_path ./test_results 
        done
    done
done


DATA_TRAIN=genome_ar53
DATA_TEST=genome_ar53
MAX_LEN=10240
for TYPE in BPE TNF; do
    python test_corr.py \
        --backbone_type $TYPE \
        --data_path ./embedding/${TYPE}_embeddings/$DATA_TEST \
        --label_path ./data/$DATA_TEST \
        --data_type_test $DATA_TEST \
        --batch_size 10 \
        --resolution $MAX_LEN \
        --max_length $MAX_LEN \
        --output_path ./test_results 
done