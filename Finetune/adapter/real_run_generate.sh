TYPE=GO
MODEL_PATH=/mnt/c/Users/11817/GenomeAI/genome_ocean
for RESO in 10240 320 160 80; do
    for MAX_LEN in 10240; do
        for DATA in genome_ar53; do
            for NAME in train val test; do
                python generate_embedding.py \
                  --type $TYPE \
                  --model_tokenizer_path $MODEL_PATH \
                  --data_path ./data/$DATA/${NAME}.txt \
                  --batch_size 4 \
                  --model_max_length $MAX_LEN \
                  --resolution $RESO \
                  --output_path ./embedding/${TYPE}_embeddings_${RESO}_${MAX_LEN}/$DATA/$NAME
            done
        done
    done
done
