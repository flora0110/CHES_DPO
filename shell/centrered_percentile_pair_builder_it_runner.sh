#!/bin/bash
gpu1=$1;
category=$2;
METRICS_LIST=(
  "ln_ches_score"
  "sequence_logprob_margin"
  "ches_score"
  "avg_token_logprob_margin"
)
base_model="meta-llama/Llama-3.2-1B-Instruct"

# FIXED PATHS
id2name_path="./eval/${category}/id2name.json"
name2id_path="./eval/${category}/name2id.json"
embeddings_path="./eval/${category}/embeddings.pt"
raw_train_dataset="./data_sprec/${category}_train.json"
raw_test_dataset="./data_sprec/${category}_test.json"

lr=1e-5
train_data_size=4096
test_data_size=1000

# Set based on category
CATEGORY_ROOT="./centered_percentile_experiments_${train_data_size}_${test_data_size}/${category}"
mkdir -p $CATEGORY_ROOT
echo "Category Root Directory: $CATEGORY_ROOT"
SAMPLED_DATA_ROOT="${CATEGORY_ROOT}/sampled_data"
DATA_ROOT="${CATEGORY_ROOT}/centered_percentile_datasets"
MODEL_ROOT="${CATEGORY_ROOT}/models/"
PREDICTS_ROOT="${CATEGORY_ROOT}/predicts/"
EVAL_ROOT="${CATEGORY_ROOT}/metrics/"

for seed in 1
do
    echo ------------------- Path setup for $category with seed $seed -------------------

    SAMPLED_DATA_seed_DIR="${SAMPLED_DATA_ROOT}/seed${seed}"
    DATA_seed_DIR="${DATA_ROOT}/seed${seed}"
    MODEL_seed_DIR="${MODEL_ROOT}/seed${seed}"
    PREDICTS_seed_DIR="${PREDICTS_ROOT}/seed${seed}"
    EVAL_seed_DIR="${EVAL_ROOT}/seed${seed}"
    mkdir -p $SAMPLED_DATA_seed_DIR
    mkdir -p $DATA_seed_DIR
    mkdir -p $MODEL_seed_DIR
    mkdir -p $PREDICTS_seed_DIR
    mkdir -p $EVAL_seed_DIR

    # Sampled data paths
    sampled_train_dataset="${SAMPLED_DATA_seed_DIR}/train.json"
    sampled_test_dataset="${SAMPLED_DATA_seed_DIR}/test.json"

    # SFT
    sft_model_path="${MODEL_seed_DIR}/SFT"
    mkdir -p $sft_model_path
    echo "sft_model_path: $sft_model_path"

    # SFT evaluation outputs
    sft_predicts_json="${PREDICTS_seed_DIR}/SFT/predicts.json"
    sft_eval_result_json="${EVAL_seed_DIR}/SFT/eval_top5.json"
    sft_recs_json="${EVAL_seed_DIR}/SFT/eval_top5_top5_recs.json"
    mkdir -p ${PREDICTS_seed_DIR}/SFT
    mkdir -p ${EVAL_seed_DIR}/SFT


    if [ ! -f "${sampled_train_dataset}" ]; then
        python ./src/data/sampled_data.py \
            --input_path $raw_train_dataset\
            --sample_size $train_data_size\
            --output_path $sampled_train_dataset\
            --seed $seed
        
        python ./src/data/sampled_data.py \
            --input_path $raw_test_dataset\
            --sample_size $test_data_size\
            --output_path $sampled_test_dataset\
            --seed $seed
    else
        echo "Sampled datasets for seed $seed already exist. Skipping sampling."
    fi
    
    echo ------------------ Running SFT for $category with seed $seed -----------------

    if [ ! -f "${sft_model_path}/adapter_config.json" ]; then
        CUDA_VISIBLE_DEVICES=$gpu1 python ./src/models/sft_for_centered_percentile.py \
            --output_dir $sft_model_path\
            --base_model $base_model \
            --train_dataset $sampled_train_dataset \
            --gradient_accumulation_steps 16 \
            --batch_size 4 \
            --num_train_epochs 4 \
            --learning_rate 0.0003 \
            --cutoff_len 512 \
            --seed 42

    
        echo ------------------ Running inference and evaluation for SFT model for $category with seed $seed -----------------
        

        CUDA_VISIBLE_DEVICES=$gpu1 python ./src/inference/inference.py \
        --base_model $base_model \
        --lora_weights $sft_model_path \
        --test_data_path $sampled_test_dataset \
        --result_json_data $sft_predicts_json \
        --num_beams 1

        
        CUDA_VISIBLE_DEVICES=$gpu1 python ./src/evaluate/evaluate_sim.py \
        --input_dir $sft_predicts_json \
        --output_dir $sft_eval_result_json \
        --topk 5 \
        --gamma 0 \
        --category $category \
        --id2name_path $id2name_path \
        --name2id_path $name2id_path \
        --embeddings_path $embeddings_path


        CUDA_VISIBLE_DEVICES=$gpu1 python ./src/evaluate/evaluate_head_tail_sim.py \
        --input_dir $sft_predicts_json \
        --output_dir $sft_eval_result_json \
        --topk 5 \
        --gamma 0 \
        --category $category \
        --train_data_for_head_tail $sampled_train_dataset \
        --id2name_path $id2name_path \
        --name2id_path $name2id_path \
        --embeddings_path $embeddings_path

        CUDA_VISIBLE_DEVICES=$gpu1 python ./src/evaluate/eval_gini.py \
        --input_dir $sft_recs_json \
        --output_dir $sft_eval_result_json \
        --topk 5 \
        --category $category
    
    else
        echo "SFT Model for seed $seed already exists. Skipping training, inference, and evaluation."
    fi


    echo ------------------ Running centered percentile pair builder for $category with seed $seed -----------------

    if [ ! -f "${DATA_seed_DIR}/full_scored_pair_metadata.json" ]; then
        CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/centered_percentile_pair_builder.py \
        --data_path $sampled_train_dataset \
        --id2name_path $id2name_path \
        --model_path $base_model \
        --lora_path $sft_model_path \
        --output_dir $DATA_seed_DIR \
        --window_size 1024 \
        --percentiles 0 25 50 75 100 \
        --random_seed 42


        python ./src/plot/plot_centered_percentile_summaries.py \
        --output_dir $DATA_seed_DIR \
        --window_size 1024
    else
        echo "Centered percentile pair metadata for seed $seed already exists. Skipping pair building."
    fi


    # DPO
    # =========================
    # DPO (loop over metrics)
    # =========================
    for METRIC in "${METRICS_LIST[@]}"
    do
        echo "===================="
        echo "Running METRIC: ${METRIC}"
        echo "===================="

        for P in 25 75
        do
            dpo_train_data_path="${DATA_seed_DIR}/centered_percentile_datasets/${METRIC}_p${P}_w1024.jsonl"
            DPO_model_path="${MODEL_seed_DIR}/${METRIC}/p${P}"

            mkdir -p $DPO_model_path
            echo "Output Directory for ${METRIC} p${P}: ${DPO_model_path}"

            CUDA_VISIBLE_DEVICES=$gpu1 python ./src/models/dpo_for_centered_percentile.py \
                --train_dataset $dpo_train_data_path \
                --output_dir $DPO_model_path \
                --base_model $base_model \
                --resume_from_checkpoint $sft_model_path \
                --batch_size 2 \
                --gradient_accumulation_steps 16 \
                --learning_rate $lr \
                --cutoff_len 512 \
                --num_epochs 3 \
                --save_strategy "epoch" \
                --sh_file_path "./shell/centrered_percentile_pair_builder_it_runner.sh"

            # =========================
            # Evaluate epoch1 / epoch2 / final
            # =========================
            for CKPT_TAG in epoch1 epoch2 final
            do
                if [ "$CKPT_TAG" = "epoch1" ]; then
                    LORA_PATH="${DPO_model_path}/checkpoint-32"
                elif [ "$CKPT_TAG" = "epoch2" ]; then
                    LORA_PATH="${DPO_model_path}/checkpoint-64"
                else
                    LORA_PATH="${DPO_model_path}"
                fi

                if [ ! -d "$LORA_PATH" ]; then
                    echo "Skip ${CKPT_TAG}: ${LORA_PATH} not found"
                    continue
                fi

                dpo_predicts_json="${PREDICTS_seed_DIR}/${METRIC}/p${P}/${CKPT_TAG}/predicts.json"
                dpo_eval_result_json="${EVAL_seed_DIR}/${METRIC}/p${P}/${CKPT_TAG}/eval_top5.json"
                dpo_recs_json="${EVAL_seed_DIR}/${METRIC}/p${P}/${CKPT_TAG}/eval_top5_top5_recs.json"

                mkdir -p "${PREDICTS_seed_DIR}/${METRIC}/p${P}/${CKPT_TAG}"
                mkdir -p "${EVAL_seed_DIR}/${METRIC}/p${P}/${CKPT_TAG}"

                echo "Evaluating ${METRIC} p${P} ${CKPT_TAG}"

                CUDA_VISIBLE_DEVICES=$gpu1 python ./src/inference/inference.py \
                    --base_model $base_model \
                    --lora_weights $LORA_PATH \
                    --test_data_path $sampled_test_dataset \
                    --result_json_data $dpo_predicts_json \
                    --num_beams 1

                CUDA_VISIBLE_DEVICES=$gpu1 python ./src/evaluate/evaluate_sim.py \
                    --input_dir $dpo_predicts_json \
                    --output_dir $dpo_eval_result_json \
                    --topk 5 \
                    --gamma 0 \
                    --category $category \
                    --id2name_path $id2name_path \
                    --name2id_path $name2id_path \
                    --embeddings_path $embeddings_path

                CUDA_VISIBLE_DEVICES=$gpu1 python ./src/evaluate/evaluate_head_tail_sim.py \
                    --input_dir $dpo_predicts_json \
                    --output_dir $dpo_eval_result_json \
                    --topk 5 \
                    --gamma 0 \
                    --category $category \
                    --train_data_for_head_tail $sampled_train_dataset \
                    --id2name_path $id2name_path \
                    --name2id_path $name2id_path \
                    --embeddings_path $embeddings_path

                CUDA_VISIBLE_DEVICES=$gpu1 python ./src/evaluate/eval_gini.py \
                    --input_dir $dpo_recs_json \
                    --output_dir $dpo_eval_result_json \
                    --topk 5 \
                    --category $category

            done
        done
    done
done