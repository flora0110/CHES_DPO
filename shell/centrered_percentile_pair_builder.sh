#!/bin/bash
gpu1=$1;
category=$2;
Metrics=$3;
base_model="meta-llama/Llama-3.2-1B-Instruct"
lora_weights="./experiments_new/models/SFT/${category}"
output_dir="./centered_percentile_pair_experiments/${category}"
data_path="./sampled_data/${category}_train.json"
id2name_path="./eval/${category}/id2name.json"
name2id_path="./eval/${category}/name2id.json"
embeddings_path="./eval/${category}/embeddings.pt"
mkdir -p $output_dir
touch $lora_weights
touch $data_path
touch $id2name_path
echo "GPU: $gpu1"
echo "Category: $category"
echo "Base Model: $base_model"
echo "LoRA Weights: $lora_weights"
echo "Output Directory: $output_dir"
echo "Data Path: $data_path"
echo "ID to Name Path: $id2name_path"


# CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/centered_percentile_pair_builder.py \
#   --data_path $data_path \
#   --id2name_path $id2name_path \
#   --model_path $base_model \
#   --lora_path $lora_weights \
#   --output_dir $output_dir \
#   --window_size 1024 \
#   --percentiles 0 25 50 75 100 \
#   --random_seed 42


# python ./src/plot/plot_centered_percentile_summaries.py \
#   --output_dir $output_dir \
#   --window_size 1024


# DPO

lr=1e-5
EPOCHS=3

DATA_DIR="${output_dir}/centered_percentile_datasets"
OUTPUT_ROOT="${output_dir}/models/"
PREDICTS_ROOT="${output_dir}/predicts/"
EVAL_ROOT="${output_dir}/metrics/"

for P in 25 75
do
    dpo_train_data_path="${DATA_DIR}/${Metrics}_p${P}_w1024.jsonl"
    DPO_model_path="${OUTPUT_ROOT}/${Metrics}/p${P}"

    mkdir -p $DPO_model_path
    echo "Output Directory for ${Metrics} p${P}: ${DPO_model_path}"

    echo "Running ${Metrics} p${P}"

    CUDA_VISIBLE_DEVICES=$gpu1 python ./src/models/dpo_for_centered_percentile.py \
        --train_dataset $dpo_train_data_path \
        --output_dir $DPO_model_path \
        --base_model $base_model \
        --resume_from_checkpoint $lora_weights \
        --batch_size 2 \
        --gradient_accumulation_steps 16 \
        --learning_rate $lr \
        --cutoff_len 512 \
        --num_epochs 3 \
        --save_strategy "epoch" \
        --sh_file_path "./shell/centrered_percentile_pair_builder.sh" \



    test_dataset="./sampled_data/${category}_test.json"
    predicts_json="${PREDICTS_ROOT}/${Metrics}/p${P}/predicts.json"
    eval_result_json="${EVAL_ROOT}/${Metrics}/p${P}/eval_top5.json"
    recs_json="${EVAL_ROOT}/${Metrics}/p${P}/eval_top5_top5_recs.json"
    mkdir -p "${PREDICTS_ROOT}/${Metrics}/p${P}"
    mkdir -p "${EVAL_ROOT}/${Metrics}/p${P}"


    CUDA_VISIBLE_DEVICES=$gpu1 python ./src/inference/inference.py \
        --base_model $base_model \
        --lora_weights $DPO_model_path \
        --test_data_path $test_dataset \
        --result_json_data $predicts_json \
        --num_beams 1


    CUDA_VISIBLE_DEVICES=$gpu1 python ./src/evaluate/evaluate_sim.py \
        --input_dir $predicts_json \
        --output_dir $eval_result_json \
        --topk 5 \
        --gamma 0 \
        --category $category \
        --id2name_path $id2name_path \
        --name2id_path $name2id_path \
        --embeddings_path $embeddings_path \

    train_data_for_head_tail="./sampled_data/${category}_train.json"

    CUDA_VISIBLE_DEVICES=$gpu1 python ./src/evaluate/evaluate_head_tail_sim.py \
        --input_dir $predicts_json \
        --output_dir $eval_result_json \
        --topk 5 \
        --gamma 0 \
        --category $category \
        --train_data_for_head_tail $train_data_for_head_tail \
        --id2name_path $id2name_path \
        --name2id_path $name2id_path \
        --embeddings_path $embeddings_path

    CUDA_VISIBLE_DEVICES=$gpu1 python ./src/evaluate/eval_gini.py \
        --input_dir $recs_json \
        --output_dir $eval_result_json \
        --topk 5 \
        --category $category

done