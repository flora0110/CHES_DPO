gpu1=$1;
i=$2;
category=$3;
metric=$4;
lr=$5

sft_lr=0.00002
train_sample_size=2048;valid_sample_size=256
base_model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=4

# category="Goodreads"
MAX_EPOCH=10
metric_for_best_model="accuracies"
metric_for_best_model_key="eval_rewards/${metric_for_best_model}"
greater_is_better=True


echo ----------------- Iteration$i of $category with $metric -----------------

echo ----------------- Training Parameters -----------------
echo "GPU: $gpu1"
echo "Iterations: $i"
echo "Train Sample Size: $train_sample_size"
echo "Valid Sample Size: $valid_sample_size"
echo "Base Model: $base_model"
echo "Category: $category"
echo "Learning Rate: $lr"  


echo ----------------- Setup paths for utils -----------------
id2name_path="./eval/${category}/id2name.json"
name2id_path="./eval/${category}/name2id.json"
embeddings_path="./eval/${category}/embeddings.pt"
train_data_for_head_tail="./sampled_data/${category}_train.json"
test_dataset="./sampled_data/${category}_test.json"
lora_weights="./experiments_new/models/SFT/${category}"

echo "id2name_path: $id2name_path"
echo "name2id_path: $name2id_path"
echo "embeddings_path: $embeddings_path"
echo "train_data_for_head_tail: $train_data_for_head_tail"
echo "test_dataset: $test_dataset"
echo "LoRA Weights: $lora_weights"
touch "${id2name_path}"
touch "${name2id_path}"
touch "${embeddings_path}"
touch "${train_data_for_head_tail}"
touch "${test_dataset}"
touch "${lora_weights}"



echo ----------------- Preparing directories for Iteration$i with ${metric} -----------------
model_it_dir="./experiments_new/models/${category}_${lr}_single_epoch/it${i}"
predicts_it_dir="./experiments_new/predicts/${category}_${lr}_single_epoch/it${i}"
metrics_it_dir="./experiments_new/metrics/${category}_${lr}_single_epoch/it${i}"
metrics_base_dir="./experiments_new/metrics/${category}_${lr}_single_epoch" # 這個是給後續agg_eval.py用的，因為它需要讀取所有iteration的eval_top5.json來做彙總
DPO_model_path="${model_it_dir}/${metric}/"
SFT_model_path="./experiments_new/models/SFT_2nd/${category}_sftlr_${sft_lr}/it${i}/"


mkdir -p "${DPO_model_path}"
mkdir -p "${predicts_it_dir}/${metric}"
mkdir -p "${metrics_it_dir}/${metric}"
echo "Model output directory: ${DPO_model_path}"
echo "Predicts output directory: ${predicts_it_dir}/${metric}"
echo "Metrics output directory: ${metrics_it_dir}/${metric}"


# SFT
echo
echo
echo ----------------- Star SFT training for Iteration$i of $category with $metric -----------------
echo

echo ----------------- Preparing data for SFT training -----------------
sft_train_data_path="./experiments_new/data/${category}/it${i}/sft/train.jsonl"
sft_valid_data_path="./experiments_new/data/${category}/it${i}/sft/valid.jsonl"

echo "SFT_model_path: $SFT_model_path"
if [ ! -f "${SFT_model_path}/adapter_config.json" ]; then
    CUDA_VISIBLE_DEVICES=$gpu1 python ./src/models/sft.py \
        --output_dir "$SFT_model_path" \
        --base_model "$base_model" \
        --train_dataset "$sft_train_data_path" \
        --valid_dataset "$sft_valid_data_path" \
        --resume_from_checkpoint "$lora_weights" \
        --gradient_accumulation_steps 16 \
        --batch_size 4 \
        --num_train_epochs 1 \
        --learning_rate "$sft_lr" \
        --cutoff_len 512 \
        --seed "$seed"
else
    echo "SFT Model for Iteration $i already exists. Skipping training."
fi

echo
echo
echo ----------------- Star DPO training for Iteration$i of $category with $metric -----------------
echo

echo ----------------- Preparing data for DPO training -----------------
dpo_train_data_path="./experiments_new/data/${category}/it${i}/${metric}/train.jsonl"
dpo_valid_data_path="./experiments_new/data/${category}/it${i}/${metric}/valid.jsonl"

echo "dpo_train_data_path: $dpo_train_data_path"
echo "dpo_valid_data_path: $dpo_valid_data_path"
touch "${dpo_train_data_path}"
touch "${dpo_valid_data_path}"

echo ----------------- Set up file paths for DPO training and evaluation -----------------
predicts_json="${predicts_it_dir}/${metric}/predicts.json"
eval_result_json="${metrics_it_dir}/${metric}/eval_top5.json"
recs_json="${metrics_it_dir}/${metric}/eval_top5_top5_recs.json"

mkdir -p "${model_it_dir}/${metric}"
mkdir -p "${predicts_it_dir}/${metric}"
mkdir -p "${metrics_it_dir}/${metric}"


echo ----------------- Training $category with DPO starting! -----------------

touch "${DPO_model_path}"
echo "DPO_path Rate: $DPO_model_path"
if [ ! -f "${DPO_model_path}/adapter_config.json" ]; then
    CUDA_VISIBLE_DEVICES=$gpu1 python ./src/models/dpo_origin.py \
        --train_dataset $dpo_train_data_path \
        --val_dataset $dpo_valid_data_path \
        --output_dir $DPO_model_path \
        --base_model $base_model \
        --resume_from_checkpoint $SFT_model_path \
        --batch_size  2\
        --gradient_accumulation_steps 16 \
        --learning_rate $lr \
        --cutoff_len 512 \
        --num_epochs 1
else
    echo "Model for Iteration $i already exists. Skipping training."
fi
# Evaluate DPO model
echo -------------------------------------- Inference and evaluation for category $category --------------------------------------

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
    --category $category \



echo "----------------- Aggregating eval_top5.json for metric: ${metric} -----------------"
python ./src/evaluate/agg_eval.py "${metrics_base_dir}" "$metric" > "${metrics_base_dir}/${metric}_head_tail_top5_table.md"
echo "Saved: ${metrics_base_dir}/${metric}_head_tail_top5_table.md"
echo SPRec for category ${category} has successfully completed!

python ./src/evaluate/agg_gini.py \
    --base_dir "${metrics_base_dir}" \
    --iterations 5 \
    --metrics ${metric} \
    --topks 5
