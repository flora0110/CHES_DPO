gpu1=$1;
i=$2;
category=$3;
metric=$4;
lr=$5

train_sample_size=2048;valid_sample_size=256
# base_model="/scratch/user/chuanhsin0110/hf_models/Llama-3.2-1B-Instruct/"
base_model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=4

# category="Goodreads"
MAX_EPOCH=10
metric_for_best_model="accuracies"
metric_for_best_model_key="eval_rewards/${metric_for_best_model}"
greater_is_better=True


echo ----------------- Training $category with DPO starting! -----------------

id2name_path="./eval/${category}/id2name.json"
name2id_path="./eval/${category}/name2id.json"
embeddings_path="./eval/${category}/embeddings.pt"
train_data_for_head_tail="./sampled_data/${category}_train.json"
test_dataset="./sampled_data/${category}_test.json"
lora_weights="./experiments/models/SFT/${category}"

    
echo ----------------- Iteration$i starts! -----------------
model_it_dir="./experiments/models/${category}_${lr}/it${i}"
predicts_it_dir="./experiments/predicts/${category}_${lr}/it${i}"
metrics_it_dir="./experiments/metrics/${category}_${lr}/it${i}"


echo ----------------- Training Parameters -----------------
echo "GPU: $gpu1"
echo "Iterations: $its"
echo "Train Sample Size: $train_sample_size"
echo "Valid Sample Size: $valid_sample_size"
echo "Base Model: $base_model"
echo "LoRA Weights: $lora_weights"
echo "Category: $category"
echo "Learning Rate: $lr"   
echo "test_dataset: $test_dataset"
echo "id2name_path: $id2name_path"
echo "name2id_path: $name2id_path"
echo "embeddings_path: $embeddings_path"     
    
    
echo ----------------- Training ${metric} -----------------
dpo_train_data_path="./experiments/data/${category}/it${i}/${metric}/train.jsonl"
dpo_valid_data_path="./experiments/data/${category}/it${i}/${metric}/valid.jsonl"


# mkdir -p $it_output_dir
echo "dpo_train_data_path: $dpo_train_data_path"
echo "dpo_valid_data_path: $dpo_valid_data_path"
touch "${dpo_train_data_path}"
touch "${dpo_valid_data_path}"

# DPO
model_path="${model_it_dir}/${metric}/"
predicts_json="${predicts_it_dir}/${metric}/predicts.json"
eval_result_json="${metrics_it_dir}/${metric}/eval_top5.json"
recs_json="${metrics_it_dir}/${metric}/eval_top5_top5_recs.json"

mkdir -p "${model_it_dir}/${metric}"
mkdir -p "${predicts_it_dir}/${metric}"
mkdir -p "${metrics_it_dir}/${metric}"

touch "${model_path}"
echo "DPO_path Rate: $DPO_path"
if [ ! -f "${model_path}/adapter_config.json" ]; then
    CUDA_VISIBLE_DEVICES=$gpu1 python ./src/models/dpo.py \
        --train_dataset $dpo_train_data_path \
        --val_dataset $dpo_valid_data_path \
        --output_dir $model_path \
        --base_model $base_model \
        --resume_from_checkpoint $lora_weights \
        --batch_size  2\
        --gradient_accumulation_steps 16 \
        --learning_rate $lr \
        --cutoff_len 512 \
        --num_epochs $MAX_EPOCH \
        --eval_strategy "epoch" \
        --save_strategy "epoch" \
        --sh_file_path "./shell/RN1_SPRec_runner.sh" \
        --metric_for_best_model $metric_for_best_model_key \
        --greater_is_better $greater_is_better
else
    echo "Model for Iteration $i already exists. Skipping training."
fi
# Evaluate DPO model
echo -------------------------------------- Inference and evaluation for category $category --------------------------------------

CUDA_VISIBLE_DEVICES=$gpu1 python ./src/inference/inference.py \
    --base_model $base_model \
    --lora_weights $model_path \
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
python ./src/evaluate/agg_eval.py "./experiments/metrics/${category}_${lr}" "$metric" > ""./experiments/metrics/${category}_${lr}/${metric}_head_tail_top5_table.md"
echo "Saved: "./experiments/metrics/${category}_${lr}/${metric}_head_tail_top5_table.md"
echo SPRec for category ${category} has successfully completed!

python ./src/evaluate/agg_gini.py \
    --base_dir "./experiments/metrics/${category}_${lr}" \
    --iterations 5 \
    --metrics ${metric} \
    --topks 5
