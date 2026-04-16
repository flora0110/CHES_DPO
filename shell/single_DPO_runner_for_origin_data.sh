gpu1=$1;
i=$2;
# metric=$4;

# 先寫死，若要修改Category或metric，需要同步修改DPO 訓練資料的路徑
category="Goodreads"
metric="DPO_RN1"
lr=1e-6

train_sample_size=2048;valid_sample_size=256
# base_model="/scratch/user/chuanhsin0110/hf_models/Llama-3.2-1B-Instruct/"
base_model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=4

# category="Goodreads"
MAX_EPOCH=10
metric_for_best_model="accuracies"
metric_for_best_model_key="eval_rewards/${metric_for_best_model}"
greater_is_better=True


echo ----------------- Training Parameters -----------------
echo "GPU: $gpu1"
echo "Iterations: $its"
echo "Train Sample Size: $train_sample_size"
echo "Valid Sample Size: $valid_sample_size"
echo "Base Model: $base_model"
echo "Category: $category"
echo "Learning Rate: $lr"   


# Inference, evaluates會需要的路徑
# 以及base model, resume_from_checkpoint的路徑
# 現在先改成用舊的
echo ----------------- Training $category with DPO starting! -----------------
id2name_path="../SPRec/eval/${category}/id2name.json"
name2id_path="../SPRec/eval/${category}/name2id.json"
embeddings_path="../SPRec/eval/${category}/embeddings.pt"
train_data_for_head_tail="../SPRec/data/${category}/train.json"
test_dataset="./sampled_data/${category}_test.json"
lora_weights="../SPRec/models/SFT_4096/${category}"
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
    
echo ----------------- Iteration$i starts! -----------------

echo ----------------- Training ${metric} -----------------

# DPO 訓練資料，現在先改用舊的，
echo ----------------- Preparing data for DPO training -----------------
# dpo_train_data_path="./experiments/data/${category}/it${i}/${metric}/train.jsonl"
# dpo_valid_data_path="./experiments/data/${category}/it${i}/${metric}/valid.jsonl"
dpo_train_data_path="../SPRec/models/SPRec/Goodreads_2048_0.00002/it${i}/data/DPO_RN1_new/train.jsonl"
dpo_valid_data_path="../SPRec/models/SPRec/Goodreads_2048_0.00002/it${i}/data/DPO_RN1_new/valid.jsonl"
echo "dpo_train_data_path: $dpo_train_data_path"
echo "dpo_valid_data_path: $dpo_valid_data_path"
touch "${dpo_train_data_path}"
touch "${dpo_valid_data_path}"


# DPO 訓練、推論、評估的輸出路徑，建立一個暫時的測試路徑到experiment_Origin_data
echo ----------------- Preparing directories for Iteration$i with ${metric} -----------------
model_it_dir="./experiment_Origin_data/models/${category}_${lr}/it${i}"
predicts_it_dir="./experiment_Origin_data/predicts/${category}_${lr}_but_with_new_test/it${i}"
metrics_it_dir="./experiment_Origin_data/metrics/${category}_${lr}_but_with_new_test/it${i}"
metrics_base_dir="./experiment_Origin_data/metrics/${category}_${lr}_but_with_new_test" # 這個是給後續agg_eval.py用的，因為它需要讀取所有iteration的eval_top5.json來做彙總
model_path="${model_it_dir}/${metric}/"

mkdir -p "${model_path}"
mkdir -p "${predicts_it_dir}/${metric}"
mkdir -p "${metrics_it_dir}/${metric}"
echo "Model output directory: ${model_path}"
echo "Predicts output directory: ${predicts_it_dir}/${metric}"
echo "Metrics output directory: ${metrics_it_dir}/${metric}"



echo ----------------- Set up file paths for DPO training and evaluation -----------------
predicts_json="${predicts_it_dir}/${metric}/predicts.json"
eval_result_json="${metrics_it_dir}/${metric}/eval_top5.json"
recs_json="${metrics_it_dir}/${metric}/eval_top5_top5_recs.json"


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
        --sh_file_path "./shell/single_DPO_runner_for_origin_data.sh" \
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
python ./src/evaluate/agg_eval.py "${metrics_base_dir}" "$metric" > "${metrics_base_dir}/${metric}_head_tail_top5_table.md"
echo "Saved: ${metrics_base_dir}/${metric}_head_tail_top5_table.md"
echo SPRec for category ${category} has successfully completed!

python ./src/evaluate/agg_gini.py \
    --base_dir "${metrics_base_dir}" \
    --iterations 5 \
    --metrics ${metric} \
    --topks 5
