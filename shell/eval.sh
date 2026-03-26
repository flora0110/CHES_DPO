gpu1=$1;

category="Goodreads";


echo ----------------- Training $category with DPO starting! -----------------

id2name_path="./eval/${category}/id2name.json"
name2id_path="./eval/${category}/name2id.json"
embeddings_path="./eval/${category}/embeddings.pt"
train_data_for_head_tail="./sampled_data/${category}_train.json"
test_dataset="./sampled_data/${category}_test.json"


    




predicts_it_dir="/data2/chuanhsin0110/SASRec.pytorch/semantic_expansion_results/"
metrics_it_dir="/data2/chuanhsin0110/SASRec.pytorch/semantic_expansion_results/"




# mkdir -p $it_output_dir
echo "dpo_train_data_path: $dpo_train_data_path"
echo "dpo_valid_data_path: $dpo_valid_data_path"
touch "${dpo_train_data_path}"
touch "${dpo_valid_data_path}"

# DPO
predicts_json="${predicts_it_dir}/${metric}/predicts.json"
eval_result_json="${metrics_it_dir}/${metric}/eval_top5.json"
recs_json="${metrics_it_dir}/${metric}/eval_top5_top5_recs.json"

mkdir -p "${model_it_dir}/${metric}"
mkdir -p "${predicts_it_dir}/${metric}"
mkdir -p "${metrics_it_dir}/${metric}"



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

