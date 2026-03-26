gpu1=$1; 
# gpu2=$2; gpu3=$3; gpu4=$4; 
its=$2
train_sample_size=2048;valid_sample_size=256
base_model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=4

# Only change the parameters above if needed
# "farthest_last_hidden_embedding_inner_prods" "nearest_last_hidden_embedding_inner_prods"
for category in "MovieLens"
do
    id2name_path="./eval/${category}/id2name.json"
    lora_weights="./experiments/models/SFT/${category}"
    output_dir="./experiments/data/${category}"
    echo ----------------- Training Parameters -----------------
    echo "GPU: $gpu1"
    echo "Iterations: its"
    echo "Train Sample Size: $train_sample_size"
    echo "Valid Sample Size: $valid_sample_size"
    echo "Base Model: $base_model"
    echo "LoRA Weights: $lora_weights"
    echo "Category: $category"

    for ((i=1;i<$its;i++))
    do
        # echo ----------------- Iteration$i starts! -----------------
        it_output_dir="${output_dir}/it${i}/"
        sprec_train_data_path="${it_output_dir}/SPRec/train.jsonl"
        sprec_valid_data_path="${it_output_dir}/SPRec/valid.jsonl"
        sft_train_data_path="${it_output_dir}/sft/train.jsonl"
        sft_valid_data_path="${it_output_dir}/sft/valid.jsonl"
        dpo_rn1_train_data_path="${it_output_dir}/DPO_RN1/train.jsonl"
        dpo_rn1_valid_data_path="${it_output_dir}/DPO_RN1/valid.jsonl"

        # mkdir -p $it_output_dir
        mkdir -p "${it_output_dir}/SPRec"
        mkdir -p "${it_output_dir}/sft"
        mkdir -p "${it_output_dir}/DPO_RN1"

        # Data Generation

        mkdir -p "${it_output_dir}/data/DPO_RN1_w_new_prompt"
        CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/data_generate.py \
            --train_json_file ./sampled_data/${category}_train.json \
            --valid_json_file ./sampled_data/${category}_valid.json \
            --result_json_dpo_data_train $sprec_train_data_path \
            --result_json_dpo_data_valid $sprec_valid_data_path \
            --result_json_dpo_rn1_data_train $dpo_rn1_train_data_path \
            --result_json_dpo_rn1_data_valid $dpo_rn1_valid_data_path \
            --id2name_json_file "./eval/${category}/id2name.json" \
            --result_json_sft_data_train $sft_train_data_path \
            --result_json_sft_data_valid $sft_valid_data_path \
            --base_model $base_model \
            --lora_weights $lora_weights \
            --batch_size 64 \
            --train_sample_size $train_sample_size \
            --valid_sample_size $valid_sample_size \
            --seed $i \
        
        # CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/compute_similarity.py \
        #     --data_path $sft_train_data_path \
        #     --id2name_path $id2name_path \
        #     --model_path $base_model \
        #     --lora_path $lora_weights \
        #     --output_dir $it_output_dir \
        #     --output_prefix "train_item_pref_similarity" \
        #     --num_random_items 50 \
        #     --random_seed 42 \
        #     --chunk_size 100

        # CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/compute_similarity.py \
        #     --data_path $sft_valid_data_path \
        #     --id2name_path $id2name_path \
        #     --model_path $base_model \
        #     --lora_path $lora_weights \
        #     --output_dir $it_output_dir \
        #     --output_prefix "valid_item_pref_similarity" \
        #     --num_random_items 50 \
        #     --random_seed 42 \
        #     --chunk_size 100

        # CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/select_reject_base_sim.py \
        #     --similarity_chunk_dir $it_output_dir \
        #     --output_dir $it_output_dir \
        #     --data_type "train"

        # CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/select_reject_base_sim.py \
        #     --similarity_chunk_dir $it_output_dir \
        #     --output_dir $it_output_dir \
        #     --data_type "valid"
        
    done
done