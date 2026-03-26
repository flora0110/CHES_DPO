gpu1=$1; 
# gpu2=$2; gpu3=$3; gpu4=$4; 
its=$2
train_sample_size=2048;valid_sample_size=256
base_model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=4

# Only change the parameters above if needed
# "farthest_last_hidden_embedding_inner_prods" "nearest_last_hidden_embedding_inner_prods"
for category in "CDs_and_Vinyl"
do
    id2name_path="./eval/${category}/id2name.json"
    lora_weights="./experiments/models/SFT/${category}"
    output_dir="./experiments/data/${category}"
    echo ----------------- Training Parameters -----------------
    echo "GPU: $gpu1"
    echo "Train Sample Size: $train_sample_size"
    echo "Valid Sample Size: $valid_sample_size"
    echo "Base Model: $base_model"
    echo "LoRA Weights: $lora_weights"
    echo "Category: $category"

    for ((i=0;i<$its;i++))
    do
        echo ----------------- Iteration$i starts! -----------------
        it_output_dir="${output_dir}/it${i}/"
        sft_train_data_path="${it_output_dir}/sft/train.jsonl"
        sft_valid_data_path="${it_output_dir}/sft/valid.jsonl"

        Min_ln_ches_train_data_path="./experiments/data/${category}/it${i}/Min_ln_ches_scores/train.jsonl"
        Min_ln_ches_valid_data_path="./experiments/data/${category}/it${i}/Min_ln_ches_scores/valid.jsonl"
        
        FILES=($Min_ln_ches_train_data_path $Min_ln_ches_valid_data_path)

        all_exist=true

        for f in "${FILES[@]}"; do
            if [[ ! -f "$f" ]]; then
                echo "File $f does not exist."
                all_exist=false
                break
            fi
        done

        if [[ "$all_exist" = false ]]; then

            mkdir -p "${it_output_dir}/SPRec"
            mkdir -p "${it_output_dir}/sft"
            mkdir -p "${it_output_dir}/DPO_RN1"
            touch "./sampled_data/${category}_train.json"
            touch "./sampled_data/${category}_valid.json"
            
            echo ""
            echo ""
            echo "compute similarity for sft data $sft_train_data_path, $sft_valid_data_path"
            echo "------------------------------------------------------------------------------"
            
            # Data Generation
            # CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/compute_similarity.py \
            #     --data_path $sft_train_data_path \
            #     --id2name_path $id2name_path \
            #     --model_path $base_model \
            #     --lora_path $lora_weights \
            #     --output_dir $it_output_dir \
            #     --output_prefix "train_item_pref_similarity" \
            #     --num_random_items 50 \
            #     --random_seed $i \
            #     --chunk_size $valid_sample_size

            # CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/compute_similarity.py \
            #     --data_path $sft_valid_data_path \
            #     --id2name_path $id2name_path \
            #     --model_path $base_model \
            #     --lora_path $lora_weights \
            #     --output_dir $it_output_dir \
            #     --output_prefix "valid_item_pref_similarity" \
            #     --num_random_items 50 \
            #     --random_seed $i \
            #     --chunk_size $valid_sample_size
            
            echo ""
            echo ""
            echo "selecting samples based on similarity"
            echo "------------------------------------------------------------------------------"
            CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/select_reject_base_sim.py \
                --similarity_chunk_dir $it_output_dir \
                --output_dir $it_output_dir \
                --data_type "train"

            CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/select_reject_base_sim.py \
                --similarity_chunk_dir $it_output_dir \
                --output_dir $it_output_dir \
                --data_type "valid"
        fi
        
        echo ""
        echo ""
        echo ----------------- Training SPRec -----------------
        bash ./shell/single_DPO_runner.sh $gpu1 $i $category "Min_ln_ches_scores" 1e-6

    done
done