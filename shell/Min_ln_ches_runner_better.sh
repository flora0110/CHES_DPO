gpu1=$1; 
# gpu2=$2; gpu3=$3; gpu4=$4; 
its=$2
train_sample_size=2048;valid_sample_size=256
base_model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=4
metric="Min_ln_ches_scores"
lr=(1e-6 1e-5)
# Only change the parameters above if needed
# "farthest_last_hidden_embedding_inner_prods" "nearest_last_hidden_embedding_inner_prods"
for category in "Steam"
do
    id2name_path="./eval/${category}/id2name.json"
    lora_weights="./experiments/models/SFT/${category}"
    data_output_dir="./experiments/data/${category}"
    
    echo ----------------- Training Parameters -----------------
    echo "GPU: $gpu1"
    echo "Data Output Directory: $data_output_dir"
    echo "Model Output Directory: $model_output_dir"
    echo "Base Model: $base_model"
    echo "LoRA Weights: $lora_weights"
    echo "Category: $category"

    sampled_train_data_path="./sampled_data/${category}_train.json"
    sampled_valid_data_path="./sampled_data/${category}_valid.json"

    CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/compute_similarity.py \
        --data_path $sampled_train_data_path \
        --id2name_path $id2name_path \
        --model_path $base_model \
        --lora_path $lora_weights \
        --output_dir $data_output_dir \
        --output_prefix "train_item_pref_similarity" \
        --num_random_items 50 \
        --random_seed 42 \
        --chunk_size $valid_sample_size

    CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/compute_similarity.py \
        --data_path $sampled_valid_data_path \
        --id2name_path $id2name_path \
        --model_path $base_model \
        --lora_path $lora_weights \
        --output_dir $data_output_dir \
        --output_prefix "valid_item_pref_similarity" \
        --num_random_items 50 \
        --random_seed 42 \
        --chunk_size $valid_sample_size

    for ((i=0;i<$its;i++))
    do
        echo ----------------- Iteration$i starts! -----------------
        data_it_output_dir="${data_output_dir}/it${i}/"

        
        sft_train_data_path="${data_it_output_dir}/sft/train.jsonl"
        sft_valid_data_path="${data_it_output_dir}/sft/valid.jsonl"

        echo "data_it_output_dir: $data_it_output_dir"
        echo "sft_train_data_path: $sft_train_data_path"
        echo "sft_valid_data_path: $sft_valid_data_path"

        Min_ln_ches_train_data_path="${data_it_output_dir}/${metric}/train.jsonl"
        Min_ln_ches_valid_data_path="${data_it_output_dir}/${metric}/valid.jsonl"

        echo "Checking if files exist..."
        echo "Checking Min_ln_ches_train_data_path: $Min_ln_ches_train_data_path"
        echo "Checking Min_ln_ches_valid_data_path: $Min_ln_ches_valid_data_path"

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

            # mkdir -p "${it_output_dir}/SPRec"
            # mkdir -p "${it_output_dir}/sft"
            # mkdir -p "${it_output_dir}/DPO_RN1"
            # touch "./sampled_data/${category}_train.json"
            # touch "./sampled_data/${category}_valid.json"
            
            # Data Generation
            echo ""
            echo ""
            echo "selecting samples based on similarity"
            echo "------------------------------------------------------------------------------"
            CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/select_reject_base_sim_from_all_sim.py \
                --similarity_chunk_dir $data_output_dir \
                --output_dir $data_it_output_dir \
                --data_type "train" \
                --it_path $sft_train_data_path

            CUDA_VISIBLE_DEVICES=$gpu1 python ./src/data/select_reject_base_sim_from_all_sim.py \
                --similarity_chunk_dir $data_output_dir \
                --output_dir $data_it_output_dir \
                --data_type "valid" \
                --it_path $sft_valid_data_path
        fi

        echo "Checking if model for Iteration $i already exists..."

        for f in "${lr[@]}"; do
            model_output_dir="./experiments/models/${category}_${f}"
            metrics_output_dir="./experiments/metrics/${category}_${f}"
            model_it_output_dir="${model_output_dir}/it${i}/"
            metrics_it_output_dir="${metrics_output_dir}/it${i}/"

            echo "Looking for file: ${metrics_it_output_dir}/${metric}/eval_top5.json"
            if [ ! -f "${metrics_it_output_dir}/${metric}/eval_top5.json" ]; then
                echo ""
                echo ""
                echo ----------------- Training DPO -----------------
                
                echo "Training with learning rate: $f"
                bash ./shell/single_DPO_runner.sh $gpu1 $i $category $metric $f
            
            else
                echo "Model for Iteration $i with learning rate $f already exists. Skipping training."
            fi
        done
    done
done