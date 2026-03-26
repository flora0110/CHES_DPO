gpu1=$1; 
# gpu2=$2; gpu3=$3; gpu4=$4; 
its=$2
train_sample_size=2048;valid_sample_size=256
base_model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=4
lr=(1e-5)

# Only change the parameters above if needed
# "farthest_last_hidden_embedding_inner_prods" "nearest_last_hidden_embedding_inner_prods"
for category in "Goodreads"
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
        sprec_train_data_path="${it_output_dir}/SPRec/train.jsonl"
        sprec_valid_data_path="${it_output_dir}/SPRec/valid.jsonl"
        sft_train_data_path="${it_output_dir}/sft/train.jsonl"
        sft_valid_data_path="${it_output_dir}/sft/valid.jsonl"
        dpo_rn1_train_data_path="${it_output_dir}/DPO_RN1/train.jsonl"
        dpo_rn1_valid_data_path="${it_output_dir}/DPO_RN1/valid.jsonl"

        FILES=($sprec_train_data_path $sprec_valid_data_path $sft_train_data_path $sft_valid_data_path $dpo_rn1_train_data_path $dpo_rn1_valid_data_path)

        all_exist=true

        for f in "${FILES[@]}"; do
            if [[ ! -f "$f" ]]; then
                echo "File $f does not exist."
                all_exist=false
                break
            fi
        done

        if [[ "$all_exist" = false ]]; then
            echo "One or more required files are missing. Running data generation."
            # mkdir -p $it_output_dir
            mkdir -p "${it_output_dir}/SPRec"
            mkdir -p "${it_output_dir}/sft"
            mkdir -p "${it_output_dir}/DPO_RN1"
            touch "./sampled_data/${category}_train.json"
            touch "./sampled_data/${category}_valid.json"


            # Data Generation
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
                --seed $i
        fi


        for f in "${lr[@]}"; do
            if [ ! -f "./experiments/metrics/${category}_${f}/it${i}/${metric}/eval_top5.json" ]; then
                echo ""
                echo ""
                echo ----------------- Training DPO -----------------
                
                echo "Training with learning rate: $f"
                bash ./shell/single_DPO_runner.sh $gpu1 $i $category "DPO_RN1" $f
                bash ./shell/single_DPO_runner.sh $gpu1 $i $category "SPRec" $f
                
            else
                echo "Model for Iteration $i with learning rate $f already exists. Skipping training."
            fi
        done
    done
done