gpu1=$1;
its=$2
train_sample_size=2048;valid_sample_size=256
# base_model="/scratch/user/chuanhsin0110/hf_models/Llama-3.2-1B-Instruct/"
base_model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=4
lr=1e-6
# category="Goodreads"
MAX_EPOCH=10
metric_for_best_model="accuracies"
metric_for_best_model_key="eval_rewards/${metric_for_best_model}"
greater_is_better=True

for category in "Goodreads"
do
    echo ----------------- Training $category with DPO starting! -----------------
    if [ "$category" == "Goodreads" ]; then
        id2name_path="../SPRec/eval/${category}/id2name.json"
        name2id_path="../SPRec/eval/${category}/name2id.json"
        embeddings_path="../SPRec/eval/${category}/embeddings.pt"
        test_dataset="../SPRec/data/${category}/test.json"
        lora_weights="../SPRec/models/SFT_4096/${category}"

    else
        id2name_path="./eval/${category}/id2name.json"
        name2id_path="./eval/${category}/name2id.json"
        embeddings_path="./eval/${category}/embeddings.pt"
        train_data_for_head_tail="./sampled_data/${category}_train.json"
        test_dataset="./sampled_data/${category}_test.json"
        lora_weights="./experiments/models/SFT/${category}"
    fi
    


    
    train_data_for_head_tail="../SPRec/data/${category}/train.json"
    for ((i=2;i<$its;i++))
    do
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
        for metric in "DPO_RN1" "SPRec"
        do
            
            
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


            mkdir -p $model_path
            touch "${model_path}"
            echo "DPO_path Rate: $DPO_path"
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
                --sh_file_path "./shell/DPO.sh" \
                --metric_for_best_model $metric_for_best_model_key \
                --greater_is_better $greater_is_better
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
            
        done
    done

    for metric in "DPO_RN1" "SPRec"
    do
        echo "----------------- Aggregating eval_top5.json for metric: ${metric} -----------------"
        python ./src/evaluate/agg_eval.py "./experiments/metrics/${category}_${lr}" "$metric" > ""./experiments/metrics/${category}_${lr}/${metric}_head_tail_top5_table.md"
        echo "Saved: "./experiments/metrics/${category}_${lr}/${metric}_head_tail_top5_table.md"
        echo SPRec for category ${category} has successfully completed!

        python ./src/evaluate/agg_gini.py \
            --base_dir "./experiments/metrics/${category}_${lr}" \
            --iterations 5 \
            --metrics ${metric} \
            --topks 5
    done
done