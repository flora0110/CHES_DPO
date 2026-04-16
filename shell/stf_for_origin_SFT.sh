base_model="meta-llama/Llama-3.2-1B-Instruct"
gpu1=$1;
# gpu3=$3; gpu4=$4 
train_sample_size=4096
valid_sample_size=512
test_sample_size=1000
seed=42
# for category in "MovieLens"  "Goodreads" "CDs_and_Vinyl" "Steam"
for category in "Goodreads"
do
    echo ---------------------- SFT for category $category starting! ---------------------- 
    
    echo --------------------- Path for Inference and evaluation for category $category ----------------------
    id2name_path="../SPRec/eval/${category}/id2name.json"
    name2id_path="../SPRec/eval/${category}/name2id.json"
    embeddings_path="../SPRec/eval/${category}/embeddings.pt"

    touch "${id2name_path}"
    touch "${name2id_path}"
    touch "${embeddings_path}"
    echo "id2name_path: $id2name_path"
    echo "name2id_path: $name2id_path"
    echo "embeddings_path: $embeddings_path"


    echo ---------------------- Paths RAW training/inference DATA for category $category ----------------------
    raw_train_dataset="./data_sprec/${category}_train.json"
    raw_valid_dataset="./data_sprec/${category}_valid.json"
    raw_test_dataset="./data_sprec/${category}_test.json"

    touch "${raw_train_dataset}"
    touch "${raw_valid_dataset}"
    touch "${raw_test_dataset}"
    echo "raw_train_dataset: $raw_train_dataset"
    echo "raw_valid_dataset: $raw_valid_dataset"
    echo "raw_test_dataset: $raw_test_dataset"


    echo ---------------------- Paths SAMPLED training/inference DATA for category $category ----------------------

    train_dataset="./sampled_data/${category}_train.json"
    valid_dataset="./sampled_data/${category}_valid.json"
    test_dataset="./sampled_data/${category}_test.json"


    echo ---------------------- Setup output paths for category $category ----------------------
    predicts_json="./experiments/predicts/SFT/${category}/predicts.json"
    eval_result_json="./experiments/metrics/SFT/${category}/eval_top5.json"
    model_path="./experiments/models/SFT/${category}"
    mkdir -p $model_path
    mkdir -p "./experiments/predicts/SFT/${category}"
    mkdir -p "./experiments/metrics/SFT/${category}"

    # echo -------------------------------------- Sampling data for category $category --------------------------------------
    # python ./src/data/sampled_data.py \
    #     --input_path $raw_train_dataset\
    #     --sample_size $train_sample_size\
    #     --output_path $train_dataset\
    #     --seed $seed

    # python ./src/data/sampled_data.py \
    #     --input_path $raw_valid_dataset\
    #     --sample_size $valid_sample_size\
    #     --output_path $valid_dataset\
    #     --seed $seed

    # python ./src/data/sampled_data.py \
    #     --input_path $raw_test_dataset\
    #     --sample_size $test_sample_size\
    #     --output_path $test_dataset\
    #     --seed $seed

    # echo -------------------------------------- SFT for category $category --------------------------------------
    # # Match gpu 4 to 1, gradient_accumulation_steps 16*4=64 effective batch size
    # CUDA_VISIBLE_DEVICES=$gpu1 python ./src/models/sft.py \
    #     --output_dir $model_path\
    #     --base_model $base_model \
    #     --train_dataset $train_dataset \
    #     --valid_dataset $valid_dataset \
    #     --gradient_accumulation_steps 16 \
    #     --batch_size 4 \
    #     --num_train_epochs 4 \
    #     --learning_rate 0.0003 \
    #     --cutoff_len 512 \
    #     --seed $seed

    echo -------------------------------------- Inference and evaluation for category $category --------------------------------------

    touch "${test_dataset}"
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

