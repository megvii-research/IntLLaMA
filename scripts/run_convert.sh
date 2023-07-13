SIZE=$1
BIT_WIDTH=$2
### baseline

python3 convert.py \
    --model_name  huggyllama/llama-$1 \
    --calib_data ./sources/calibration_dataset.json \
    --nsample 128 \
    --bit_width $2 \
    --is_perchannel \
    --groupsize 128 \
    --output_dir ./train_log/converted_ckpts/llama_$1_$2w_g128 \
    --eval_tiny_data wikitext2 \
    --quant_type gptq \
    --enable_mmlu_evaluation \
    --enable_input_quant \
    --enable_smooth \
    --svd_deltaW_rank 8 \
    --enable_svd_deltaW \
    --enable_random_project \
