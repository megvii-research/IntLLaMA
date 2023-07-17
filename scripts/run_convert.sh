SIZE=$1

python3 convert.py \
    --model_name  huggyllama/llama-$1 \
    --calib_data ./sources/calibration_dataset.json \
    --nsample 128 \
    --bit_width 4 \
    --is_perchannel \
    --groupsize 128 \
    --output_dir ./train_log/converted_ckpts/llama_$1_w4_g128 \
    --eval_tiny_data wikitext2 \
    --quant_type gptq \
    --enable_mmlu_evaluation \
    --input_bit 8 \
    --enable_input_quant \
    --enable_random_project \
    --enable_smooth \
    --load_scales ./sources/smooth_scales/llama-$1-w4-g128-scale.pt \
    --svd_deltaW_rank 8 \
    --enable_svd_deltaW \
