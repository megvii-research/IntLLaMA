SIZE=$1
BIT_WIDTH=$2
### baseline

python3 convert.py \
    --model_name  huggyllama/llama-$1 \
    --calib_data ./sources/c4-eos-128-calib.json \
    --nsample 128 \
    --bit_width 4 \
    --is_perchannel \
    --groupsize 128 \
    --output_dir ./train_log/converted_ckpts/llama_$1_w4_g128 \
    --eval_tiny_data wikitext2 \
    --quant_type gptq \
    --enable_mmlu_evaluation \
    --input_bit $2 \
    --enable_input_quant \
    --enable_random_project \
    --load_awq ../sources/smooth_scales/llama-$1-w4-g128-scale.pt \
    --svd_deltaW_rank 8 \
    --enable_svd_deltaW \
    #--true_sequential \
    #--enable_smooth \

### baseline wi gscales_decompose
#python3 convert.py \
#    --model_name huggyllama/llama-$1 \
#    --calib_data ./sources/huggyllama-c4-1024-seed-0-calib.json \
#    --nsample 1024 \
#    --bit_width $2 \
#    --is_perchannel \
#    --groupsize 128 \
#    --output_dir ./train_log/converted_ckpts/llama_$1_$2w_g128_gs_decomp \
#    --eval_tiny_data wikitext2 \
#    --enable_gscales_decompose \
#    --enable_mmlu_evaluation
#
#### baseline wi gscales_decompose wi svd_deltaW
#python3 convert.py \
#    --model_name huggyllama/llama-$1 \
#    --calib_data ./sources/huggyllama-c4-1024-seed-0-calib.json \
#    --nsample 1024 \
#    --bit_width $2 \
#    --is_perchannel \
#    --groupsize 128 \
#    --output_dir ./train_log/converted_ckpts/llama_$1_$2w_g128_gs_decomp_svd8 \
#    --eval_tiny_data wikitext2 \
#    --enable_gscales_decompose \
#    --enable_svd_deltaW \
#    --svd_deltaW_rank 8 \
#    --enable_mmlu_evaluation
#
#### baseline wi gscales_decompose wi svd_deltaW wi randproj
#python3 convert.py \
#    --model_name huggyllama/llama-$1 \
#    --calib_data ./sources/huggyllama-c4-1024-seed-0-calib.json \
#    --nsample 1024 \
#    --bit_width $2 \
#    --is_perchannel \
#    --groupsize 128 \
#    --output_dir ./train_log/converted_ckpts/llama_$1_$2w_g128_gs_decomp_svd8_randproj \
#    --eval_tiny_data wikitext2 \
#    --enable_gscales_decompose \
#    --enable_svd_deltaW \
#    --svd_deltaW_rank 8 \
#    --enable_random_project \
#    --enable_mmlu_evaluation
