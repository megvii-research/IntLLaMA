# IntLLaMA: A fast and light quantization solution for LLaMA [TechReport]()


## Introduction
IntLLaMA, a fast and light quantization solution reduces gpu-memory requirement and improve computational efficiency while simultaneously preserving model intelligence. Specifically, IntLLaMA facilitates a quantization-friendly distribution of hidden-states by utilizing Random Centralization to address the asymmetry and mitigate the impact of outliers. Meanwhile, Hessian-weighted Singular Value Decomposition(HSVD) is further proposed to compensate for the performance degradation caused by representing the model weights using low bit-width. Benefits from RandC and HSVD, IntLLaMA quantize the weight into 4 bit-width, hidden-state into 8 bit-width sperately and close to full-precision performance in perplexity and MMLU accuracy. 

![fig1]()

## Update News
- 2023-07-13: Release the code for LoRA instruct fine-tuing, More information can be found in 
- 2023-07-13: Release a 4w8f ChatGLMv2-6B, which archieve in C-eval and speedup . The more detail can be found in Table1 . 
- 2023-07-12: Release the code for convert a full-precision model to quantized model


## Acknowledgement
IntLLaMA was inspired by several open source projects. We are grateful for these excellent projects and list them as follows:
- GPTQ
- AWQ
- Alpaca-LoRA
- Standard-Alpaca

## License
IntLLaMA is released under the Apache 2.0 license.
