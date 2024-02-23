# Agenerative
RAG-Chatbot connected to Autogen with TensorRT-LLM optimized inference.

## Installation:
- Install anaconda 3 on your system.
- Follow the instructions from the below repositories for installation, but do not clone their repositories, only clone TensorRT-LLM for Windows.
- https://github.com/NVIDIA/trt-llm-as-openai-windows
- https://github.com/NVIDIA/trt-llm-rag-windows
- This repository has used modified versions of the code, but still has the same dependencies.
- run pip install -r requirements.txt in trt-llm-rag-windows and trt-llm-as-openai-windows

## Building TRT Engine
Follow these steps to build your TRT engine:

Download models and quantized weights

Mistral-7B Chat Int4
- Download Mistral-7B-Instruct-v0.1 from https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
- Download Mistral-7B Chat Int4 checkpoints (mistral_tp1_rank0.npz) from https://catalog.ngc.nvidia.com/orgs/nvidia/models/mistral-7b-int4-chat/files

In the TensorRT-LLM repository:
- For Mistral engine, navigate to the examples\llama directory and run the following script.
```python
python build.py --model_dir <path to mistral model directory> --quant_ckpt_path <path to mistral_tp1_rank0.npz file> --dtype float16 --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --use_weight_only --weight_only_precision int4_awq --per_group --enable_context_fmha --max_batch_size 1 --max_input_len 3500 --max_output_len 1024 --output_dir <TRT engine folder>
```

# How to Run:
- In the terminal, run .\run.bat and input the name of your conda installation directory.
