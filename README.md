[![Demo Video](https://i.ytimg.com/an_webp/lG6T68h8yko/mqdefault_6s.webp?du=3000&sqp=CNjkxLEG&rs=AOn4CLBolG_PUMNJbOYlZXHvgx-IOTdKtw)](https://www.youtube.com/watch?v=lG6T68h8yko)

# Agenerative
Agenerative merges advanced technologies to create a RAG-based chatbot capable of code generation and execution optimized by TensorRT-LLM for inference. It uses a REST API compatible with the OpenAI API to integrate seamlessly with AutoGen using any local LLM. It leverages the llama_index and Facebook AI Similarity Search (FAISS) libraries for retrieval-augmented generation (RAG). The RAG component functions within a Flask app which AutoGen agents can use to dynamically query vectorized documents for rich context. AutoGen agents can use user-defined tools written in Python in combination with the RAG tool to access external resources. With AutoGen Studio, it offers a no-code UI, ensuring easy setup and access. Agenerative offers an efficient generative AI solution with no API costs/limits and ensures privacy with local RAG and inference for a secure way to chat with your data.

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
- Add any desired documents (.pdf, .txt, etc) for context to the dataset folder in trt-llm-rag-windows for vectorization. 
- In the terminal, run .\run.bat and input the name of your conda installation directory.
- The embeddings will be stored in the storage-default directory after being generated, so if you want to add new documents delete this folder and restart the application to generate new embeddings.
