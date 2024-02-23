@echo off

:: Prompt user for their username
set /p USERNAME="Please enter the directory name located in the Users directory which contains your anaconda3 installation: "

:: Construct the Conda path using the provided username
set CONDA_PATH=C:\Users\%USERNAME%\anaconda3\Scripts\activate.bat

:: Activate the rag environment and run the first app
start cmd /k ""%CONDA_PATH%" activate rag ^&^& python trt-llm-rag-windows\app.py --trt_engine_path model --trt_engine_name llama_float16_tp1_rank0.engine --tokenizer_dir_path model --data_dir trt-llm-rag-windows\dataset"

:: Activate the autogen environment and run the second app
start cmd /k ""%CONDA_PATH%" activate autogen ^&^& python trt-llm-as-openai-windows\app.py --trt_engine_path model --trt_engine_name llama_float16_tp1_rank0.engine --tokenizer_dir_path model --port 8081"

:: Run the autogenstudio UI in the autogen environment
start cmd /k ""%CONDA_PATH%" activate autogen ^&^& autogenstudio ui --port 8080"

:: Open the default web browser at the specified URL
start http://127.0.0.1:8080/
