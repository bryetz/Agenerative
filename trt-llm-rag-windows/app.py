# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import logging
from flask import Flask, request, jsonify
import argparse
from trt_llama_api import TrtLlmAPI #llama_index does not currently support TRT-LLM. The trt_llama_api.py file defines a llama_index compatible interface for TRT-LLM.
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import set_global_service_context
from faiss_vector_storage import FaissEmbeddingStorage

app = Flask(__name__)

# Create an argument parser
parser = argparse.ArgumentParser(description='NVIDIA Chatbot Parameters')

# Add arguments
parser.add_argument('--trt_engine_path', type=str, required=True,
                    help="Path to the TensorRT engine.", default="")
parser.add_argument('--trt_engine_name', type=str, required=True,
                    help="Name of the TensorRT engine.", default="")
parser.add_argument('--tokenizer_dir_path', type=str, required=True,
                    help="Directory path for the tokenizer.", default="")
parser.add_argument('--embedded_model', type=str,
                    help="Name or path of the embedded model. Defaults to 'sentence-transformers/all-MiniLM-L6-v2' if "
                         "not provided.",
                    default='sentence-transformers/all-MiniLM-L6-v2')
parser.add_argument('--data_dir', type=str, required=False,
                    help="Directory path for data.", default="./dataset")
parser.add_argument('--verbose', type=bool, required=False,
                    help="Enable verbose logging.", default=False)
# Parse the arguments
args = parser.parse_args()

# Use the provided arguments
trt_engine_path = args.trt_engine_path
trt_engine_name = args.trt_engine_name
tokenizer_dir_path = args.tokenizer_dir_path
embedded_model = args.embedded_model
data_dir = args.data_dir
verbose = args.verbose

# create trt_llm engine object
llm = TrtLlmAPI(
    model_path=trt_engine_path,
    engine_name=trt_engine_name,
    tokenizer_dir=tokenizer_dir_path,
    temperature=0.1,
    max_new_tokens=1024,
    context_window=3900,
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=False
)

# create embeddings model object
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=embedded_model))
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(service_context)

# load the vectorstore index
faiss_storage = FaissEmbeddingStorage(data_dir=data_dir)
query_engine = faiss_storage.get_query_engine()

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/retrieve_docs', methods=['POST'])
def retrieve_documents():
    try:
        data = request.json
        query = data.get('query')
        
        # Log the received query
        app.logger.debug(f"Received query: {query}")
        
        if not query:
            return jsonify({"error": "Query parameter is missing."}), 400
        
        try:
            retrieved_doc = query_engine.query(query)
            # Log the retrieved documents
            app.logger.debug(f"Retrieved document: {str(retrieved_doc)}")
        except Exception as e:
            app.logger.error(f"Error during document retrieval: {e}")
            return jsonify({"error": "Failed to retrieve documents."}), 500
        
        return jsonify({"documents": str(retrieved_doc)}), 200
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
