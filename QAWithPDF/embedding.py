import sys
import os
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
import google.generativeai as genai
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model,document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
    """

    GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

    genai.configure(api_key=GOOGLE_API_KEY)

    try:
        logging.info("")
        Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001",api_key=GOOGLE_API_KEY)
        node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

        nodes = node_parser.get_nodes_from_documents(
        document, show_progress=False)
        # maximum input size to the LLM
        Settings.context_window = 4096

        # number of tokens reserved for text generation.
        Settings.num_output = 256

        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

        
        
        logging.info("")
    
        
        
        logging.info("")
        query_engine = vector_index.as_query_engine(similarity_top_k=2)
        return query_engine
    except Exception as e:
        raise customexception(e,sys)