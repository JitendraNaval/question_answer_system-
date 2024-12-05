import sys
import os
from llama_index.core import SimpleDirectoryReader

# Add the project root directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exception import customexception
from logger import logging



def load_data(data):
    """
    Load PDF documents from a specified directory.

    Parameters:
    - data (str): The path to the directory containing PDF files.

    Returns:
    - A list of loaded PDF documents. The specific type of documents may vary.
    """
    try:
        logging.info("Data loading started...")
        loader = SimpleDirectoryReader(data)
        documents = loader.load_data()
        logging.info("Data loading completed.")
        return documents
    except Exception as e:
        logging.error("Exception occurred during data loading.")
        raise customexception(e, sys)