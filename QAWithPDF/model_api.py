import sys
import os

from dotenv import load_dotenv
import sys

from llama_index.llms.gemini import Gemini
from IPython.display import Markdown, display
import google.generativeai as genai
from llama_index.core import Settings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exception import customexception
from logger import logging

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

def load_model():
    
    """
    Loads a Gemini-Pro model for natural language processing.

    Returns:
    - Gemini: An instance of the Gemini class initialized with the 'gemini-pro' model.
    """
    try:
        Settings.llm=Gemini(models='gemini-pro',temperature=0.1,api_key=GOOGLE_API_KEY)
        return Settings.llm
    except Exception as e:
        raise customexception(e,sys)
        