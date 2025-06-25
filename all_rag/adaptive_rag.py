import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from typing import Dict,Any
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel,Field
from helper_functions import *
from evaluation.evalute_rag import *