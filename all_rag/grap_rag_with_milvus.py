import os
import numpy as np
from collections import defaultdict
from scipy.sparse import csc_matrix
from pymilvus import MilvusClient
from langchain_core.messages import AIMessage, HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tqdm import tqdm

os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'

milvus_client = MilvusClient(
    uri='YOUR_ENDPOINT',
    token='YOUR_TOKEN'
)

