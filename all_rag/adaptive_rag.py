import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from typing import Dict, Any
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from helper_functions import *
from evaluation.evalute_rag import *

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


class categories_options(BaseModel):
    category: str = Field(
        description='The category of query, the options are: Factual, Analytical, Opinion, or Contextual',
        exaple='Factual')


class QueryClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name='gpt-4o', max_tokens=4000)
        self.prompt = PromptTemplate(
            input_variables=['query'],
            template='Classify the following query into one of these categories: Factual, Analytical, Opinion, or Contextual.\nQuery:{query}\nCategory:'
        )
        self.chain = self.prompt | self.llm.with_structured_output(categories_options)

    def classify(self, query):
        return self.chain.invoke(query).category


class BaseRetrievalStrategy:
    def __init__(self, texts):
        self.embeddings = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        self.documents = text_splitter.create_documents(texts)
        self.db = FAISS.from_documents(self.documents, self.embeddings)
        self.llm = ChatOpenAI(temperature=0,model='gpt-4o',max_tokens=4000)

    def retrieve(self,query,k=4):
        return self.db.similarity_serch(query,k=k)
    