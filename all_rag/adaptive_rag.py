import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from typing import Dict, Any
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from helper_functions import *
from evaluation.evalute_rag import *


# load_dotenv()

# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


class categories_options(BaseModel):
    category: str = Field(
        description='The category of query, the options are: Factual, Analytical, Opinion, or Contextual',
        examples=['Factual'])


class QueryClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model='gpt-4o', max_tokens=4000)
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
        self.llm = ChatOpenAI(temperature=0, model='gpt-4o', max_tokens=4000)

    def retrieve(self, query, k=4):
        return self.db.similarity_serch(query, k=k)


class relevant_score(BaseModel):
    score: float = Field(description='The relevance score of the document to the query', example=8.0)


class FactualRetrievalStrategy(BaseRetrievalStrategy):
    '''Queries seeking specific, verifiable information'''

    def retrieve(self, query, k=4):
        enhanced_query_prompt = PromptTemplate(
            input_variables=['query'],
            template='Enhance this facutal query for better information retrieval: {qurey}'
        )
        query_chain = enhanced_query_prompt | self.llm
        enhanced_query = query_chain.invoke(query).content

        docs = self.db.similarity_serch(enhanced_query, k=k * 2)

        ranking_prompt = PromptTemplate(
            input_variables=['query', 'doc'],
            template="On a scale of 1-10, how relevant is this document to the query: '{query}'?\nDocument: {doc}\nRelevance score:"
        )
        ranked_chain = ranking_prompt | self.llm.with_structured_output(relevant_score)

        ranked_docs = []
        for doc in docs:
            input_data = {'query': enhanced_query, 'doc': doc.page_content}
            score = float(ranked_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))

        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:k]]


class SelectedIndices(BaseModel):
    indices: List[int] = Field(description='Indices of selected documents', examples=[0, 1])


class SubQueries(BaseModel):
    sub_queries: List[str] = Field(description='List of sub-queries for comprehensive analysis',
                                   examples=['What is the population of New York?', 'What is the GDP of New York?'])


class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    '''Queries requiring comprehensive analysis or explanation'''

    def retrieve(self, query, k=4):
        sub_queries_prompt = PromptTemplate(
            input_variables=['query', 'k'],
            template='Generate {k} sub-questions for:{query}'
        )

        llm = ChatOpenAI(temperature=0, model='gpt-4o', max_tokens=4000)
        sub_queries_chain = sub_queries_prompt | llm.with_structured_output(SubQueries)

        input_data = {'query': query, 'k': k}
        sub_queries = sub_queries_chain.invoke(input_data).sub_queries

        all_docs = []
        for sub_query in sub_queries:
            all_docs.extend(self.db.similarity_serch(sub_query, k=2))

        diversity_prompt = PromptTemplate(
            input_variables=['query', 'doc', 'k'],
            template='''Select the most diverse and relevant set of {k} documents for the query: '{query}'\nDocuments: {docs}\nReturn only the indices of selected documents as a list of integers.'''
        )
        diversity_chain = diversity_prompt | llm.with_structured_output(SelectedIndices)
        docs_text = '\n'.join([f'{i}:{doc.page_content[:50]}...' for i, doc in enumerate(all_docs)])
        input_data = {'query': query, 'doc': docs_text, 'k': k}
        selected_indices_result = diversity_chain.invoke(input_data).indices
        return [all_docs[i] for i in selected_indices_result if i < len(all_docs)]


class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=3):
        # Use LLM to identify potential viewpoints
        viewpoints_prompt = PromptTemplate(
            input_variables=['query', 'k'],
            template='Identify {k} distinct viewpoints or perspectives on the topic: {query}'
        )
        viewpoints_chain = viewpoints_prompt | self.llm
        input_data = {'query': query, 'k': k}
        viewpoints = viewpoints_chain.invoke(input_data).content.split('\n')

        all_docs = []
        for viewpoint in viewpoints:
            all_docs.extend(self.db.similarity_serch(f'{query} {viewpoint}', k=2))

        # Use LLM to classify and select diverse opinions
        opinion_prompt = PromptTemplate(
            input_variables=['query', 'docs', 'k'],
            template='Classify these documents into distinct opinions on {query} and select the {k} most representative and diverse viewpoints:\nDocuments: {docs}\nSelected indices:'
        )
        opinion_chain = opinion_prompt | self.llm.with_structured_output(SelectedIndices)
        docs_text = '\n'.join([f'{i}: {doc.page_content[:100]}...' for i, doc in enumerate(all_docs)])
        input_data = {'query': query, 'docs': docs_text, 'k': k}
        selected_indices = opinion_chain.invoke(input_data).indices
        return [all_docs[int(i)] for i in selected_indices if i.isdigit() and int(i) < len(all_docs)]
