from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from openai import RateLimitError
from typing import List
from rank_bm25 import BM25Okapi
import fitz
import asyncio
import random
import textwrap
import numpy as np
from enum import Enum


def replace_t_with_space(list_of_document):
    '''
    Replaces all tab characters ('\t') with spaces in the page content of each document
    :param list_of_document: A list of document objects, each with a 'page_content' attribute.
    :return: The modified list of document with tab characters replaced by spaces.
    '''

    for doc in list_of_document:
        doc.page_content = doc.page_content.replace('\t', ' ')
    return list_of_document


def text_wrap(text, width=120):
    '''
    Wraps the input text to the specified width.
    :param text: The input text to wrap.
    :param width: The width at which to wrap the text.
    :return: The wrapped text.
    '''
    return textwrap.fill(text, width=width)


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    '''
    Encodes a PDF file into a vector store using OpenAI embeddings.
    :param path: The path to the PDF file.
    :param chunk_size: The desired size of each text chunk.
    :param chunk_overlap: The amount of overlap between consecutive chunks.
    :return: A FAISS vector store containing the encoded file content.
    '''

    # load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(cleaned_texts, embeddings=embeddings)
    return vector_store


def encode_from_string(content, chunk_size=1000, chunk_overlap=200):
    '''
    Encodes a string into a vector store using OpenAI embeddings.
    Args:
        content(str): The text content to be encoded.
        chunk_size(int): The size of each chunk of text.
        chunk_overlap(int): The overlap between chunks.

    Returns:
        FAISS: A vector store containing the encoded content.

    Raises:
        ValueError: If content is not valid.
        RuntimeError: If there is an error during the encoding process.
    '''

    if not isinstance(content, str) or not content.strip():
        raise ValueError('Content must be a non-empty string.')

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('Chunk size must be a positive integer.')

    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError('Chunk overlap must be a non-negative integer.')

    try:
        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, is_separator_regex=False
        )
        chunks = text_splitter.create_documents([content])

        # Assign metadata to each chunk
        for chunk in chunks:
            chunk.metadata['relevance_score'] = 1.0

        # Generate embeddings and create the vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings=embeddings)

    except Exception as e:
        raise RuntimeError(f'An error occured during the encoding process: {str(e)}')

    return vector_store


def retrieve_context_per_question(question, chunks_query_retriever):
    '''
    Retrieves relevant context and unique URLs for a given question using the chunks query retriever.
    Args:
        question: The question for which to retrieve context and URLs.
        chunks_query_retriever:

    Returns:
        A tuple containing:
        - A string with the concatenated content of relevant documents.
        - A list of unique URLs from the metadata of the relevant documents.
    '''

    # Retrieve relevant documents for the given question
    docs = chunks_query_retriever.get_relevant_documents(question)

    # Concatenate document content
    context = [doc.page_content for doc in docs]

    return context


class QuestionAnswerFromContext(BaseModel):
    '''
    Model to generate an answer to a query based on a given context.

    Attributes:
        answer_based_on_content(str): The generated answer based on the context.
    '''
    answer_based_on_content: str = Field(description='Generates an answer to a query based on a given context.')


def create_question_answer_from_context_chain(llm):
    # Initialize the ChatOpenAI model with specific parameters
    question_answer_from_context_llm = llm

    # Define the prompt template for chain-of-thought reasoning
    question_answer_prompt_template = '''
    For the question below, provide a concise but suffice answer based ONLY on the provided context:
    {context}
    Question
    {question}
    '''

    # Create a PromptTemplate object with the specified template and input variables
    question_answer_from_context_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=['context', 'question'],
    )

    # Create a chain by combining the prompt template and the language model
    question_answer_from_context_cot_chain = question_answer_from_context_prompt | question_answer_from_context_llm.with_structured_output(
        QuestionAnswerFromContext)
    return question_answer_from_context_cot_chain


def answer_question_from_context(question, context, question_answer_from_context_chain):
    '''
    Answer a question using a given context by invoking a chain of reasoning.
    Args:
        question: The question to be answered.
        context: The context to be used for answering the question.
        question_answer_from_context_chain:

    Returns:
        A dictionary containing the answer, context, and question.
    '''

    input_data = {
        'question': question,
        'context': context
    }
    output = question_answer_from_context_chain.invoke(input_data)
    answer = output.answer_based_on_content
    return {'answer': answer, 'context': context, 'question': question}


def show_context(context):
    '''
    Display the contents of the provided context list.
    Args:
        context(list): A list of context items to be displayed.

    Prints each context item in the list with a heading indicating its position.
    '''

    for i, c in enumerate(context):
        print(f'Context {i + 1}:')
        print(c)
        print('\n')


def read_pdf_to_string(path):
    '''
    Read a PDF document from the specified path and return its content as a string.
    Args:
        path(str): The file path to the PDF document.

    Returns:
        str: The concatenated text content of all pages in the PDF document.
    '''

    doc = fitz.open(path)
    content = ''
    for page_num in range(len(doc)):
        page = doc[page_num]
        content += page.get_text()
    return content


def bm25_retrieval(bm25: BM25Okapi, cleaned_texts: List[str], query: str, k: int = 5) -> List[str]:
    '''
    Perform BM25 retrieval and return the top K cleaned text chunks.
    Args:
        bm25: Pre-computed BM25 index.
        cleaned_texts(List[str]): List of text chunks corresponding to the BM25 index.
        query(str): The query string.
        k(int): The number of text chunks to retrieve.

    Returns:
        List[str]: The top K cleaned text chunks based on BM25 scores.
    '''

    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]
    top_k_texts = [cleaned_texts[i] for i in top_k_indices]
    return top_k_texts


async def exponential_backoff(attempt):
    '''
    Implement exponential backoff with a jitter.
    Args:
        attempt: The current retry attempt number.

    Waits for a period of time before retrying the operation.
    The wait time is calculated as (2^attempt)+a random fraction of a second.
    '''

    wait_time = (2 ** attempt) + random.uniform(0, 1)
    await asyncio.sleep(wait_time)


async def retry_with_exponential_backoff(coroutine, max_retries=5):
    '''
    Retries a coroutine using exponential backoff upon encountering a RateLimitError.
    Args:
        coroutine: The coroutine to be executed.
        max_retries: The maximum number of retry attempts.

    Returns:
        The last encountered exception if all retry attempts fail.
    '''

    for attempt in range(max_retries):
        try:
            return await coroutine
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise e

            await exponential_backoff(attempt)

    raise Exception('Max retries reached')


class EmbeddingProvider(Enum):
    OPENAI = 'openai'
    COHERE = 'cohere'
    AMAZON_BEDROCK = 'bedrock'


class ModelProvider(Enum):
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    AMAZON_BEDROCK = "bedrock"


def get_langchain_embedding_provider(provider: EmbeddingProvider, model_id: str = None):
    '''
    Returns an embedding provider based on the specified provider and model ID.
    Args:
        provider(EmbeddingProvider): The embedding provider to use.
        model_id(str): Optional - The specific embedding model ID to use.

    Returns:
        A LangChain embedding provider instance.
    '''

    if provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    elif provider == EmbeddingProvider.COHERE:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings()
    elif provider == EmbeddingProvider.AMAZON_BEDROCK:
        from langchain_community.embeddings import BedrockEmbeddings
        return BedrockEmbeddings(model_id=model_id) if model_id else BedrockEmbeddings(
            model_id='amazon.titan-embed-text-v2:0')
    else:
        raise ValueError(f'Unsupported embedding provider: {provider}')
