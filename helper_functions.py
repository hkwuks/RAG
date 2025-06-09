from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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