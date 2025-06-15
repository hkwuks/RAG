import os
import sys
import argparse
import time

from contourpy.util.data import simple
from dotenv import load_dotenv
from helper_functions import *
from evaluation.evalute_rag import *

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


class SimpleRAG:
    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retieved=2):
        '''
        Initializes the SimpleRAGRetriever by encoding the PDF document and creating the retriever.
        Args:
            path: Path to the PDF file to encode.
            chunk_size: Size of each text chunk.
            chunk_overlap: Overlap between consecutive chunks.
            n_retieved: Number of chunks to retrieve for each query.
        '''

        start_time = time.time()
        self.vector_store = encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.time_records = {'Chunking': time.time() - start_time}

        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={'k': n_retieved})

    def run(self, query):
        '''
        Retrieves and displays the context for the given query.
        Args:
            query: The query to retrieve context for.

        Returns:
            tuple: The retrieval time.
        '''

        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        show_context(context)


def validate_args(args):
    if args.chunk_size <= 0:
        raise ValueError('Chunk size must be a positive integer.')
    if args.chunk_overlap < 0:
        raise ValueError('Chunk overlap must be a non-negative integer.')
    if args.n_retrieved <= 0:
        raise ValueError('n_retrieved must be a positive integer.')
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data/understanding_climate_change.pdf',
                        help='Path to the PDF file to encode.')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Size of each text chunk.')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Overlap between consecutive chunks.')
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="Number of chunks to retrieve for each query (default: 2).")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever (default: 'What is the main cause of climate change?').")
    parser.add_argument("--evaluate", action="store_true",
                        help="Whether to evaluate the retriever's performance (default: False).")
    return validate_args(parser.parse_args())


def main(args):
    simple_rag = SimpleRAG(
        path=args.path,
        chunk_size=args.chunk,
        chunk_overlap=args.chunk_overlap,
        n_retieved=args.n_retieved
    )

    simple_rag.run(args.query)

    if args.evaluate:
        evaluate_rag(simple_rag.chunks_query_retriever)


if __name__ == '__main__':
    main(parse_args())
