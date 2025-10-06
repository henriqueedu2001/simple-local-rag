import argparse
import os
from pathlib import Path

class CLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='CLI for interacting with the local RAG (Retrieval-Augmented Generation).')
        subparsers = self.parser.add_subparsers(dest='command', required=True)
        
        # Ingestion CMD
        parser_ingest = subparsers.add_parser(
            'ingest',
            help='Processes a PDF, performs chunking, and builds the vector index.'
        )
        parser_ingest.add_argument(
            'file_name',
            type=str,
            help='The file name in the .data directory'
        )
        parser_ingest.add_argument(
            'extraction',
            type=str,
            choices=['plain', 'layout'],
            default='plain',
            help='The type of text extraction. (default: \'plain\')'
        )
        parser_ingest.set_defaults(func=self.cmd_ingest)
        
        # Splitting CMD
        parser_split = subparsers.add_parser(
            'split',
            help='Splits the raw text in to text chunks and its embeddings'
        )
        parser_split.add_argument(
            'file_name',
            type=str,
            help='The file name in the .data directory'
        )
        parser_split.add_argument(
            'chunk_size',
            type=int,
            default=150,
            help='The size of each text chunk'
        )
        parser_split.set_defaults(func=self.cmd_split)
        
        # Retrieve CMD
        parser_retrieval = subparsers.add_parser(
            'retrieve',
            help='Retrieves k relevant text chunks for a given query'
        )
        parser_retrieval.add_argument(
            'file_name',
            type=str,
            help='The file name in the .data directory'
        )
        parser_retrieval.add_argument(
            'query',
            type=str,
            help='The query string for retrieval'
        )
        parser_retrieval.add_argument(
            'top_k',
            type=int,
            default=10,
            help='The number of retrieved text chunks'
        )
        parser_retrieval.set_defaults(func=self.cmd_retrieve)
        
        # Ask CMD
        parser_ask = subparsers.add_parser(
            'ask',
            help='Ask the local LLM with RAG'
        )
        parser_ask.add_argument(
            'file_name',
            type=str,
            help='The file name in the .data directory'
        )
        parser_ask.add_argument(
            'query',
            type=str,
            help='The query string for the RAG'
        )
        parser_ask.add_argument(
            'top_k',
            type=int,
            default=10,
            help='The number of retrieved text chunks'
        )
        parser_ask.set_defaults(func=self.cmd_ask)
    
    
    def run(self):
        args = self.parser.parse_args()
        args.func(args)
        return
    
    
    def cmd_ingest(self, args):
        file_name = args.file_name
        extraction_mode = args.extraction
        
        print(f'Ingesting pdf file...')
        print(f'\t- file_name: {file_name}')
        print(f'\t- extraction_mode: {extraction_mode}')
        return
    
    
    def cmd_split(self, args):
        file_name = args.file_name
        chunk_size = args.chunk_size
        
        print(f'Splitting raw text...')
        print(f'\t- file_name: {file_name}')
        print(f'\t- chunk_size: {chunk_size}')
        return
    
    
    def cmd_retrieve(self, args):
        file_name = args.file_name
        query = args.query
        top_k = args.top_k
        
        print(f'Retrieving relevant chunks for the given query')
        print(f'\t- file_name: {file_name}')
        print(f'\t- query: {query}')
        print(f'\t- top_k: {top_k}')
        return
    
    
    def cmd_ask(self, args):
        file_name = args.file_name
        query = args.query
        top_k = args.top_k
        
        print(f'Answering the question')
        print(f'\t- file_name: {file_name}')
        print(f'\t- query: {query}')
        print(f'\t- top_k: {top_k}')
        return
    

def main():
    cli = CLI()
    cli.run()
    return


if __name__ == '__main__':
    main()