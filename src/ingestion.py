import os
from typing import *
from pypdf import PdfReader
import pandas as pd
import ollama

DEFAULT_EMBEDDING_MODEL_NAME = 'nomic-embed-text'

class IngestionHandler:
    def __init__(self, path: os.path, extract_on_load: bool = True):
        self.path: os.path = path
        self.reader: PdfReader = PdfReader(self.path)
        self.raw_text: str = 'empty'
        self.chunks: List[str] = None
        self.embedding_model_name = DEFAULT_EMBEDDING_MODEL_NAME
        if extract_on_load: self.extract_raw_text()
        return


    def extract_raw_text(self):
        pages = [page.extract_text() for page in self.reader.pages]
        self.raw_text = ''.join(pages)
        return
    

    def split_into_chunks(self):
        chunks = self.raw_text.split(sep='.') # spliting the text by '.'
        chunks = [chunk for chunk in chunks if chunk != ''] # removing empty strings
        for bad_char in ['\n', '\t', '  ', '   ', '    ', '       ']: # removing bad characters
            chunks = [chunk.replace(bad_char, '').strip() for chunk in chunks]
        self.chunks = chunks
        return
    

    def normalize_chunks_lenghts(self, minimum_chunk_length: int = 50, join_str: str = ' '):
        normalized_chunks = ['']
        
        for chunk in self.chunks:
            if len(normalized_chunks[-1]) < minimum_chunk_length:
                normalized_chunks[-1] = normalized_chunks[-1] + join_str + chunk
            else:
                normalized_chunks.append(chunk)

        self.chunks = normalized_chunks
        return
    

    def save_chunks(self, path: os.path):
        data = {
            'chunk_str': self.chunks
        }
        df = pd.DataFrame(data)
        df.to_csv(path)
        return
            

def main():
    pdf_path = '.docs/memoriasBras.pdf'
    ingestion_handler = IngestionHandler(path=pdf_path)
    ingestion_handler.split_into_chunks()
    ingestion_handler.normalize_chunks_lenghts()
    ingestion_handler.save_chunks('.docs/memoriaBrasChunks.csv')
    return


main()