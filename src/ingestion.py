from llm import EmbeddingModel
from pypdf import PdfReader
import pandas as pd
import numpy as np
import os
from typing import *


class IngestionHandler:
    def __init__(self, path: os.path, extract_on_load: bool = True):
        """
        Handle PDF ingestion and optional text extraction.

        Args:
            path (os.path): Path to the PDF file.
            extract_on_load (bool, optional): If True, extract text automatically
                when the object is initialized. Defaults to True.

        Examples:
            >>> handler = IngestionHandler("document.pdf")
            >>> handler.extract_raw_text()
            >>> print(handler.raw_text[:200])  # Print first 200 characters
        """
        self.path: os.path = path
        self.reader: PdfReader = PdfReader(self.path)
        self.raw_text: str = 'empty'
        self.chunks: List[str] = None
        if extract_on_load: self.extract_raw_text()
        return


    def extract_raw_text(self, extraction_mode: Literal['plain', 'layout'] = 'plain'):
        """
        Extract raw text from all pages of the PDF.

        Args:
            extraction_mode (Literal['plain', 'layout'], optional): Extraction mode
                for text. 'plain' extracts without formatting, while 'layout'
                preserves layout information. Defaults to 'plain'.

        Examples:
            >>> handler = IngestionHandler("document.pdf")
            >>> handler.extract_raw_text(extraction_mode="layout")
            >>> print(handler.raw_text[:500])
        """
        raw_text = [page.extract_text(extraction_mode=extraction_mode) for page in self.reader.pages]
        raw_text = ''.join(raw_text)
        self.raw_text = raw_text
        return
    
    
    def save_raw_text(self, path: os.path):
        """
        Save the extracted raw text to a file.

        Args:
            path (os.path): Path where the text file should be saved.

        Examples:
            >>> handler = IngestionHandler("document.pdf")
            >>> handler.extract_raw_text()
            >>> handler.save_raw_text("output.txt")
        """
        with open(path, 'w', encoding='utf-8') as file:
            file.write(self.raw_text)
        return


class ChunkingHandler:
    def __init__(self, raw_text: str = None, file_path: os.path = None):
        """
        Handle splitting and normalizing raw text into chunks.

        Args:
            raw_text (str, optional): A string of raw text to be chunked.
                Defaults to None.
            file_path (os.path, optional): Path to a text file containing raw text.
                If provided, the file will be loaded automatically. Defaults to None.

        Examples:
            >>> chunker = ChunkingHandler(file_path="output.txt")
            >>> chunker.split()
            >>> print(chunker.chunks[:5])  # Show first 5 chunks
        """
        self.raw_text = raw_text
        self.file_path = file_path
        self.chunks: List[str] = None
        self.chunk_embeddings: np.typing.ArrayLike = None
        self.load_text()
        pass
    
    
    def load_text(self):
        """
        Load text from a file if `file_path` was provided.

        Examples:
            >>> chunker = ChunkingHandler(file_path="output.txt")
            >>> chunker.load_text()
            >>> print(chunker.raw_text[:200])
        """
        if self.file_path:
            with open(self.file_path, mode='r', encoding='utf-8') as file:
                self.raw_text = file.read()
        return


    def split(self):
        """
        Split raw text into chunks separated by newline characters.
        Empty lines are ignored.

        Examples:
            >>> chunker = ChunkingHandler(file_path="output.txt")
            >>> chunker.split()
            >>> print(len(chunker.chunks))  # Number of chunks
        """
        chunks = self.raw_text.split(sep='\n')
        chunks = [chunk.strip() for chunk in chunks]
        chunks = [chunk for chunk in chunks if chunk not in ['']]
        self.chunks = chunks
        return
    
    
    def embed(self):
        """
        Generate embeddings for the current text chunks using the default embedding model.

        This method uses the `EmbeddingModel` class to convert each chunk of text
        stored in `self.chunks` into a numerical vector representation and stores
        the result in `self.chunk_embeddings`.

        Examples:
            >>> handler = ChunkingHandler(file_path="output.txt")
            >>> handler.split()
            >>> handler.normalize_lengths(minimum_chunk_length=300)
            >>> handler.embed()
            >>> print(handler.chunk_embeddings.shape)  # e.g., (num_chunks, embedding_dim)
        """
        embedding_model = EmbeddingModel()
        self.chunk_embeddings = embedding_model.embed(self.chunks)
        return
    

    def normalize_lengths(self, minimum_chunk_length: int = 500, join_str: str = ' '):
        """
        Normalize chunk lengths by concatenating small chunks until the minimum length is reached.

        Args:
            minimum_chunk_length (int, optional): Minimum character length for each chunk.
                Defaults to 500.
            join_str (str, optional): String used to join smaller chunks. Defaults to ' '.

        Examples:
            >>> chunker = ChunkingHandler(file_path="output.txt")
            >>> chunker.split()
            >>> chunker.normalize_lengths(minimum_chunk_length=300)
            >>> print(chunker.chunks[0])  # First normalized chunk
        """
        normalized_chunks = ['']
        
        for chunk in self.chunks:
            if len(normalized_chunks[-1]) < minimum_chunk_length:
                normalized_chunks[-1] = normalized_chunks[-1] + join_str + chunk
            else:
                normalized_chunks.append(chunk)

        self.chunks = normalized_chunks
        return
    

    def save_chunks(self, path: os.path):
        """
        Save chunks to a CSV file.

        Args:
            path (os.path): Path where the CSV file should be saved.

        Examples:
            >>> chunker = ChunkingHandler(file_path="output.txt")
            >>> chunker.split()
            >>> chunker.save_chunks("chunks.csv")
        """
        data = {
            'chunk_str': self.chunks
        }
        df = pd.DataFrame(data)
        df.to_csv(path)
        return
    
    
    def save_embeddings(self, path: os.path):
        """
        Save the generated embeddings to a file in NumPy `.npy` format.

        Args:
            path (os.path): Path where the embeddings file should be saved.

        Examples:
            >>> handler = ChunkingHandler(file_path="output.txt")
            >>> handler.split()
            >>> handler.normalize_lengths(minimum_chunk_length=300)
            >>> handler.embed()
            >>> handler.save_embeddings("embeddings.npy")
        """
        np.save(file=path, arr=self.chunk_embeddings)
        return
    