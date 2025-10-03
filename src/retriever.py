from llm import EmbeddingModel
from typing import *
import numpy as np
import pandas as pd
import os

class Retriever:
    def __init__(self, chunks_path: os.path, chunks_embeddings_path: os.path, top_k: int = 10):
        """
        Initialize a retriever for performing similarity search over precomputed embeddings.

        Args:
            chunks_path (os.path): Path to the CSV file containing text chunks.
            chunks_embeddings_path (os.path): Path to the NumPy `.npy` file containing
                precomputed embeddings for the chunks.
            top_k (int, optional): Number of top relevant chunks to return during search.
                Defaults to 10.

        Examples:
            >>> retriever = Retriever("chunks.csv", "embeddings.npy", top_k=5)
            >>> retriever.load_chunks()
        """
        self.chunks_path: os.path = chunks_path
        self.chunks_embeddings_path: os.path = chunks_embeddings_path
        self.top_k: int = top_k
        self.chunks: List[str] = None
        self.chunk_embeddings = None
        pass
    
    
    def search(self, query: str) -> List[str]:
        """
        Retrieve the most relevant chunks for a given query based on cosine similarity.

        Args:
            query (str): The query text to search for relevant chunks.

        Returns:
            List[str]: A list of top-k most relevant chunks sorted by relevance (most relevant first).

        Examples:
            >>> retriever = Retriever("chunks.csv", "embeddings.npy", top_k=3)
            >>> retriever.load_chunks()
            >>> results = retriever.search("What is deep learning?")
            >>> print(results)
            ["Deep learning is a subset of machine learning ...",
             "Neural networks are used in deep learning ...",
             "Applications of deep learning include image recognition ..."]
        """
        embedding_model = EmbeddingModel()
        query_embedding = embedding_model.embed(query)
        scores = np.linalg.matmul(self.chunk_embeddings, query_embedding)
        top_k_indexes = np.argpartition(scores, -self.top_k)[-self.top_k:]
        top_k_indexes = np.sort(top_k_indexes)[::-1]
        relevant_chunks = [self.chunks[index] for index in top_k_indexes]
        return relevant_chunks
    
    
    def load_chunks(self):
        """
        Load text chunks and their embeddings from disk.

        The CSV file specified in `chunks_path` should contain a column named 'chunk_str'.
        The embeddings file specified in `chunks_embeddings_path` should be a NumPy `.npy` array.

        Examples:
            >>> retriever = Retriever("chunks.csv", "embeddings.npy")
            >>> retriever.load_chunks()
            >>> print(len(retriever.chunks))  # Number of chunks loaded
            >>> print(retriever.chunk_embeddings.shape)  # Shape of embeddings array
        """
        self.chunk_embeddings = np.load(self.chunks_embeddings_path)
        chunks = pd.read_csv(self.chunks_path)
        chunks = chunks['chunk_str'].to_list()
        self.chunks = chunks


