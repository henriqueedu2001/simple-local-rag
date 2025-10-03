import ollama
import numpy as np
from typing import *

DEFAULT_EMBEDDING_MODEL_NAME = 'mxbai-embed-large'
DEFAULT_GENERATIVE_MODEL_NAME = 'gemma3:4b'

class GenerativeModel:
    def __init__(self, model_name: str = DEFAULT_GENERATIVE_MODEL_NAME):
        """
        Initialize a wrapper for a generative language model.

        Args:
            model_name (str, optional): The name of the generative model to use.
                Defaults to 'gemma3:4b'.

        Examples:
            >>> model = GenerativeModel()
            >>> response = model.ask("Tell me a joke about cats.")
            >>> print(response)
            "Why did the cat sit on the computer? Because it wanted to keep an eye on the mouse."
        """
        self.model_name = model_name
        pass
    
    
    def ask(self, prompt: str) -> str:
        """
        Generate a response from the generative model given a prompt.

        Args:
            prompt (str): The input text prompt to send to the model.

        Returns:
            str: The model's generated response as plain text.

        Examples:
            >>> model = GenerativeModel()
            >>> response = model.ask("Summarize the plot of Hamlet in one sentence.")
            >>> print(response)
            "Prince Hamlet seeks revenge for his father's murder, leading to tragedy."
        """
        response = ollama.generate(model=self.model_name, prompt=prompt)
        response = response.get('response')
        return response
    

class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL_NAME):
        """
        Initialize a wrapper for an embedding model.

        Args:
            model_name (str, optional): The name of the embedding model to use.
                Defaults to 'mxbai-embed-large'.

        Examples:
            >>> embed_model = EmbeddingModel()
            >>> vector = embed_model.embed("Artificial intelligence is fascinating.")
            >>> print(vector.shape)
            (1024,)   # Example dimension of the embedding
        """
        self.model_name = model_name
        pass
    
    
    def embed(self, input_text: Union[str, List[str]]) -> np.typing.ArrayLike:
        """
        Generate vector embeddings for input text using the embedding model.

        Args:
            input_text (Union[str, List[str]]): A single string or a list of strings
                for which to generate embeddings.

        Returns:
            np.typing.ArrayLike: A NumPy array of embeddings. If a single string
                is provided, returns a 1D array. If a list of strings is provided,
                returns a 2D array where each row corresponds to an embedding.

        Examples:
            >>> embed_model = EmbeddingModel()
            >>> # Single string input
            >>> vector = embed_model.embed("Deep learning models are powerful.")
            >>> print(vector.shape)
            (1024,)

            >>> # List of strings input
            >>> vectors = embed_model.embed(["Cat", "Dog", "Elephant"])
            >>> print(vectors.shape)
            (3, 1024)
        """
        embeddings = ollama.embed(model=self.model_name, input=input_text)
        embeddings = embeddings.get('embeddings')
        if type(input_text) == str: embeddings = embeddings[0]
        embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings