from llm import GenerativeModel, EmbeddingModel
from retriever import Retriever
from ingestion import IngestionHandler, ChunkingHandler
from prompt import Prompt

import numpy as np

import os

from typing import *

class RAG:
    def __init__(self, file_name: os.path, dir_name: os.path):
        self.file_name: os.path = file_name
        self.dir_name: os.path = dir_name
        self.pdf_path: os.path = os.path.join(dir_name, file_name + '.pdf')
        self.raw_text_path: os.path = os.path.join(dir_name, file_name + '_raw_text' '.txt')
        self.chunks_path: os.path = os.path.join(dir_name, file_name + '_chunks' + '.csv')
        self.embeddings_path: os.path = os.path.join(dir_name, file_name + '_embeddings' + '.npy')
        
        # checkpoints
        self.raw_text_extracted = False
        self.chunks_extracted = False
        self.embeddings_extracted = False
        
        self.raw_text: str = None
        self.chunks: List[str] = None
        self.embeddings: np.typing.ArrayLike = None
        
        self.relevant_chunks: List[str] = None
        self.answer: str = None
        
        self.prompt: Prompt = Prompt()
        pass
    
    
    def get_checkpoints(self):
        if os.path.exists(self.raw_text_path): self.raw_text_extracted = True
        if os.path.exists(self.chunks_path): self.chunks_extracted = True
        if os.path.exists(self.embeddings_path): self.embeddings_extracted = True
        return
    
    
    def ingest(self):
        ingestion_handler = IngestionHandler(path=self.pdf_path)
        ingestion_handler.extract_raw_text('plain')
        self.raw_text = ingestion_handler.raw_text
        ingestion_handler.save_raw_text(self.raw_text_path)
        self.raw_text_extracted = True
    
    
    def split(self):
        chunker = ChunkingHandler(raw_text=self.raw_text)
        chunker.split()
        chunker.normalize_lengths()
        chunker.embed()
        chunker.save_chunks(self.chunks_path)
        self.chunks = chunker.chunks
        self.chunks_extracted = True
        chunker.save_embeddings(self.embeddings_path)
        self.embeddings = chunker.chunk_embeddings
        self.embeddings_extracted = True
    
    
    def retrieve(self, query: str):
        retriever = Retriever(
            chunks_path=self.chunks_path,
            chunks_embeddings_path=self.embeddings_path,
            top_k=20
        )
        retriever.load_chunks()
        self.relevant_chunks = retriever.search(query)
        return
    
    
    def ask_llm(self, query: str = None):
        generative_model = GenerativeModel()
        if query: answer = generative_model.ask(query)
        else: answer = generative_model.ask(self.prompt.compiled_prompt)
        self.answer = answer

rag = RAG(file_name='sample_paper', dir_name='./.data')

rag.get_checkpoints()
if not rag.raw_text_extracted: rag.ingest()
else: print('CHECKPOINT: raw text already extracted')
if not rag.chunks_extracted: rag.split()
else: print('CHECKPOINT: chunks already extracted')

rag.retrieve('autoencoder')

rag.prompt.set_context(
    """You are a research assistant specialized in artificial intelligence, medical imaging, and foundation models. 
Your role is to help the user analyze the paper "FOUNDATIONAL MODELS IN MEDICAL IMAGING: A COMPREHENSIVE SURVEY AND FUTURE VISION"."""
)
rag.prompt.set_instructions("""1. Use **only** the information contained in the text chunks above. 
2. If the answer is explicitly stated in the chunks, cite the exact line(s) or phrase(s) that support your answer. 
3. If the answer requires interpretation, provide a short explanation but always ground it by quoting or referring directly to the retrieved lines. 
4. If the retrieved chunks do not provide enough information to answer the question, respond only with: **"Not found in the provided text chunks."**
5. Do not use external knowledge or prior training data about the poem — rely only on the text chunks above.""")

rag.prompt.set_chunks(rag.relevant_chunks)

rag.prompt.set_question('What is the purpose of the foundational models in this text?')
rag.prompt.compile()

rag.ask_llm()

print(rag.prompt.compiled_prompt)
print(rag.answer)