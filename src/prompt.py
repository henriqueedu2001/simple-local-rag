from typing import *

class Prompt:
    def __init__(self):
        self.context: str = None
        self.instructions: str = None
        self.chunks: List[str] = None
        self.question: str = None
        pass
    
    
    def set_context(self, context: str):
        self.context = context
        return
    
    
    def set_instructions(self, instructions: str):
        self.instructions = instructions
        return
    
    
    def set_chunks(self, chunks: List[str]):
        self.chunks = chunks
        return
    
    def set_question(self, question: str):
        self.question = question
    
    def _compile_chunks(self):
        compiled_chunks = ''
        for index, chunk in enumerate(self.chunks):
            compiled_chunks += f'- Chunk #{index + 1}: {chunk}\n'
        self.compiled_chunks = compiled_chunks
        return
    
    
    def compile(self):
        self._compile_chunks()
        compiled_prompt = ''
        compiled_prompt += f'# Context\n{self.context}\n\n'
        compiled_prompt += f'# Instructions\n{self.instructions}\n\n'
        compiled_prompt += f'# Original Text Chunks \n{self.compiled_chunks}\n\n'
        compiled_prompt += f'# Question \n{self.question}\n\n'
        self.compiled_prompt = compiled_prompt
        return