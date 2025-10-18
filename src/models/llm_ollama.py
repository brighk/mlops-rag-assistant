"""
LLM using Ollama API - Much easier!
"""
import requests
import yaml

class OllamaLLM:
    """Use Ollama for LLM inference"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.api_url = "http://localhost:11434/api/generate"
        self.model = "phi3"
        print("✓ Using Ollama API")
    
    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate using Ollama"""
        
        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": max_length
                }
            }
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Ollama error: {response.text}")
    
    def format_prompt(self, query: str, context: str) -> str:
        """Format prompt"""
        return f"""Use the following context to answer the question concisely.

Context:
{context}

Question: {query}

Answer:"""

if __name__ == "__main__":
    llm = OllamaLLM()
    test = llm.generate("What is 2+2?")
    print(test)
