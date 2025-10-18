"""
LLM Model - handles text generation
"""
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional

class TinyLLM:
    """Wrapper for small LLM model"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['llm']['device']
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, using CPU")
            self.device = 'cpu'
        
        print(f"Loading LLM: {self.config['llm']['model_name']}...")
        print("This may take a few minutes on first run...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['llm']['model_name'],
            trust_remote_code=True
        )
        
        # Quantization config for efficiency (optional, only with GPU)
        quantization_config = None
        if self.device == 'cuda':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['llm']['model_name'],
            quantization_config=quantization_config,
            device_map=self.device if self.device == 'cuda' else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("✓ LLM loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """Generate text from prompt"""
        
        max_length = max_length or self.config['llm']['max_length']
        temperature = temperature or self.config['llm']['temperature']
        top_p = top_p or self.config['llm']['top_p']
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from output
        generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def format_prompt(self, query: str, context: str) -> str:
        """Format prompt with context for RAG"""
        prompt = f"""<|system|>
You are a helpful AI assistant. Use the following context to answer the user's question. If the answer is not in the context, say so.
</s>
<|user|>
Context:
{context}

Question: {query}
</s>
<|assistant|>
"""
        return prompt

if __name__ == "__main__":
    # Test the LLM
    llm = TinyLLM()
    
    test_prompt = "What is machine learning? Explain briefly."
    print(f"\nPrompt: {test_prompt}")
    print("\nGenerating response...")
    
    response = llm.generate(test_prompt, max_length=200)
    print(f"\nResponse:\n{response}")
