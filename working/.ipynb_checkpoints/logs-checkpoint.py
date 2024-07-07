import torch
import gc
import time
import psutil
from functools import wraps
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from utils import track_resources
import datetime

def log_separator(message):
    print("\n" + "="*50)
    print(f" {message} ".center(50, "="))
    print("="*50 + "\n")

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class ModelManager:
    def __init__(self, model_name, quantization_config=None):
        self.model_name = model_name
        self.pipe = None
        self.quantization_config = quantization_config
        
    @track_resources
    def __enter__(self):
        print(f"[{timestamp()}] Loading model: {self.model_name}")
        model_kwargs = {}
        if self.quantization_config:
            model_kwargs["quantization_config"] = self.quantization_config
            print(f"[{timestamp()}] Using quantization: {self.quantization_config}")
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=True,
            model_kwargs=model_kwargs
        )
        print(f"[{timestamp()}] Model loaded successfully")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free_memory()

    @track_resources
    def generate(self, messages, generation_args):
        print(f"[{timestamp()}] Generating response...")
        output = self.pipe(messages, **generation_args)
        print(f"[{timestamp()}] Response generated")
        return output

    @track_resources
    def free_memory(self):
        print(f"[{timestamp()}] Freeing memory...")
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[{timestamp()}] Memory freed")

@track_resources
def main(quantization_config=None):
    model_name = 'microsoft/Phi-3-mini-128k-instruct'
    with ModelManager(model_name, quantization_config) as manager:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the number that rhymes with the word we use to describe a tall plant?"}
        ]
        generation_args = {
            "max_new_tokens": 1024,
        }
        output = manager.generate(messages, generation_args)
        print(f"\n[{timestamp()}] AI Response:")
        print(output[0]["generated_text"][-1]["content"])

if __name__ == "__main__":
    modes = [
        ("No quantization", None),
        # ("4-bit quantization", BitsAndBytesConfig(load_in_4bit=True)),
        # ("8-bit quantization", BitsAndBytesConfig(load_in_8bit=True))
    ]

    for mode, config in modes:
        log_separator(f"Running with {mode}")
        start_time = time.time()
        main(config)
        end_time = time.time()
        print(f"\n[{timestamp()}] Total execution time: {end_time - start_time:.2f} seconds")
        log_separator(f"Finished {mode}")