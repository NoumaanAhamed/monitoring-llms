import torch
import gc
import time
import psutil
from functools import wraps
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from utils import track_resources

class ModelManager:
    def __init__(self, model_name):
        self.model_name = model_name
        self.pipe = None

        
    @track_resources
    # @profile
    def __enter__(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=True,
            # model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free_memory()

    @track_resources
    def generate(self, messages, generation_args):
        output = self.pipe(messages, **generation_args)
        return output

    @track_resources
    # @profile
    def free_memory(self):
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory freed.")

@track_resources
def main():
    model_name = 'microsoft/Phi-3-mini-128k-instruct'
    with ModelManager(model_name) as manager:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Who are you?"}
        ]
        generation_args = {
            "max_new_tokens": 1024,
        }
        output = manager.generate(messages, generation_args)
        print("Response:", output[0]["generated_text"][-1]["content"])

if __name__ == "__main__":
    main()