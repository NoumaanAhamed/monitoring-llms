import torch
import gc
import time
import psutil
from functools import wraps
from transformers import pipeline, BitsAndBytesConfig
from sample_utils import track_resources

class ModelManager:
    def __init__(self, model_name):
        self.model_name = model_name
        self.pipe = None

        
    @track_resources('time', 'sys_mem', 'proc_mem', 'cpu')
    def __enter__(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=True,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free_memory()

    @track_resources('time', 'sys_mem', 'proc_mem', 'cpu')
    def generate(self, messages):
        output = self.pipe(messages)
        return output

    @track_resources('time', 'sys_mem', 'proc_mem', 'cpu')
    def free_memory(self):
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory freed.")

@track_resources('time', 'sys_mem', 'proc_mem', 'cpu')
def main():
    model_name = 'gpt2'
    with ModelManager(model_name) as manager:
        messages = "Who are you?"
        output = manager.generate(messages)
        print("Response:", output[0]['generated_text'])

if __name__ == "__main__":
    main()