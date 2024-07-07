import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
# from memory_profiler import profile

def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1024**2  # Convert to MB

def get_gpu_max_memory():
    return torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB

def track_gpu_memory(func):
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()  # Reset peak stats
        start_mem = get_gpu_memory()
        result = func(*args, **kwargs)
        end_mem = get_gpu_memory()
        max_mem = get_gpu_max_memory()
        print(f"GPU Memory: {start_mem:.2f}MB -> {end_mem:.2f}MB (Peak: {max_mem:.2f}MB)")
        return result
    return wrapper


class ModelManager:
    def __init__(self, model_name):
        self.model_name = model_name
        self.pipe = None

    @track_gpu_memory
    # @profile
    def __enter__(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=True,
            model_kwargs={"quantization_config":                 BitsAndBytesConfig(load_in_4bit=True)}
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free_memory()

    @track_gpu_memory
    # @profile
    def generate(self, messages,generation_args):
        output = self.pipe(messages,**generation_args)
        return output

    @track_gpu_memory    
    # @profile
    def free_memory(self):
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory freed.")

@track_gpu_memory
# @profile
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
        output = manager.generate(messages,generation_args)
        print("Response:", output[0]["generated_text"][-1]["content"])

if __name__ == "__main__":
    main()