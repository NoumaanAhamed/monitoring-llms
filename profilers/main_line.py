import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from line_profiler import profile

class ModelManager:
    def __init__(self, model_name):
        self.model_name = model_name
        self.pipe = None

    @profile
    def __enter__(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=True
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free_memory()

    @profile
    def generate(self, messages,generation_args):
        output = self.pipe(messages,**generation_args)
        return output

    @profile
    def free_memory(self):
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory freed.")

@profile
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