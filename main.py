import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from memory_profiler import profile

class ModelManager:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipe = None

    @profile
    def __enter__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map='auto',
            trust_remote_code=True
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free_memory()

    @profile
    def generate(self, messages, generation_args):
        output = self.pipe(messages, **generation_args)
        return output

    @profile
    def free_memory(self):
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
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
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        output = manager.generate(messages, generation_args)
        print(output[0]['generated_text'])

if __name__ == "__main__":
    main()