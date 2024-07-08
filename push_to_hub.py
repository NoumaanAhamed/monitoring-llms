from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "microsoft/Phi-3-mini-128k-instruct"

from huggingface_hub import login

# login(token=)

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True)

# Load the model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


hf_username = "Noumaan"
repo_name = "phi3-mini-128k-instruct-4bit-quantized"

# Create the full repository name
repo_id = f"{hf_username}/{repo_name}"

# Push the model and tokenizer to the Hub
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)