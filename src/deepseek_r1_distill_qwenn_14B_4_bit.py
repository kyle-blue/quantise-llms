import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = AutoModelForCausalLM.from_pretrained()

# Load dataset for GPTQ algorithm

# Use 'oneshot' to apply smart quantisation

# Save new quantised weigths
