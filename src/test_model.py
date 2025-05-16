from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # type: ignore
import os
import fileinput
import torch
from gptqmodel import GPTQModel

device = "cuda" if torch.cuda.is_available() else "cpu"


script_dir = os.path.realpath(os.path.dirname(__file__))
pretrained_model_path = os.path.normpath(
    os.path.join(script_dir, "..", "quantised_models", "ds_r1_14b_qwen_4bit")
)

# model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
model = GPTQModel.load(pretrained_model_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

model.to(device)


print(
    "Hello! I'm a dumber (4bit) quantised version of Deekseek R1 14B param Qwen. How can I help you?"
)
for line in fileinput.input():
    prompt = f"{line}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output_ids = model.generate(
        input_ids,
        max_length=2000,  # Maximum length of the generated sequence
        num_beams=1,  # Number of beams for beam search (for better quality)
        no_repeat_ngram_size=2,  # Prevent repeating n-grams
        top_k=50,  # Sample from the top k tokens
        top_p=0.95,  # Sample within the top p probability mass
        temperature=1.0,  # Controls randomness (higher is more random)
        do_sample=True,  # Whether to use sampling; if False, uses greedy decoding
        pad_token_id=tokenizer.eos_token_id,  # Some models require explicit padding token
    )

    generated_text = tokenizer.decode(output_ids[0])

    print()
    print(generated_text)
    print()
