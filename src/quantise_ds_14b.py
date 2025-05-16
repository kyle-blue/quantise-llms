import typing
import torch
import os
from datasets import IterableDataset, load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

script_dir = os.path.realpath(os.path.dirname(__file__))
pretrained_model_path = os.path.normpath(
    os.path.join(script_dir, "..", "downloads", "deepseek_distill_qwen_14b")
)

quant_config = QuantizeConfig(bits=4, group_size=64, v2=True)
model = GPTQModel.load(pretrained_model_path, quant_config)

dataset_num_rows = 10_000
buffer_size = 10_000
dataset_name = "data-is-better-together/10k_prompts_ranked"
reasoning_dataset = typing.cast(
    IterableDataset,
    load_dataset(dataset_name, streaming=True, split="train"),
)
shuffled_dataset = reasoning_dataset.shuffle(buffer_size=buffer_size)

num_samples = 2000
calibration_sample = list(reasoning_dataset.take(num_samples))
calibration_sample = [x["prompt"] for x in calibration_sample]

model.quantize(calibration_sample, batch_size=8)

quant_model_path = os.path.normpath(
    os.path.join(script_dir, "../quantised_models/ds_r1_14b_qwen_4bit_opt")
)

model.save(quant_model_path)
