import typing
import random
import torch
import os
from datasets import IterableDataset, load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

script_dir = os.path.realpath(os.path.dirname(__file__))
pretrained_model_path = os.path.normpath(os.path.join(script_dir, "..", "downloads"))

# Group size is the like a batch size for GTPQ
# Each group is processed together and will share the same quant params e.g. scaling factor, zero point
# TODO: When 3.0.0 is released, add v2=True flag for GTPQ v2 quant with higher accuracy recovery
quant_config = QuantizeConfig(bits=4, group_size=128)
model = GPTQModel.load(pretrained_model_path, quant_config)

dataset_num_rows = 1_900_000  # Hard to know num rows of streamed dataset up front. Lets just hardcode it since I know approx
buffer_size = 1000  # Used for in memory shuffling
reasoning_dataset = typing.cast(
    IterableDataset,
    load_dataset("PrimeIntellect/SYNTHETIC-1", streaming=True, split="train"),
)
n_skip = random.randint(0, dataset_num_rows - 1 - buffer_size)
shuffled_dataset = reasoning_dataset.skip(n_skip).shuffle(buffer_size=buffer_size)

num_samples = 300  # Enough for calibration
calibration_sample = list(reasoning_dataset.take(num_samples))
# We only need the prompt inputs. GPTQ will calibrate and attempt to mimic based on that
calibration_sample = [x["prompt"] for x in calibration_sample]

model.quantize(calibration_sample, batch_size=1)

quant_model_path = os.path.normpath(
    os.path.join(script_dir, "../quantised_models/ds_r1_14b_qwen_4bit")
)
