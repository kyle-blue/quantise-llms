# test-pretrained-models

Testing some pre-trained models available on huggingface

## Pre-requisites

GPTQ requires the CUDA toolkit to be installed on the system!
This is why GPTQ must be built without build-isolation

Pytorch, an some other libs package a CUDA runtime (but not necessarily the entire toolkit) alongside the package
But GPTQ doesn't!
