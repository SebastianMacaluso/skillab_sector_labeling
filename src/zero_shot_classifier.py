# from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained(
#     "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
# )

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipeline("Hey how are you doing today?")