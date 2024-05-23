import transformers
import torch

# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "meta-llama/Llama-2-13b-chat-hf"
access_token = "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"

# model_name = "lmsys/vicuna-13b-v1.5"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# model_name = "chavinlo/alpaca-native"

cache_dir = "/home/iouzzani/.cache"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    token=access_token,
    cache_dir=cache_dir
)

model = transformers.AutoModel.from_pretrained(
    model_name,
    token=access_token,
    trust_remote_code=True,
    cache_dir=cache_dir
)