import openai
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:3636/v1",
    api_key="EMPTY",
)

completion = client.chat.completions.create(
#   model="/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf",
    # model="/data/user_data/jiaruil5/.cache/models--chavinlo--alpaca-native/snapshots/3bf09cbff2fbd92d7d88a0f70ba24fca372befdf/",
    # model="/data/user_data/jiaruil5/.cache/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d",
    # model="/data/user_data/jiaruil5/.cache/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2",
    model="/data/user_data/jiaruil5/.cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45/",
  messages=[
    {"role": "user", "content": """What is the colour of soccer?

Try to answer concisely with the key phrase, ideally in 10 words."""}
  ],
  temperature=0,
  max_tokens=1024
)

print(completion.choices[0].message)