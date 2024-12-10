import json
import pandas as pd
from tqdm import tqdm
import random

import time
import boto3
from botocore.exceptions import ClientError


class Claude:
   def __init__(self, model_id):
       # Create a Bedrock Runtime client in the AWS Region you want to use.
       self.client = boto3.client("bedrock-runtime", region_name="us-east-1")


       # Set the model ID, e.g., Claude 3 Haiku.
       self.model_id = model_id
       self.max_retries = 5
  
   def converse(self, conversation, inference_config):
       curr_retries = 0
       while True:
           try:
               response = self.client.converse(
                   modelId = self.model_id,
                   messages = conversation,
                   inferenceConfig = inference_config
               )
               return response['output']['message']['content'][0]['text']
           except (ClientError, Exception) as e:
               print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
               if curr_retries >= self.max_retries:
                   print("EXITING...")
                   exit(1)
               else:
                   print("RETRYING...")
                   curr_retries += 1
                   time.sleep(5)

prompt = """Translate the following AI scientific term directly into {tgt_lang} based on its context. If the term is an abbreviation that could reduce confusion by keeping it as is, provide the original English term. Directly provide your translated term without any explanations.

- Term: {term}
- Context: "{context}"
"""




df = pd.read_csv("growing_dict/final_terms_with_context.csv")
data = df.to_dict(orient='records')
translator = Claude("anthropic.claude-3-sonnet-20240229-v1:0")

res = {
    "English": [],
    "Arabic": [],
    "Chinese": [],
    "French": [],
    "Japanese": [],
    "Russian": []
}
for item in tqdm(data):
    term = item['processed_term']
    context = eval(item['context'])
    if len(context) > 3:
        context = random.sample(context, 3)
    res['English'].append(term)
    for lang in res.keys():
        if lang == "English":
            continue
        
        prompt = prompt.format(
            tgt_lang=lang,
            term = term,
            context = context
        )
        
        term_trans = translator.converse(
            conversation=[{
                'role': 'user',
                'content': [{'text': prompt}]
            }],
            inference_config={
                "maxTokens": 128,
                "temperature": 0.8,
            }
        )
        res[lang].append(term_trans)
    
df = pd.DataFrame()
for lang in res.keys():
    df[lang] = res[lang]
df.to_csv("growing_dict/final_terms_gpt35.csv")