import time
import re
import json
import pandas as pd
import openai
from tqdm import tqdm
import sys

def openai_setup(key_path='/home/ubuntu/openai_key_r3lit.txt'):
	with open(key_path) as f:
		key, org_id = f.read().strip().split("\n")

	print("Read key from", key_path)
	openai.api_key = key.strip()
	openai.organization = org_id.strip()
 
openai_setup()

def openai_prompt(src_text, tgt_text, relevant_terms_dict, tgt_lang, model='gpt-4o'):
    src_lang = 'English'
    
    terms = "- ".join([f"{src_lang}: {src_term}, {tgt_lang}: {tgt_term}" for src_term, tgt_term in relevant_terms_dict])
    
    prompt = f"""For the following translation into {tgt_lang}, please use the specified {tgt_lang} terms for the corresponding {src_lang} terms, while keeping the other content unchanged.

Term dictionary:
{terms}

{src_lang} text:
{src_text}

{tgt_lang} translation:
{tgt_text}

If multiple terms are nested or overlap with the context in {src_lang}, select the longest span that matches the context. Provided the updated translation only.
"""
    while True:
        try:
            resp = openai.chat.completions.create(
                model = model,
                messages = [{"role": "user", "content": prompt}],
                temperature = 0,
                max_tokens = 1024
            ).choices[0].message.content
            break
        except Exception as e:
            print(e)
        time.sleep(5)
    
    # extract valid_term, explanation
    return resp

def check_substring_in_string(string, substring):
    # Construct the regular expression to look for the substring surrounded by non-alphabet characters or boundaries
    pattern = r'(?<![a-zA-Z])' + re.escape(substring) + r'(?![a-zA-Z])'
    
    # Search for the pattern in the string
    match = re.search(pattern, string)
    
    # Return True if a match is found, False otherwise
    return bool(match)

def get_relevant_term_list(en_text, term_dict):
    en_text = en_text.lower()
    relevant_term_dict = []
    for key, item in term_dict.items():
        if check_substring_in_string(en_text, key):
            relevant_term_dict.append([key, item])
        
    print(relevant_term_dict)
    return relevant_term_dict
    

data = [json.loads(i) for i in open("/home/ubuntu/multilingual-model-card/src/data_eval_6060/output/predictions_dev_seamless.jsonl", 'r').readlines()]
lang = 'Japanese'
out_f = open("/home/ubuntu/multilingual-model-card/src/data_eval_6060/output/predictions_dev_seamless_prompt_japanese.jsonl", 'a')
df = pd.read_csv("/home/ubuntu/multilingual-model-card/src/dictionary_collection/mturk/analysis/Japanese_validated.csv")
term_dict = {row['word'].lower(): row['validated_translation'] for _, row in df.iterrows()}

for line in data:
    translation = openai_prompt(
        line['text'],
        line['text_Japanese'],
        get_relevant_term_list(line['text'], term_dict),
        tgt_lang = lang,
        model='gpt-4o-mini'
    )
    info = {
        "text": line['text'],
        "text_Japanese": translation
    }
    print(info)
    
    json.dump(info, out_f, ensure_ascii=False)
    out_f.write("\n")
    out_f.flush()