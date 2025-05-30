import time
import re
import json
import pandas as pd
import openai
from tqdm import tqdm
import sys
import argparse

def openai_setup(key_path='/home/jiaruil5/openai_key_r3lit.txt'):
	with open(key_path) as f:
		key, org_id = f.read().strip().split("\n")

	print("Read key from", key_path)
	openai.api_key = key.strip()
	openai.organization = org_id.strip()

# def openai_setup(key_path='/home/jiaruil5/openai_key_ralf_misleading.txt'):
# 	with open(key_path) as f:
# 		key = f.read().strip().split("\n")[0]

# 	print("Read key from", key_path)
# 	openai.api_key = key.strip()


def openai_prompt(term, tgt_lang, model='gpt-4o-mini'):
    prompt = f"""You are an expert in {tgt_lang} AI scientific literature. Determine whether the term "{term}" is more commonly:

A: Translated into {tgt_lang}.
B: Borrowed from English as a loanword.

Choose B only if none of the words in the term are translated and the entire term is used as-is in English. Output only "A" or "B", based on what is most prevalent in {tgt_lang} AI academic and technical contexts.
""" 

    while True:
        try:
            resp = openai.ChatCompletion.create(
                model = model,
                messages = [{"role": "user", "content": prompt}],
                temperature = 0,
                max_tokens = 1024
            ).choices[0].message.content
            break
        except Exception as e:
            print(e)
        time.sleep(5)
    
    return resp

if __name__ == "__main__":
    openai_setup()
    
    tgt_langs = [
        "Chinese",
        "Arabic",
        "French",
        "Japanese",
        "Russian",
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_lang", type=str, required=True)
    parser.add_argument("--out_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/rebuttal/loanwords.jsonl")
    parser.add_argument("--term_file_path", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/")
    
    args = parser.parse_args()
    assert args.tgt_lang in tgt_langs
    
    args.out_file = args.out_file.replace(".jsonl", args.tgt_lang + ".jsonl")
    args.out_file = open(args.out_file, 'a')
    
    args.term_file_path = args.term_file_path + args.tgt_lang + ".csv"
    
    terms_en = pd.read_csv(args.term_file_path)['English']
    
    for term in tqdm(terms_en):
        res = openai_prompt(term, args.tgt_lang, model='gpt-4o-mini')
        is_translated = True
        
        if "B" in res:
            is_translated = False
        
        
        info = {
            "English": term,
            "is_translated": is_translated
        }
        
        json.dump(info, args.out_file, ensure_ascii=False)
        args.out_file.write("\n")
        args.out_file.flush()