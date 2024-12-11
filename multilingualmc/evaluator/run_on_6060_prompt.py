import time
import re
import json
import pandas as pd
import openai
from tqdm import tqdm
import sys
import argparse
from multilingualmc.translator.get_terms import TermCollector
import os

# def openai_setup(key_path='/home/jiaruil5/openai_key_r3lit.txt'):
# 	with open(key_path) as f:
# 		key, org_id = f.read().strip().split("\n")

# 	print("Read key from", key_path)
# 	openai.api_key = key.strip()
# 	openai.organization = org_id.strip()

def openai_setup(key_path=''):
    if (key_path == ""):
        key = os.environ["openai_api_key"]
    else:
        with open(key_path) as f:
            key = f.read().strip().split("\n")[0]
    
    print("Read key from", key_path)
    openai.api_key = key.strip()

def openai_prompt(src_text, tgt_text, relevant_terms_dict, tgt_lang, model='gpt-4o'):
    src_lang = 'English'
    
    tmp = []
    for src_term, tgt_term in relevant_terms_dict.items():
        if tgt_lang in tgt_term:
            tmp.append(f"{src_lang}: {src_term}, {tgt_lang}: {tgt_term[tgt_lang]}")
    
    if len(tmp) == 0:
        return None
    terms = "- ".join(tmp)
    
    prompt = f"""For the following translation into {tgt_lang}, please use the specified {tgt_lang} terms for the corresponding {src_lang} terms, while keeping the other content unchanged.

Term dictionary:
{terms}

{src_lang} text:
{src_text}

{tgt_lang} translation:
{tgt_text}

If multiple terms are nested or overlap with the context in {src_lang}, select the longest span that matches the context. Additionally, if a term has multiple meanings, only replace the term if its original context is relevant to the AI field. Provided the updated translation only.
"""
    print(prompt)
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
    
    # extract valid_term, explanation
    return resp


if __name__ == "__main__":
    openai_setup()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="/home/iouzzani/research/multilingual_model_cards/multilingualmc/data_eval_6060/output/predictions_dev_seamless.jsonl")
    parser.add_argument("--out_file", type=str, default="/home/iouzzani/research/multilingual_model_cards/multilingualmc/dictionary_collection/mturk/analysis/Japanese_validated.csv")
    parser.add_argument("--term_file_path", type=str, default=None, help="used only when the method is constrained_beam_search.")
    parser.add_argument("--model", type=str, default='gpt-4o-mini')
    
    args = parser.parse_args()
    args.in_file = [json.loads(i) for i in open(args.in_file, 'r').readlines()]
    args.out_file = open(args.out_file, 'a')
    
    tgt_langs = [
        "Chinese",
        "Arabic",
        "French",
        "Japanese",
        "Russian",
    ]
    
    term_collector = TermCollector(args.term_file_path, tgt_langs)


    for line in args.in_file:
        info = {
            "text": line['text']
        }
        relevant_terms_dict = {}
        for key in term_collector.find_terminology(line['text']):
            relevant_terms_dict[key] = term_collector.terms_dict[key]
            
        for lang in tgt_langs:
            translation = openai_prompt(
                line['text'],
                line[f'text_{lang}'],
                relevant_terms_dict = relevant_terms_dict,
                tgt_lang = lang,
                model=args.model
            )
            if translation is None:
                info[f"text_{lang}"] = line[f'text_{lang}']
            else:
                info[f"text_{lang}"] = translation
        print(info)
        
        json.dump(info, args.out_file, ensure_ascii=False)
        args.out_file.write("\n")
        args.out_file.flush()