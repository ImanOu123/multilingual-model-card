import time
import re
import json
import pandas as pd
import openai
from tqdm import tqdm
import sys
import argparse
import random

random.seed(42)

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

def openai_prompt(term, tgt_lang, context, model='gpt-4o-mini'):
    prompt = f"""Translate the following AI scientific term directly into {tgt_lang} based on its context. If the term is an abbreviation that could reduce confusion by keeping it as is, provide the original English term. Directly provide your translated term without any explanations.

- Term: {term}
- Context: "{context}"
"""
    print("Prompt:")
    print(prompt)

    while True:
        try:
            resp = openai.ChatCompletion.create(
                model = model,
                messages = [{"role": "user", "content": prompt}],
                temperature = 0.8,
                max_tokens = 128
            ).choices[0].message.content
            break
        except Exception as e:
            print(e)
        time.sleep(5)
    
    print("Response:")
    print(resp)
    return resp


if __name__ == "__main__":
    openai_setup()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--sample_n", type=int, default=0)
    parser.add_argument("--model", type=str, default='gpt-4o-mini')
    parser.add_argument("--tgt_lang", type=str)
    args = parser.parse_args()

    df = pd.read_csv("/home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/growing_dict/final_terms_with_context.csv")

    # get English terms to translate
    tgt_langs = [
        "Chinese",
        "Arabic",
        "French",
        "Japanese",
        "Russian",
    ]

    dfs_term = {}
    for tgt_lang in tgt_langs:
        tmp = pd.read_csv(f"intersected_term_to_translate_{tgt_lang}.csv")
        dfs_term[tgt_lang] = tmp['English'].values.tolist()
        dfs_term[tgt_lang + "_dict"] = dict(zip(tmp['English'], tmp['English_human']))
    
    data = list(set(dfs_term[tgt_langs[0]]) | set(dfs_term[tgt_langs[1]]) | set(dfs_term[tgt_langs[2]]) | set(dfs_term[tgt_langs[3]]) | set(dfs_term[tgt_langs[4]]))

    if args.sample:
        # random sample 500 words
        data = json.load(open(f"translations/sample_{args.sample_n}.json", 'r'))

    in_f = [json.loads(i)['English'] for i in open(f"translations/{args.model}_{args.tgt_lang}.jsonl", 'r').readlines()]
    
    out_f = open(f"translations/{args.model}_{args.tgt_lang}.jsonl", 'a')
    data = [item for item in data if item not in in_f]
    
    for item in tqdm(data):
        
        term = item
        context = df.loc[df['processed_term'] == term]['context'].tolist()
        if len(context) == 0 or term in ['morphology', 'decision stump']:
            continue
        
        try:
            context = eval(context[0])
        except:
            context = ""
        
        if len(context) > 3:
            context = random.sample(context, 3)
        
        tgt_lang = args.tgt_lang
        term_trans = openai_prompt(
            dfs_term[tgt_lang + "_dict"][term],
            tgt_lang,
            context,
            model = args.model
        )
        
        info = {
            "English": term,
            "tgt_lang": tgt_lang,
            "translation": term_trans
        }

        json.dump(info, out_f, ensure_ascii=False)
        out_f.write("\n")
        out_f.flush()

# python3 translate_by_gpt4o.py --tgt_lang Chinese --model gpt-4o --sample --sample_n 500
# python3 translate_by_gpt4o.py --tgt_lang Arabic --model gpt-4o --sample --sample_n 500
# python3 translate_by_gpt4o.py --tgt_lang French --model gpt-4o --sample --sample_n 500
# python3 translate_by_gpt4o.py --tgt_lang Japanese --model gpt-4o --sample --sample_n 500
# python3 translate_by_gpt4o.py --tgt_lang Russian --model gpt-4o --sample --sample_n 500