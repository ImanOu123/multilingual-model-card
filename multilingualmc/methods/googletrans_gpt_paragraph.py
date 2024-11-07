from tqdm import tqdm
import json
import os
import argparse
from utils import DocProcessor, split_paragraph, extract_terms, translate_terms
from typing import Any

import httpcore
setattr(httpcore, 'SyncHTTPTransport', Any)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_file", type=str, default="../dataset/info.json")
    parser.add_argument("--in_file", type=str, default="../data_ai_papers/translated/googletrans_paragraph.jsonl")
    parser.add_argument("--out_file", type=str, default="../data_ai_papers/translated/googletrans_gpt_paragraph.jsonl")
    args = parser.parse_args()
    
    tgt_langs = [
        "Chinese",
        "Arabic",
        "French",
        "Russian",
        "Japanese"
    ]
    
    out_f = open(args.out_file, 'a')
    
    json_info = json.load(open(args.info_file, 'r'))
    chunk_lst = [json.loads(i) for i in open(args.in_file, 'r').readlines()]
    
    for chunk in tqdm(chunk_lst):
        info = chunk.copy()
        english_terms = extract_terms_llm(
            args,
            item['text']
        )
        info['terms'] = english_terms
        
        for tgt_lang in tqdm(tgt_langs):
            terms = translate_terms(
                args,
                item['text'], 
                item[f'text_{tgt_lang}'], 
                tgt_lang, 
                english_terms
            )
            info[f'terms_{tgt_lang}'] = terms
            
            answer_chunk = refine_chunk(
                args,
                item['text'], 
                item[f'text_{tgt_lang}'], 
                tgt_lang, 
                english_terms, 
                terms
            )
            info[f'text_{tgt_lang}'] = answer_chunk
                
            json.dump(info, out_f, ensure_ascii=False)
            out_f.write("\n")
            out_f.flush()