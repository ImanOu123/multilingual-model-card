from tqdm import tqdm
import json
import os
import argparse
from utils import DocProcessor, split_paragraph
from typing import Any

import sys
sys.path.append("../")
from translator.config import GoogleTranslatorConfig
from translator.translator import GoogleTranslator

import httpcore
setattr(httpcore, 'SyncHTTPTransport', Any)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_file", type=str, default="../dataset/info.json")
    parser.add_argument("--in_dir", type=str, default="../data_model_cards/original/claude3_json/")
    parser.add_argument("--out_file", type=str, default="../data_model_cards/translated/googletrans_paragraph.jsonl")
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
    
    trans_args = GoogleTranslatorConfig
    translator = GoogleTranslator(trans_args)
    for file in tqdm(os.listdir(args.in_dir)):
        print("Start translating ", file)
        filepath = args.in_dir + file
        json_list = json.load(open(filepath, 'r'))
        new_json_list = []
        for idx, item in tqdm(enumerate(json_list)):
            content_chunks = split_paragraph(item['answer'])
            for chunk in tqdm(content_chunks):
                info = {
                    "file": file,
                    "heading": item['question'],
                    "text": chunk,
                }
                
                for tgt_lang in tqdm(tgt_langs):
                    answer_chunk = translator.translate(
                        chunk,
                        src_lang='English',
                        tgt_lang=tgt_lang
                    )
                    info[f'text_{tgt_lang}'] = answer_chunk
                json.dump(info, out_f, ensure_ascii=False)
                out_f.write("\n")
                out_f.flush()