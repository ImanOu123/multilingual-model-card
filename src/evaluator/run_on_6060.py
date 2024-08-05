from tqdm import tqdm
import json
import os
import argparse
from typing import Any
import sys
sys.path.append("../")

def choose_translator(args):
    if args.method == 'googletrans':
        from translator.config import GoogleTranslatorConfig
        from translator.translator import GoogleTranslator
        import httpcore
        setattr(httpcore, 'SyncHTTPTransport', Any)
        trans_args = GoogleTranslatorConfig
        translator = GoogleTranslator(trans_args)
        return translator
    elif args.method == 'seamless':
        from translator.config import M4TLargeTranslatorConfig
        from translator.translator import SeamlessTranslator
        trans_args = M4TLargeTranslatorConfig
        translator = SeamlessTranslator(trans_args)
        return translator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/2/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt")
    parser.add_argument("--out_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_googletrans.jsonl")
    parser.add_argument("--method", type=str, default='googletrans', choices=['googletrans', 'seamless'])
    args = parser.parse_args()
    
    tgt_langs = [
        "Chinese",
        "Arabic",
        "French",
        "Japanese",
        "Russian",
    ]
    
    gt_dict = {
        "English": [i for i in open(args.in_file, 'r').readlines()],
    }
    
    translator = choose_translator(args)
    
    out_f = open(args.out_file, 'a')
    for idx, item in tqdm(enumerate(gt_dict['English'])):
        info = {
            'text': item
        }
        for tgt_lang in tqdm(['Chinese', 'Arabic', 'French', 'Japanese', 'Russian']):
            answer = translator.translate(
                item,
                src_lang = 'English',
                tgt_lang = tgt_lang,
            )
            info[f'text_{tgt_lang}'] = answer
        
        json.dump(info, out_f, ensure_ascii=False)
        out_f.write("\n")
        out_f.flush()

    