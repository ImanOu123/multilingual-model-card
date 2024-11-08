from tqdm import tqdm
import json
import os
from typing import Any
from multilingualmc.translator.config import GoogleTranslatorConfig
from multilingualmc.translator.translator import GoogleTranslator

import httpcore
setattr(httpcore, 'SyncHTTPTransport', Any)

if __name__ == "__main__":
    tgt_langs = [
        "Chinese",
        "Arabic",
        "French",
        "Russian",
        "Japanese"
    ]
    
    out_f = open("eval_data_google.jsonl", 'a')
    
    json_info = json.load(open("eval_data.json", 'r'))
    
    trans_args = GoogleTranslatorConfig
    translator = GoogleTranslator(trans_args)
    for chunk in tqdm(json_info):
        
        for tgt_lang in tqdm(tgt_langs):
            answer_chunk = translator.translate(
                chunk['text'],
                src_lang='English',
                tgt_lang=tgt_lang
            )
            chunk[f'text_{tgt_lang}'] = answer_chunk
        json.dump(chunk, out_f, ensure_ascii=False)
        out_f.write("\n")
        out_f.flush()