import json
import pandas as pd
from tqdm import tqdm
from typing import Any
import httpcore
setattr(httpcore, 'SyncHTTPTransport', Any)
class GoogleTranslate:
    def __init__(self):
        import sys
        sys.path.append("../")
        from translator.config import GoogleTranslatorConfig
        from translator.translator import GoogleTranslator
        args = GoogleTranslatorConfig
        self.model = GoogleTranslator(args)
        # text, src_lang='English, tgt_lang=tgt_lang

df = pd.read_csv("growing_dict/final_terms_with_context.csv")
data = df.to_dict(orient='records')
translator = GoogleTranslate()


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
    res['English'].append(term)
    for lang in res.keys():
        if lang == "English":
            continue
        term_trans = translator.model.translate(
            term,
            src_lang='English',
            tgt_lang=lang
        )
        res[lang].append(term_trans)
    
df = pd.DataFrame()
for lang in res.keys():
    df[lang] = res[lang]
df.to_csv("growing_dict/final_terms_google.csv")