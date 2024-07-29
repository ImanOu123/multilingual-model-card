import json
import pandas as pd
from tqdm import tqdm
import random

class GPTTermTranslator():
    def __init__(self, args):
        self.args = args
        self.prepare_model()
    
    def prepare_model(self):
        import sys

        sys.path.append("../")
        from models.config import get_model_config
        from models.llm import call
        self.config = get_model_config(self.args.model_name)
        def llm_config_func(llm):
            llm.temperature = 0.8
            llm.max_tokens = 128
            return llm
        self.llm_config_func = llm_config_func
        self.call = call
    
    
    def translate_term(self, prompt):
        try:
            res = self.call(
                [prompt],
                self.llm_config_func,
                has_system_prompt=False,
                model_version=self.args.model_name,
                verbose=True,
                **self.config
            )
        
            return res
        except Exception as e:
            print("Error in processing outputs:", e)
            return None

prompt = """Translate the following AI scientific term directly into {tgt_lang} based on its context. If the term is an abbreviation that could reduce confusion by keeping it as is, provide the original English term. Directly provide your translated term without any explanations.

- Term: {term}
- Context: "{context}"
"""


class Args:
    model_name = 'gpt-3.5-turbo'

df = pd.read_csv("growing_dict/final_terms_with_context.csv")
data = df.to_dict(orient='records')
translator = GPTTermTranslator(Args)

out_f = open("growing_dict/tmp.txt", 'a')

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
        term_trans = translator.translate_term(
            prompt.format(
                tgt_lang=lang,
                term = term,
                context = context
            )
        )
        res[lang].append(term_trans)
        out_f.write(str(term_trans) + "\n")
        out_f.flush()
    
df = pd.DataFrame()
for lang in res.keys():
    df[lang] = res[lang]
df.to_csv("growing_dict/final_terms_gpt35.csv")