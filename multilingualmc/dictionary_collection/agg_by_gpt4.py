import pandas as pd
import json
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


data = pd.read_csv("growing_dict/final_terms_agg.csv").to_dict(orient='records')
data_context = pd.read_csv("growing_dict/final_terms_with_context.csv").to_dict(orient='records')
class Args:
    model_name = 'gpt-4o'

translator = GPTTermTranslator(Args)

prompt = """Translate the following AI scientific term directly into {tgt_lang} based on its context and three candidate translations. Return the original term only if the term is an abbreviation that would be obviously clearer in English. Directly provide your translated term without any explanations.

- Term: "{term}"
- Context: {context}
- Candidate term translation 1: "{candidate1}"
- Candidate term translation 2: "{candidate2}"
- Candidate term translation 3: "{candidate3}"
"""

def get_data(data, data_context):
    new_data = []
    assert len(data) == len(data_context)
    for i, j in zip(data, data_context):
        i['context'] = eval(j['context'])
        new_data.append(i)
    return new_data

def find_repeated_element(lst):
    count_dict = {}
    for item in lst:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1
    
    for key, val in count_dict.items():
        if val >= 2:
            return key
    
    return None
    

data = get_data(data.copy(), data_context)
out_f = open("growing_dict/final_terms_agg.jsonl", 'a')

for item in tqdm(data[345:]):
    info = {
        
    }
    for lang in tqdm(['Arabic', 'Chinese', 'French', 'Japanese', 'Russian']):
        res = find_repeated_element([item[f'google_{lang}'], item[f'gpt35_{lang}'], item[f'claude3_{lang}']])
        info[lang] = translator.translate_term(prompt.format(
            tgt_lang=lang,
            term=item[f'google_{lang}'],
            context=random.choice(item['context']) if len(item['context']) > 0 else "[]",
            candidate1=item[f'google_{lang}'],
            candidate2=item[f'gpt35_{lang}'],
            candidate3=item[f'claude3_{lang}'],
        ))
    
    json.dump(info, out_f, ensure_ascii=False)
    out_f.write("\n")
    out_f.flush()