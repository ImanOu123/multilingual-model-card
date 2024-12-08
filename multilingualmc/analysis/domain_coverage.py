import pandas as pd
import openai
import time
from tqdm import tqdm
import json

def openai_setup(key_path='/home/jiaruil5/openai_key_ralf_misleading.txt'):
	with open(key_path) as f:
		key = f.read().strip().split("\n")[0]

	print("Read key from", key_path)
	openai.api_key = key.strip()

def openai_prompt(term, model='gpt-4o-mini'):
    prompt = """
Categorize the following AI terminology into one of the predefined subfields:
Natural language processing, Human computer interaction, Biology, Medical, Data science, Knowledge graph, Information retrieval, Psychology, Network science, Internet of things, Computer vision, Math, Neuroscience, Signal processing, Computer science, Statistics and probability, Physics, Robotics, Other.

Term: {term}

Output only the category name without any explanation:
""".format(term=term)
    
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

    lang = 'Chinese'
    data_6060 = pd.read_csv(f"/home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_results_6060/{lang}_validated.csv").rename(columns={"validated_translation": "validated_translation_old", "gold": "validated_translation"}, inplace=False).drop_duplicates(subset='word', inplace=False)
    data_mturk = pd.read_csv(f"/home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_results_crawled/{lang}_validated.csv").drop_duplicates(subset='word', inplace=False)
    terms = pd.concat([data_6060[['word', 'validated_translation']], data_mturk[['word', 'validated_translation']]], axis=0, ignore_index=True)['word'].tolist()
    
    out_f = open("domain_res.jsonl", 'a')
    
    for term in tqdm(terms):
        result = openai_prompt(
            term,
            model = 'gpt-4o-mini'
        )
        info = {"term": term, "domain": result.strip()}
        json.dump(info, out_f)
        out_f.write("\n")
        out_f.flush()