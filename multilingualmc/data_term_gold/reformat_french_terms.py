import re
import json
import openai
from tqdm import tqdm
from openai import AzureOpenAI
import backoff
from database_utils import TinyDataBase

in_f = open("french.json", 'r')
data = json.load(in_f)

client = AzureOpenAI(api_key="f8e0702ca34d420b8b52258a81a28660",
                    api_version="2024-02-01",
                    azure_endpoint="https://openai-zjin-5.openai.azure.com/")


# Function to remove parentheses and their contents from a given text
def remove_parentheses_content(text):
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'（.*?）', '', text)
    text = re.sub(r'/.*', '', text)
    return text


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def call_azure_gpt(**kwargs):
   resp = client.chat.completions.create(**kwargs)
   
   return resp.choices[0].message.content.strip()


db = TinyDataBase("term_conversion.json")

prompt_selection = """You are provided one French AI terminology and its coresponding English translations. Please output the best English translation following these instructions:

1. Domain relevance check: If the French term is not relevant to the AI field at all, just output "None".
2. Select the best translation: If the French term has multiple English translations under "terms_english", select only the one you think is most accurate. If the list is empty or none of the provided translations seems appropriate enough, just output "None".
3. Output format: Directly return the output either as "None" or as a string of the best English translation.

Example input 1:
{
    "term_french": "Tolérance",
    "terms_english": [
        "tolerance",
        "error margin",
        "margin of error"
    ]
}

Example output 1:
tolerance

Example input 2:
{
    "term_french": "Filtrage des courriels",
    "terms_english": [
        "Email filtering"
    ]
}

Example output 2:
None

Task input:
"""

prompt_selection_suffix = """

Task output:
"""



prompt = """Convert the following term to its normal form, appropriate for use in the middle of a sentence. This means:

Capitalizing only proper nouns and the first letter of any hyphenated proper names (e.g., "Vapnik-Chervonenkis").
Lowercasing all other words that are not proper nouns or parts of proper names, including the first letter.

Example:
Term: Accumulated Error Backpropagation
Normal Form: accumulated error backpropagation

Example:
Term: Vapnik-Chervonenkis Dimension
Normal Form: Vapnik-Chervonenkis dimension

Example:
Term: Von Neumann Architecture
Normal Form: Von Neumann architecture

Example:
Term: GAN
Normal Form: GAN

Example:
Term: Word Sense Disambiguation
Normal Form: word sense disambiguation

Example:
Term: Markov
Normal Form: Markov

Task:
Term: {term}
Normal Form: """

res_data = []
for item in tqdm(data):
    terms_english = [remove_parentheses_content(i) for i in item['terms_english']]
    term_french = remove_parentheses_content(item['term_french'])
    term_english = None
    
    if not db.safe_contains(**{"type": "selection", "term_french": term_french, "term_english": term_english}):
        try:
            
            input_dict_str = json.dumps({"term_french": term_french, "terms_english": terms_english}, ensure_ascii=False, indent=4)
            output_str = call_azure_gpt(
                messages=[
                    {"role": "user", "content": prompt_selection + input_dict_str + prompt_selection_suffix}
                ],
                max_tokens=256,
                temperature=0.0,
                model="z-gpt-4o-2024-08-0"
            ).strip()
            
            if "none" in output_str.lower():
                print("none", output_str)
                db.safe_insert_unique({"type": "selection", "term_french": term_french, "term_english": "None"})
                continue
            else:
                term_english = output_str
                db.safe_insert_unique({"type": "selection", "term_french": term_french, "term_english": term_english})
            
        except Exception as e:
            print(e)
            continue
    else:
        tmp = db.safe_search(**{"type": "selection", "term_french": term_french})[0]
        term_english = tmp['term_english']
    
    info = {}
    if not db.safe_contains(**{"language": "English", "term": term_english}):
        try:
            term_reformatted = call_azure_gpt(
                messages=[
                    {"role": "user", "content": prompt.format(term=term_english)}
                ],
                max_tokens=128,
                temperature=0.0,
                model="z-gpt-4o-2024-08-0"
            )
            
            
            info_english = {"language": "English", "term": term_english, "term_reformatted": term_reformatted}
            db.safe_insert_unique(info_english)
            
            info['term'] = info_english['term']
            info['term_reformatted'] = info_english['term_reformatted']
            
        except Exception as e:
            print(e)
            continue
    else:
        tmp = db.safe_search(**{"language": "English", "term": term_english})[0]
        info['term'] = tmp["term"]
        info['term_reformatted'] = tmp['term_reformatted']
    
    
    if not db.safe_contains(**{"language": "French", "term": term_french}):
        try:
            term_reformatted = call_azure_gpt(
                messages=[
                    {"role": "user", "content": prompt.format(term=term_french)}
                ],
                max_tokens=128,
                temperature=0.0,
                model="z-gpt-4o-2024-08-0"
            )
            
            info_french = {"language": "French", "term": term_french, "term_reformatted": term_reformatted}
            db.safe_insert_unique(info_french)
            
            
            info['term_french'] = info_french['term']
            info['term_french_reformatted'] = info_french['term_reformatted']
            
            print(info)
            res_data.append(info)
        except Exception as e:
            print(e)
            continue
    else:
        tmp = db.safe_search(**{"language": "French", "term": term_french})[0]
        info['term_french'] = tmp['term']
        info['term_french_reformatted'] = tmp['term_reformatted']
        print(info)
        res_data.append(info)

out_f = open("/home/jiaruil5/multilingual/multilingual-model-card/src/data_term_gold/processed/french.json", 'w')
json.dump(res_data, out_f, ensure_ascii=False, indent=2)
out_f.flush()
        