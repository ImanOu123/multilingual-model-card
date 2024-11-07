import re
import json
import openai
from tqdm import tqdm
from openai import AzureOpenAI
import backoff
from database_utils import TinyDataBase

in_f = open("chinese.json", 'r')
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
    term = remove_parentheses_content(item['term_english'])
    
    
    if db.safe_contains(**{"language": "English", "term": term}):
        pass
    
    else:
        try:
            term_reformatted = call_azure_gpt(
                messages=[
                    {"role": "user", "content": prompt.format(term=term)}
                ],
                max_tokens=128,
                temperature=0.0,
                model="z-gpt-4o-2024-08-0"
            )
            
            info = {"language": "English", "term": term, "term_reformatted": term_reformatted}
            db.safe_insert_unique(info)
            
            info['term_Chinese'] = remove_parentheses_content(item['term_chinese'])
            print(info)
            res_data.append(info)
        except Exception as e:
            print(e)

out_f = open("/home/jiaruil5/multilingual/multilingual-model-card/src/data_term_gold/processed/chinese.json", 'w')
json.dump(res_data, out_f, ensure_ascii=False, indent=2)
out_f.flush()
        