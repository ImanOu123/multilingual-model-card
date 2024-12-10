import re
import json
import openai
from tqdm import tqdm
from openai import AzureOpenAI
import backoff
from database_utils import TinyDataBase

in_f = open("arabic.json", 'r')
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


prompt_arabic = """Given an Arabic noun or noun phrase, convert it to its base form. This means:
1. For noun phrases (الإضافة), remove any definite article (ال) from all nouns.
2. For singularizing, convert plural or dual nouns to their singular form.
Retain only the first noun in iḍāfa constructions, converting the second noun if necessary to singular.
3. For any adjectives following a noun, apply the same rules: remove any definite articles, singularize if plural, and ensure the adjective follows the base form of the noun in gender and number.

Example:
Term: الكتاب الكبير
Base Form: كتاب كبير

Example:
Term: كتب المعلمين
Base Form: كتاب معلم

Task:
Term: {term}
Base Form: """


res_data = []
for item in tqdm(data):
    term = remove_parentheses_content(item['term_english'])
    term_arabic = remove_parentheses_content(item['term_arabic'])
    
    info = {}
    if not db.safe_contains(**{"language": "English", "term": term}):
        try:
            term_reformatted = call_azure_gpt(
                messages=[
                    {"role": "user", "content": prompt.format(term=term)}
                ],
                max_tokens=128,
                temperature=0.0,
                model="z-gpt-4o-2024-08-0"
            )
            
            info_english = {"language": "English", "term": term, "term_reformatted": term_reformatted}
            db.safe_insert_unique(info_english)
            info['term'] = info_english['term']
            info['term_reformatted'] = info_english['term_reformatted']
            
        except Exception as e:
            print(e)
            continue
    else:
        tmp = db.safe_search(**{"language": "English", "term": term})[0]
        info['term'] = tmp["term"]
        info['term_reformatted'] = tmp['term_reformatted']

    if not db.safe_contains(**{"language": "Arabic", "term": term_arabic}):
        try:
            term_reformatted = call_azure_gpt(
                messages=[
                    {"role": "user", "content": prompt_arabic.format(term=term_arabic)}
                ],
                max_tokens=128,
                temperature=0.0,
                model="z-gpt-4o-2024-08-0"
            )
            
            info_arabic = {"language": "Arabic", "term": term_arabic, "term_reformatted": term_reformatted}
            db.safe_insert_unique(info_arabic)
            
            
            info['term_arabic'] = info_arabic['term']
            info['term_arabic_reformatted'] = info_arabic['term_reformatted']
            
            print(info)
            res_data.append(info)
        except Exception as e:
            print(e)
            continue
    else:
        tmp = db.safe_search(**{"language": "Arabic", "term": term_arabic})[0]
        info['term_arabic'] = tmp['term']
        info['term_arabic_reformatted'] = tmp['term_reformatted']
        print(info)
        res_data.append(info)
        

out_f = open("/home/jiaruil5/multilingual/multilingual-model-card/src/data_term_gold/processed/arabic.json", 'w')
json.dump(res_data, out_f, ensure_ascii=False, indent=2)
out_f.flush()
        