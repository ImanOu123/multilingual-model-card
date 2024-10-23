import time
import re
import pandas as pd
import openai
from tqdm import tqdm
from googletrans import Translator
google_translator = Translator()

google_translator_lang_dict = {
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "English": "en",
    "French": "fr",
    "Japanese": "ja",
    "Russian": "ru",
}

def google_translate(text, src_lang, tgt_lang):
    result = google_translator.translate(
        text,
        dest=google_translator_lang_dict[tgt_lang],
        src=google_translator_lang_dict[src_lang],
    ).text
    return result

def openai_setup(key_path='/home/ubuntu/openai_key.txt'):
	with open(key_path) as f:
		key, org_id = f.read().strip().split("\n")

	print("Read key from", key_path)
	openai.api_key = key.strip()
	openai.organization = org_id.strip()
 
openai_setup()

context_df = pd.read_json("/home/ubuntu/multilingual-model-card/src/dictionary_collection/growing_dict/mturk.json")

def extract_response(response_text):
    # Regex pattern to capture the first part (ranked candidates) and the second part (explanation)
    pattern = r"1\. Candidate:\s*(.*)\n2\. Explanation\s*(.*)"
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        ranked_candidate = match.group(1).strip()
        return ranked_candidate, response_text
    else:
        return None, response_text

def openai_prompt(en_text, context, translations, back_translations, tgt_lang, model='gpt-4o'):
    tuple_str = "- ".join([f"""("{term}", "{back_term}")""" for term, back_term in zip(translations, back_translations)])
    prompt = f"""You are an expert in English and {tgt_lang} especially in the AI terminology translation. Select the best translation candidate based on the semantic accuracy and back translation accuracy for contextual fit. Explain why the candidate is the best fit considering the AI domain-specific usage. Here is the provided information about the terminology:
```
English term: {en_text}

Context:
{context}

{tgt_lang} (translation candidate, back translation) tuples:
- {tuple_str}
```

Output format:
```
1. Candidate: <The best translation candidate.>
2. Explanation: <Explanation of why the first translation is the best fit.>
```
"""
    while True:
        try:
            resp = openai.chat.completions.create(
                model = model,
                messages = [{"role": "user", "content": prompt}],
                temperature = 0,
                max_tokens = 1024
            ).choices[0].message.content
            break
        except Exception as e:
            print(e)
        time.sleep(5)
    
    # extract valid_term, explanation
    return extract_response(resp)

def get_validated_term(row, lang):
    # Use google translate for back translation of all the candidates to English
    ratio_dict = eval(row['prediction_ratio'])
    back_candidates = []
    for ratio in ratio_dict:
        if ratio['word'] is not None:
            try:
                back_candidate = google_translate(
                    ratio['word'],
                    lang,
                    'English'
                )
            except:
                back_translate = ratio['word']
        else:
            back_translate = ""
        back_candidates.append(back_candidate)
    
    # Proposed method (old)
    ## Combined with the context, let two LLMs decide which translation is the best (gpt-4o, claude3.5-sonnet, gemini-v1.5-pro)
    ## Return valid_term if it's the same across different llms. Otherwise expert in the loop
    # Proposed method (new)
    ## use gpt-4o to decide, return (valid_term, explanation)
    print(row['word'])
    context = context_df.loc[context_df['English'] == row['word']]['context'].item()
    
    valid_term, explanation = openai_prompt(
        row['word'],
        context,
        [ratio['word'] for ratio in ratio_dict],
        back_candidates,
        tgt_lang = lang,
        model = 'gpt-4o'
    )
    return valid_term, explanation, back_candidates
    

def get_valid_list(df, lang, threshold):
    valid_lst = []
    reason_lst = []
    back_candidates = []
    out_f = open("tmp.txt", 'a')
    
    for idx, row in tqdm(df.iterrows()):
        ratio_dict = eval(row['prediction_ratio'])
        if ratio_dict[0]['ratio'] >= threshold:
            valid_lst.append(ratio_dict[0]['word'])
            reason_lst.append({})
            back_candidates.append([])
        else:
            res = get_validated_term(row, lang)
            valid_lst.append(res[0])
            reason_lst.append(res[1])
            back_candidates.append(res[2])
            print(res)
            out_f.write(row['word'] + " " + lang + "\n")
            out_f.write(str(res)+"\n")
            out_f.flush()
    
    return valid_lst, reason_lst, back_candidates


def validate_translation(lang, threshold):
    df = pd.read_csv(f"{lang}.csv")
    res = get_valid_list(df, lang, threshold)
    df['validated_translation'] = res[0]
    df['reason'] = res[1]
    df['back_translations'] = res[2]
    df.to_csv(f"{lang}_validated.csv")
    
if __name__ == "__main__":
    # langs = ['Chinese', 'Arabic', 'French', 'Japanese', 'Russian']
    langs = ['Chinese']
    threshold = 0.5

    for lang in tqdm(langs):
        validate_translation(lang, threshold)