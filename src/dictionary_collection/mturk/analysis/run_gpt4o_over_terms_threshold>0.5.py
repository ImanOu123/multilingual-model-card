import time
import re
import pandas as pd
import openai
from tqdm import tqdm
import sys
import httpcore
from typing import Any
setattr(httpcore, 'SyncHTTPTransport', Any)
from googletrans import Translator
google_translator = Translator()
import random
seed = 42
random.seed(seed)  # Set the seed

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

def openai_setup(key_path='/home/jiaruil5/openai_key_r3lit.txt'):
	with open(key_path) as f:
		key, org_id = f.read().strip().split("\n")

	print("Read key from", key_path)
	openai.api_key = key.strip()
	openai.organization = org_id.strip()
 
openai_setup()

context_df = pd.read_csv("/home/jiaruil5/multilingual/multilingual-model-card/src/dictionary_collection/mturk/mturk_with_6060.csv")

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
2. Explanation: <Short explanation of why the first translation is the best fit.>
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

def get_validated_term(row, lang, split):
    # Use google translate for back translation of all the candidates to English
    ratio_dict = eval(row['prediction_ratio'])
    candidates = list(set([ratio['word'] for ratio in ratio_dict] + [google_translate(row['word'], 'English', lang)]))
    print(candidates)
    back_candidates = []
    for candidate in candidates:
        if candidate is not None:
            try:
                back_candidate = google_translate(
                    candidate,
                    lang,
                    'English'
                )
            except:
                back_candidate = candidate
        else:
            back_candidate = ""
        back_candidates.append(back_candidate)
    
    # Proposed method (old)
    ## Combined with the context, let two LLMs decide which translation is the best (gpt-4o, claude3.5-sonnet, gemini-v1.5-pro)
    ## Return valid_term if it's the same across different llms. Otherwise expert in the loop
    # Proposed method (new)
    ## use gpt-4o to decide, return (valid_term, explanation)
    print(row['word'])
    context = context_df.loc[(context_df['English'] == row['word']) & (context_df['source'] == split)]['context'].item()
    
    
    valid_term, explanation = openai_prompt(
        row['word'],
        context,
        candidates,
        back_candidates,
        tgt_lang = lang,
        model = 'gpt-4o'
    )
    return valid_term, explanation, back_candidates
    
def remove_surrounding_quotes(s):
    # Check if the string starts and ends with quotes
    if s is None:
        return ""
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]  # Remove the surrounding quotes
    return s  # Return the original string if no surrounding quotes

def sample_indices(a, sample_size=200):
    if a < sample_size:
        return list(range(a))  # Return all indices if 'a' is less than sample_size
    else:
        return random.sample(range(a), sample_size)  # Randomly sample 'sample_size' indices from range(a)

def get_valid_list(df, lang, threshold, log_file_path, split, sample_size):
    valid_lst = []
    reason_lst = []
    back_candidates = []
    # read existing content
    out_f_content = open(log_file_path, 'r').readlines()
    existing_dict = {}
    for line1, line2 in zip(out_f_content[::2], out_f_content[1::2]):
        line1, line2 = line1.strip(), line2.strip()
        assert line1.endswith(lang)
        line1 = line1[:-len(lang)].strip()
        line2 = eval(line2)
        existing_dict[line1] = line2
        
    # check if one exists or is now. Append if it is new
    out_f = open(log_file_path, 'a')
    
    cnt = 0
    for idx, row in tqdm(df.iterrows()):
        ratio_dict = eval(row['prediction_ratio'])
        if ratio_dict[0]['ratio'] >= threshold:
            cnt += 1
    
    indices = sample_indices(cnt, sample_size)
    
    cnt = 0
    for idx, row in tqdm(df.iterrows()):
        ratio_dict = eval(row['prediction_ratio'])
        if ratio_dict[0]['ratio'] >= threshold:
            if cnt in indices:
                if row['word'] in existing_dict:
                    print("Find in existing dict...")
                    print(existing_dict[row['word']])
                    res = existing_dict[row['word']]
                    valid_lst.append(remove_surrounding_quotes(res[0]))
                    reason_lst.append(res[1])
                    back_candidates.append(res[2])
                else:
                    res = get_validated_term(row, lang, split)
                    res0 = remove_surrounding_quotes(res[0])
                    valid_lst.append(res0)
                    reason_lst.append(res[1])
                    back_candidates.append(res[2])
                    print(res)
                    out_f.write(row['word'] + " " + lang + "\n")
                    out_f.write(str((res0, res[1], res[2]))+"\n")
                    out_f.flush()
            else:
                valid_lst.append("")
                reason_lst.append("")
                back_candidates.append([])
            cnt += 1
        else:
            valid_lst.append("")
            reason_lst.append("")
            back_candidates.append([])
    return valid_lst, reason_lst, back_candidates


def validate_translation(lang, threshold, in_csv_path, log_file_path, out_csv_path, split, sample_size):
    df = pd.read_csv(in_csv_path)
    res = get_valid_list(df, lang, threshold, log_file_path, split, sample_size)
    df['validated_translation'] = res[0]
    df['reason'] = res[1]
    df['back_translations'] = res[2]
    df.to_csv(out_csv_path)
    
if __name__ == "__main__":
    langs = ['Chinese', 'Arabic', 'French', 'Japanese', 'Russian']
    threshold = 0.5
    sample_size = 200
    
    for lang in langs:
        in_csv_path = f"annotation_results_crawled/{lang}.csv"
        log_file_path = f"annotation_results_crawled/tmp_{lang}_gpt4o_>0.5_sample200.txt"
        out_csv_path = in_csv_path.replace(".csv", "_validated_>0.5_sample200.csv")
        split = "whole"

        validate_translation(lang, threshold, in_csv_path, log_file_path, out_csv_path, split, sample_size)