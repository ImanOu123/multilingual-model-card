import json
import csv
import pandas as pd

df_terms = pd.read_csv("growing_dict/final_terms_with_original.csv")
data_terms = df_terms.to_dict(orient='records')

data_context = dict()
for i in open("growing_dict/terms.jsonl", 'r').readlines():
    data = json.loads(i)
    for term in data['processed_result']:
        if term in data_context:
            data_context[term].append(data['context'])
        else:
            data_context[term] = [data['context']]

contexts = []
for term in data_terms:
    term = term['original_term']
    if term in data_context:
        contexts.append(data_context[term])
    else:
        contexts.append([])

df_terms['context'] = contexts
df_terms.to_csv("growing_dict/final_terms_with_context.csv")