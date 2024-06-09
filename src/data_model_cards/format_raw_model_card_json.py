import json
import re
import os

# 1. use regex to remove (reference(s) xx)
# 2. only select generation dicts

def get_json_list(f):
    with open(f, 'r') as file:
        json_list = []
        for line in file.readlines():
            item = json.loads(line)
            if item['chain'] == 'generation':
                new_item = item.copy()
                del new_item['prompt']
                json_list.append(new_item)
    return json_list

# def sub_content(text):
#     pattern = r'\s*[\(\[]References?[\s\d,]*[\)\]]'
#     cleaned_text = re.sub(pattern, '', text)
#     return cleaned_text

# in_dir = 'claude3_raw_json/'
# out_dir = 'claude3_json/'

in_dir = 'gpt4_raw_json/'
out_dir = 'gpt4_json/'

for file in os.listdir(in_dir):
    filepath = in_dir + file
    json_list = get_json_list(filepath)
    new_json_list = []
    for item in json_list:
        new_item = item.copy()
        # new_item['answer'] = sub_content(item['answer'])
        new_json_list.append(new_item)
    with open(out_dir + file, 'w') as f:
        json.dump(new_json_list, f, indent=2)