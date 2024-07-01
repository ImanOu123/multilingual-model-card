import os
import json
from tqdm import tqdm
from utils import split_paragraph
import sys
sys.path.append("../")

from translator.config import GoogleTranslatorConfig
from translator.translator import GoogleTranslator
args = GoogleTranslatorConfig
translator = GoogleTranslator(args)

in_dir = "../data_model_cards/original/claude3_json/"
out_dir = "../data_model_cards/translated/googletrans_claude3_json/"

tgt_langs = [
    "Chinese",
    "Arabic",
    "French",
    "Russian",
    "Japanese"
]

for file in tqdm(os.listdir(in_dir)):
    print(f"Start translating {file}..")
    filepath = in_dir + file
    json_list = json.load(open(filepath, 'r'))
    new_json_list = []
    for idx, item in tqdm(enumerate(json_list)):
        print(f"Reach item {idx}..")
        new_item = item.copy()
        for tgt_lang in tqdm(tgt_langs):
            chunks = split_paragraph(item['answer'])
            answer_chunks = []
            for chunk in chunks:
                answer_chunk = translator.translate(
                    chunk,
                    src_lang='English',
                    tgt_lang=tgt_lang
                )
                answer_chunks.append(answer_chunk)
        new_item[f"answer_{tgt_lang}"] = " ".join(answer_chunks)
        new_json_list.append(new_item)
    out_filepath = out_dir + file
    json.dump(
        new_json_list,
        open(out_filepath, 'w'),
        ensure_ascii=False,
        indent=2
    )