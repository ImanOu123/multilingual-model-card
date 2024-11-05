from tqdm import tqdm
import json
import os
import argparse
from utils import DocProcessor, split_paragraph
from typing import Any

import sys
sys.path.append("../")
from translator.config import M4TLargeTranslatorConfig
from translator.translator import SeamlessTranslator
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_file", type=str, default="../dataset/info.json")
    parser.add_argument("--in_dir", type=str, default="../data_ai_papers/text/")
    parser.add_argument("--out_file", type=str, default="../data_ai_papers/translated/seamless_paragraph.jsonl")
    args = parser.parse_args()
    
    tgt_langs = [
        "Chinese",
        "Arabic",
        "French",
        "Russian",
        "Japanese"
    ]
    
    out_f = open(args.out_file, 'a')
    
    json_info = json.load(open(args.info_file, 'r'))
    paper_paths = [args.in_dir + paper for paper in json_info['test']['paper']]
    
    trans_args = M4TLargeTranslatorConfig
    translator = SeamlessTranslator(trans_args)
    ## tmp fix
    found_paper = False
    found_heading = False
    ##
    for paper_path in tqdm(paper_paths):
        ## tmp fix
        if paper_path.split("/")[-1] == "6379bd6e-b2f3-53b5-816f-31529cb70e91.json":
            found_paper = True
        if not found_paper:
            continue
        ##
        print("Start translating paper ", paper_path)
        doc_store = DocProcessor(paper_path)
        doc = doc_store.doc
        for item in tqdm(doc):
            if item['heading'] not in ['title', 'authors']:
                ## tmp fix
                if paper_path.split("/")[-1] == "6379bd6e-b2f3-53b5-816f-31529cb70e91.json" and item['heading'] == "Dialogue Behavior":
                    found_heading = True
                if not found_heading:
                    continue
                ##
                content_chunks = split_paragraph(item['content'])
                for chunk in tqdm(content_chunks):
                    
                    info = {
                        "file": paper_path.split("/")[-1],
                        "heading": item['heading'],
                        "text": chunk,
                    }
                    
                    for tgt_lang in tqdm(tgt_langs):
                        answer_chunk = translator.translate(
                            chunk,
                            src_lang='English',
                            tgt_lang=tgt_lang
                        )
                        info[f'text_{tgt_lang}'] = answer_chunk
                    json.dump(info, out_f, ensure_ascii=False)
                    out_f.write("\n")
                    out_f.flush()