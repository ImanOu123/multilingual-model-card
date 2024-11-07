from tqdm import tqdm
import json
import os
import argparse
from utils import DocProcessor, split_paragraph

import sys
sys.path.append("../translator/")
from term_extractor import LLAMATermExtractor

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_file", type=str, default="../dataset/info.json")
    parser.add_argument("--in_dir", type=str, default="../data_ai_papers/text/")
    parser.add_argument("--out_file", type=str, default="../dictionary_collection/growing_dict/terms.jsonl")
    args = parser.parse_args()
    
    class ExtractorArgs:
        model_name = "llama3_70b"
        
    extractor = LLAMATermExtractor(ExtractorArgs)

    out_f = open(args.out_file, 'a')

    json_info = json.load(open(args.info_file, 'r'))
    paper_paths = [args.in_dir + paper for paper in json_info['dev']['paper']]

    for paper_path in tqdm(paper_paths):
        print("Start extracting terms from ", paper_path)
        doc_store = DocProcessor(paper_path)
        doc = doc_store.doc
        for item in tqdm(doc):
            if item['heading'] not in ["title", "authors"]:
                content_chunks = split_paragraph(item['content'])
                for chunk in content_chunks:            
                    res, processed_res = extractor.extract_terms(chunk, prompt_version="term_extraction_llama3")
                    info = {
                        "paper_path": paper_path.split("/")[-1],
                        "heading": item['heading'],
                        "prompt_version": "term_extraction_llama3",
                        "context": chunk,
                        "result": res,
                        "processed_result": processed_res,
                    }
                    json.dump(info, out_f)
                    out_f.write("\n")
                    out_f.flush()