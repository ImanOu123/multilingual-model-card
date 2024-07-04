"""

Go to the installed grobid-0.6.2/ folder, and run `./gradlew run` to start the grobid server. The server will be on port 8070 by default.


"""

import glob
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import json
import subprocess
import torch
import logging
import scipdf
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pdf2md")
log.addHandler(logging.FileHandler(str(Path(os.getcwd()) / "logs")))

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir_path", type=str) # source pdf directory
    parser.add_argument("--target_dir_path", type=str) # target markdown directory
    parser.add_argument("--error_file", type=str) # error logging
    args = parser.parse_args() # exclusive end index
    return args

# python3 pdf2json.py --source_dir_path pdf/ --target_dir_path text/ --error_file error_scipdf.jsonl

if __name__ == "__main__":
    args = arg_parse()
    
    
    f = open(args.error_file, "a")
    source_file_list = glob.glob(str(Path(args.source_dir_path) / "*.pdf"))
    for file in tqdm(source_file_list):
        filename = os.path.splitext(os.path.basename(file))[0]
        text_path = Path(args.target_dir_path) / f"{filename}.json"
        if not os.path.isfile(text_path):
            try:
                print(f"Processing {filename}:")
                article_dict = scipdf.parse_pdf_to_dict(file, grobid_url="http://localhost:8075")
                with open(text_path, 'w', encoding='utf-8') as out_file:
                    json.dump(article_dict, out_file)
                print(f"Finished..")
            except Exception as e:
                error_info = {
                    "pdf_path": file,
                    "error": str(e),
                }
                json.dump(error_info, f, ensure_ascii=False)
                f.write("\n")