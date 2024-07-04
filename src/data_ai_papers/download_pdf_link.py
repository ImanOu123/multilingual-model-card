import random
import argparse
import sys
import json
import os
import re
import uuid
import time
from tqdm import tqdm
from spider import ArxivSpider, PdfSpider, GoogleSpider

arxiv_id_regex = r"(?:[0-9]{4}\.[0-9]{4,5})|(?:[a-z\-]+\/[0-9]{7})"

def hash_url(url: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))

# python3 download_pdf_link.py

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--error_file", type=str, default="error.jsonl")
    parser.add_argument("--in_dir", type=str, default="json/")
    parser.add_argument("--out_dir", type=str, default="new_json/")
    parser.add_argument("--pdf_dir", type=str, default="pdf/")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = arg_parse()
    
    arxiv_spider = ArxivSpider(error_file = args.error_file)
    pdf_spider = PdfSpider(error_file=args.error_file)
    google_spider = GoogleSpider(error_file=args.error_file)
    
    # for file in tqdm(os.listdir(args.in_dir)):
    for file in ['iclr_best_papers.json', 'www_best_papers.json', 'sigir_best_papers.json', 'neurips_best_papers.json']:
        in_file = os.path.join(args.in_dir, file)
        out_file = os.path.join(args.out_dir, file + "l")
        out_f = open(out_file, 'a')
        print("Current file:", in_file)
        
        json_list = json.load(open(in_file, 'r'))
        for paper in tqdm(json_list):
            print(paper['title'])
            time.sleep(random.randint(1, 5))
            
            try:
                paper['urls'] = None
                if paper['link'] is not None:
                    url = paper['link']
                    file_hash = hash_url(url)
                    pdf_spider.download_pdf(
                        url = url,
                        dir_path = args.pdf_dir,
                        file_hash = file_hash,
                    )
                    paper['file_hash'] = file_hash
                else:
                    urls = google_spider.get_pdf_link(paper['title'])
                    paper['urls'] = urls
                    url = google_spider.process_results(urls)
                    if url is not None:
                        if 'arxiv.org' in url:
                            arxiv_id = re.findall(arxiv_id_regex, url)[0]
                            arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}"
                            file_hash = hash_url(arxiv_url)
                            arxiv_spider.download_pdf(
                                arxiv_id = arxiv_id,
                                dir_path = args.pdf_dir,
                                file_hash = file_hash,
                            )
                            paper['link'] = arxiv_url
                        else:
                            file_hash = hash_url(url)
                            pdf_spider.download_pdf(
                                url = url,
                                dir_path = args.pdf_dir,
                                file_hash = file_hash,
                            )
                            paper['link'] = url
                            
                        paper['file_hash'] = file_hash
                    else:
                        paper['link'] = None
                        paper['file_hash'] = None
                print(json.dumps(paper))
                json.dump(paper, out_f)
                out_f.write("\n")
                out_f.flush()
            except Exception as e:
                print("Encountered error...")
                print(e)
                continue