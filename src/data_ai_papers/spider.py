import arxiv
# logging.basicConfig(level=logging.INFO)
import requests
import json
import time
from pathlib import Path
import os

class Spider():
    def __init__(self, error_file):
        self.error_file = error_file
        self.f = open(self.error_file, 'a')
    
    def log_error(self, error_info):
        json.dump(error_info, self.f, ensure_ascii=False)
        self.f.write("\n")
        

class ArxivSpider(Spider):
    def __init__(self, error_file, max_tries=2):
        super().__init__(error_file)
        self.max_tries = max_tries
    
    def get_arxiv_id(self, paper_title):
        curr_tries = 0
        while curr_tries < self.max_tries:
            try:
                search = arxiv.Search(
                    query=paper_title,
                    max_results=1,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                for result in search.results():
                    if result.title.lower() == paper_title.lower():
                        return result.entry_id.split('/')[-1]
                else:
                    return None
            except Exception as e:
                print(e)
            curr_tries += 1
            time.sleep(5)
        if curr_tries == self.max_tries:
            self.log_error({
                "type": "arxiv_spider_id",
                "arxiv_id": paper_title,
            })
    
    def download_pdf(self, arxiv_id: str, dir_path, file_hash):
        curr_tries = 0
        while curr_tries < self.max_tries:
            try:
                paper = next(arxiv.Search(id_list=[arxiv_id]).results())
                paper.download_pdf(dirpath=dir_path, filename=str(Path(dir_path) / f"{file_hash}.pdf"))
                break
            except Exception as e:
                print(e)
            curr_tries += 1
            time.sleep(5)
        if curr_tries == self.max_tries:
            self.log_error({
                "type": "arxiv_spider_pdf",
                "arxiv_id": arxiv_id,
                "dir_path": dir_path,
                "file_hash": file_hash,
            })

class PdfSpider(Spider):
    def __init__(self, error_file, max_tries=3):
        super().__init__(error_file)
        self.max_tries = max_tries
    
    def download_pdf(self, url, dir_path, file_hash):
        curr_tries = 0
        while curr_tries < self.max_tries:
            try: 
                r = requests.get(url, timeout=1, verify=True) 
                r.raise_for_status()
                with open(str(Path(dir_path) / f"{file_hash}.pdf"), 'wb') as f:
                    f.write(r.content)
                break
            except requests.exceptions.HTTPError as errh: 
                print("HTTP Error", errh) 
                print(errh.args[0]) 
            except requests.exceptions.ReadTimeout as errrt: 
                print("Time out", errrt) 
            except requests.exceptions.ConnectionError as conerr: 
                print("Connection error", conerr) 
            except requests.exceptions.RequestException as errex: 
                print("Exception request", errex) 
            except Exception as e:
                print(e)
            curr_tries += 1
            time.sleep(5)
        if curr_tries == self.max_tries:
            self.log_error({
                "type": "pdf_spider",
                "url": url,
                "dir_path": dir_path,
                "file_hash": file_hash,
            })