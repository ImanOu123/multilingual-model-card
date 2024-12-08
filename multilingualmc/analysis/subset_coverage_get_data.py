import json
from tqdm import tqdm
import re

# if __name__ == "__main__":
#     data = [json.loads(i) for i in open("/home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/growing_dict/terms.jsonl", 'r').readlines()]
    
#     papers = json.load(open("/home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/info.json", 'r'))
#     papers = papers['dev']['paper']
    
    
#     all_papers = {paper: set() for paper in papers}
#     for item in tqdm(data):
#         paper = item['paper_path']
        
#         paper_terms = [re.sub(r'\(.*?\)', '', i.lower()).strip() for i in item['processed_result']]
#         all_papers[paper].update(paper_terms)
        
#     res = []
#     for key, val in all_papers.items():
#         res.append(list(val))
    
#     out_f = open("subset_coverage_data.json", 'w')
#     json.dump(res, out_f)
#     out_f.flush()


if __name__ == "__main__":
    data = [json.loads(i) for i in open("/home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/growing_dict/terms.jsonl", 'r').readlines()]
    
    papers = json.load(open("/home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/info.json", 'r'))
    papers = papers['dev']['paper']
    
    
    all_papers = {paper: [] for paper in papers}
    for item in tqdm(data):
        paper = item['paper_path']
        
        paper_terms = [re.sub(r'\(.*?\)', '', i.lower()).strip() for i in item['processed_result']]
        all_papers[paper].extend(paper_terms)
        
    res = []
    for key, val in all_papers.items():
        res.append(list(val))
    
    out_f = open("subset_coverage_data_list.json", 'w')
    json.dump(res, out_f)
    out_f.flush()