import json
import re

def extract_paper_term_lst(jsonl_path):
    with open(jsonl_path) as f:
        json_str_lst = list(f)
    
    json_lst = list(map(lambda s: json.loads(s), json_str_lst))
    
    # make into list of paper id and terms only
    return list(map(lambda x: {"paper_path" : x["paper_path"], 
                               "terms" : list(set(x["processed_result"]))},
                    json_lst))

def remove_paper_unique_terms(paper_term_lst):
    # remove terms that only exist in one paper - terms already unique in one paper
    temp_term_lst = []
    term_lst = []
    
    for dct in paper_term_lst:
        for term in dct["terms"]:
            # if seeing a term for a second or more time add to the term_lst
            if term in temp_term_lst and term not in term_lst:
                term_lst.append(term)
            # if haven't seen term before, add to temp_term_lst
            elif term not in temp_term_lst:
                temp_term_lst.append(term)
                
    return term_lst

def write_list_to_file(file_path, lst):
    f = open(file_path,'w')
    for w in lst:
        f.write(w+"\n")
    f.close()
    
def read_list_from_file(file_path):
    with open(file_path) as f:
        lines = [line.strip() for line in f]
    return lines

def remove_terms_w_abbr(term_lst):

    def check_abbr(term):
        if re.search(r"([A-Z]\.*){2,}s?", term) or term == "": 
            return False 
        else:
            return True
    
    # remove terms with abbreviations
    return list(filter(check_abbr, term_lst))

def remove_terms_w_nonce():
    # remove terms with nonce words
    
    return


if __name__ == "__main__":
    
    # paper_term_lst = extract_paper_term_lst('growing_dict/terms.jsonl')

    # term_lst = remove_paper_unique_terms(paper_term_lst)
    
    # add terms to a text file
    # write_list_to_file('growing_dict/terms.txt', term_lst)
    
    # ----------------------------------------------------------------------

    term_lst = read_list_from_file('growing_dict/terms.txt')
    term_lst_wo_abbr = remove_terms_w_abbr(term_lst)
    term_lst_wo_abbr.sort()
    write_list_to_file('growing_dict/processed_terms.txt', term_lst_wo_abbr)
    