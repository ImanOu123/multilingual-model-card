import re
import spacy
import csv
import pandas as pd
from pyinflect import getAllInflections

def remove_punct(term):
    return re.sub(r'[^0-9a-zA-Z /]+', "", term.lower()).replace("  ", " ").strip()

def remove_term_inflections(term_lst):
    """removes terms that repeat"""
    
    term_lst.sort()
    
    def flatten(lsts):
        tmp = []
        for lst in lsts:
            for e in lst:
                tmp.append(e)
        return tmp
    
    tmp_term_lst = []
    processed_term_lst = []
    inflect_lst = []
    
    for term in term_lst:
        termWOpunct = remove_punct(term)
        # if the term or an inflection of the term wasn't seen before
        if termWOpunct.replace(" ", "") not in tmp_term_lst and len(termWOpunct) > 1:
                                
            # add to tmp_term_lst without punctuation to use as a check
            tmp_term_lst.append(termWOpunct.replace(" ", ""))
            
            # finds inflections of final word in term
            inflect_lst = flatten(list(map(lambda x: list(x), getAllInflections(termWOpunct.split()[-1]).values())))
            
            # add inflections of full term to tmp_term_lst 
            tmp_term_lst += list(map(lambda lastWord: (" ". join(termWOpunct.split()[:-1]) + " " + lastWord).replace(" ", ""), inflect_lst))
            tmp_term_lst.append(termWOpunct.replace(" ", "")+"s")
            
            # add to final term list
            processed_term_lst.append(term)
    
    return processed_term_lst

def deduplicate_term_lst(term_lst):
    nlp = spacy.load("en_core_web_trf")
    
    process_term_lst = []
    for i, term in enumerate(term_lst):
        doc = nlp(term)
        new_token = "".join([token.lemma_ + token.whitespace_ for token in doc])
        process_term_lst.append(new_token)
        # process_term_lst.append({
        #     "term": new_token,
        #     "original_term": term
        # })
        
        print(i)
        
    print("removing duplicates...")
    final_term_lst = remove_term_inflections(process_term_lst)
    return final_term_lst
    
def read_list_from_file(file_path):
    with open(file_path) as f:
        lines = [line.strip() for line in f]
    return lines

def write_list_to_csv(file_path, lst):
    # put terms into a dictionary
    term_dict_lst = []
    for i, term in enumerate(lst):
        term_dict_lst.append({"idx" : i, "term" : term})
    fields = ["idx", "term"]
    
    # writing to csv file
    with open(file_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(term_dict_lst)

term_lst = read_list_from_file('growing_dict/man_final_terms.txt')

fin_term_lst = deduplicate_term_lst(term_lst)

write_list_to_csv('growing_dict/final_terms_2.csv', fin_term_lst)
