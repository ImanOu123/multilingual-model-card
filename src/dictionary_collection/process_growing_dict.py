import json
import re
from pyinflect import getAllInflections
import nltk
from confirm_terms_gpt import GPTTermConfirmer

def remove_punct(term):
    return re.sub(r'[^0-9a-zA-Z /]+', "", term.lower()).replace("  ", " ").strip()

def extract_paper_term_lst(jsonl_path):
    with open(jsonl_path) as f:
        json_str_lst = list(f)
    
    json_lst = list(map(lambda s: json.loads(s), json_str_lst))
    
    processed_json = {}
    for line in json_lst:
        if line['paper_path'] not in processed_json:
            processed_json[line['paper_path']] = line['processed_result']
        else:
            processed_json[line['paper_path']] += line['processed_result']
    
    # make into list of paper id and terms only
    res = []
    for paper in processed_json:
        res.append({"paper_path" : paper, "terms" : remove_term_inflections(processed_json[paper])})
        
    return res

def remove_paper_unique_terms(paper_term_lst):
    """ removes terms that only exist in one paper """
    # remove terms that only exist in one paper - terms already unique in one paper
    temp_term_lst = []
    term_lst = []
    
    for dct in paper_term_lst:
        for term in dct["terms"]:
            # if seeing a term for a second or more time add to the term_lst
            if term in temp_term_lst and term not in term_lst and term.strip() != "":
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
    """removes abbreviations (continuous stream of capital letters)"""
    def check_abbr(term):
        if re.search(r"([A-Z]\.*){2,}s?", term) or len(remove_punct(term)) <= 2: 
            return False 
        else:
            return True
    
    # remove terms with abbreviations
    return list(filter(check_abbr, term_lst))

def remove_terms_w_nonce(term_lst):
    """removes terms that start with special characters"""
    
    # remove terms with nonce words
    
    # remove any terms after terms starting with z and before terms
    # starting with number 0 in sorted list
    term_lst.sort()
    
    postZidx = len(term_lst)
    firstNumidx = 0
    
    for idx, term in enumerate(term_lst):
        if idx != len(term_lst)-1 and term != "":
            if term[0] == 'z' and term_lst[idx+1][0] != 'z':
                # index of term after last term starting with z
                postZidx = idx+1
        
        if idx != 0:
            if term[0] == '0' and term_lst[idx-1][0] != '0':
                # index of first term starting with 0
                firstNumidx = idx
             
    return term_lst[firstNumidx:postZidx]
    
def remove_term_inflections(term_lst):
    """removes terms that repeat or their inflections - e.g. singular/plural"""
    
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

def remove_non_nouns(term_lst):
    """removes terms that aren't nouns"""
    processed_term_lst = []
    
    for term in term_lst:
        
        # https://stackoverflow.com/questions/33587667/extracting-all-nouns-from-a-text-file-using-nltk
        is_noun = lambda pos: pos[:2] == 'NN'
        tokenized = nltk.word_tokenize(term)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
        if nouns != []:
            processed_term_lst.append(term)
    
    return processed_term_lst
            

def confirm_AI_terms(term_lst):
    """remove terms that are not considered AI terms"""
    
    print("running GPT confirmer...")
    
    class ConfirmerArgs:
        model_name = "gpt-3.5-turbo"
    confirmer = GPTTermConfirmer(ConfirmerArgs)
    
    prompt = [
        """ Your task is to identify whether the following term below can be classified as AI scientific terminology:

            Here is the definition of AI scientific terminology:
            '''
            AI scientific terminology refers to specialized nouns or noun phrases within Artificial Intelligence, encompassing essential concepts, methods, models, algorithms, and systems. These terms must:
            1. Be composed of nouns, adjectives, and occasionally prepositions.
            2. Be context-specific to AI, having either no meaning or a different meaning outside this field.
            Additionally, these terms often pose significant challenges for accurate translation by machine translation models due to their technical specificity.
            '''
            
            Provide only “True” or “False” in your result, nothing else.


            Here is the term:
            '''
            {term}
            '''
        """
    ]
    
    # uses GPT to confirm whether each term falls under the definition of an AI terminology
    
    processed_term_lst = []
    false_lst = []
    for term in term_lst:
        res, processed_res = confirmer.confirm_term(prompt=[prompt[0].format(term=term)])
        
        if processed_res:
            processed_term_lst.append(term)
        
        elif processed_res == False:
            false_lst.append(term)
            
    print(false_lst)
    return processed_term_lst
        
def final_changes(term_lst):
    
    processed_term_lst = []
    for term in term_lst:
        # remove any terms with a capital in the middle of the term with a lower case letter preceding it
        if re.search(r"\b[A-z]?[a-z]+[A-Z][a-z]*\b", term) == None:
            processed_term_lst.append(term)

    return processed_term_lst

if __name__ == "__main__":
    
    # paper_term_lst = extract_paper_term_lst('growing_dict/terms.jsonl')
    
    # term_lst = remove_paper_unique_terms(paper_term_lst)
    # term_lst.sort()
    # # add terms to a text file
    # write_list_to_file('growing_dict/terms.txt', term_lst)
    
    # ----------------------------------------------------------------------
    
    # term_lst = read_list_from_file('growing_dict/terms.txt')
    
    # term_lst_wo_abbr = remove_terms_w_abbr(term_lst)
    # term_lst_wo_nonce = remove_terms_w_nonce(term_lst_wo_abbr)
    # term_lst_wo_repeat = remove_term_inflections(term_lst_wo_nonce)
    # term_lst_w_noun = remove_non_nouns(term_lst_wo_repeat)
        
    # ----------------------------------------------------------------------

    # term_lst_w_ai = confirm_AI_terms(term_lst_w_noun)

    # write_list_to_file('growing_dict/processed_terms.txt', term_lst_w_ai)
    
    # ----------------------------------------------------------------------
    
    # term_lst = read_list_from_file('growing_dict/processed_terms.txt')
    
    # final_term_lst = final_changes(term_lst)
    
    # write_list_to_file('growing_dict/final_terms.txt', final_term_lst)
    
    print("done")