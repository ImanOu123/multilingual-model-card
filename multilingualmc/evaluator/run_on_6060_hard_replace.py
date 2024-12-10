import time
import re
import json
import pandas as pd
import openai
from tqdm import tqdm
import sys
import argparse
from multilingualmc.translator.get_terms import TermCollector
import torch
import jieba
import MeCab
import nltk
from nltk.tokenize import word_tokenize
import transformers
import itertools

class HardReplacement():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = transformers.BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased', cache_dir="/data/user_data/jiaruil5/.cache/")
        self.model = transformers.BertModel.from_pretrained('google-bert/bert-base-multilingual-cased', cache_dir="/data/user_data/jiaruil5/.cache/").to(self.device)
        
    
    def split_text(self, text, lang, terms_dict=None):
        if lang in ['English'] and terms_dict is not None:
            preserve_list = [i.lower() for i in terms_dict.keys()]
            placeholders = {term: f"__PLACEHOLDER_{i}__" for i, term in enumerate(preserve_list)}
            for term, placeholder in placeholders.items():
                # text = text.replace(term, placeholder)
                text = re.sub(re.escape(term), placeholder, text, flags=re.IGNORECASE)
        
        if lang in ['English', 'French', 'Arabic', 'Russian']:
            text = word_tokenize(text)
            text = ['"' if token in ['``', "''"] else token for token in text]
        elif lang in ['Chinese']:
            text = [i for i in jieba.cut(text, cut_all=False)]
        elif lang in ['Japanese']:
            mecab = MeCab.Tagger('-Owakati')
            text = mecab.parse(text.strip()).split()
            
        if lang in ['English'] and terms_dict is not None:
            res_text = []
            for token in text:
                for term, placeholder in placeholders.items():
                    if token == placeholder:
                        res_text.append(term)
                        break
                else:
                    res_text.append(token)
            return res_text
         
        return text
    
    def get_word_indices(self, sentence, words):
        indices = []
        position = 0  # Track the character position in the original string
        
        for word in words:
            # Locate the word by iterating forward
            while position < len(sentence):
                # Check if the substring matches the word
                if sentence[position:position+len(word)] == word:
                    indices.append([position, position+len(word)])
                    position += len(word)  # Move position past the word
                    break
                position += 1  # Move forward to next character
            print(word, position)
        
        return indices
    
    def find_word_alignment(self, text_src, text_tgt, src_lang, tgt_lang, terms_dict):
        text_src = self.split_text(text_src, src_lang, terms_dict)
        text_tgt = self.split_text(text_tgt, tgt_lang)
        
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in text_src], [self.tokenizer.tokenize(word) for word in text_tgt]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=self.tokenizer.model_max_length, truncation=True)['input_ids'], self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=self.tokenizer.model_max_length)['input_ids']
        sub2word_map_src = []
        for i, word_list in enumerate(token_src):
            sub2word_map_src += [i for x in word_list]
        sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt += [i for x in word_list]

        # alignment
        align_layer = 8
        threshold = 1e-4
        self.model.eval()
        with torch.no_grad():
            out_src = self.model(ids_src.unsqueeze(0).to(self.device), output_hidden_states=True)[2][align_layer][0, 1:-1]
            out_tgt = self.model(ids_tgt.unsqueeze(0).to(self.device), output_hidden_states=True)[2][align_layer][0, 1:-1]

            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

            softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        align_words = set()
        for i, j in align_subwords:
            align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]) )

        return text_src, text_tgt, align_words
        
        
    def hard_replace(
        self,
        text_src,
        text_tgt,
        terms_dict,
        tgt_lang,
    ):
        def is_continuous(sorted_lst):
            
            for i in range(1, len(sorted_lst)):
                if sorted_lst[i] != sorted_lst[i - 1] + 1:
                    return False
            return True
        def find_overlap_indices(larger_list, smaller_list):
            n, m = len(larger_list), len(smaller_list)
            
            # Loop through the larger list to find where the smaller list begins to overlap
            for i in range(n - m + 1):
                # Check if the sublist starting at index i matches the smaller list
                if larger_list[i:i + m] == smaller_list:
                    start_index = i
                    end_index = i + m - 1
                    return start_index, end_index

            # If no overlap is found
            return None, None
        def modify_multiple(original, modifications):
            
            # Sort modifications in descending order of start index
            modifications.sort(key=lambda x: x[0], reverse=True)
            
            # Apply each modification
            for start_index, end_index, new_content in modifications:
                # Ensure indices are within valid range
                if start_index < 0 or end_index >= len(original) or start_index > end_index:
                    return original
                
                # Apply the modification using string slicing
                original = original[:start_index] + new_content + original[end_index:]
            print(original)
            return original
        
        sep_src, sep_tgt, align_words = self.find_word_alignment(
            text_src,
            text_tgt,
            src_lang='English',
            tgt_lang=tgt_lang,
            terms_dict=terms_dict
        )
        for i, j in sorted(align_words):
            print(f"{sep_src[i]}: {sep_tgt[j]}, ")
        indices = self.get_word_indices(text_tgt, sep_tgt)
        print(sep_tgt)
        print(indices)
        
        new_contents = []
        for term_orig in terms_dict:
            if tgt_lang not in terms_dict[term_orig]:
                continue
            term = term_orig.lower()
            
            indices_tgt = []
            for (i_src, i_tgt) in align_words:
                if sep_src[i_src].lower() == term:
                    indices_tgt.append(i_tgt)
            
            indices_tgt = sorted(indices_tgt)
            if len(indices_tgt) > 0 and is_continuous(indices_tgt):

                start_idx, end_idx = find_overlap_indices(sep_tgt, [sep_tgt[i] for i in indices_tgt])
                if start_idx is not None:
                    print("Entering modify multiple:")
                    print(text_tgt)
                    try:
                        new_contents.append([indices[start_idx][0], indices[end_idx][1], terms_dict[term_orig][tgt_lang]])
                    except:
                        return text_tgt
                    print(new_contents)

        res_tgt = modify_multiple(
            text_tgt,
            new_contents           
        )
        return res_tgt



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="home/ubuntu/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless.jsonl")
    parser.add_argument("--out_file", type=str, default="/home/ubuntu/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/Japanese_validated.csv")
    parser.add_argument("--term_file_path", type=str, default=None, help="used only when the method is constrained_beam_search.")
    
    args = parser.parse_args()
    args.in_file = [json.loads(i) for i in open(args.in_file, 'r').readlines()]
    args.out_file = open(args.out_file, 'a')
    
    tgt_langs = [
        "Chinese",
        "Arabic",
        "French",
        "Japanese",
        "Russian",
    ]
    
    term_collector = TermCollector(args.term_file_path, tgt_langs)

    hard_replace_worker = HardReplacement()

    for line in args.in_file:
        info = {
            "text": line['text']
        }
        relevant_terms_dict = {}
        for key in term_collector.find_terminology(line['text']):
            relevant_terms_dict[key] = term_collector.terms_dict[key]
            
        
        for lang in tgt_langs:
            translation = hard_replace_worker.hard_replace(
                line['text'],
                line[f'text_{lang}'],
                terms_dict = relevant_terms_dict,
                tgt_lang = lang
            )
            if translation is None:
                info[f"text_{lang}"] = line[f'text_{lang}']
            else:
                info[f"text_{lang}"] = translation

            info[f"terms_dict"] = relevant_terms_dict
            print("Before editing:", line[f'text_{lang}'])
            print("After editing:", translation)
        # import pdb
        # pdb.set_trace()
        print(info)
        
        json.dump(info, args.out_file, ensure_ascii=False)
        args.out_file.write("\n")
        args.out_file.flush()