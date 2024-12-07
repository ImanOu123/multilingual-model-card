# Adapted from run_on_6060.py. This is not a post-hoc method.
from tqdm import tqdm
import json
import os
import argparse
from typing import Any
import sys

def choose_translator(args):
    if args.model == 'googletrans':
        from multilingualmc.translator.config import GoogleTranslatorConfig
        from multilingualmc.translator.translator import GoogleTranslator
        import httpcore
        setattr(httpcore, 'SyncHTTPTransport', Any)
        trans_args = GoogleTranslatorConfig
        translator = GoogleTranslator(trans_args)
        return translator
    elif args.model == 'seamless':
        from multilingualmc.translator.config import M4TLargeTranslatorConfig
        from multilingualmc.translator.translator import SeamlessTranslator
        from multilingualmc.translator.config import M4TLargeTranslatorConfig
        from multilingualmc.translator.translator import SeamlessTranslator
        trans_args = M4TLargeTranslatorConfig
        trans_args.method = args.method # type: ignore
        translator = SeamlessTranslator(trans_args)
        return translator
    elif args.model == 'nllb':
        from multilingualmc.translator.config import NLLBTranslatorConfig
        from multilingualmc.translator.translator import NLLBTranslator
        trans_args = NLLBTranslatorConfig
        trans_args.method = args.method # type: ignore
        translator = NLLBTranslator(trans_args)
        return translator        
        return translator
    elif args.model == 'nllb':
        from multilingualmc.translator.config import NLLBTranslatorConfig
        from multilingualmc.translator.translator import NLLBTranslator
        trans_args = NLLBTranslatorConfig
        trans_args.method = args.method # type: ignore
        translator = NLLBTranslator(trans_args)
        return translator        
    elif 'gpt' in args.model:
        from multilingualmc.translator.config import GPTTranslatorConfig
        from multilingualmc.translator.translator import VLLMTranslator
        from multilingualmc.translator.config import GPTTranslatorConfig
        from multilingualmc.translator.translator import VLLMTranslator
        trans_args = GPTTranslatorConfig
        trans_args.model_name = args.model
        translator = VLLMTranslator(trans_args)
        return translator
    elif "llama3" in args.model:
        # llama3_8b, llama3_70b
        from multilingualmc.translator.config import LLAMA3TranslatorConfig
        from multilingualmc.translator.translator import LLAMATranslator
        from multilingualmc.translator.config import LLAMA3TranslatorConfig
        from multilingualmc.translator.translator import LLAMATranslator
        trans_args = LLAMA3TranslatorConfig
        if args.model == "llama3_8b":
            trans_args.model_name = "/data/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/"
        elif args.model == "llama3_70b":
            trans_args.model_name = "/data/models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct/"
        elif args.model == "llama31_8b":
            trans_args.model_name = "/data/user_data/jiaruil5/.cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"
        elif args.model == "llama31_70b":
            trans_args.model_name = "/compute/babel-8-7/jiaruil5/.cache/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393/"
        else:
            raise NotImplementedError
        translator = LLAMATranslator(trans_args)
        return translator
    elif "qwen" in args.model:
        from multilingualmc.translator.config import QWENTranslatorConfig
        from multilingualmc.translator.translator import QWENTranslator
        from multilingualmc.translator.config import QWENTranslatorConfig
        from multilingualmc.translator.translator import QWENTranslator
        trans_args = QWENTranslatorConfig
        if args.model == "qwen2_7b":
            trans_args.model_name = "/compute/babel-8-7/jiaruil5/.cache/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c/" # type: ignore
        else:
            raise NotImplementedError
        translator = QWENTranslator(trans_args)
        return translator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/2/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt")
    parser.add_argument("--out_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_googletrans.jsonl")
    parser.add_argument("--model", type=str, default='googletrans', choices=['googletrans', 'seamless', 'nllb', 'gpt-4o-mini', 'gpt-3.5-turbo', 'llama3_8b', 'llama31_8b', 'llama3_70b', 'llama31_70b', 'qwen2_7b'])
    parser.add_argument("--method", type=str, default='none', choices=['constraint_soft'])
    parser.add_argument("--term_file_path", type=str, default=None, help="used only when the method is constraint soft.")
    parser.add_argument("--soft_penalty", type=float, default=0.8)
    args = parser.parse_args()
    
    args.out_file = args.out_file.replace(".jsonl", "_" + str(args.soft_penalty) + ".jsonl")
    
    tgt_langs = [
        "Chinese",
        "Arabic",
        "French",
        "Japanese",
        "Russian",
    ]
    
    from multilingualmc.translator.get_terms import TermCollector
    term_collector = TermCollector(args.term_file_path, tgt_langs)
    
    
    if args.in_file.endswith(".txt"):
        gt_dict = {
            "English": [i for i in open(args.in_file, 'r').readlines()],
        }
    elif args.in_file.endswith(".json"):
        gt_dict = {
            "English": [i['text'] for i in json.load(open(args.in_file, 'r'))]
        }
    
    translator = choose_translator(args)
    
    out_f = open(args.out_file, 'a')
    for idx, item in tqdm(enumerate(gt_dict['English'])):
        info = {
            'text': item
        }
        for tgt_lang in tqdm(['Chinese', 'Arabic', 'French', 'Japanese', 'Russian']):
            
            kwargs = {}
            if 'gpt' in args.model or 'llama' in args.model or 'qwen' in args.model:
                kwargs['prompt_version'] = "simple"

            if args.method == 'constraint_soft':
                
                relevant_terms_dict = {}
                for key in term_collector.find_terminology(item):
                    relevant_terms_dict[key] = term_collector.terms_dict[key]
                    
                answer = translator.translate_constraint_soft( # type: ignore
                    item,
                    src_lang = 'English',
                    tgt_lang = tgt_lang,
                    terms_dict = list(set([val[tgt_lang] for key, val in relevant_terms_dict[key].items()])),
                    soft_penalty = args.soft_penalty,
                    **kwargs
                )
                info['relevant_terms_dict'] = relevant_terms_dict # type: ignore
            else:
                exit(0)
            
            
            info[f'text_{tgt_lang}'] = answer # type: ignore
        
        json.dump(info, out_f, ensure_ascii=False)
        out_f.write("\n")
        out_f.flush()

    