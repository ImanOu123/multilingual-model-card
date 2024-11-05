from tqdm import tqdm
import json
import os
import argparse
from typing import Any
import sys
sys.path.append("../")

def choose_translator(args):
    if args.model == 'googletrans':
        from translator.config import GoogleTranslatorConfig
        from translator.translator import GoogleTranslator
        import httpcore
        setattr(httpcore, 'SyncHTTPTransport', Any)
        trans_args = GoogleTranslatorConfig
        translator = GoogleTranslator(trans_args)
        return translator
    elif args.model == 'seamless':
        from translator.config import M4TLargeTranslatorConfig
        from translator.translator import SeamlessTranslator
        trans_args = M4TLargeTranslatorConfig
        trans_args.method = args.method
        translator = SeamlessTranslator(trans_args)
        return translator
    elif 'gpt' in args.model:
        from translator.config import GPTTranslatorConfig
        from translator.translator import LLMTranslator
        trans_args = GPTTranslatorConfig
        trans_args.model_name = args.model
        translator = LLMTranslator(trans_args)
        return translator
    elif "llama3" in args.model:
        # llama3_8b, llama3_70b
        from translator.config import LLAMA3TranslatorConfig
        from translator.translator import LLAMATranslator
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
        from translator.config import QWENTranslatorConfig
        from translator.translator import QWENTranslator
        trans_args = QWENTranslatorConfig
        if args.model == "qwen2_7b":
            trans_args.model_name = "/compute/babel-8-7/jiaruil5/.cache/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c/"
        else:
            raise NotImplementedError
        translator = QWENTranslator(trans_args)
        return translator

# gpt-4o-mini: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_gpt4omini.jsonl --model gpt-4o-mini
# gpt-3.5-turbo: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_gpt35turbo.jsonl --model gpt-3.5-turbo
# llama3_8b: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama3_8b.jsonl --model llama3_8b
# llama3_70b: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama3_70b.jsonl --model llama3_70b
# llama31_8b: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama31_8b.jsonl --model llama31_8b
# llama31_70b: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama31_70b.jsonl --model llama31_70b
# qwen2_7b: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_qwen2_7b.jsonl --model qwen2_7b
# seamless: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_seamless.jsonl --model seamless
# seamless_cbs: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_seamless_cbs.jsonl --model seamless --method constrained_beam_search --term_file /home/jiaruil5/multilingual/multilingual-model-card/src/dictionary_collection/growing_dict/mturk.json

# qwen2_7b_cbs: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_qwen2_7b_cbs.jsonl --model qwen2_7b --method constrained_beam_search --term_file /home/jiaruil5/multilingual/multilingual-model-card/src/dictionary_collection/growing_dict/mturk.json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/2/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt")
    parser.add_argument("--out_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_googletrans.jsonl")
    parser.add_argument("--model", type=str, default='googletrans', choices=['googletrans', 'seamless', 'gpt-4o-mini', 'gpt-3.5-turbo', 'llama3_8b', 'llama31_8b', 'llama3_70b', 'llama31_70b', 'qwen2_7b'])
    parser.add_argument("--method", type=str, default='none', choices=['none', 'constrained_beam_search'])
    parser.add_argument("--term_file", type=str, default=None, help="used only when the method is constrained_beam_search.")
    args = parser.parse_args()
    
    if args.method == 'constrained_beam_search':
        from translator.get_terms import TermCollector
        term_collector = TermCollector(args.term_file)
    
    tgt_langs = [
        "Chinese",
        "Arabic",
        "French",
        "Japanese",
        "Russian",
    ]
    
    gt_dict = {
        "English": [i for i in open(args.in_file, 'r').readlines()],
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


            if args.method == 'none':
                answer = translator.translate(
                    item,
                    src_lang = 'English',
                    tgt_lang = tgt_lang,
                    **kwargs
                )
            elif args.method == 'constrained_beam_search':
                force_words = list(set([val[tgt_lang] for key, val in term_collector.get_occurrences(item).items()]))
                answer = translator.translate_cbs(
                    item,
                    src_lang = 'English',
                    tgt_lang = tgt_lang,
                    force_words = force_words,
                    **kwargs
                )
                info['force_words'] = force_words
                
            info[f'text_{tgt_lang}'] = answer
        
        json.dump(info, out_f, ensure_ascii=False)
        out_f.write("\n")
        out_f.flush()

    