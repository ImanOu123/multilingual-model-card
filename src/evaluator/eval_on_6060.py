from tqdm import tqdm
import json
import os
import argparse
import pandas as pd
from collections import defaultdict
import sacrebleu

class Scores:
    def __init__(self):
        self.bleu = sacrebleu.BLEU(effective_order=True)
        self.bleu_jp = sacrebleu.BLEU(effective_order=True, tokenize='ja-mecab')
        self.bleu_zh = sacrebleu.BLEU(effective_order=True, tokenize='zh')
        
        self.chrf = sacrebleu.CHRF()

        self.ter = sacrebleu.TER(normalized=True)
        self.ter_asian = sacrebleu.TER(normalized=True, asian_support=True)
        
        self.chrfpp = sacrebleu.CHRF(word_order=2)
    
    def get_scores(self, pred_str, ref_str, tgt_lang):
        ref_lst = [ref_str]

        if tgt_lang == 'Japanese':
            bleu_score = self.bleu_jp.corpus_score(pred_str, ref_lst)
        elif tgt_lang == 'Chinese':
            bleu_score = self.bleu_zh.corpus_score(pred_str, ref_lst)
        else:
            bleu_score = self.bleu.corpus_score(pred_str, ref_lst)

        chrf_score = self.chrf.corpus_score(pred_str, ref_lst)

        if tgt_lang in ['Japanese', 'Chinese']:
            ter_score = self.ter_asian.corpus_score(pred_str, ref_lst)
        else:
            ter_score = self.ter.corpus_score(pred_str, ref_lst)

        chrfpp_score = self.chrfpp.corpus_score(pred_str, ref_lst)
        
        info = {
            "bleu": bleu_score,
            "chrf": chrf_score,
            "ter": ter_score,
            "chrfpp": chrfpp_score
        }
        return info

def get_mean_std_dict(df):
    mean_std_dict = {}

    for column in df.columns:
        mean_std_dict[column + "_mean"] = df[column].mean()
        mean_std_dict[column + "_std"] = df[column].std()

    return mean_std_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_googletrans.jsonl")
    parser.add_argument("--gt_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/2/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt")
    parser.add_argument("--out_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_googletrans.csv")
    args = parser.parse_args()
    
    tgt_langs = [
        "Chinese",
        "Arabic",
        "French",
        "Japanese",
        "Russian",
    ]
    
    # create gt_dict
    gt_dict = {
        "English": [i for i in open(args.gt_file, 'r').readlines()],
        "Chinese": [i for i in open(args.gt_file.replace("en.txt", "zh.txt"), 'r').readlines()],
        "Arabic": [i for i in open(args.gt_file.replace("en.txt", "ar.txt"), 'r').readlines()],
        "French": [i for i in open(args.gt_file.replace("en.txt", "fr.txt"), 'r').readlines()],
        "Japanese": [i for i in open(args.gt_file.replace("en.txt", "ja.txt"), 'r').readlines()],
        "Russian": [i for i in open(args.gt_file.replace("en.txt", "ru.txt"), 'r').readlines()],
    }
    
    # create pred_dict
    pred_dict = {
        "Chinese": [],
        "Arabic": [],
        "French": [],
        "Japanese": [],
        "Russian": [],
    }
    in_data = [json.loads(i) for i in open(args.in_file, 'r').readlines()]
    for item in in_data:
        for tgt_lang in tgt_langs:
            pred_dict[tgt_lang].append(item[f'text_{tgt_lang}'])
    
    # get scores
    scores_dict = defaultdict(list)
    scorer = Scores()
    for tgt_lang in tqdm(tgt_langs):
        for gt_item, pred_item in zip(gt_dict[tgt_lang], pred_dict[tgt_lang]):
            scores = scorer.get_scores(pred_item, gt_item, tgt_lang)
            for score in scores:
                if score == "bleu":
                    scores_dict[f"{tgt_lang}_{score}_score"].append(scores[score].score)
                    scores_dict[f"{tgt_lang}_{score}_1gram_precision"].append(scores[score].precisions[0])
                    scores_dict[f"{tgt_lang}_{score}_2gram_precision"].append(scores[score].precisions[1])
                    scores_dict[f"{tgt_lang}_{score}_3gram_precision"].append(scores[score].precisions[2])
                    scores_dict[f"{tgt_lang}_{score}_4gram_precision"].append(scores[score].precisions[3])
                    scores_dict[f"{tgt_lang}_{score}_ref_len"].append(scores[score].ref_len)
                    scores_dict[f"{tgt_lang}_{score}_pred_len"].append(scores[score].sys_len)
                    scores_dict[f"{tgt_lang}_{score}_bp"].append(scores[score].bp)
                    scores_dict[f"{tgt_lang}_{score}_ratio"].append(scores[score].ratio)
                else:
                    scores_dict[f"{tgt_lang}_{score}_score"].append(scores[score].score)
    
    # get scores into a dataframe
    df = pd.DataFrame(scores_dict)
    
    # store scores to csv
    df.to_csv(args.out_file, index=False)