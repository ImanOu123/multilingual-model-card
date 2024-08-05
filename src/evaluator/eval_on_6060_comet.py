from tqdm import tqdm
import json
import os
import argparse
import pandas as pd
from collections import defaultdict
from comet import download_model, load_from_checkpoint

from eval_on_6060 import get_mean_std_dict

os.environ['HUGGINGFACE_API_TOKEN'] = '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'

class COMET:
    def __init__(self, model_id, cache_dir=None):
        """
        Supported model_id:
        - `Unbabel/wmt22-comet-da`
        - `Unbabel/wmt23-cometkiwi-da-xxl`
        - `Unbabel/wmt23-cometkiwi-da-xl`
        - `Unbabel/wmt22-cometkiwi-da`
        - `Unbabel/XCOMET-XXL`
        - `Unbabel/XCOMET-XL`
        
        
        HUGGINGFACE_API_TOKEN=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
        """
        self.model_id = model_id
        model_path = download_model(model_id, saving_directory=cache_dir)
        self.model = load_from_checkpoint(model_path)
    
    def get_scores(self, data, batch_size=8, gpus=1):
        """
        If using `Unbabel/wmt22-comet-da`:
        ```
        data: [
            {
                "src": "Dem Feuer konnte Einhalt geboten werden",
                "mt": "The fire could be stopped",
                "ref": "They were able to control the fire."
            },
            {
                "src": "Schulen und Kindergärten wurden eröffnet.",
                "mt": "Schools and kindergartens were open",
                "ref": "Schools and kindergartens opened"
            }
        ]
        ```
        
        If using `Unbabel/wmt23-cometkiwi-da-xxl` (44GB GPU memory) or `Unbabel/wmt23-cometkiwi-da-xl` (15GB GPU memory) or `Unbabel/wmt22-cometkiwi-da`:
        ```
        data = [
            {
                "src": "The output signal provides constant sync so the display never glitches.",
                "mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."
            },
            {
                "src": "Kroužek ilustrace je určen všem milovníkům umění ve věku od 10 do 15 let.",
                "mt": "Кільце ілюстрації призначене для всіх любителів мистецтва у віці від 10 до 15 років."
            },
            {
                "src": "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
                "mt": "その後、1994年の選挙でアフリカ国民会議派が勝利し、南アフリカ初の黒人大統領となった。"
            }
        ]
        ```
        
        If using `Unbabel/XCOMET-XXL` (44GB GPU memory) or `Unbabel/XCOMET-XL` (15GB GPU memory):
        ```
        data = [
            {
                "src": "Boris Johnson teeters on edge of favour with Tory MPs", 
                "mt": "Boris Johnson ist bei Tory-Abgeordneten völlig in der Gunst", 
                "ref": "Boris Johnsons Beliebtheit bei Tory-MPs steht auf der Kippe"
            }
        ]
        ```
        
        Usage: 
        - https://huggingface.co/Unbabel/wmt22-comet-da
        - https://huggingface.co/Unbabel/wmt22-cometkiwi-da
        - https://huggingface.co/Unbabel/XCOMET-XXL
        """
        model_output = self.model.predict(data, batch_size=batch_size, gpus=gpus)
        return model_output

# default model:
# python3 eval_on_6060_comet.py
# python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_seamless.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_seamless_comet.json

# reference-free model: Unbabel/wmt23-cometkiwi-da-xl
# python3 eval_on_6060_comet.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_googletrans_comet_noref_xl.json --model_id Unbabel/wmt23-cometkiwi-da-xl
# python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_seamless.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_seamless_comet_noref_xl.json --model_id Unbabel/wmt23-cometkiwi-da-xl

# explanable comet model: Unbabel/XCOMET-XL
# python3 eval_on_6060_comet.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_googletrans_xcomet_xl.json --model_id Unbabel/XCOMET-XL
# python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_seamless.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_seamless_xcomet_xl.json --model_id Unbabel/XCOMET-XL

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_googletrans.jsonl")
    parser.add_argument("--gt_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/2/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt")
    parser.add_argument("--out_file", type=str, default="/home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_googletrans_comet.json")
    parser.add_argument("--model_id", type=str, default="Unbabel/wmt22-comet-da")
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
    scores_dict = {}
    comet = COMET(
        model_id = args.model_id,
        cache_dir = "/data/user_data/jiaruil5/.cache/",
    )
    for tgt_lang in tqdm(tgt_langs):
        data = []
        # for gt_item, pred_item, eng_item in zip(gt_dict[tgt_lang][:5], pred_dict[tgt_lang][:5], gt_dict['English'][:5]):
        for gt_item, pred_item, eng_item in zip(gt_dict[tgt_lang], pred_dict[tgt_lang], gt_dict['English']):
            info = {
                "src": eng_item,
                "mt": pred_item,
            }
            if "cometkiwi" not in comet.model_id:
                info['ref'] = gt_item
            data.append(info)
            
        scores = comet.get_scores(
            data,
            batch_size = 8,
            gpus = 1,
        )
        
        # `Unbabel/wmt22-comet-da`:
        if "XCOMET" not in args.model_id:
            scores_dict[f"{tgt_lang}_scores"] = scores.scores
            scores_dict[f"{tgt_lang}_system_score"] = scores.system_score
        else:
            scores_dict[f"{tgt_lang}"] = scores.__dict__

    # store scores to json
    with open(args.out_file, 'w') as file:
        json.dump(scores_dict, file, indent=2, ensure_ascii=False)
        file.flush()