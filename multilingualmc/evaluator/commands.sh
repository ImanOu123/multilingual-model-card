# run_on_6060.py

# gpt-4o-mini: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_gpt4omini.jsonl --model gpt-4o-mini
# gpt-3.5-turbo: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_gpt35turbo.jsonl --model gpt-3.5-turbo
# llama3_8b: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_llama3_8b.jsonl --model llama3_8b
# llama3_70b: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_llama3_70b.jsonl --model llama3_70b
# llama31_8b: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_llama31_8b.jsonl --model llama31_8b
# llama31_70b: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_llama31_70b.jsonl --model llama31_70b
# qwen2_7b: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_qwen2_7b.jsonl --model qwen2_7b

# seamless: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless.jsonl --model seamless
# seamless_cbs: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless_cbs.jsonl --model seamless --method constrained_beam_search --term_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/growing_dict/mturk.json


# nllb: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_nllb.jsonl --model nllb

# qwen2_7b_cbs: python3 run_on_6060.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_qwen2_7b_cbs.jsonl --model qwen2_7b --method constrained_beam_search --term_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/growing_dict/mturk.json

## eval_data
gpt-4o-mini: python3 run_on_6060.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data.json --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/gpt4omini.jsonl --model gpt-4o-mini

seamless: python3 run_on_6060.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data.json --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/seamless.jsonl --model seamless

nllb: python3 run_on_6060.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data.json --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/nllb.jsonl --model nllb

# run_on_6060_prompt.py

# python run_on_6060_prompt.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless_prompt_gpt4omini.jsonl --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/

# python run_on_6060_prompt.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_nllb.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_nllb_prompt_gpt4omini.jsonl --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/

# python run_on_6060_prompt.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_qwen2_7b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_qwen2_7b_prompt_gpt4omini.jsonl --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/

# python run_on_6060_prompt.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_gpt4omini_prompt_gpt4omini.jsonl --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/

python run_on_6060_prompt.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/gold_dev.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/gold_dev_prompt_gpt4o.jsonl --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/


python3 run_on_6060_prompt.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data_google.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data_gold.jsonl --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/ --model gpt-4o

# eval_on_6060_comet.py

# default model: Unbabel/wmt22-comet-da

# python3 eval_on_6060_comet.py
# python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_seamless.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_seamless_comet.json
# python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_gpt4omini_comet.json

python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless_prompt_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/predictions_dev_seamless_prompt_gpt4omini_comet.jsonl


python3 eval_on_6060_comet_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/gpt4omini.jsonl --gt_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data_gold.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval/gpt4omini_comet.jsonl


# reference-free model: Unbabel/wmt23-cometkiwi-da-xl
# python3 eval_on_6060_comet.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_googletrans_comet_noref_xl.json --model_id Unbabel/wmt23-cometkiwi-da-xl
# python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_seamless.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_seamless_comet_noref_xl.json --model_id Unbabel/wmt23-cometkiwi-da-xl
# python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_gpt4omini_comet_noref_xl.json --model_id Unbabel/wmt23-cometkiwi-da-xl

# python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless_prompt_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/predictions_dev_seamless_prompt_gpt4omini_comet_noref_xl.jsonl --model_id Unbabel/wmt23-cometkiwi-da-xl

# python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_nllb_prompt_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/predictions_dev_nllb_prompt_gpt4omini_comet_noref_xl.jsonl --model_id Unbabel/wmt23-cometkiwi-da-xl

# python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_nllb.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/predictions_dev_nllb_comet_noref_xl.jsonl --model_id Unbabel/wmt23-cometkiwi-da-xl

# python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_gpt4omini_prompt_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/predictions_dev_gpt4omini_prompt_gpt4omini_comet_noref_xl.jsonl --model_id Unbabel/wmt23-cometkiwi-da-xl

# python3 eval_on_6060_comet_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/gold_dev_prompt_gpt4o.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_dev_prompt_gpt4o_comet.json

# python3 eval_on_6060_comet_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_seamless_comet.json

# python3 eval_on_6060_comet_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless_prompt_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_seamless_prompt_gpt4omini_comet.json

# python3 eval_on_6060_comet_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_nllb.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_nllb_comet.json

# python3 eval_on_6060_comet_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_nllb_prompt_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_nllb_prompt_gpt4omini_comet.json

# python3 eval_on_6060_comet_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_gpt4omini_comet.json

# python3 eval_on_6060_comet_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_gpt4omini_prompt_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_gpt4omini_prompt_gpt4omini_comet.json


# python3 eval_on_6060_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_seamless.csv

python3 eval_on_6060_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless_prompt_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_seamless_prompt_gpt4omini.csv

python3 eval_on_6060_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_nllb.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_nllb.csv

python3 eval_on_6060_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_nllb_prompt_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_nllb_prompt_gpt4omini.csv

python3 eval_on_6060_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_gpt4omini.csv

python3 eval_on_6060_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_gpt4omini_prompt_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_gpt4omini_prompt_gpt4omini.csv


# Hard replacement

## eval gold 6060
python3 eval_on_6060_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless_hard_replace.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_seamless_hard_replace.csv

python3 eval_on_6060_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_nllb_hard_replace.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_nllb_hard_replace.csv

python3 eval_on_6060_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_gpt4omini_hard_replace.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_gpt4omini_hard_replace.csv

## eval comet 6060
python3 eval_on_6060_comet_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless_hard_replace.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_seamless_hard_replace_comet.csv

python3 eval_on_6060_comet_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_nllb_hard_replace.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_nllb_hard_replace_comet.csv

python3 eval_on_6060_comet_gold.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_gpt4omini_hard_replace.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_gpt4omini_hard_replace_comet.csv

## eval model card
CUDA_VISIBLE_DEVICES=0 bash eval_papers.sh seamless_hard_replace
CUDA_VISIBLE_DEVICES=0 bash eval_papers.sh nllb_hard_replace
CUDA_VISIBLE_DEVICES=0 bash eval_papers.sh gpt4omini_hard_replace

# explanable comet model: Unbabel/XCOMET-XL
# python3 eval_on_6060_comet.py --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_googletrans_xcomet_xl.json --model_id Unbabel/XCOMET-XL
# python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_seamless.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_seamless_xcomet_xl.json --model_id Unbabel/XCOMET-XL