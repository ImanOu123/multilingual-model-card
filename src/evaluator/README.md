https://huggingface.co/spaces/evaluate-metric/bleu

https://huggingface.co/spaces/evaluate-metric/comet

https://huggingface.co/spaces/evaluate-metric/chrf



# eval_on_6060

python3 eval_on_6060.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_gpt35turbo.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_gpt35turbo.jsonl

python3 eval_on_6060.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama3_8b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_llama3_8b.jsonl

python3 eval_on_6060.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama3_70b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_llama3_70b.jsonl

python3 eval_on_6060.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama31_8b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_llama31_8b.jsonl

python3 eval_on_6060.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama31_70b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_llama31_70b.jsonl

python3 eval_on_6060.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_qwen2_7b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_qwen2_7b.jsonl

## default comet
python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_gpt4omini_comet.json

python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_gpt35turbo.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_gpt35turbo_comet.json

python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama3_8b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_llama3_8b_comet.json

python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama3_70b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_llama3_70b_comet.json

python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_qwen2_7b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_qwen2_7b_comet.json


python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama31_8b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_llama31_8b_comet.json

python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama31_70b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_llama31_70b_comet.json

## reference free comet

python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_gpt4omini.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_gpt4omini_comet_noref_xl.json --model_id Unbabel/wmt23-cometkiwi-da-xl

python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_gpt35turbo.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_gpt35turbo_comet_noref_xl.json --model_id Unbabel/wmt23-cometkiwi-da-xl

python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama3_8b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_llama3_8b_comet_noref_xl.json --model_id Unbabel/wmt23-cometkiwi-da-xl

python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama3_70b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_llama3_70b_comet_noref_xl.json --model_id Unbabel/wmt23-cometkiwi-da-xl

python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_qwen2_7b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_qwen2_7b_comet_noref_xl.json --model_id Unbabel/wmt23-cometkiwi-da-xl


python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama31_8b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_llama31_8b_comet_noref_xl.json --model_id Unbabel/wmt23-cometkiwi-da-xl

python3 eval_on_6060_comet.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/output/predictions_dev_llama31_70b.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/src/data_eval_6060/eval/predictions_dev_llama31_70b_comet_noref_xl.json --model_id Unbabel/wmt23-cometkiwi-da-xl