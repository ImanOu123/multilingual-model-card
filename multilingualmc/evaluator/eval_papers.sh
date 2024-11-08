model=${1:-"gpt4omini"}

python3 eval_on_6060_comet_gold.py \
    --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/${model}.jsonl \
    --gt_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data_gold.jsonl \
    --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval/${model}_comet.json \
    --model_id Unbabel/wmt22-comet-da

python3 eval_on_6060_gold.py \
    --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/${model}.jsonl \
    --gt_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data_gold.jsonl \
    --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval/${model}.csv