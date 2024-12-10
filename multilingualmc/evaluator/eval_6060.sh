model=${1:-"gpt4omini"}

python3 eval_on_6060_comet_gold.py \
    --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}.jsonl \
    --gt_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/gold_dev_prompt_gpt4o.jsonl \
    --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_${model}_comet.json \
    --model_id Unbabel/wmt22-comet-da

python3 eval_on_6060_gold.py \
    --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}.jsonl \
    --gt_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/gold_dev_prompt_gpt4o.jsonl \
    --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/eval/gold_predictions_dev_${model}.csv