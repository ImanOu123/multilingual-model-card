model=${1:-"gpt4omini"}

if [[ $model == 'gpt4omini' ]]; then
    model_name=gpt-4o-mini
else
    model_name=${model}
fi


# python3 run_on_6060.py \
#     --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data.json \
#     --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/${model}.jsonl \
#     --model ${model_name}

# python3 run_on_6060_prompt.py \
#     --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/${model}.jsonl \
#     --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/${model}_prompt_gpt4omini.jsonl \
#     --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/ \
#     --model gpt-4o-mini

# python3 run_on_6060_hard_replace.py \
#     --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/${model}.jsonl \
#     --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/${model}_hard_replace.jsonl \
#     --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/


python3 run_on_6060_constraint_soft.py \
    --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data.json \
    --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/${model}_constraint_soft.jsonl \
    --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/ \
    --model ${model_name} \
    --method constraint_soft \
    --soft_penalty 0.9

python3 run_on_6060_constraint_soft.py \
    --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data.json \
    --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/${model}_constraint_soft.jsonl \
    --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/ \
    --model ${model_name} \
    --method constraint_soft \
    --soft_penalty 0.8

python3 run_on_6060_constraint_soft.py \
    --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data.json \
    --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/${model}_constraint_soft.jsonl \
    --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/ \
    --model ${model_name} \
    --method constraint_soft \
    --soft_penalty 0.7


## create gold data
# python3 run_on_6060_prompt.py \
#     --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/eval_data_google.jsonl \
#     --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dataset/output/eval_data_gold.jsonl \
#     --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/ \
#     --model gpt-4o