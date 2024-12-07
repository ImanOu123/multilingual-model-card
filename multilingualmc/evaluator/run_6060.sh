model=${1:-"gpt4omini"}

if [[ $model == 'gpt4omini' ]]; then
    model_name=gpt-4o-mini
else
    model_name=${model}
fi


# python3 run_on_6060.py \
#     --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/2/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt \
#     --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}.jsonl \
#     --model ${model_name}

python3 run_on_6060_prompt.py \
    --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}.jsonl \
    --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}_prompt_gpt4omini.jsonl \
    --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/ \
    --model gpt-4o-mini

python3 run_on_6060_hard_replace.py \
    --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}.jsonl \
    --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}_hard_replace.jsonl \
    --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/


## only for nllb and seamless
# python3 run_on_6060_constraint_soft.py \
#     --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/2/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt \
#     --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}_constraint_soft.jsonl \
#     --model ${model_name} \
#     --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/ \
#     --method constraint_soft \
#     --soft_penalty 0.8

# python3 run_on_6060_constraint_soft.py \
#     --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/2/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt \
#     --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}_constraint_soft.jsonl \
#     --model ${model_name} \
#     --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/ \
#     --method constraint_soft \
#     --soft_penalty 0.7

# python3 run_on_6060_constraint_soft.py \
#     --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/2/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt \
#     --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}_constraint_soft.jsonl \
#     --model ${model_name} \
#     --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/ \
#     --method constraint_soft \
#     --soft_penalty 0.6






## generate gold 
# python3 run_on_6060_prompt.py \
#     --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/gold_dev.jsonl \
#     --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/gold_dev_prompt_gpt4o.jsonl \
#     --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/ \
#     --model gpt-4o