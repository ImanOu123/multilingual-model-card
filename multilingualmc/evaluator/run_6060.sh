model=${1:-"gpt4omini"}

if [[ $model == 'gpt4omini' ]]; then
    model_name=gpt-4o-mini
else
    model_name=${model}
fi


python3 run_on_6060.py \
    --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/2/acl_6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt \
    --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}.jsonl \
    --model ${model_name}

python3 run_on_6060_prompt.py \
    --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}.jsonl \
    --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_${model}_prompt_gpt4omini.jsonl \
    --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/ \
    --model gpt-4o-mini