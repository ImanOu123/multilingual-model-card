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



# run_on_6060_prompt.py

python run_on_6060_prompt2.py --in_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless.jsonl --out_file /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/data_eval_6060/output/predictions_dev_seamless_prompt_gpt4omini.jsonl --term_file_path /home/jiaruil5/multilingual/multilingual-model-card/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/