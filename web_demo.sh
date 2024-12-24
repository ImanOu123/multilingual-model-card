# !/bin/bash

python3 -m multilingualmc.evaluator.run_on_6060 \
        --in_file "/home/iouzzani/research/multilingual_model_cards/multilingualmc/web_demo/2024.naacl-long.97_abs.txt" \
        --out_file "/home/iouzzani/research/multilingual_model_cards/multilingualmc/web_demo/2024.naacl-long.97_abs_seamless.jsonl" \
        --model "seamless"     

python3 -m multilingualmc.evaluator.run_on_6060_prompt \
        --in_file "/home/iouzzani/research/multilingual_model_cards/multilingualmc/web_demo/2024.naacl-long.97_abs_seamless.jsonl" \
        --out_file "/home/iouzzani/research/multilingual_model_cards/multilingualmc/web_demo/2024.naacl-long.97_abs.jsonl" \
        --term_file_path "/home/iouzzani/research/multilingual_model_cards/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/"