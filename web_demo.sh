# !/bin/bash
# SBATCH --job-name=myjob
# SBATCH --partition=general
# SBATCH --output=myjob.out
# SBATCH --error=myjob.err
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=1
# SBATCH --gres=gpu:A6000:2
# SBATCH --time=1-00:00:00
# SBATCH --mem=40G

python3 -m multilingualmc.evaluator.run_on_6060 \
        --in_file "/home/iouzzani/research/multilingual_model_cards/multilingualmc/web_demo/2024.naacl-long.97_abs.txt" \
        --out_file "/home/iouzzani/research/multilingual_model_cards/multilingualmc/web_demo/2024.naacl-long.97_abs_seamless.jsonl" \
        --model "seamless"     

python3 -m multilingualmc.evaluator.run_on_6060_prompt \
        --in_file "/home/iouzzani/research/multilingual_model_cards/multilingualmc/web_demo/2024.naacl-long.97_abs_seamless.jsonl" \
        --out_file "/home/iouzzani/research/multilingual_model_cards/multilingualmc/web_demo/2024.naacl-long.97_abs.jsonl" \
        --term_file_path "/home/iouzzani/research/multilingual_model_cards/multilingualmc/dictionary_collection/mturk/analysis/annotation_final/"