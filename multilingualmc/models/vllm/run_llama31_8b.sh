# llama3.1 8b chat hf
# CUDA_VISIBLE_DEVICES=0 bash run_llama31_8b.sh > logs/stdout_llama31_8b.txt 2> logs/stderr_llama31_8b.txt
MODEL_DIR="/data/user_data/jiaruil5/.cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=3637 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --tensor-parallel-size=1 \
    --max-num-batched-tokens=8192