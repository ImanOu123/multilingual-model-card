# llama31 70b instruct
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_llama31_70b.sh > logs/stdout_llama31_70b.txt 2> logs/stderr_llama31_70b.txt
MODEL_DIR=""
test -d "$MODEL_DIR"
python -O -u -m vllm.entrypoints.openai.api_server \
    --port=9570 \
    --model=$MODEL_DIR \
    --tokenizer=$MODEL_DIR \
    --chat-template "chat_templates/llama31.jinja" \
    --tensor-parallel-size=4 \
    --max-num-batched-tokens=8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 32

# sources: https://github.com/vllm-project/vllm/pull/2249