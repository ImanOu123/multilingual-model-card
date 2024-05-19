# Summary

This codebase supports the calling of OpenAI chat models and LLAMA3 instruct models. It uses the VLLM Python package and its implementation of the [OpenAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) to call LLAMA3.

# Usage

1. Install the necessary Python packages

```bash
# (Recommended) Create a new conda environment.
conda create -n myenv python=3.9 -y
conda activate myenv

# Install vLLM
pip install vllm==0.3.3

# Install LangChain
pip install langchain==0.0.327
```

Reference: https://docs.vllm.ai/en/latest/getting_started/installation.html

2. If you use OpenAI models:

    1. Copy the OpenAI API key and Organization ID to your local `~/.bashrc` file:
        ```bash
        export openai_api_key="<api_key>"
        export openai_api_org_1="<api_org>"
        ```
    2. Refresh the environment variable configuration
        ```bash
        source ~/.bashrc
        ```
    3. Done! Since here we specify `openai_api_org_1` as the organization id, remember to modify the code block in `config.py` to reflect the corresponding configuration:
        ```python
        if 'gpt' in model:
            api_key = None
            org_id = 1
            model_path = None
        ```
    4. Start testing! In `test_inference.ipynb`, comment the configuration of other models and uncomment the configuration of gpt3.5, for example:
        ```python
        model = 'gpt-3.5-turbo'
        def llm_config_func(llm):
            llm.temperature = 0.8
            llm.max_tokens = 4096
            return llm
        config = get_model_config(model)
        ```

3. If you use LLAMA3 8B Inst:

    1. Download the model checkpoint from HuggingFace to a local folder, following `vllm/download_models.py`. Remember to specify `model_name` and `cache_dir`.
    2. If you are using LLAMA3 8B Inst model and you have one GPU available, run the commented line in `run_llama3_8b.sh` after specifying 1) the `MODEL_DIR` to be the checkpoint snapshot folder and 2) the port number.
    3. In `config.py`, modify the variables `org_id` and `model_path` for the specified model. Note that `model_path` is the same as the `MODEL_DIR` in the previous step.
    4. Start testing! In `test_inference.ipynb`, comment the configuration of other models and uncomment the configuration of `llama3_8b`:
        ```python
        model = 'llama3_8b'
        def llm_config_func(llm):
            llm.temperature = 0
            llm.max_tokens = 4096
            return llm
        config = get_model_config(model)
        ```