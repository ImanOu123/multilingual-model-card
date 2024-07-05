import random

def get_model_config(model):
    if 'gpt' in model:
        api_key = None
        # org_id = random.sample([0, 1], 1)
        org_id = 1
        model_path = None
    elif model == 'llama3_8b':
        api_key = "EMPTY"
        org_id = "http://127.0.0.1:3636/v1"
        model_path = "/data/user_data/jiaruil5/.cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45/"
    elif model == 'llama3_70b':
        api_key = "EMPTY"
        org_id = "http://127.0.0.1:9570/v1"
        model_path = "/compute/babel-1-31/jiaruil5/.cache/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/7129260dd854a80eb10ace5f61c20324b472b31c/"
    elif model == 'llama2_7b':
        api_key = "EMPTY"
        org_id = "http://127.0.0.1:2525/v1"
        model_path = "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf"
    return {
        'api_key': api_key, 
        'org_id': org_id, 
        'model_path': model_path
    }