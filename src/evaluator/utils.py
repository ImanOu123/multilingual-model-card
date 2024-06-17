import evaluate
import torch
import json
import rich

class GeneralEvaluator():
    """
    args: an instance of General Evluator
    -- prediction: a list of model's output
    -- reference: a list of groundtrue sentences
    """
    def __init__(self, args):
        import evaluate
        self.args = args
        
    def metrics(self):
        self.metric = evaluate.load(self.args.metric)

    def evaluate(self):
        self.metric.compute(predictions=self.args.predictions, references=self.args.references)

class ScientificTermDiscoveryEvaluator():
    """
    args: an instance of Scientific Term Discovery Evluator
    -- prediction: a list of model's output
    -- reference: a list of groundtrue sentences
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prepare_model()

    def prepare_model(self):
        import sys
        sys.path.append('../')
        
        from models.config import get_model_config
        from models.llm import call
        self.config = get_model_config(self.args.model_name)
        def llm_config_func(llm):
            llm.temperature = 0
            llm.max_tokens = 2048
            return llm
        # self.get_prompt = get_prompt
        self.llm_config_func = llm_config_func
        self.call = call
    
    def discovery(self, text, prompt_version: str):
        from evaluation_prompt import get_prompt

        prompt = get_prompt(
            text,
            prompt_version
        )
        # rich.print(f"Prompt: {prompt}")
        res = self.call(
            prompt,
            self.llm_config_func,
            has_system_prompt = True,
            model_version = self.args.model_name,
            verbose=False,
            **self.config
        )
        # print("The following context would be the output of scientic terms list: ")
        # rich.print(res)
        return res

