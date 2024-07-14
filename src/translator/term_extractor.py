import re
import torch


class TermExtractor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prepare_model()
        
    def prepare_model(self):
        """Prepare necessary configurations of using the translator model."""
        pass
    
    def extract_terms(self):
        """Extract terms."""
        pass
    
class LLAMATermExtractor(TermExtractor):
    def prepare_model(self):
        import sys
        from llm_prompts import get_prompt
        self.get_prompt = get_prompt

        sys.path.append("../")
        from models.config import get_model_config
        from models.llm import call
        self.config = get_model_config(self.args.model_name)
        def llm_config_func(llm):
            llm.temperature = 0
            llm.max_tokens = 4096
            return llm
        self.llm_config_func = llm_config_func
        self.call = call  
    
    def extract_terms(self, text, prompt_version: str):
        """
        prompt_version:
        - term_extraction_llama3
        """
        
        prompt_info = self.get_prompt(prompt_version)
        prompt = [prompt_info['system_prompt'], prompt_info['prompt'].format(text=text)]
        process_outputs_func = prompt_info['process_outputs_func']
        
        res = ""
        try:
            res = self.call(
                prompt,
                self.llm_config_func,
                has_system_prompt = True,
                model_version = self.args.model_name,
                verbose=True,
                **self.config
            )
            processed_res = process_outputs_func(res)
            return res, processed_res
        except Exception as e:
            print("Error in generating and processing outputs:", e)
            return res, []
