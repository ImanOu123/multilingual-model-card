import torch

class TermConfirmer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prepare_model()
        
    def prepare_model(self):
        pass
    
    def extract_terms(self):
        pass
    
class GPTTermConfirmer(TermConfirmer):
    def prepare_model(self):
        import sys

        sys.path.append("../")
        from models.config import get_model_config
        from models.llm import call
        self.config = get_model_config(self.args.model_name)
        def llm_config_func(llm):
            llm.temperature = 0.8
            llm.max_tokens = 4096
            return llm
        self.llm_config_func = llm_config_func
        self.call = call  
    
    def process_output(self, output):
        if output.strip().lower() == "true":
            return True
        elif output.strip().lower() == "false":
            return False
        else:
            raise Exception("ineligble output")
        
    def confirm_term(self, prompt):
        res = self.call(
            prompt,
            self.llm_config_func,
            has_system_prompt=True,
            model_version=self.args.model_name,
            verbose=True,
            **self.config
        )
        
        try:
            processed_res = self.process_output(res)
            return res, processed_res
        except Exception as e:
            print("Error in processing outputs:", e, "returned", res)
            return res, None 
    