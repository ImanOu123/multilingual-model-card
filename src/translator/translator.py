import re

class Translator:
    def __init__(self, args):
        import torch
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prepare_model()
        
    def prepare_model(self):
        """Prepare necessary configurations of using the translator model."""
        pass
    
    def translate(self):
        """Translate text from source language to target language."""
        pass
    
class SeamlessTranslator(Translator):
    """
    args: an instance of M4TLarge from config.py.
    - model_name: "facebook/hf-seamless-m4t-Large"
    - cache_dir: "/data/user_data/jiaruil5/.cache/"
    
    """
    def prepare_model(self):
        from transformers import AutoProcessor
        # from transformers import SeamlessM4TModel
        from transformers_customized.models.seamless_m4t.modeling_seamless_m4t import SeamlessM4TModel
        self.processor = AutoProcessor.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir, use_fast=False)
        self.model = SeamlessM4TModel.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir).to(self.device)

    def translate(self, text, src_lang, tgt_lang):
        text_inputs = self.processor(
            text = text,
            src_lang = self.args.lang_dict[src_lang],
            return_tensors="pt"
        )
        output_tokens = self.model.generate(
            **text_inputs.to(self.device),
            tgt_lang=self.args.lang_dict[tgt_lang],
            generate_speech=False
        )
        translated_text = self.processor.decode(
            output_tokens[0].tolist()[0],
            skip_special_tokens=True
        )
        return translated_text

class MaskedSeamlessTranslator(SeamlessTranslator):
    """
    Masked decoding Seamless Translator implementation
    """      
    def translate(self, text, src_lang, tgt_lang, tgt_list):
        from transformers import PhrasalConstraint
        constraints = [
            PhrasalConstraint(
                self.processor(
                    item,
                    src_lang=self.args.lang_dict[tgt_lang]
                ).input_ids
            ) for item in tgt_list
        ]
        text_inputs = self.processor(
            text = text,
            src_lang = self.args.lang_dict[src_lang],
            return_tensors="pt"
        )
        output_tokens = self.model.generate(
            **text_inputs.to(self.device),
            tgt_lang=self.args.lang_dict['Chinese'],
            generate_speech=False,
            constraints=constraints,
            num_beams=10,
            # num_return_sequences=1,
        )
        translated_text = self.processor.decode(
            output_tokens[0].tolist()[0],
            skip_special_tokens=True
        )
        return translated_text
            
class GoogleTranslator(Translator):
    def prepare_model(self):
        from googletrans import Translator as TLR
        self.model = TLR()
        
    def translate(self, text, src_lang, tgt_lang):
        translated_text = self.model.translate(
            text,
            dest=self.args.lang_dict[tgt_lang],
            src=self.args.lang_dict[src_lang]
        )
        return translated_text.text

class LLMTranslator(Translator):
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
    
    def translate(self, text, src_lang, tgt_lang, prompt_version: str):
        
        prompt = self.get_prompt(prompt_version).format(
            text=text,
            src_lang=self.args.lang_dict[src_lang],
            tgt_lang=self.args.lang_dict[tgt_lang]
        )
        
        res = self.call(
            prompt,
            self.llm_config_func,
            has_system_prompt = False,
            model_version = self.args.model_name
        )
        return res

class LLMTermTranslator(LLMTranslator):
    def __init__(self, args):
        """
        term_dict: {"english_term": <>, "arabic_term": <>, "chinese_term": <>, "french_term": <>, "japanese_term": <>, "russian_term": <>, "context": <>, "explanation": <>}
        """
        super().__init__(args)
        # if args.use_
        self.term_dict = {}
        
        
        

    

    
    def translate_terms(self, text, context, src_lang, tgt_lang, prompt_version: str):
        prompts = self.get_prompt(prompt_version)
        prompt = [prompts['system_prompt'], prompts['prompt_with_context']]
            
    # def translate_with_dict(self, text, src_lang, tgt_lang, prompt_version: str):
    
    # def translate_with_llm(self, text, src_lang, tgt_lang, prompt_version: str):
        
    