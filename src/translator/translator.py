import re
import time
import transformers
import torch
from transformers import PhrasalConstraint

class Translator:
    def __init__(self, args):
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
        try:
            from transformers_customized.models.seamless_m4t.modeling_seamless_m4t import SeamlessM4TModel
        except:
            from translator.transformers_customized.models.seamless_m4t.modeling_seamless_m4t import SeamlessM4TModel
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
    
    def translate_cbs(self, text, src_lang, tgt_lang, force_words):
        text_inputs = self.processor(
            text = text,
            src_lang = self.args.lang_dict[src_lang],
            return_tensors="pt"
        )
        
        print(force_words)
        constraints = [
            PhrasalConstraint(self.processor(
                force_word,
                src_lang = tgt_lang,
                add_special_tokens=False
            ).input_ids) for force_word in force_words
        ]
        if len(constraints) == 0:
            constraints = None
        
        output_tokens = self.model.generate(
            **text_inputs.to(self.device),
            tgt_lang=self.args.lang_dict[tgt_lang],
            generate_speech=False,
            constraints=constraints,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
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
        max_tries = 5
        curr_tries = 0
        while True:
            try:
                translated_text = self.model.translate(
                    text,
                    dest=self.args.lang_dict[tgt_lang],
                    src=self.args.lang_dict[src_lang]
                )
                return translated_text.text
            except Exception as e:
                print(e)
                if curr_tries < max_tries:
                    curr_tries += 1
                    time.sleep(5)
                else:
                    return None

class LLAMATranslator(Translator):
    def prepare_model(self):
        from translator.llm_prompts import get_prompt
        self.get_prompt = get_prompt
        
        model_id = self.args.model_name
        self.model_id = model_id
        
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map='auto'
        )
        
    def translate(self, text, src_lang, tgt_lang, prompt_version: str = ""):
        
        prompt = self.get_prompt(prompt_version).format(
            text=text,
            src_lang=self.args.lang_dict[src_lang],
            tgt_lang=self.args.lang_dict[tgt_lang]
        )
        
        kwargs = {}
        if "llama-3-" in self.model_id:
            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            kwargs['eos_token_id'] = terminators
        
        print(prompt)
        res = self.pipeline(
            [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ],
            max_new_tokens=4096,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            **kwargs,
        )[0]['generated_text'][-1]['content']
        
        print(res)
        return res

class QWENTranslator(Translator):
    def prepare_model(self):
        from translator.llm_prompts import get_prompt
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.get_prompt = get_prompt
        self.device = "cuda"
        
        model_id = self.args.model_name
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    def translate(self, text, src_lang, tgt_lang, prompt_version: str = ""):
        prompt = self.get_prompt(prompt_version).format(
            text=text,
            src_lang=self.args.lang_dict[src_lang],
            tgt_lang=self.args.lang_dict[tgt_lang]
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        templated_messages = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        model_inputs = self.tokenizer([templated_messages], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response

    def translate_cbs(self, text, src_lang, tgt_lang, force_words, prompt_version: str = ""):
        prompt = self.get_prompt(prompt_version).format(
            text=text,
            src_lang=self.args.lang_dict[src_lang],
            tgt_lang=self.args.lang_dict[tgt_lang]
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        templated_messages = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        model_inputs = self.tokenizer([templated_messages], return_tensors="pt").to(self.device)
        
        print(force_words)
        constraints = [
            PhrasalConstraint(self.tokenizer(
                force_word,
                add_special_tokens=False
            ).input_ids) for force_word in force_words
        ]
        if len(constraints) == 0:
            constraints = None
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            constraints=constraints,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response

class LLMTranslator(Translator):
    def prepare_model(self):
        import sys
        from translator.llm_prompts import get_prompt
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
    
    def translate(self, text, src_lang, tgt_lang, prompt_version: str = ""):
        
        prompt = self.get_prompt(prompt_version).format(
            text=text,
            src_lang=self.args.lang_dict[src_lang],
            tgt_lang=self.args.lang_dict[tgt_lang]
        )
        
        res = self.call(
            [prompt],
            self.llm_config_func,
            has_system_prompt = False,
            model_version = self.args.model_name,
            verbose = True,
            api_key = self.config['api_key'] if 'gpt' not in self.args.model_name else None,
            org_id = self.config['org_id'] if 'gpt' not in self.args.model_name else None,
            model_path = self.config['model_path'] if 'gpt' not in self.args.model_name else None,
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
        
        
        

    

    
    def translate_terms(self, text, context, src_lang, tgt_lang, prompt_version: str = ""):
        prompts = self.get_prompt(prompt_version)
        prompt = [prompts['system_prompt'], prompts['prompt_with_context']]
            
    # def translate_with_dict(self, text, src_lang, tgt_lang, prompt_version: str):
    
    # def translate_with_llm(self, text, src_lang, tgt_lang, prompt_version: str):
        
    