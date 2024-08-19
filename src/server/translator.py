import re
import time
import json
from functools import wraps
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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
    - cache_dir
    
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

class GPTTranslator(Translator):
    def prepare_model(self):
        from prompt_utils import prompt_simple
        self.prompt = prompt_simple
        
        from gpt_utils import gpt_completion
        self.call = gpt_completion
    
    def translate(self, text, src_lang, tgt_lang):
        prompt = self.prompt.format(
            text=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        return self.call(
            prompt = prompt,
            model=self.args.model,
            temperature = 0,
            max_tokens = 1024,
            openai_key_path = self.args.openai_key_path
        )


class TermAwareRefiner:
    def __init__(self, args, term_path):
        self.terms = json.load(open(term_path, 'r'))
        self.args = args
        
    def find_terminology(self, paragraph, terminology):
        def preprocess_text(text):
            lemmatizer = WordNetLemmatizer()
            tokens = word_tokenize(text.lower())
            return ' '.join([lemmatizer.lemmatize(token) for token in tokens])
        preprocessed_paragraph = preprocess_text(paragraph)
        preprocessed_terminology = preprocess_text(terminology)
        return fuzz.partial_ratio(preprocessed_terminology, preprocessed_paragraph) > 80  # Adjust threshold as needed
        

    def get_related_term_lst(self, result, tgt_lang):
        terms_dict = {}
        for term in self.terms:
            if self.find_terminology(result, term['English']):
                terms_dict[term['English']: term[tgt_lang]]
        return terms_dict
    
    def format_term_str(self, term_dict):
        return "- " + "- ".join([f"{key}: {val}" for key, val in term_dict.items()])

    def refine_translation(result, text, src_lang, tgt_lang):
        from prompt_utils import prompt_refine
        from gpt_utils import gpt_completion
        term_lst = self.get_related_term_lst(result, tgt_lang)
        term_str = self.format_term_str(term_lst)
        prompt = prompt_refine.format(
            text=text,
            result=result,
            term_str=term_str,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        return gpt_completion(
            prompt = prompt,
            model=self.args.model,
            temperature = 0,
            max_tokens = 1024,
            openai_key_path = self.args.openai_key_path
        )
        
        