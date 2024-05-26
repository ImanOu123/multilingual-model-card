

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
        from transformers import AutoProcessor, SeamlessM4TModel
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
        translated_text = self.model.translate(
            text,
            dest=self.args.lang_dict[tgt_lang],
            src=self.args.lang_dict[src_lang]
        )