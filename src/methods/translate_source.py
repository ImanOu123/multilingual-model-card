import tqdm
import json
import os
from utils import split_paragraph

import sys
sys.path.append("../")

class TranslateSource:
    def __init__(self, in_dir, out_dir, source: str):
        """
        - source: either "model_card" or "paper"
        - in_dir: "../data_model_cards/original/claude3_json/"
        - out_dir: "../data_model_cards/translated/googletrans_claude3_json/"
        """
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.source = source
        self.translator = None
        self.translation_mode = None
        
        self.tgt_langs = [
            "Chinese",
            "Arabic",
            "French",
            "Russian",
            "Japanese"
        ]
        
        if self.source == 'model_card':
            self.translate = self.translate_model_card
        elif self.source == 'paper':
            self.translate = self.translate_paper
    
    def load_translator(self, translation_mode: str):
        """
        translator: select one from ['seamless', 'googletrans', 'googletrans_dict_gpt', 'googletrans_gpt', 'googletrans_paragraph']
        """
        self.translation_mode = translation_mode
        if translation_mode == 'seamless':
            from translator.config import M4TLargeTranslatorConfig
            from translator.translator import SeamlessTranslator
            args = M4TLargeTranslatorConfig
            self.translator = SeamlessTranslator(args)
        elif translation_mode in ['googletrans', 'googletrans_paragraph']:
            from translator.config import GoogleTranslatorConfig
            from translator.translator import GoogleTranslator
            args = GoogleTranslatorConfig
            self.translator = GoogleTranslator(args)
        elif translation_mode == 'googlatrans_gpt':
            from translator.config import LLAMA370BTermTranslatorConfig
            from translator.translator import LLMTermTranslator
            args = LLAMA370BTermTranslatorConfig
            self.translator = LLMTermTranslator(args)
        elif translation_mode == 'googletrans_growing_dict_gpt':
            from translator.config import LLAMA370BTermGrowingDictTranslatorConfig
            from translator.translator import LLMTermTranslator
            args = LLAMA370BTermGrowingDictTranslatorConfig
            self.translator = LLMTermTranslator(args)
        elif translation_mode == 'googletrans_dict_gpt':
            from translator.config import LLAMA370BTermDictTranslatorConfig
            from translator.translator import LLMTermTranslator
            args = LLAMA370BTermDictTranslatorConfig
            self.translator = LLMTermTranslator(args)
        else:
            raise NotImplementedError
    
    def _translate_chunks(self, in_text, tgt_lang):
        
        if self.translation_mode in ['googletrans', 'seamless']:
            chunks = split_paragraph(in_text)
            answer_chunks = []
            for chunk in chunks:
                answer_chunk = self.translator.translate(
                    chunk,
                    src_lang='English',
                    tgt_lang=tgt_lang
                )
                answer_chunks.append(answer_chunk)
            out_text = " ".join(answer_chunks)
            return out_text
        elif self.translation_mode == 'googletrans_paragraph':
            return self.translator.translate(
                in_text,
                src_lang='English',
                tgt_lang=tgt_lang
            )
        elif self.translation_mode == 'googletrans_gpt':
            chunks = split_paragraph(in_text)
            answer_chunks = []
            for chunk in chunks:
                answer_chunk = self.translator.translate(
                    chunk,
                    src_lang='English',
                    tgt_lang=tgt_lang
                )
                answer_chunks.append(answer_chunk)
            out_text = " ".join(answer_chunks)
            return out_text
        elif self.translation_mode == 'googletrans_dict_gpt':
            pass
        else:
            raise NotImplementedError
    
    def _translate_langs(self, in_text, tgt_langs):
        pass
    
    def translate_model_card(self):
        for file in tqdm(os.listdir(self.in_dir)):
            print(f"Start translating {file}..")
            filepath = self.in_dir + file
            json_list = json.load(open(filepath, 'r'))
            new_json_list = []
            for idx, item in tqdm(enumerate(json_list)):
                print(f"Reach item {idx}..")
                new_item = item.copy()
                for tgt_lang in tqdm(self.tgt_langs):
                    new_item[f"answer_{tgt_lang}"] = self._translate_chunks(item['answer'], tgt_lang)
                new_json_list.append(new_item)
            out_filepath = self.out_dir + file
            json.dump(
                new_json_list,
                open(out_filepath, 'w'),
                ensure_ascii=False,
                indent=2
            )
    
    def translate_paper(self, translator_func):
        # for file in tqdm(os.listdir(self.in_dir)):
        #     print(f"Start translating {file}..")
        #     filepath = self.in_dir + file
        raise NotImplementedError
    
    def translate(self):
        pass