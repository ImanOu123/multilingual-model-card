from transformers import AutoProcessor, SeamlessM4TModel
from nltk.tokenize import sent_tokenize
import torch
import json
import os
# from google.cloud import translate_v2 as translate
# from googletrans import Translator
import nltk
nltk.download('punkt')
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path_to_your_service_account.json"


class Translator:
    def __init__(self, method = "m4t"):
        if method == "m4t":
            model_name = "facebook/hf-seamless-m4t-Large"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
            self.model = SeamlessM4TModel.from_pretrained(model_name).to(self.device)
            # self.translate_client = Translator()
            self.tag_lang_map_m4t = {"Chinese": "cmn", "English": "eng", "French": "fra", "German": "deu", "Italian": "ita", "Japanese": "jpn", "Korean": "kor", "Portuguese": "por", "Russian": "rus", "Spanish": "spa", "Arabic":"arb"}
            self.tag_lang_map_google = {"Chinese": "zh", "English": "en", "French": "fr", "German": "de", "Italian": "it", "Japanese": "ja", "Korean": "ko", "Portuguese": "pt", "Russian": "ru", "Spanish": "es", "Arabic":"ar"}
            
    def translate_term_m4t(self, src_dict, full_src_lang, full_tag_lang):
        src_lang = self.tag_lang_map_m4t.get(full_src_lang)
        tag_lang = self.tag_lang_map_m4t.get(full_tag_lang)
        src_lang_dicts = src_dict[full_tag_lang]
        for term in src_lang_dicts.keys():
            text_inputs = self.processor(text=term, src_lang=src_lang, return_tensors="pt")
            output_tokens = self.model.generate(**text_inputs, tgt_lang=tag_lang, generate_speech=False)
            translated_text = self.processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
            src_dict[full_tag_lang][term] = translated_text
    
    def translate_term_google(self, src_dict, full_src_lang, full_tag_lang):
        src_lang = self.tag_lang_map_google.get(full_src_lang)
        tag_lang = self.tag_lang_map_google.get(full_tag_lang)
        src_lang_dicts = src_dict[full_tag_lang]
        for term in src_lang_dicts.keys():
                translated_text = self.google_translate(term, src_lang, tag_lang)
                src_dict[full_tag_lang][term] = translated_text
                
    def google_translate(self, text, src_lang, target_lang):
        result = self.translate_client.translate(text, source_language=src_lang, target_language=target_lang)
        return result["translatedText"]

    @staticmethod
    def read_json_file(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def write_file_to_json(content, fp):
        with open(fp, "w", encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)

            
# def arg_parse():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument("--source_text", type=str) # source text directory
#     parser.add_argument("--file_path", type=str) # source dir directory
#     parser.add_argument("--target_dir", type=str) # target directory
#     parser.add_argument("--source_lang", type=str) # source language
#     parser.add_argument("--target_lang", type=str) # target language
#     parser.add_argument("--error_file", type=str) # error logging
#     parser.add_argument("--start_index", type=int, default=0) # start index
#     parser.add_argument("--end_index", type=int) # end index
#     args = parser.parse_args() # exclusive end index
#     return args

# python translation_pipeline_several_model.py --source_dir /home/wenkail/model-card/data/dataset_cards_text/ --target_dir /home/wenkail/model-card/data/dataset_translated_text --source_lang eng --target_lang cmn --error_file /home/wenkail/model-card/tools/translator/translate_error.txt
# if __name__ == "__main__":
#     args = arg_parse()
#     # files = read_json_file(args.source_text)
#     files = read_json_file(args.file_path)
#     translate_write_json(files, args.target_dir, args.source_lang, args.target_lang, args.start_index, args.end_index)





# def translate_write_json(files, target_dir, src_lang, tgt_lang, start_index, end_index):
    
#     for file in tqdm(files[start_index:end_index], desc="Translating files"):
#         for file_name, file_content in file.items():
#             for sec in file_content.keys():
#                 if sec == "abstract" or sec == "title":
#                     file_content[sec] = translate_text_to_text(text = file_content[sec], src_lang= src_lang, tgt_lang=tgt_lang)
#                     continue
#                 if sec == "figures":
#                     for fig in file_content[sec]:
#                         fig['figure_caption'] = translate_text_to_text(text = fig["figure_caption"], src_lang= src_lang, tgt_lang=tgt_lang)
#                     continue
#                 if sec == "sections":
#                     for sub_sec in file_content[sec]:
#                         sub_sec['heading'] = translate_text_to_text(text = sub_sec["heading"], src_lang= src_lang, tgt_lang=tgt_lang)
#                         sub_sec['text'] = translate_text_to_text(text = sub_sec["text"], src_lang= src_lang, tgt_lang=tgt_lang)
#                     continue
#             # write_file_to_json(file_content, os.path.join(target_dir, file_name.replace(".json", "_translated.json")))
#             write_file_to_json(file_content, os.path.join(target_dir, file_name))
#     return files