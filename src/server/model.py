import json
from nltk import word_tokenize, sent_tokenize
from dataclasses import dataclass
from enum import Enum

class DocProcessor():
    def __init__(self, paper_path):
        self.doc = []
        self.doc.extend(
            self.read_doc(
                paper_path
            )
        )
    
    def format_str(self, doc):
        return [{"heading": d['heading'].replace(r"\{|\}", ""), "content": d['content'].replace(r"\{|\}", "")} for d in doc]
    
    def read_doc(self, doc_path):
        print(f"Loading the doc {doc_path}..")
        doc = None
        with open(doc_path, 'r') as file:
            doc = json.load(file)
            res = []
            # heading: xxx, content: xxx
            info = {
                'heading': None,
                'content': None,
                'source': 'paper',
                # 'code_blocks': 
            }
            
            title = doc.get("title", '')
            if title != '':
                info['heading'] = 'title'
                info['content'] = title
                res.append(info.copy())
            
            authors = doc.get("authors", '')
            if authors != '':
                info['heading'] = 'authors'
                info['content'] = authors
                res.append(info.copy())
            
            abstract = doc.get('abstract', '')
            if abstract != '':
                info['heading'] = 'abstract'
                info['content'] = abstract
                res.append(info.copy())
            
            for section in doc.get("sections", []):
                heading = section.get("heading", "")
                txt = section.get("text", "")
                if heading != "" and txt != "":
                    info['heading'] = heading
                    info['content'] = txt
                    res.append(info.copy())
            return self.format_str(res)

def split_paragraph(paragraph, max_words=64):
    # Step 1: Split the paragraph into sentences
    sentences = sent_tokenize(paragraph)
    
    # Step 2: Handle newline characters within sentences
    split_sentences = []
    for sentence in sentences:
        sub_sentences = re.split(r'(\n+)', sentence)
        split_sentences.extend(sub_sentences)
    
    chunks = []
    current_chunk = []

    def add_sentence_to_chunk(sentence, chunk):
        words = word_tokenize(sentence)
        if len(words) > max_words:
            # If the sentence itself is longer than max_words, split it further
            sub_chunks = split_long_sentence(sentence, max_words)
            chunk.extend(sub_chunks)
        else:
            chunk.append(sentence)
        return chunk

    def split_long_sentence(sentence, max_words):
        
        words = word_tokenize(sentence)
        sub_chunks = []
        current_sub_chunk = []
        for word in words:
            current_sub_chunk.append(word)
            if len(current_sub_chunk) >= max_words:
                sub_chunks.append(' '.join(current_sub_chunk))
                current_sub_chunk = []
        if current_sub_chunk:
            sub_chunks.append(' '.join(current_sub_chunk))
        return sub_chunks

    # Step 3: Join sentences into chunks
    for sentence in split_sentences:
        current_chunk = add_sentence_to_chunk(sentence, current_chunk)
        current_chunk_str = ' '.join(current_chunk)
        if len(word_tokenize(current_chunk_str)) > max_words:
            # Remove the last added sentence to keep the chunk under max_words
            removed_sentence = current_chunk.pop()
            chunks.append(' '.join(current_chunk))
            current_chunk = [removed_sentence]

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

class ModelMT(Enum):
    default = "seamless"
    seamless = "seamless"
    gpt_4o_mini = "gpt-4o-mini"
    googletrans = "googletrans"

class ModelRefine(Enum):
    default = "gpt-4o-mini"
    gpt_4o_mini = "gpt-4o-mini"

class Mode(Enum):
    default = "direct"
    direct = "direct"
    term_aware = "term_aware"

@dataclass
class Args:
    model_mt: ModelMT = ModelMT.default
    model_refine: ModelRefine = ModelRefine.default
    mode: Mode = Mode.default
    cache_dir: str = None
    openai_key_path: str = None
    term_path: str = "../dictionary_collection/growing_dict/mturk.json"

class Translator:
    def __init__(self, args: Args):
        self.args = args
        
        self.tgt_langs = [
            "Chinese",
            "Arabic",
            "French",
            "Russian",
            "Japanese"
        ]
        
        self.set_model()
        from translator import TermAwareRefiner
        class RefinerArgs:
            model = self.args.model_refine
            openai_key_path = self.args.openai_key_path
        refiner_args = RefinerArgs
        self.refiner = TermAwareRefiner(refiner_args, self.args.term_path)
    
    def set_model(self):
        if self.args.model_mt == ModelMT.seamless:
            from translator import SeamlessTranslator
            class Config:
                model_name = "facebook/hf-seamless-m4t-Large"
                cache_dir = self.args.cache_dir
                lang_dict = {
                    "Arabic": "arb",
                    "Chinese": "cmn",
                    "English": "eng",
                    "French": "fra",
                    "Japanese": "jpn",
                    "Russian": "rus",
                }
            config = Config
            self.model = SeamlessTranslator(config)
        elif self.args.model_mt == ModelMT.googletrans:
            from translator import GoogleTranslator
            from typing import Any
            import httpcore
            setattr(httpcore, 'SyncHTTPTransport', Any)
            class Config:
                lang_dict = {
                    "Arabic": "ar",
                    "Chinese": "zh-cn",
                    "English": "en",
                    "French": "fr",
                    "Japanese": "ja",
                    "Russian": "ru",
                }
            config = Config
            self.model = GoogleTranslator(config)
        elif "gpt" in str(self.args.model_mt).lower():
            from translator import GPTTranslator
            class Config:
                model_name = self.args.model_mt
                openai_key_path = self.args.openai_key_path
            config = Config
            self.model = GPTTranslator(config)
        
    def translate(self, text, src_lang, tgt_lang):
        result = self.model.translate(text, src_lang, tgt_lang)
        
        if self.args.mode == Mode.direct:
            return result
        elif self.args.mode == Mode.term_aware:
            return self.refiner.refine_translation(result, text, src_lang, tgt_lang)