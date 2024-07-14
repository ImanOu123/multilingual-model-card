# source: https://huggingface.co/facebook/seamless-m4t-v2-large
seamless_lang_dict = {
    "Arabic": "arb",
    "Chinese": "cmn",
    "English": "eng",
    "French": "fra",
    "Japanese": "jpn",
    "Russian": "rus",
}

# source: https://py-googletrans.readthedocs.io/en/latest/
google_translator_lang_dict = {
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "English": "en",
    "French": "fr",
    "Japanese": "ja",
    "Russian": "ru",
}

llm_lang_dict = {
    "Arabic": "Arabic", # standard modern Arabic
    "Chinese": "Chinese", # simplified Chinese
    "English": "English",
    "French": "French",
    "Japanese": "Japanese",
    "Russian": "Russian",
}

# translator

class M4TLargeTranslatorConfig:
    model_name = "facebook/hf-seamless-m4t-Large"
    cache_dir = "/data/user_data/jiaruil5/.cache/"
    lang_dict = seamless_lang_dict
    
class GoogleTranslatorConfig:
    lang_dict = google_translator_lang_dict

class GPT35TranslatorConfig:
    model_name = "gpt-3.5-turbo"
    lang_dict = llm_lang_dict
    
class GPT4TranslatorConfig:
    model_name = "gpt-4o"
    lang_dict = llm_lang_dict

class LLAMA38BTranslatorConfig:
    model_name = "llama3_8b"
    lang_dict = llm_lang_dict
    
class LLAMA370BTranslatorConfig:
    model_name = "llama3_70b"
    lang_dict = llm_lang_dict
    
# term detector + translator

class GPT35TermTranslatorConfig:
    model_name = "gpt-3.5-turbo"
    lang_dict = llm_lang_dict

class LLAMA370BTermTranslatorConfig:
    model_name = "llama3_70b"
    lang_dict = llm_lang_dict

class LLAMA370BTermDictTranslatorConfig:
    from utils import get_json_list
    model_name = "llama3_70b"
    lang_dict = llm_lang_dict
    term_dict = get_json_list("../dictionary_collection/growing_dict/terms.jsonl")
    dict_is_growing = False
    
class LLAMA370BTermGrowingDictTranslatorConfig:
    from utils import get_json_list
    model_name = "llama3_70b"
    lang_dict = llm_lang_dict
    term_dict = get_json_list("../dictionary_collection/growing_dict/terms.jsonl")
    dict_is_growing = True