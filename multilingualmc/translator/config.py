def get_json_list(path):
    import json
    f = open(path, 'r')
    info = []
    for line in f.readlines():
        info.append(json.loads(line))
    return info



# source: https://huggingface.co/facebook/seamless-m4t-v2-large
seamless_lang_dict = {
    "Arabic": "arb",
    "Chinese": "cmn",
    "English": "eng",
    "French": "fra",
    "Japanese": "jpn",
    "Russian": "rus",
}

# source: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
nllb_lang_dict = {
    "Arabic": "arb_Arab",
    "Chinese": "zho_Hans",
    "English": "eng_Latn",
    "French": "fra_Latn",
    "Japanese": "jpn_Jpan",
    "Russian": "rus_Cyrl"
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

llama_lang_dict = {
    "Arabic": "Arabic", # standard modern Arabic
    "Chinese": "Simplified Chinese", # simplified Chinese
    "English": "English",
    "French": "French",
    "Japanese": "Japanese",
    "Russian": "Russian",
}

# translator

class M4TLargeTranslatorConfig:
    model_name = "facebook/hf-seamless-m4t-Large"
    cache_dir = "/home/iouzzani/.cache/"
    lang_dict = seamless_lang_dict

class NLLBTranslatorConfig:
    model_name = "facebook/nllb-200-3.3B"
    cache_dir = "/home/iouzzani/.cache/"
    lang_dict = nllb_lang_dict

class AyaTranslatorConfig:
    model_name = "CohereForAI/aya-expanse-8b"
    cache_dir = "/home/iouzzani/.cache/"
    lang_dict = llm_lang_dict

class GoogleTranslatorConfig:
    lang_dict = google_translator_lang_dict

class GPT35TranslatorConfig:
    model_name = "gpt-3.5-turbo"
    lang_dict = llm_lang_dict
    
class GPT4TranslatorConfig:
    model_name = "gpt-4o"
    lang_dict = llm_lang_dict

class GPTTranslatorConfig:
    lang_dict = llm_lang_dict

class LLAMA3TranslatorConfig:
    lang_dict = llama_lang_dict

class QWENTranslatorConfig:
    lang_dict = llama_lang_dict

# term detector + translator

class GPT35TermTranslatorConfig:
    model_name = "gpt-3.5-turbo"
    lang_dict = llm_lang_dict

class LLAMA370BTermTranslatorConfig:
    model_name = "llama3_70b"
    lang_dict = llm_lang_dict

class LLAMA370BTermDictTranslatorConfig:
    model_name = "llama3_70b"
    lang_dict = llm_lang_dict
    # term_dict = get_json_list("/home/jiaruil5/multilingual/multilingual-model-card/src/dictionary_collection/growing_dict/terms.jsonl")
    dict_is_growing = False
    
class LLAMA370BTermGrowingDictTranslatorConfig:
    model_name = "llama3_70b"
    lang_dict = llm_lang_dict
    # term_dict = get_json_list("/home/jiaruil5/multilingual/multilingual-model-card/src/dictionary_collection/growing_dict/terms.jsonl")
    dict_is_growing = True