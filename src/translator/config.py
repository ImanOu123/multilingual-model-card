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
    "French": "fr"
    "Japanese": "ja",
    "Russian": "ru",
}

class M4TLarge:
    model_name = "facebook/hf-seamless-m4t-Large"
    cache_dir = "/data/user_data/jiaruil5/.cache/"
    lang_dict = seamless_lang_dict
    
class GoogleTrans:
    lang_dict = google_translator_lang_dict