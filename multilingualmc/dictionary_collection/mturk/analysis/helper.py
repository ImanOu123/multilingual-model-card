import json
import pandas as pd
from tqdm import tqdm
from typing import Any
import httpcore
setattr(httpcore, 'SyncHTTPTransport', Any)

class GoogleTranslate:
    def __init__(self):
        import sys
        sys.path.append("../../../")
        from translator.config import GoogleTranslatorConfig
        from translator.translator import GoogleTranslator
        args = GoogleTranslatorConfig
        self.model = GoogleTranslator(args)
