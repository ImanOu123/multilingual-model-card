import sys

from models.config import get_model_config
from models.llm import call

from llm_prompts import PromptTemplate

def translate_term(term, tgt_lang, context=None, explanation=None, model='gpt-4o'):
    def llm_config_func(llm):
        llm.temperature = 0
        llm.max_tokens = 4096
        return llm
    config = get_model_config(model)
    prompt = [PromptTemplate.term_translation['system_prompt']]
    if context is not None:
        prompt.append(PromptTemplate.term_translation['prompt_with_context'].format(
            term=term,
            tgt_lang=tgt_lang,
            context=context
        ))
    elif explanation is not None:
        prompt.append(PromptTemplate.term_translation['prompt_with_explanation'].format(
            term=term,
            tgt_lang=tgt_lang,
            explanation=explanation
        ))
    res = call(
        prompt,
        llm_config_func,
        has_system_prompt=True,
        model_version=model,
        verbose=False,
        **config
    )
    return res