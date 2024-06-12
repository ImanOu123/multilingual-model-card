def get_prompt(text, src_lang, tgt_lang, prompt_version):
    prompt_str = None
    try:
        prompt_str = getattr(PromptTemplate, prompt_version)
    except AttributeError:
        print("The prompt you called doesn't exist in the PromptTemplate class")
        raise AttributeError
    
    prompt = prompt_str.format(
        text=text,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )
    return prompt


class PromptTemplate:
    
    simple = """Translate the below text from {src_lang} to {tgt_lang}. Provide the answer only.

```
{text}
```
"""

    cot = """Translate the below text from {src_lang} to {tgt_lang}. Pay attention to the translation of AI specific scientific terminologies. Provide the answer only.

```
{text}
```
"""

    term_translation = {
        "system_prompt": "You are a helpful assistant.",
        "prompt_with_explanation": """Translate the term "{term}" from English to {tgt_lang}. Provide the answer only. Here is an explanation of the term:

```
{explanation}
```""",
        "prompt_with_context": """Translate the term "{term}" from English to {tgt_lang}. Provide the answer only. Here is a sentence where the term appears:

```
{context}
```"""
    }