import re

######## Term extraction process_outputs_func

def process_outputs_comma(res):
    """
    Seperated by comma
    """
    res = res.strip()
    if res == "None":
        return []
    if "Here are the identified AI scientific terms:" in res:
        res = res.split("Here are the identified AI scientific terms:")[-1].strip()
    if "\n" in res or "here are" in res.lower() or "here is" in res.lower():
        return []

    terms = re.split(r"\s*,\s*", res)
    return terms

######### Prompts

def get_prompt(prompt_version):
    prompt_str = None
    try:
        prompt_str = getattr(PromptTemplate, prompt_version)
    except AttributeError:
        print("The prompt you called doesn't exist in the PromptTemplate class")
        raise AttributeError
    
    return prompt_str


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

    term_extraction_llama3 = {
        "system_prompt": "You are a helpful assistant.",
        "prompt": """Your task is to identify AI scientific terminology from the paragraph below, based on the provided definition:

Here is the definition of AI scientific terminology:
```
AI scientific terminology refers to specialized nouns or noun phrases within Artificial Intelligence, encompassing essential concepts, methods, models, algorithms, and systems. These terms must:
1. Be composed of nouns, adjectives, and occasionally prepositions.
2. Be context-specific to AI, having either no meaning or a different meaning outside this field.
Additionally, these terms often pose significant challenges for accurate translation by machine translation models due to their technical specificity.
```

Provide only the identified terms in your response, separated by commas. If no scientific terms are found, respond with "None" only. Example: transformer, batch normalization, embedding.

Here is the paragraph:
```
{text}
```
""",
        "process_outputs_func": process_outputs_comma
    }
    
    term_verification_claude3 = {
        "system_prompt": "You are a helpful assistant.",
        "prompt": """"""
    }
    
    
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
    
    
    
