def get_prompt(text, prompt_version = "simple"):
    prompt_str = None
    try:
        prompt_str = getattr(PromptTemplate, prompt_version)
    except AttributeError:
        print("The prompt you called doesn't exist in the PromptTemplate class")
        raise AttributeError
    
    prompt = prompt_str.format(
        text=text,
    )
    return prompt


class PromptTemplate:
    
    simple = """You are a language expert with a focus on machine translation. Your task is to analyze the following model card and identify a scientific jargon or terminology within the field of machine learning that might be challenging for machine translation software to accurately translate to Chinese, Russian, Japanese, French, and Arabic.

    Respond with only one term. If you don't find any scientific jargon or terminology, respond with "None".

    The following text is:
    '''{text}'''
    """


    term_list =  """You are a language expert with a focus on machine translation. Your task is to analyze the following model card and identify scientific jargon or terminology within the field of machine learning that might be challenging for machine translation software to accurately translate to Chinese, Russian, Japanese, French, and Arabic.

            Respond in the following format (as a list): [Term 1, Term 2, Term 3, ...] Respond only as shown, with no additional discursive or explanatory text. 
            An example of a response is: [artificial intelligence, machine learning]. 
            If you don't find any scientific jargon or terminology respond as follows: [],
            
        The following text is:
        '''{text}'''
        """


#     cot = """Translate the below text from {src_lang} to {tgt_lang}. Pay attention to the translation of AI specific scientific terminologies. Provide the answer only.

# ```
# {text}
# ```
# """