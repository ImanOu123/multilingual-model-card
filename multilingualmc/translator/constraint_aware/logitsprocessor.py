from transformers import LogitsProcessor

class TerminologyAwareLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, en_text, lang, term_dict, soft_penalty):
        self.tokenizer = tokenizer
        self.en_text = en_text
        self.lang = lang
        self.soft_penalty = soft_penalty

    def get_all_terms(self, term):
        return terms
    
    def get_token_ids_of_terms(self, terms):
        tokens_set = []
        for term in terms:
            tokens = self.tokenizer.encode(term, add_special_tokens=False)
            tokens_set.extend(tokens)
        
        return list(set(tokens_set))
    
    def __call__(self, input_ids, scores, term_to_regenerate):
        # get current index
        # check if the current index is the start of some terms that need to regenerate (get feedbacks from previous runs)
        # if true, then adjust the token logits for m next tokns
        
        
        for idx in range(scores.shape[-1]):
            if idx in tokens:  # Penalize tokens not in the translation
                scores[:, idx] /= self.soft_penalty  # Apply soft penalty
        return scores