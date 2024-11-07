Dictionary collection on the dev set:
- `extract_terms_llama3.py`: Prompting LLAMA3 70B to extract terms from the whole dev set

Ground truth on the test set:
1. `googletrans_paragraph`: Google Translate API, per paragraph

Baselines on the test set:
1. `googletrans_gpt_paragraph`: Google Translate API + GPT for extracting terms + GPT for updating the overall translation, per paragraph
2. `seamless_paragraph`: Google Translate API + GPT for extracting terms + GPT for updating the overall translation, per paragraph
3. `seamless_gpt_paragraph`: Google Translate API + GPT for extracting terms + GPT for updating the overall translation, per paragraph

Our method: 
1. `googletrans_dict_gpt_paragraph`: Google Translate API + searching terms in dictionary + GPT for updating the overall translation, per paragraph
2. `seamless_dict_gpt_paragraph`: Google Translate API + searching terms in dictionary + GPT for updating the overall translation, per paragraph


googletrans_paragraph: md-1




(Previous?) Baselines on the test set:
2. `googletrans`: Google Translate API, per sub-paragraph
3. `googletrans_gpt`: Google Translate API + GPT for extracting terms + GPT for updating the overall translation, per sub-paragraph
4. `googletrans_dict_gpt`: Google Translate API + searching terms in dictionary + GPT for updating the overall translation, per sub-paragraph
5. `seamless`: Seamless Large, per sub-paragraph

"per sub-paragraph" here means per chunk of 64 words in maximum.

If a paragraph has 500 words, then we will chunk into the first 64 words, the next 64 words, etc.