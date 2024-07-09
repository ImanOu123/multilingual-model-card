Dictionary collection on the dev set:
- `extract_terms_llama3.py`: Prompting LLAMA3 70B to extract terms from the whole dev set

Baselines on the test set:
1. `googletrans_paragraph`: Google Translate API, per paragraph
2. `googletrans`: Google Translate API, per sentence
3. `googletrans_gpt`: Google Translate API + GPT for extracting terms + GPT for updating the overall translation, per sentence
4. `googletrans_dict_gpt`: Google Translate API + searching terms in dictionary + GPT for updating the overall translation, per sentence
5. `seamless`: Seamless Large, per sentence

"per sentence" here means per chunk of 64 words in maximum.