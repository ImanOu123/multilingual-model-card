

```
conda install -c conda-forge jupyter
conda install -c anaconda ipykernel


python3
import nltk
nltk.download('punkt_tab')


```


`evaluator/`
- `run_on_6060.py`: use translator in `translator/` on the 6060 **test** set.
- `eval_on_6060.py`: assess the generations.
- `eval_on_6060_comet.py`: assess the generations using the comet model.


Dictionary collection on the dev set:
- `extract_terms_llama3.py`: Prompting LLAMA3 70B to extract terms from the whole dev set

