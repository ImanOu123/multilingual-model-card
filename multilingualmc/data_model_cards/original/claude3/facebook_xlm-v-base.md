# Model Card for facebook/xlm-v-base

facebook/xlm-v-base is a multilingual language model with a one million token vocabulary trained on 2.5TB of data from Common Crawl. It outperforms XLM-R on a range of natural language tasks, particularly for low-resource languages, by improving vocabulary coverage for individual languages while reducing over-tokenization.

## Model Details

### Model Description

Model Details:
- Architecture: XLM-V is a multilingual language model based on a transformer encoder architecture. It has 12 layers.
- Vocabulary: XLM-V uses a 1 million token vocabulary that provides better coverage for individual languages compared to previous multilingual models like XLM-R. This results in more semantically meaningful and shorter tokenizations.
- Training Data: XLM-V was trained on the CC100 dataset, which consists of 2.5TB of data from Common Crawl. A sampling temperature of 0.3 was used to increase the amount of low- and medium-resource language examples seen during training.
- Training Procedure: 
[More Information Needed]
- Training Compute: The model was trained for a total of [More Information Needed] steps using a batch size of 8,192 distributed across 256 A100 GPUs. Float16 precision was used.
- Parameters: [More Information Needed]
- Important Disclaimers: [More Information Needed]

- **Developed by:** Davis Liang; Hila Gonen; Yuning Mao; Rui Hou; Naman Goyal; Marjan Ghazvininejad; Luke Zettlemoyer; Madian Khabsa; Meta Ai
- **Funded by:** Based on the provided references, the XLM-V model was developed by researchers from Meta AI (formerly Facebook AI Research). The key people involved in the project are:

- Davis Liang (first author of the XLM-V paper)
- Hila Gonen
- Yuning Mao
- Rui Hou
- Naman Goyal
- Marjan Ghazvininejad
- Luke Zettlemoyer
- Madian Khabsa

The model is hosted under the Meta AI organization on Hugging Face: https://huggingface.co/facebook/xlm-v-base

However, there is no explicit information in the provided references about the specific funding sources for this project. To accurately answer this question, [More Information Needed] regarding the funding organizations or grants that supported the development of the XLM-V model.
- **Shared by:** Based on the references provided, the contributors who made the model facebook/xlm-v-base available online are:

Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, and Madian Khabsa from Meta AI.

The model was introduced in their paper "XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models" and is available under the Meta AI organization on Hugging Face.
- **Model type:** The facebook/xlm-v-base model is a multilingual masked language model trained on 2.5TB of Common Crawl data using a one million token vocabulary, which improves upon XLM-R by assigning vocabulary capacity to achieve sufficient coverage for each individual language.
- **Language(s):** The facebook/xlm-v-base model is a multilingual language model that uses a shared vocabulary of one million tokens to process over 100 languages.
- **License:** [More Information Needed]

The provided references do not contain any information about the specific license being used for the facebook/xlm-v-base model. More details would be needed from the model creators or official documentation to determine the license under which this model is released.
- **Finetuned from model:** Based on the information provided in the references, the facebook/xlm-v-base model is not fine-tuned from another model. It is a new multilingual language model with a one million token vocabulary that was trained from scratch on 2.5TB of data from Common Crawl, as mentioned in this excerpt:

"XLM-V is multilingual language model with a one million token vocabulary trained on 2.5TB of data from Common Crawl (same as XLM-R)."

The references do not indicate that XLM-V was fine-tuned from any other pre-existing model. Therefore, for the model card description, the answer to the question "If the model facebook/xlm-v-base is fine-tuned from another model, provide the name and link to that base model" would be:

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/stefan-it/xlm-v-experiments
- **Paper:** https://arxiv.org/pdf/2301.10472.pdf
- **Demo:** [More Information Needed]

Based on the provided references, there is no mention of a demo link for the facebook/xlm-v-base model. The references discuss the model's training data, vocabulary size, performance on various tasks, and its availability on the Hugging Face Model Hub. However, no specific demo link is provided.
## Uses

### Direct Use

The model facebook/xlm-v-base can be used without fine-tuning, post-processing, or plugging into a pipeline for fill-mask tasks. Here's a code snippet demonstrating how to use it:

```python
from transformers import pipeline

unmasker = pipeline('fill-mask', model='stefan-it/xlm-v-base')
unmasker("Paris is the <mask> of France.")
```

This code loads the XLM-V model and uses it to predict the masked word in the given sentence.

[More Information Needed] on other tasks that XLM-V can perform without fine-tuning or additional processing.

### Downstream Use

The facebook/xlm-v-base model can be fine-tuned for various downstream tasks such as Named Entity Recognition (NER) and Question Answering (QA). Here are a couple of examples:

1. Fine-tuning for NER:
```bash
python3 flair-fine-tuner.py ./configs/xlm_v_base.json
```
This command fine-tunes the XLM-V base model on the English WikiANN dataset using the Flair library. The fine-tuned model can then be used for NER tasks.

2. Fine-tuning for QA:
```bash
python3 run_qa.py \
--model_name_or_path stefan-it/xlm-v-base \
--dataset_name squad \
--do_train \
--do_eval \
--max_seq_length 512 \
--doc_stride 128 \
--per_device_train_batch_size 6 \
--learning_rate 3e-5 \
--weight_decay 0.0 \
--warmup_steps 0 \
--num_train_epochs 2 \
--seed 1 \
--output_dir xlm-v-1 \
--fp16 \
--save_steps 14618
```
This command fine-tunes the XLM-V base model on the SQuAD dataset using the Hugging Face Transformers library. The fine-tuned model can then be used for question answering tasks, even in a zero-shot manner on other languages like Hindi using the XQuAD dataset.

[More Information Needed] on how the model can be plugged into a larger ecosystem or app.

### Out-of-Scope Use

Based on the provided references about the XLM-V model (facebook/xlm-v-base), here are some potential ways the model could be misused and what users ought not do:

The model should not be used to generate or analyze content that promotes hate speech, discrimination, or biases against protected groups. While the references don't directly discuss this, large language models can potentially pick up and amplify biases present in training data.

Users should not rely solely on the model's outputs for critical decision-making that impacts people's lives, rights or access to opportunities, without human oversight and without considering the model's limitations. The references focus on the model's capabilities but do not cover its constraints or failure modes in detail. [More Information Needed]

The model should not be used in high-stakes domains like healthcare, finance or legal settings without extensive testing, as the references do not discuss its suitability or potential risks in these areas. [More Information Needed]

Users should not assume the model's multilingual capabilities extend equally well to all languages, especially low-resource ones. While the references show the model performs well on benchmarks, real-world performance may vary. More testing is needed on the long tail of languages.

In general, the model should be used in a transparent and accountable way, with clear communication to end users about its abilities and limitations. Potential negative impacts on society should be carefully monitored. However, [More Information Needed] in the provided references to draw firm conclusions. More interdisciplinary analysis of the model in real-world contexts would help surface additional misuse cases and guidance for responsible use.

### Bias, Risks, and Limitations

Based on the provided references about the XLM-V model (facebook/xlm-v-base), here are some known or foreseeable issues:

1. Scalability issues: While scaling the vocabulary is an efficient form of conditional compute, it can result in increased pre-training times due to the computational complexity of the softmax over the entire vocabulary (Reference 1). This can be mitigated by adopting approximation techniques like adaptive softmax and adaptive inputs.

2. Increased memory footprint: Scaling the vocabulary can significantly increase the memory footprint of the model. For XLM-V, the 1M token vocabulary accounts for 93% of the model's trainable parameters (Reference 7). However, this becomes less of a problem as we begin to work with larger models, where the number of non-embedding parameters vastly outweigh the size of the vocabulary embedding matrix.

3. Inference latency: While XLM-V is faster than mT5 and mT5-XL, it still remains approximately 1.6 times slower than a comparable BERT baseline (Reference 8). Simply using a larger sentencepiece vocabulary can improve downstream performance with a smaller impact on inference latency.

4. [More Information Needed] on potential biases, fairness issues, or misuse of the model.

5. [More Information Needed] on the model's performance and limitations for specific downstream tasks and domains.

6. [More Information Needed] on the environmental impact and computational resources required for training and deploying the model at scale.

Further research and analysis are needed to fully understand the sociotechnical implications and potential harms associated with the XLM-V model. Continuous monitoring, responsible deployment, and engaging with diverse stakeholders can help identify and mitigate risks.

### Recommendations

Based on the provided references about the XLM-V model (facebook/xlm-v-base), here are some foreseeable issues and recommendations:

1. Scalability: While scaling the vocabulary is an efficient form of conditional compute, it can result in increased pre-training times due to the computational complexity of the softmax over the entire vocabulary. Approximation techniques like adaptive softmax and adaptive inputs can help mitigate this issue. (Reference 1)

2. Memory footprint: Scaling the vocabulary can significantly increase the memory footprint of the model. However, this becomes less of a problem as we work with larger models, where the number of non-embedding parameters vastly outweigh the size of the vocabulary embedding matrix. (Reference 3)

3. Diminishing utility of increasing vocabulary size: Increasing the vocabulary size beyond a certain point (e.g., 1M tokens) may lead to diminishing returns in performance on downstream tasks. This is because further expanding the vocabulary inherits tokens from the long tail of a Zipfian distribution, which learn sub-optimal representations due to insufficient training data. (References 4 and 5)

4. Inference latency: While XLM-V's larger vocabulary can improve downstream performance and reduce over-tokenization for low-resource languages, it may impact inference latency compared to models with smaller vocabularies like BERT. (Reference 7)

5. [More Information Needed] on the specific ethical and societal implications of the XLM-V model, as the provided references do not cover these aspects in detail.

Recommendations:
- Explore approximation techniques to address scalability issues during pre-training.
- Monitor the memory footprint and consider the trade-off between vocabulary size and model performance.
- Be cautious when increasing the vocabulary size beyond 1M tokens, as it may lead to diminishing returns in performance.
- Consider the impact on inference latency when deploying the model in real-world applications.
- Conduct further research on the ethical and societal implications of the XLM-V model to ensure responsible deployment and use.

## Training Details

### Training Data

The training data for the facebook/xlm-v-base model is the CC100 dataset, which contains 2.5 TB of data split between 116 languages, created from one Common Crawl dump for English and twelve dumps for all other languages. The data was sampled with a temperature of 0.3 to increase the amount of low-and medium-resource language examples seen during training.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about the tokenization and preprocessing for the data of the model facebook/xlm-v-base:

Tokenization:
- The tokenizer in 🤗 Transformers should output the same ids/subtokens as the `fairseq` tokenizer. (Reference 1)
- The final vocabulary consists of 901,629 tokens. (Reference 2)
- The Unigram Language Model (ULM) from Kudo and Richardson (2018) is used as the subword segmentation algorithm to construct the vocabulary. (References 4, 7, 10)
- The authors train individual sentencepiece models for each language with a vocabulary size of 30,000 per language. (Reference 5)
- The sentencepiece model reduces over-tokenization and can split sentences into pseudo-meaningful segments out-of-the-box. (Reference 8)

Preprocessing:
- The authors use the WikiANN dataset, which covers 176 languages, to compare the tokenization between the `fairseq` tokenizer and the 🤗 Transformers tokenizer. (Reference 1)
- [More Information Needed] about specific preprocessing steps like resizing or rewriting, as the references do not provide direct information about these aspects.

The references suggest that the XLM-V model uses an improved vocabulary compared to XLM-R, resulting in more semantically meaningful and shorter tokenizations. (Reference 9)

#### Training Hyperparameters

Here are the training hyperparameters for the facebook/xlm-v-base model based on the provided references:

- Dataset: CC100 dataset with a sampling temperature of 0.3 to increase low- and medium-resource language examples
- Optimizer: Adam optimizer with default (β1, β2) = (0.9, 0.98) and ϵ = 1e-6
- Learning rate: 6e-4
- Warmup: 15,000 steps
- Batch size: 8,192 distributed across 256 A100 GPUs
- Training iterations: 1.5M
- Maximum sequence length: 512
- Pretraining task: Masked Language Model (MLM) with standard 15% masking rate

[More Information Needed] on:
- Weight decay
- Dropout
- Gradient clipping
- Learning rate scheduler

The model was trained using float16 precision. No approximation tricks like adaptive softmax or adaptive inputs were used for the softmax layer despite the large 1M vocabulary size.

#### Speeds, Sizes, Times

Here are the details about the model facebook/xlm-v-base based on the provided references:

Training Data:
- The model was trained on 2.5TB of data from Common Crawl, the same dataset as XLM-R.

Training Configuration:
- The model was trained for 1.5M iterations with a batch size of 8,192 on CC100 data.
- It was trained for 12 epochs on 8 A100 GPUs with float16 precision.

Model Architecture:
- The model has a vocabulary size of 1 million tokens.
- The vocabulary embedding matrix contains 93% of the model's trainable parameters.

[More Information Needed] for the following:
- Throughput
- Start and end training time 
- Checkpoint sizes

The model outperformed XLM-R on various benchmarks:
- Outperformed XLM-R on all datasets in Table 2 by an average of 3.5 points absolute.
- Outperformed XLM-R on all languages in cross-lingual transfer (Table 3).
- Consistently outperformed XLM-R on low-resource languages, e.g. 4.7% and 2.9% accuracy improvement on Swahili and Urdu on XNLI, and 11.2% F1 gain on average on MasakhaNER.
- On Americas NLI zero-shot cross-lingual transfer (Table 4), obtained 18.2% and 17.2% absolute F1 improvement on Quechua and Guaraní respectively.

The model resulted in shorter sequences post-tokenization compared to XLM-R, with the largest drops for low-resource languages like Quechua and Guaraní, suggesting XLM-R over-tokenized them.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the references provided, the facebook/xlm-v-base model evaluates on the following benchmarks and datasets:

1. MLQA (Multilingual Question Answering) dataset: It consists of over 12K QA instances in English and 5K in each of the 6 target languages. The training set used is SQuAD v1.1.

2. XQuAD (Cross-lingual Question Answering Dataset): The fine-tuned XLM-V model is zero-shot evaluated on the 11 languages in XQuAD. An example is provided for evaluation on Hindi:

```bash
python3 run_qa.py --model_name_or_path xlm-r-1 \
--dataset_name xquad \
--dataset_config_name xquad.hi \
--do_eval \
--max_seq_length 512 \
--doc_stride 128 \
--output_dir xlm-r-1-hi \
--fp16
```

3. XNLI (Cross-lingual Natural Language Inference) dataset: It evaluates whether a premise sentence entails, contradicts, or is neutral toward a hypothesis sentence. The crowd-sourced English data is translated to 10 other languages by professional human translators and used for evaluation.

[More Information Needed] for other potential benchmarks like TyDiQA and WikiAnn, as they are mentioned in the XLM-V paper abstract but not explicitly stated as evaluation datasets for the facebook/xlm-v-base model.

#### Factors

Based on the provided references about the XLM-V model (facebook/xlm-v-base), here are some foreseeable characteristics that may influence the model's behavior:

1. Language coverage and resource availability: XLM-V is a multilingual model trained on a large corpus covering many languages. However, the performance may vary across high-resource, medium-resource, and low-resource languages due to differences in training data availability and quality. Evaluation should be disaggregated across languages to uncover potential disparities.

2. Domain and context: The model's performance may be influenced by the domain and context of the input text. The references mention training on Common Crawl data and evaluation on tasks like natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), and named entity recognition (WikiAnn). Performance may vary across different domains and contexts not well-represented in the training data.

3. Tokenization and vocabulary: XLM-V uses a large vocabulary of one million tokens, which can lead to more semantically meaningful and shorter tokenizations compared to models like XLM-R. This may impact the model's ability to handle longer contexts and out-of-vocabulary words, especially for low-resource languages.

4. [More Information Needed] on specific population subgroups that may experience disparities in model performance, such as demographic groups based on age, gender, ethnicity, or socioeconomic status.

5. [More Information Needed] on potential biases or fairness concerns that may arise from the training data or model architecture.

To better understand the model's behavior and uncover potential disparities, it is important to conduct disaggregated evaluation across various factors such as language, domain, context, and relevant population subgroups. More information and analysis would be needed to fully characterize the model's performance and fairness.

#### Metrics

Based on the provided references about the XLM-V model (facebook/xlm-v-base), the key evaluation metrics used are:

1. Accuracy and F1 scores on various cross-lingual understanding tasks, including:
   - Natural Language Inference (XNLI, AmericasNLI)
   - Question Answering (MLQA, XQuAD, TyDiQA) 
   - Named Entity Recognition (WikiAnn, MasakhaNER)

2. Performance comparison against the baseline XLM-R model, with a focus on:
   - Overall average improvement across all tasks
   - Improvements on low-resource languages (e.g., Swahili, Urdu, Quechua, Guaraní)

3. Relative difference in average number of tokens post-tokenization between XLM-R and XLM-V, as shorter sequences can indicate more efficient and semantically meaningful tokenization.

The references do not provide specific details on how the tradeoffs between different types of errors are handled in the evaluation. [More Information Needed]

### Results

Here are the evaluation results for the facebook/xlm-v-base model based on the provided references:

Performance:
- Outperforms XLM-R on all datasets by an average of 3.5 points absolute
- Outperforms XLM-R on all languages in cross-lingual transfer 
- Outperforms XLM-R on low-resource languages like Swahili (4.7% accuracy improvement on XNLI) and Urdu (2.9% accuracy improvement on XNLI)
- 11.2% average F1 gain on MasakhaNER (low-resource African language NER dataset)
- Outperforms XLM-R on Americas NLI in zero-shot crosslingual transfer, with 18.2% absolute F1 improvement on Quechua and 17.2% absolute improvement on Guaraní

Factors:
- Trained on CC100 for 1.5M iterations with batch size 8,192
- 1 million token vocabulary
- Trained for 12 epochs on 8 A100 GPUs with float16 precision
- Uses AdamW optimizer for finetuning

Metrics:
- Evaluated on natural language inference (XNLI), question answering (MLQA, TyDiQA, XQuAD), named entity recognition (WikiAnn), low resource tasks (AmericasNLI, MasakhaNER) 
- Metrics include accuracy, F1 score
- [More Information Needed] on specific metric values for each dataset

Tokenization:
- Outputs shorter sequences post-tokenization compared to XLM-R
- Tokenizations are more semantically meaningful
- Largest token count reduction for Quechua and Guaraní, suggesting XLM-R over-tokenizes these languages

#### Summary

Based on the provided references, here is a summary of the evaluation results for the facebook/xlm-v-base model:

- XLM-V outperforms XLM-R on all datasets tested by an average of 3.5 points absolute. This includes natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), and named entity recognition (WikiAnn).

- In cross-lingual transfer (training on English and evaluating on other languages), XLM-V outperforms XLM-R on all languages. Similar improvements are seen on translate-train-all (finetuning on both English and translated training sets). 

- XLM-V shows particularly strong performance on low-resource languages. On XNLI, it achieves 4.7% and 2.9% accuracy improvements on Swahili and Urdu. On MasakhaNER (African language NER), it obtains an average 11.2% F1 gain.

- On the Americas NLI dataset in zero-shot cross-lingual transfer, XLM-V outperforms XLM-R by a wide margin, with 18.2% and 17.2% absolute F1 improvements on Quechua and Guaraní.

- The XLM-V tokenizer results in fewer tokens on average compared to XLM-R, especially for languages where XLM-R over-tokenizes. The tokenizations are often more semantically meaningful.

[More Information Needed] on the exact model and training details, inference speed comparisons, and performance on high-resource languages.

## Model Examination

Based on the provided references about the XLM-V model (facebook/xlm-v-base), here is the information I can provide for the experimental section on explainability/interpretability:

The XLM-V model uses an improved vocabulary compared to XLM-R, which results in tokenizations that are typically more semantically meaningful and shorter. This can potentially aid in the interpretability of the model's outputs, as the tokens may align more closely with human-understandable semantic units.

[More Information Needed] on specific techniques or approaches used for explainability or interpretability of the XLM-V model.

The references mention that XLM-V outperforms XLM-R on various tasks across different languages, including low-resource languages. However, [More Information Needed] on how this performance relates to the model's explainability or interpretability.

The `xlm_v_tokenizer_comparison.py` script is used to compare the tokenizer outputs between the 🤗 Transformers implementation and the original `fairseq` tokenizer, ensuring consistency. [More Information Needed] on how this comparison could be leveraged for explainability or interpretability purposes.

## Environmental Impact

- **Hardware Type:** [More Information Needed]

The provided references do not mention the specific hardware type that the facebook/xlm-v-base model was trained on. More information would be needed from the model developers or the associated research paper to determine the training hardware.
- **Software Type:** Based on the references provided, the XLM-V model (including the facebook/xlm-v-base variant) was trained on 2.5TB of data from Common Crawl. However, the specific software type or framework used for training is not explicitly mentioned. To provide a complete answer, [More Information Needed] about the software stack and libraries used in the training process.
- **Hours used:** Based on the provided references, the total training time for the facebook/xlm-v-base model is not explicitly stated. The references mention some training details, but not the full training time:

- The model was pretrained on the CC100 dataset for 1.5M iterations with a batch size of 8,192 (reference 3)
- A smaller 12-layer model was pretrained for 300,000 steps with a batch size of 2,048 for comparison experiments (reference 4)

However, the total wall-clock training time for the full facebook/xlm-v-base model is not provided. To include the training time in the model card, [More Information Needed] from the model developers or training logs.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider that the model facebook/xlm-v-base was trained on. More information would be needed to determine the cloud provider used for training this model.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the facebook/xlm-v-base model. To answer this question, more specific details would be needed, such as the total compute used, energy consumption, and carbon footprint of the training hardware.
## Technical Specification

### Model Architecture and Objective

Model Architecture and Objective:

XLM-V is a multilingual language model with a one million token vocabulary trained on 2.5TB of data from Common Crawl, following the same training procedure as XLM-R. The model uses a new approach for scaling to very large multilingual vocabularies by de-emphasizing token sharing between languages with little lexical overlap and assigning vocabulary capacity to achieve sufficient coverage for each individual language.

The model is pretrained using the Masked Language Model (MLM) task with the standard masking rate of 15%. The training details are as follows:

- Optimizer: Adam with default (β1, β2) and ϵ parameters of (0.9, 0.98) and 1e-6, respectively
- Learning rate: 6e-4
- Warmup: 15,000 steps
- Batch size: 8,192 distributed across 256 A100 GPUs
- Training iterations: 1.5M
- Maximum sequence length: 512

[More Information Needed] (Specific model architecture details are not provided in the given references)

The objective of XLM-V is to overcome the vocabulary bottleneck in multilingual masked language models and improve the representational capabilities compared to models like XLM-R. XLM-V aims to provide more semantically meaningful and shorter tokenizations, especially for low-resource languages, leading to improved performance on various multilingual language understanding tasks.

### Compute Infrastructure

Based on the provided references, the compute infrastructure used for training the facebook/xlm-v-base model is as follows:

- The model was trained using 256 A100 GPUs with a batch size of 8,192 distributed across the GPUs.
- The training was done using float16 precision.
- The model was trained for a total of [More Information Needed] steps.

For finetuning on downstream tasks:
- Named entity recognition (WikiAnn) and low resource language tasks (AmericasNLI, MasakhaNER) were trained using float16 precision on [More Information Needed] GPUs.
- XQuAD was finetuned for 2 epochs on a single A100 GPU with float16 precision, using a learning rate of 3e-5, max sequence length of 512, batch size of 6, no weight decay, and no warmup.

## Citation

```
@misc{davis-xlmv,
    author = {Davis Liang and
              Hila Gonen and
              Yuning Mao and
              Rui Hou and
              Naman Goyal and
              Marjan Ghazvininejad and
              Luke Zettlemoyer and
              Madian Khabsa and
              Meta Ai},
    title  = {XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models},
    url    = {https://arxiv.org/pdf/2301.10472.pdf}
}
```

