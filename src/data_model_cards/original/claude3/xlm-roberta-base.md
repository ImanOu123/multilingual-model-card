# Model Card for xlm-roberta-base

XLM-RoBERTa (XLM-R) is a generic cross-lingual sentence encoder trained on 2.5TB of filtered CommonCrawl data in 100 languages. It obtains state-of-the-art results on many cross-lingual understanding (XLU) benchmarks, significantly outperforming multilingual BERT (mBERT).

## Model Details

### Model Description

Model Details for xlm-roberta-base:

Model Architecture:
- Transformer model with 12 layers, 768 hidden states, 12 attention heads
- Total parameters: 270M
- Vocabulary size: 250K tokens

Training Procedure:
- Trained with the multilingual MLM (masked language modeling) objective using only monolingual data
- Trained on 2.5TB of filtered CommonCrawl data in 100 languages
- Languages are sampled using a distribution with α = 0.3
- Tokenization directly on raw text data using SentencePiece with a unigram language model
- Trained for 1.5 million updates on five-hundred 32GB Nvidia V100 GPUs with a batch size of 8192

Important Disclaimers:
- [More Information Needed]

- **Developed by:** Alexis Conneau; Kartikay Khandelwal; Naman Goyal; Vishrav Chaudhary; Guillaume Wenzek; Francisco Guzmán; Edouard Grave; Myle Ott; Luke Zettlemoyer; Veselin Stoyanov; Facebook Ai
- **Funded by:** [More Information Needed]

The provided references do not contain any information about the funding sources for the XLM-RoBERTa-base model. More details would be needed from the model developers or their organizations to determine what people or groups financially supported this specific project.
- **Shared by:** Based on the provided references, there is no direct information about the specific contributors who made the XLM-RoBERTa-base model available online as a GitHub repo. The references discuss the XLM-R model in general, its training data, and performance on various benchmarks, but do not mention the individuals involved in publishing the model.

[More Information Needed]
- **Model type:** XLM-RoBERTa-base is a transformer-based multilingual masked language model pre-trained on monolingual text data in 100 languages using the unsupervised cross-lingual representation learning approach.
- **Language(s):** XLM-RoBERTa (XLM-R) is a multilingual model trained on CommonCrawl data in 100 languages.
- **License:** [More Information Needed]

The references provided do not contain any information about the license being used for the XLM-RoBERTa base model. More details would be needed from the model developers or official documentation to determine the specific license that applies to this model.
- **Finetuned from model:** XLM-RoBERTa (XLM-R) is not fine-tuned from another model, but rather trained from scratch on a large corpus of filtered CommonCrawl data in 100 languages, as mentioned in Reference 9:

"XLM-R (XLM-RoBERTa) is a generic cross lingual sentence encoder that obtains state-of-the-art results on many cross-lingual understanding (XLU) benchmarks. It is trained on 2.5T of filtered CommonCrawl data in 100 languages (list below)."

The model follows the XLM approach closely, with some improvements and scaling modifications, but it is not fine-tuned from a previously released model.
### Model Sources

- **Repository:** https://github.com/pytorch/fairseq/tree/master/examples/xlmr
- **Paper:** https://arxiv.org/pdf/1911.02116.pdf
- **Demo:** [More Information Needed]

The references provided do not contain a direct link to a demo of the XLM-RoBERTa-base model. More information would be needed to provide a demo link in the model card description.
## Uses

### Direct Use

The xlm-roberta-base model can be used without fine-tuning, post-processing, or plugging into a pipeline for tasks like extracting features, encoding and decoding text in multiple languages. Here are a few examples:

1. Extracting features from the last layer:
```python
last_layer_features = xlmr.extract_features(zh_tokens)
assert last_layer_features.size() == torch.Size([1, 6, 1024])
```

2. Extracting features from all layers:
```python
all_layers = xlmr.extract_features(zh_tokens, return_all_hiddens=True)
assert len(all_layers) == 25
assert torch.all(all_layers[-1] == last_layer_features)
```

3. Encoding and decoding text in various languages:
```python
en_tokens = xlmr.encode('Hello world!')
assert en_tokens.tolist() == [0, 35378,  8999, 38, 2]
xlmr.decode(en_tokens)  # 'Hello world!'

zh_tokens = xlmr.encode('你好，世界')
assert zh_tokens.tolist() == [0, 6, 124084, 4, 3221, 2]
xlmr.decode(zh_tokens)  # '你好，世界'

hi_tokens = xlmr.encode('नमस्ते दुनिया')
assert hi_tokens.tolist() == [0, 68700, 97883, 29405, 2]
xlmr.decode(hi_tokens)  # 'नमस्ते दुनिया'

ar_tokens = xlmr.encode('مرحبا بالعالم')
assert ar_tokens.tolist() == [0, 665, 193478, 258, 1705, 77796, 2]
xlmr.decode(ar_tokens) # 'مرحبا بالعالم'

fr_tokens = xlmr.encode('Bonjour le monde')
assert fr_tokens.tolist() == [0, 84602, 95, 11146, 2]
xlmr.decode(fr_tokens) # 'Bonjour le monde'
```

To use the model, it can be loaded using PyTorch hub:
```python
import torch
xlmr = torch.hub.load('pytorch/fairseq:main', 'xlmr.large')
xlmr.eval()  # disable dropout (or leave in train mode to finetune)
```

[More Information Needed] on other potential use cases without fine-tuning, post-processing or using a pipeline.

### Downstream Use

The XLM-RoBERTa-base model can be used for various cross-lingual understanding tasks when fine-tuned, such as:

1. Cross-lingual Natural Language Inference (XNLI): The model can be fine-tuned on the English XNLI training set and evaluated on the dev and test sets of 15 languages for cross-lingual transfer.

2. Named Entity Recognition (NER): The model can be fine-tuned on the CoNLL-2002 and CoNLL-2003 datasets in English, Dutch, Spanish, and German for cross-lingual transfer, per-language performance, or multilingual learning.

3. Question Answering: [More Information Needed]

4. GLUE benchmark: The model can be fine-tuned on the GLUE benchmark to evaluate its English performance and compare it to other state-of-the-art models.

When plugged into a larger ecosystem or app, the XLM-RoBERTa-base model can be used as a generic cross-lingual sentence encoder. Here's a code snippet demonstrating how to use the model in PyTorch:

```python
import torch
xlmr = torch.hub.load('pytorch/fairseq:main', 'xlmr.large')
xlmr.eval()  # disable dropout (or leave in train mode to finetune)

ar_tokens = xlmr.encode('مرحبا بالعالم')
assert ar_tokens.tolist() == [0, 665, 193478, 258, 1705, 77796, 2]
xlmr.decode(ar_tokens) # 'مرحبا بالعالم'

fr_tokens = xlmr.encode('Bonjour le monde')
assert fr_tokens.tolist() == [0, 84602, 95, 11146, 2]
xlmr.decode(fr_tokens) # 'Bonjour le monde'
```

### Out-of-Scope Use

Based on the provided references about the XLM-RoBERTa model, here are some potential ways the model could be misused and what users should not do:

The XLM-RoBERTa model is a powerful multilingual language model trained on a large amount of data in 100 languages. As such, it could potentially be misused to:

- Generate fake news, misinformation, hate speech, or other harmful content in many languages. Users should not use the model to create deceptive or abusive multilingual content.

- Impersonate real people or organizations across languages in generated text. The model should not be used for identity theft or fraud.

- Automate spam, phishing, or social engineering attacks by adapting malicious content to target many languages. This is unethical and users must not do this.

- Unfairly favor high-resource languages over low-resource languages in applications, worsening inequities. Developers should carefully evaluate language-specific performance.

[More Information Needed] on specific ways the model architecture or training data could enable misuse.

In general, the XLM-RoBERTa model is a tool that can be used for both beneficial and harmful purposes in many languages. Users have an ethical responsibility to deploy it only for legitimate applications that help rather than deceive people. Careful testing and monitoring for misuse across languages is advised.

### Bias, Risks, and Limitations

Based on the provided references about the XLM-R model, here are some known or foreseeable issues, limitations, and potential harms:

Capacity Dilution and Curse of Multilinguality:
- As the number of languages increases in the multilingual model, the per-language capacity decreases for a fixed model size. This leads to a trade-off between positive transfer and capacity dilution. (Ref 5, 7)
- The performance on downstream tasks suffers as more languages are added, known as the "curse of multilinguality". Increasing model capacity can help mitigate this to some extent. (Ref 5, 7)

High-Resource vs Low-Resource Language Trade-off: 
- There is a trade-off between the performance on high-resource and low-resource languages. Improving low-resource language performance by adding similar high-resource languages during pretraining can hurt overall downstream performance. (Ref 7)

Vocabulary Dilution:
- Using a fixed vocabulary size for all models leads to vocabulary dilution as more languages are added. (Ref 6) 
[More Information Needed] on specifics of how this impacts performance.

Monolingual vs Multilingual Performance:
- Multilingual models are often criticized for obtaining worse performance than their monolingual counterparts. (Ref 10)
[More Information Needed] on direct comparisons between XLM-R and monolingual models.

Sampling and Pretraining Data:
- The sampling rate of training examples from each language and the pretraining data used can significantly impact performance, especially for low-resource languages. (Ref 6) 
[More Information Needed] on the specifics of XLM-R's sampling strategy and pretraining data.

Potential Misuse and Biases:
[More Information Needed] The references do not directly address potential misuse cases or biases in XLM-R. Further analysis would be required to identify any biases in the pretraining data or potentially harmful applications of the model.

Technical Limitations:
[More Information Needed] on any technical limitations like inference speed, memory requirements, etc. Detailed system specifications and benchmarks would be needed to identify these.

In summary, the key limitations seem to revolve around the trade-offs in multilingual modeling between the number of languages, positive transfer, capacity dilution, and high-resource vs low-resource language performance. More information is needed to draw conclusions about potential biases, misuse, and technical constraints.

### Recommendations

Based on the information provided in the references, here are some recommendations regarding foreseeable issues with the XLM-RoBERTa-base model:

1. Capacity dilution: As the model is trained on 100 languages, the per-language capacity decreases compared to monolingual models. This can lead to reduced performance, especially for low-resource languages. Increasing model capacity (e.g., hidden size) can help mitigate this issue.

2. High-resource vs. low-resource language trade-off: The model's performance on high-resource languages may come at the cost of lower performance on low-resource languages. Careful tuning of the language sampling rate (controlled by the α parameter) is necessary to balance this trade-off.

3. Vocabulary size limitations: The fixed vocabulary size of 150K tokens may not be sufficient to capture the full diversity of all 100 languages, leading to vocabulary dilution. [More Information Needed] on the impact of increasing vocabulary size on the model's performance.

4. Language-specific performance: While XLM-RoBERTa-base may outperform previous multilingual models on cross-lingual tasks, it may still lag behind monolingual models for specific languages. [More Information Needed] on the model's performance compared to state-of-the-art monolingual models for each language.

5. Bias and fairness: [More Information Needed] on potential biases in the training data and their impact on the model's performance across different languages, dialects, and demographic groups.

To address these issues, consider increasing model capacity, optimizing language sampling rates, and conducting thorough evaluations on a diverse set of languages and tasks. Additionally, ensure that the model is used responsibly and that its limitations are clearly communicated to users.

## Training Details

### Training Data

The XLM-RoBERTa model was trained on 2.5TB of filtered CommonCrawl data in 100 languages. The data was filtered using language models trained on each language and an internal language identification model in combination with fastText.

[More Information Needed] for links to documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the references provided, here are the details about the preprocessing for the data of the XLM-RoBERTa Base model:

Tokenization:
- Used SentencePiece model (SPM) directly on raw text data for all languages, instead of language-specific tokenization tools (Reference 1)
- Trained a unigram language model for SentencePiece tokenization (Reference 3)
- Used a large vocabulary size of 250K tokens with a full softmax (Reference 3)

Resizing/Rewriting:
[More Information Needed]

Other preprocessing details:
- Sampled batches from different languages using the same sampling distribution as Lample and Conneau (2019), but with α = 0.3 (Reference 3)
- Did not use language embeddings, which allows the model to better deal with code-switching (Reference 3)
- Built a clean CommonCrawl Corpus in 100 languages using an internal language identification model in combination with the one from fastText (Reference 4)
- Trained language models in each language and used it to filter documents as described in Wenzek et al. (2019) (Reference 4)
- Considered one CommonCrawl dump for English and twelve dumps for all other languages, which significantly increased dataset sizes, especially for low-resource languages (Reference 4)

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the XLM-R Base model:

Model Architecture:
- L = 12 (number of layers)
- H = 768 (hidden size) 
- A = 12 (number of attention heads)

Tokenization:
- Trained SentencePiece tokenization directly on raw text data
- Used a unigram language model 
- Vocabulary size of 250K with a full softmax

Training Data:
- Trained on the CommonCrawl corpus in 100 languages
- Used an internal language identification model in combination with fastText for language identification
- Trained language models in each language and used them to filter documents
- Considered one CommonCrawl dump for English and twelve dumps for all other languages

Training Objective:
- Used the multilingual MLM (Masked Language Model) objective with only monolingual data

Sampling:
- Sampled batches from different languages using the same sampling distribution as Lample and Conneau (2019), but with α = 0.3

Other:
- Did not use language embeddings to better deal with code-switching

[More Information Needed] on specific hyperparameters like learning rate, batch size, number of training steps, optimizer, etc.

#### Speeds, Sizes, Times

Based on the provided references, here are the details I can provide about the xlm-roberta-base model:

Model architecture:
- Uses the BERT-base architecture

Model size:
- 250M parameters
- 250k vocabulary size

Download link:
[xlm.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz)

Training data:
- Trained on 2.5T of filtered CommonCrawl data in 100 languages

Performance:
- Obtains state-of-the-art results on many cross-lingual understanding (XLU) benchmarks

[More Information Needed]:
- Throughput details
- Training start and end times
- Checkpoint sizes

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the XLM-RoBERTa-base model is evaluated on the following benchmarks and datasets:

1. Cross-lingual Natural Language Inference (XNLI) dataset, which covers 15 languages.

2. Named Entity Recognition (NER) datasets: CoNLL-2002 (Dutch and Spanish) and CoNLL-2003 (English and German).

3. Cross-lingual Question Answering: MLQA benchmark, which extends the English SQuAD dataset to Spanish, German, Arabic, Hindi, Vietnamese, and Chinese.

4. GLUE (General Language Understanding Evaluation) benchmark for evaluating the model's performance on English natural language understanding tasks, such as MNLI, SST-2, and QNLI.

The model is fine-tuned and evaluated on these datasets to assess its cross-lingual transfer capabilities, per-language performance, and multilingual learning abilities.

#### Factors

Based on the provided references about the XLM-RoBERTa-base model, here are some foreseeable characteristics that may influence the model's behavior:

Domain and context:
- The model was pretrained on the CommonCrawl (CC) dataset, which improved performance on low-resource languages compared to pretraining on just Wikipedia (Reference 1). So the model may perform better on domains and contexts similar to web crawl data.
- The model was evaluated on cross-lingual natural language inference (XNLI), named entity recognition (NER), question answering (QA), and the GLUE benchmark (Reference 4). Performance may vary for other NLP tasks and domains.

Population subgroups:
- The model shows strong cross-lingual transfer and performance on low-resource languages like Swahili and Urdu when pretrained on CC data (Reference 1). However, performance disparities likely still exist between high-resource and low-resource languages.
- [More Information Needed] on performance across other population subgroups, such as dialects, age groups, etc. Evaluation should ideally be disaggregated across these factors.

Other factors:
- Model performance is impacted by the number of languages it was pretrained on, the size of the shared vocabulary, and the sampling rate of training examples from each language (Reference 9).
- Using a SentencePiece model for tokenization instead of language-specific tools makes the model easier to use on raw text without hurting performance (Reference 11).

To uncover potential performance disparities, more disaggregated evaluation is needed across languages, domains, and population subgroups. The provided references do not contain enough information to fully characterize the model's behavior across all relevant factors.

#### Metrics

Based on the provided references, the following metrics will be used to evaluate the XLM-RoBERTa-base model:

1. Cross-lingual Natural Language Inference (XNLI): Accuracy
2. Named Entity Recognition (NER): F1 score
3. Cross-lingual Question Answering (MLQA): F1 score and Exact Match (EM) score
4. GLUE benchmark: [More Information Needed] (The specific metrics for each GLUE task are not mentioned in the provided references)

The references also mention the trade-off between transfer learning and per-language capacity, known as the "transfer-dilution trade-off" or "curse of multilinguality". This suggests that while adding more languages can improve performance on low-resource languages, it may lead to an overall decrease in downstream performance due to limited model capacity.

### Results

Based on the provided references, here are the evaluation results for the XLM-RoBERTa-base model:

Cross-lingual Natural Language Inference (XNLI):
- Evaluated on cross-lingual transfer from English to 14 other languages
- Used machine translation baselines: translate-test, translate-train (per-language), and translate-train-all (multi-language)
- [More Information Needed] for specific scores

Named Entity Recognition (NER):
- Evaluated on CoNLL-2002 and CoNLL-2003 datasets in English, Dutch, Spanish, and German
- Fine-tuned for (1) cross-lingual transfer from English, (2) per-language performance, and (3) multilingual learning
- Reported F1 score
- [More Information Needed] for specific scores and comparison to baselines

Cross-lingual Question Answering (MLQA):
- Evaluated on MLQA benchmark, extending SQuAD to Spanish, German, Arabic, Hindi, Vietnamese, and Chinese
- Reported F1 score and exact match (EM) score for cross-lingual transfer from English
- [More Information Needed] for specific scores

GLUE Benchmark:
- Evaluated English performance on GLUE benchmark
- [More Information Needed] for comparison to RoBERTa and specific scores

The references also mention a comparison between XLM and BERT on seven languages using a BERT-base architecture, but [More Information Needed] for the specific results of this comparison for the XLM-RoBERTa-base model.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the XLM-RoBERTa-base (XLM-R) model:

- Cross-lingual Natural Language Inference (XNLI): XLM-R sets a new state of the art, obtaining 80.9% accuracy on cross-lingual transfer. It outperforms XLM-100 and mBERT models by 10.2% and 14.6% average accuracy, respectively. On low-resource languages like Swahili and Urdu, XLM-R significantly outperforms both XLM-100 and mBERT.

- Named Entity Recognition (NER): XLM-R performs on par with the state of the art, even without using a linear-chain CRF on top of the representations. It outperforms Akbik et al. (2018) on Dutch by 2.09 points.

- Question Answering (MLQA): XLM-R achieves new state of the art results, obtaining F1 and accuracy scores of 70.7% and 52.7%, outperforming the previous best by a significant margin. It also outperforms mBERT by 13.0% F1-score and 11.1% accuracy, and even surpasses BERT-Large on English.

- GLUE Benchmark: [More Information Needed]

The references also suggest that multilingual models like XLM can outperform their monolingual BERT counterparts on the XNLI benchmark, as demonstrated by a comparison on 7 languages.

## Model Examination

Based on the provided references, here is the experimental section on explainability/interpretability for the XLM-RoBERTa-base model:

XLM-RoBERTa-base is a multilingual masked language model trained on a large corpus of filtered CommonCrawl data in 100 languages. The model follows the XLM approach closely, with improvements in the learning procedure inspired by RoBERTa to achieve better performance.

Explainability and interpretability are important aspects to consider when developing and deploying deep learning models. For XLM-RoBERTa-base, [More Information Needed] on specific techniques or methods used to analyze and interpret the model's predictions and internal representations.

Further research and experiments could be conducted to gain insights into how XLM-RoBERTa-base captures and processes cross-lingual information. This may involve techniques such as:

- [More Information Needed]
- [More Information Needed]
- [More Information Needed]

By investigating the model's behavior and decision-making process, we can better understand its strengths, limitations, and potential biases. This knowledge can help in developing more transparent and trustworthy multilingual models.

As the development of XLM-RoBERTa-base progresses, we aim to incorporate more explainability and interpretability techniques to provide a clearer understanding of the model's inner workings and improve its reliability in cross-lingual natural language understanding tasks.

## Environmental Impact

- **Hardware Type:** Based on the provided references, XLM-R (XLM-RoBERTa) was trained on five-hundred 32GB Nvidia V100 GPUs with a batch size of 8192, as stated in reference 6:

"We use the multilingual MLM loss and train our XLM-R model for 1.5 Million updates on five-hundred 32GB Nvidia V100 GPUs with a batch size of 8192."
- **Software Type:** Based on the references provided, the XLM-RoBERTa (XLM-R) model is trained on CommonCrawl data that has been filtered and cleaned. Specifically, from reference 6:

"It is trained on `2.5T` of filtered CommonCrawl data in 100 languages."

So in summary, the software type that the XLM-RoBERTa Base model is trained on is filtered and cleaned CommonCrawl data.
- **Hours used:** [More Information Needed]

The provided references do not contain specific information about the training time for the XLM-RoBERTa-base model. More details would be needed from the model training logs or documentation to determine the exact amount of time used for training.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the XLM-RoBERTa-base model. More information would be needed to determine which cloud provider, if any, was used in the training process.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the carbon emissions from training the XLM-RoBERTa-base model. To answer this question, we would need more specific details on the computational resources used during training, such as the hardware specs, training duration, energy consumption, etc. Without that information, it is not possible to estimate the carbon footprint of the model training process.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
- XLM-R Base has an architecture similar to BERT Base, with L=12 layers, H=768 hidden size, A=12 attention heads, and 270M parameters total.
- It uses a large vocabulary size of 250K with a full softmax.

Training Objective:
- XLM-R is trained with the multilingual MLM (masked language modeling) objective, following the XLM approach.
- It only uses monolingual data, with training data sampled from different languages.
- The model is trained to predict the masked tokens in the input.

Other Details:
- XLM-R is trained on CommonCrawl data in 100 languages, with a total of 2.5T tokens after filtering.
- It uses SentencePiece tokenization with a unigram language model.
- Language embeddings are not used, allowing the model to better handle code-switching.
- [More Information Needed] on the specific training hyperparameters like batch size, learning rate, etc.

### Compute Infrastructure

Based on the provided references, here is the compute infrastructure information I could find about the XLM-RoBERTa Base model:

The XLM-RoBERTa Base model has an architecture with:
- L = 12 layers
- H = 768 hidden states 
- A = 12 attention heads

The model was trained for 1.5 million updates on five-hundred 32GB Nvidia V100 GPUs with a batch size of 8192.

[More Information Needed] on the exact training time and any other compute infrastructure details.

## Citation

```
@misc{alexis-unsupervised,
    author = {Alexis Conneau and
              Kartikay Khandelwal and
              Naman Goyal and
              Vishrav Chaudhary and
              Guillaume Wenzek and
              Francisco Guzmán and
              Edouard Grave and
              Myle Ott and
              Luke Zettlemoyer and
              Veselin Stoyanov and
              Facebook Ai},
    title  = {Unsupervised Cross-lingual Representation Learning at Scale},
    url    = {https://arxiv.org/pdf/1911.02116.pdf}
}
```

