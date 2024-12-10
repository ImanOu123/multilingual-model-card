# Model Card for bert-base-multilingual-uncased

The BERT-Base, Multilingual Uncased model is a language representation model that supports 102 languages and is designed to pretrain deep bidirectional representations from unlabeled text, which can then be fine-tuned for various downstream tasks. It features 12 layers, 768 hidden units, 12 attention heads, and 110 million parameters, and is pre-trained using a "masked language model" (MLM) objective to understand context from both left and right directions.

## Model Details

### Model Description

Model Name: BERT-Base, Multilingual Uncased

Model Architecture:
The BERT-Base, Multilingual Uncased model is built upon a multi-layer bidirectional Transformer encoder, as described in Vaswani et al. (2017). It consists of 12 layers (i.e., Transformer blocks), with a hidden size of 768 and 12 self-attention heads, totaling approximately 110 million parameters. This architecture allows the model to process text by jointly conditioning on both left and right context in all layers, which is a key differentiator from unidirectional models.

Training Procedures:
BERT-Base, Multilingual Uncased was pre-trained on a large corpus comprising the BooksCorpus (800M words) and Wikipedia in 102 languages (2,500M words). The pre-training involved two unsupervised tasks: Masked Language Model (MLM) and Next Sentence Prediction (NSP). For the MLM task, 15% of the input tokens were masked at random, and the model was trained to predict these masked tokens. The model was trained for 1 million steps with a batch size of 128,000 words. During pre-training, masked words were replaced with the [MASK] token 80% of the time, with a random token 10% of the time, and the remaining 10% of the time, the original token was kept unchanged.

Parameters:
- Number of layers (L): 12
- Hidden size (H): 768
- Number of self-attention heads (A): 12
- Total parameters: 110M

Important Disclaimers:
- The model is the uncased variant, which means it does not distinguish between upper and lower case letters. This may affect performance on tasks where case information is important.
- The model is pre-trained on multiple languages, but it is not recommended for use in new projects as there is a newer, recommended cased variant available. The cased variant may perform better on tasks involving languages with case distinctions.
- There is a potential mismatch between pre-training and fine-tuning because the [MASK] token used during pre-training does not appear during fine-tuning. This is mitigated by not always replacing masked words with the actual [MASK] token during training.

For further inquiries or updates regarding the model card, please contact the project organizer. [More Information Needed] for the contact details.

- **Developed by:** Jacob Devlin; Ming-Wei Chang; Kenton Lee; Kristina Toutanova
- **Funded by:** The references provided do not explicitly mention the people or organizations that funded the project for the model bert-base-multilingual-uncased. However, given that BERT was introduced by researchers from Google and the pre-trained models are available on Google's storage (as indicated by the URLs provided in reference 7), it is reasonable to infer that Google, or more specifically Google Research, funded the development of the BERT model, including its multilingual uncased variant.

If more specific information on funding is required, it would be necessary to consult the original paper or additional documentation related to the project for details on funding sources. Without such information in the provided references, the answer would be:

[More Information Needed]
- **Shared by:** The contributors that made the model `bert-base-multilingual-uncased` available online as a GitHub repo are Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, as mentioned in reference 5.
- **Model type:** The bert-base-multilingual-uncased model is a deep bidirectional Transformer encoder trained using a masked language model (MLM) pre-training method, which is a type of unsupervised learning, and it is designed for natural language processing (NLP) tasks across multiple languages (modality).
- **Language(s):** The model bert-base-multilingual-uncased is designed to process text in 102 different languages, including both high-resource and low-resource languages, as part of its pre-training on a multilingual corpus.
- **License:** [More Information Needed]
- **Finetuned from model:** The model `bert-base-multilingual-uncased` is not fine-tuned from another model; rather, it is a base model itself. It is one of the BERT (Bidirectional Encoder Representations from Transformers) models that has been pre-trained on a large corpus of multilingual data. The references do not provide a link to a different base model from which `bert-base-multilingual-uncased` was fine-tuned. Instead, they describe the BERT model's pre-training process and its capabilities.

However, if you are looking for the link to download the `bert-base-multilingual-uncased` model, it is provided in reference 6:

[`BERT-Base, Multilingual Uncased (Orig, not recommended)`](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip)
### Model Sources

- **Repository:** https://github.com/google-research/bert/blob/master/multilingual.md
- **Paper:** https://arxiv.org/pdf/1810.04805.pdf
- **Demo:** [More Information Needed]
## Uses

### Direct Use

The bert-base-multilingual-uncased model can be used without fine-tuning for feature-based approaches, as mentioned in reference 2. This involves extracting token representations from the pre-trained model and using them as features in a downstream model. For example, one could extract the token representations from the top four hidden layers of the Transformer and concatenate them to use as input features for a separate classifier.

However, the references provided do not include a direct code snippet for using bert-base-multilingual-uncased without fine-tuning. Therefore, I cannot provide a code snippet without additional information. If you need to use the model without fine-tuning, you would typically load the pre-trained model, pass your input data through it, and extract the necessary features from the hidden states.

Since the references do not contain a specific code example for this use case, the response is: [More Information Needed].

### Downstream Use

The `bert-base-multilingual-uncased` model is a versatile deep learning model that can be fine-tuned for a wide range of language understanding tasks across multiple languages. As indicated in the references, fine-tuning involves initializing the BERT model with pre-trained parameters and then training it further on labeled data specific to the downstream task. This process adapts the model to the nuances of the particular task, whether it be text classification, question answering, or any other task that can be framed as involving single text or text pairs.

For example, in the case of text classification or sequence tagging, the model takes a single piece of text as input and outputs the classification labels or tags for each token. In question answering tasks, the model takes a pair of texts (a question and a passage) and predicts the answer from the passage. Similarly, for entailment, the model would take a pair of sentences and predict whether the second sentence logically follows from the first.

The fine-tuning process is straightforward and involves selecting task-specific hyperparameters such as batch size, learning rate, and number of training epochs, which have been found to work well across various tasks. The dropout probability is typically kept at 0.1 during fine-tuning.

Once fine-tuned, the `bert-base-multilingual-uncased` model can be easily integrated into a larger ecosystem or application. For instance, it can be used to power a multilingual chatbot, perform sentiment analysis on user feedback across different languages, or enhance search engines by improving their understanding of queries in various languages.

Here is a general example of how to fine-tune the model on a text classification task using the Huggingface Transformers library. Note that this is a conceptual code snippet and may require additional context such as the dataset and task-specific details:

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased')

# Tokenize the input data
inputs = tokenizer("Example input text", return_tensors="pt", padding=True, truncation=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # provide the training dataset
    eval_dataset=eval_dataset,    # provide the evaluation dataset
)

# Train the model
trainer.train()
```

Please note that the actual code for fine-tuning will depend on the specific task, dataset, and the hyperparameters chosen based on the development set performance. The above code is a simplified example and does not include the actual dataset or the full fine-tuning process.

### Out-of-Scope Use

As a sociotechnic on the deep learning model development team, it is my responsibility to consider the potential societal impacts and misuse of the `bert-base-multilingual-uncased` model. Here are some considerations for how the model may be misused and guidance on what users should avoid doing with the model:

1. **Biased Output**: Given that BERT models learn from large datasets of text from the internet, there is a risk that the model may inadvertently learn and perpetuate biases present in the training data. Users should not use the model in applications where biased outputs could lead to discrimination or unfair treatment of individuals based on race, gender, sexual orientation, or other protected characteristics.

2. **Misinformation and Manipulation**: The model's ability to generate and modify text can be misused to create convincing fake news, impersonate individuals online, or manipulate public opinion. Users should not employ the model for generating or spreading misinformation or for deceptive purposes.

3. **Privacy Violations**: The model could potentially be used to infer private information from text data if not handled properly. Users should not use the model to analyze sensitive text data in a way that could violate privacy rights or data protection laws.

4. **Content Generation for Malicious Purposes**: The model could be used to automate the generation of harmful content, such as hate speech, phishing emails, or spam. Users should not use the model to create or disseminate content that is illegal, unethical, or intended to cause harm.

5. **Intellectual Property Infringement**: The model's ability to generate text based on input could lead to the creation of derivative works that infringe on copyright. Users should not use the model to generate content that violates intellectual property rights.

6. **Language and Cultural Sensitivity**: As a multilingual model, it is important to consider the cultural context when using the model across different languages. Users should not use the model in a way that disregards cultural nuances and sensitivities, which could lead to miscommunication or offense.

7. **Security Risks**: The model could be used in the development of systems that are vulnerable to adversarial attacks, where small changes to input data can lead to incorrect outputs. Users should not rely on the model for security-critical applications without thorough testing and safeguards.

In conclusion, while `bert-base-multilingual-uncased` is a powerful tool for natural language processing tasks, it is crucial that users employ the model responsibly, with consideration for ethical implications, societal norms, and legal constraints. Users should actively work to mitigate potential misuse and ensure that applications of the model are aligned with the principles of fairness, accountability, and transparency.

### Bias, Risks, and Limitations

The model card for bert-base-multilingual-uncased should address several known and foreseeable issues related to the model's performance, potential harms, misunderstandings, and technical and sociotechnical limitations:

1. **Performance on Low-Resource Languages**: While BERT models have shown improvements for low-resource tasks (Reference 1), the performance of bert-base-multilingual-uncased on languages with limited training data may still lag behind that of high-resource languages. This could lead to inequities in the quality of NLP applications across different languages.

2. **Model Size and Fine-Tuning**: Reference 4 and 5 suggest that increasing model size can lead to improvements on both large-scale and small-scale tasks. However, the bert-base-multilingual-uncased model has a fixed size (Reference 8), which means that it may not capture the nuances of all languages equally well. Additionally, fine-tuning on specific tasks may require careful hyperparameter tuning to achieve optimal performance.

3. **Bias and Fairness**: Since BERT models are pre-trained on large text corpora that may contain biases, bert-base-multilingual-uncased could inadvertently perpetuate or amplify these biases. This is a significant sociotechnical issue as it can affect fairness and representation in downstream applications.

4. **Contextual Understanding**: The use of a "masked language model" (MLM) pre-training objective (Reference 9) allows the model to understand context bidirectionally. However, there may still be limitations in the model's ability to fully grasp the nuances of all languages, especially those with complex syntax or less representation in the training data.

5. **Generalization Across Tasks**: While BERT models are designed to be fine-tuned on a variety of downstream tasks (Reference 10), there may be limitations in how well the bert-base-multilingual-uncased model generalizes across different NLP tasks, especially those that are not well-represented in the pre-training corpus or that require a deep understanding of cultural context.

6. **Technical Limitations**: The model has a significant number of parameters (Reference 8), which may pose challenges for deployment in resource-constrained environments. Additionally, the computational cost of fine-tuning and inference may limit accessibility for some users.

7. **Misunderstandings and Misuse**: There is a risk that users may misunderstand the capabilities of the model or use it in contexts for which it was not intended or adequately evaluated. This could lead to incorrect or harmful outcomes, particularly if the model's limitations are not clearly communicated.

8. **Language Representation**: The model's representation of 102 languages (Reference 8) may not be equitable, as some languages may be underrepresented in the training corpus. This could lead to poorer performance for speakers of those languages and a lack of linguistic diversity in NLP applications.

In conclusion, while bert-base-multilingual-uncased represents a significant advancement in multilingual language processing, it is important to be aware of and address its limitations and potential for harm in order to ensure responsible and equitable use.

### Recommendations

As the sociotechnic on the deep learning model development team, my recommendations with respect to the foreseeable issues about the model `bert-base-multilingual-uncased` are as follows:

1. **Language Representation and Cultural Sensitivity**: The uncased model strips accent markers, which can lead to a loss of meaning in languages where accents are significant. While BERT's contextual abilities may compensate for some ambiguity, it is important to acknowledge that this may not be sufficient for all cases. Users should be aware that for languages heavily reliant on accent markers, the model might not perform optimally, potentially leading to misinterpretations or biases in language understanding.

2. **Model Generalization**: The model has been trained on 102 languages, which is an impressive coverage. However, users should consider the depth and quality of the training data for each language. Some languages may have been represented with less data or in less diverse contexts, which could affect the model's performance in those languages. It is recommended to evaluate the model's performance on a case-by-case basis, especially for low-resource languages.

3. **Ethical Considerations and Fairness**: Given the broad application of the model across different languages and cultural contexts, it is crucial to consider the ethical implications of its deployment. Users should assess whether the model's predictions might perpetuate or amplify biases present in the training data. It is recommended to conduct fairness assessments and to be transparent about the limitations of the model in this regard.

4. **Recommendation on Usage**: The reference indicates that the cased version of the multilingual BERT model is recommended over the uncased version. Users should take this into consideration and prefer the cased model when possible, especially for tasks where case information could be critical for understanding the input text.

5. **Ablation Studies and Model Understanding**: It is important for users to understand the impact of different model components and training objectives on performance. The reference to ablation studies suggests that certain facets of BERT, such as the bidirectionality and the MLM objective, are crucial. Users should review these studies to better understand how the model processes language and to make informed decisions about its application.

6. **Legal and Compliance Issues**: When deploying the model in applications that may affect individuals or groups, it is important to consider legal and compliance issues, particularly around privacy and data protection laws. Users should ensure that the use of the model complies with all relevant regulations and that any data used for fine-tuning or inference respects the rights of individuals.

In summary, while `bert-base-multilingual-uncased` is a powerful tool for understanding and processing multiple languages, it is important to be mindful of its limitations and the potential societal impact of its deployment. Users should conduct thorough testing and consider ethical, cultural, and legal aspects when applying the model to real-world tasks.

## Training Details

### Training Data

The training data for the model `bert-base-multilingual-uncased` consists of text from the BooksCorpus and English Wikipedia, as well as Wikipedia dumps from the top 100 languages by size, excluding user and talk pages. The data is balanced using exponentially smoothed weighting to prevent over-representation of high-resource languages and under-representation of low-resource languages.

### Training Procedure

#### Preprocessing

The preprocessing for the data of the model bert-base-multilingual-uncased involves several steps:

1. **Tokenization**: We use a shared WordPiece tokenizer with a vocabulary size of 110,000 tokens. The tokenizer performs lower casing and accent removal for all languages as part of its normalization process. This is done to reduce the effective vocabulary size and is expected to be compensated by BERT's strong contextual modeling capabilities. [Reference 1, 4]

2. **Sentence Pair Construction**: During pre-training, we create input sequences by sampling two spans of text from the corpus, which we refer to as "sentences." These spans can be longer or shorter than typical sentences. One span is assigned the A embedding and the other the B embedding. In 50% of cases, the second span (B) is the actual next sentence that follows the first span (A), and in the other 50%, it is a random sentence. This is done for the next sentence prediction task. [Reference 2]

3. **Sequence Length**: The combined length of the two spans is limited to 512 tokens or fewer. [Reference 2]

4. **Masking**: After tokenization, we apply a uniform masking rate of 15% to the tokens for the masked language model (MLM) pre-training objective. No special consideration is given to partial word pieces when applying the mask. [Reference 3]

5. **Batch Size and Training Steps**: We train with a batch size of 256 sequences for 1,000,000 steps, which is approximately 40 epochs over the 3.3 billion word corpus. [Reference 3]

6. **Optimization**: We use the Adam optimizer with a learning rate of 1e-4, β1 = 0.9, β2 = 0.999, and L2 weight decay of 0.01. We also employ a learning rate warmup over the first 10,000 steps followed by linear decay. [Reference 3]

7. **Data Weighting**: To address the representation of low-resource languages, we perform exponentially smoothed weighting during pre-training data creation and WordPiece vocabulary creation. This means that the probability of selecting a language for training is adjusted to prevent under-representation of low-resource languages. [Reference 6, 7]

8. **Special Tokens**: We use special tokens such as [CLS] at the beginning of every sequence and [SEP] to separate sentence pairs. Additionally, a learned embedding is added to every token to indicate whether it belongs to sentence A or B. [Reference 8]

9. **Fine-tuning**: For fine-tuning on specific tasks, we use the pre-trained model and adjust all parameters end-to-end based on the task-specific inputs and outputs. [Reference 9]

10. **Masked Language Model (MLM)**: The MLM objective involves predicting the original vocabulary id of randomly masked tokens based solely on their context, which allows for bidirectional context understanding. [Reference 10]

In summary, the preprocessing for bert-base-multilingual-uncased involves tokenization with lower casing and accent removal, sentence pair construction with special embeddings, sequence length limitation, uniform masking for MLM, and data weighting to balance language representation.

#### Training Hyperparameters

The training hyperparameters for the model bert-base-multilingual-uncased are as follows:

- The model was trained on the BooksCorpus (800M words) and Wikipedia (2,500M words) as per reference 2.
- It was trained for 1M steps with a batch size of 128,000 words, also mentioned in reference 2.
- A task-specific fine-tuning learning rate was chosen, which performs the best on the development set, rather than using a single learning rate for all fine-tuning experiments. The exact learning rate is not specified in the provided references, so [More Information Needed] for the exact value.
- For the "masked LM" (MLM) pre-training objective, 15% of all WordPiece tokens in each sequence were masked, as stated in reference 3.
- When masking tokens, 80% of the time the token was replaced with the [MASK] token, 10% of the time with a random token, and the remaining 10% of the time the original token was kept unchanged. This detail is provided in reference 4 but the exact behavior for the remaining 10% is not specified in the provided references, so [More Information Needed] for that detail.
- The model architecture is a 12-layer Transformer with 768 hidden units, 12 attention heads, and a total of 110M parameters, as described in reference 6.

Additional hyperparameters such as the optimizer used, learning rate schedule, and exact values for fine-tuning learning rates are not provided in the references, so [More Information Needed] for those specifics.

#### Speeds, Sizes, Times

The model `bert-base-multilingual-uncased` is a multilingual variant of the BERT model that has been trained on text from 102 languages. It is designed with 12 layers, 768 hidden units, 12 attention heads, and has a total of 110 million parameters. This model is the uncased version, meaning that the text has been lowercased before WordPiece tokenization, and it is recommended for multilingual applications.

Regarding the specific details requested:

- Throughput: [More Information Needed]
- Start or End Time: [More Information Needed]
- Checkpoint Sizes: The model has 110 million parameters, which suggests a significant checkpoint size, but the exact size in megabytes (MB) or gigabytes (GB) is not provided in the references and would depend on the precision of the weights (e.g., 32-bit vs. 16-bit floating-point).

It's important to note that the references provided do not include specific information about the throughput, start or end time, or checkpoint sizes. These details are typically obtained during the training process or from the model hosting platform and would need to be recorded by the development team during the model's training and evaluation phases.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model `bert-base-multilingual-uncased` has been evaluated on the following benchmarks and datasets:

1. **XNLI dataset**: This dataset is a version of MultiNLI where the development and test sets have been translated into 15 languages. The training set was machine-translated, and this dataset was used to evaluate the performance of the model across multiple languages (Reference 2).

2. **GLUE benchmark**: The General Language Understanding Evaluation benchmark is a collection of diverse natural language understanding tasks. The model's performance on this benchmark has been reported, although the specific results for the multilingual uncased model are not detailed in the provided references (Reference 8).

3. **SQuAD v1.1 and v2.0**: These are question answering datasets. While the references mention improvements in performance on these datasets with BERT models, they do not specify the exact performance of the `bert-base-multilingual-uncased` model on these datasets (Reference 9).

For more detailed performance metrics and evaluations on additional datasets, [More Information Needed].

#### Factors

The performance and behavior of the `bert-base-multilingual-uncased` model are influenced by several factors, including language resources, pre-training methodology, and the nature of the datasets used for evaluation. Here are some characteristics to consider:

1. **Language Resources**: The model is trained on multiple languages, which means that for high-resource languages like English or Chinese, the multilingual model may perform somewhat worse than a single-language model. This is because the model has to generalize across multiple languages, which can dilute its performance on any single language. For low-resource languages, however, the multilingual model may be the best available option since single-language models may not exist due to the lack of sufficient training data.

2. **Pre-training Methodology**: The model uses bidirectional pre-training, which allows it to learn deep bidirectional representations by conditioning on both left and right context. This is in contrast to unidirectional models, which may only condition on one direction of context. The bidirectional nature of BERT is crucial for understanding the context of words in sentences, which can significantly affect the model's performance across different languages and domains.

3. **Quality of Machine Translation**: The training set for the model was machine-translated into various languages. The quality of this machine translation can affect the model's understanding and representation of those languages. If the machine translation is poor, it may lead to lower accuracy when the model is used for tasks in those languages.

4. **Domain and Context**: The model's performance can vary depending on the domain and context in which it is used. For example, if the model is fine-tuned on legal or medical texts, its performance on general language tasks may not be as high. Similarly, if the model is used in a domain that is underrepresented in the training data, it may not perform as well.

5. **Population Subgroups**: The model's performance may also vary across different population subgroups, particularly if those subgroups use language in ways that are not well represented in the training data. This could include regional dialects, sociolects, or language used by specific communities.

6. **Evaluation Disaggregation**: To uncover disparities in performance, evaluation should be disaggregated across factors such as language, domain, and demographic characteristics of the population. This can help identify where the model performs well and where it may need further fine-tuning or additional training data.

In summary, the `bert-base-multilingual-uncased` model's behavior is influenced by the diversity of languages it is trained on, the bidirectional nature of its pre-training, the quality of machine translations used for training, the domain and context of its application, and the representation of various population subgroups in the training data. Disaggregated evaluation is essential to understand these influences and to improve the model's performance across different factors.

#### Metrics

For evaluating the `bert-base-multilingual-uncased` model, we will primarily use the XNLI dataset as mentioned in reference 2. The metrics for evaluation will include accuracy on the XNLI test sets, which have been translated into multiple languages. This will allow us to assess the model's performance across different languages.

From reference 3, we see that the model's performance is compared with both a multilingual baseline and single-language models. Therefore, we will also consider the performance difference between the multilingual model and single-language models as a metric, particularly for high-resource languages like English and Chinese, where the single-language models perform better.

Reference 4 introduces the concept of "Translate Train," which implies that we should be aware of the potential impact of machine translation quality on training data and its subsequent effect on model accuracy. While this does not provide a direct metric, it suggests that we should be cautious when interpreting performance in non-English languages.

Reference 5 discusses the "Zero Shot" setting, where the model is fine-tuned on English data and then evaluated on foreign language data without any machine translation involved. This setting allows us to evaluate the model's cross-lingual transfer capabilities, which is another important metric for our multilingual model.

Lastly, reference 6 and 7 mention the importance of the model's performance on a wide range of tasks and the effect of different masking strategies during pre-training. While these references do not provide specific metrics for our model, they highlight the need to evaluate the model on diverse NLP tasks and consider the impact of pre-training strategies on downstream task performance.

In summary, the metrics for evaluating `bert-base-multilingual-uncased` will include:
- Accuracy on the XNLI test sets across multiple languages.
- Performance comparison with single-language models, especially for high-resource languages.
- Evaluation of cross-lingual transfer capabilities in a "Zero Shot" setting.
- Consideration of the impact of machine translation quality on training data accuracy.
- Performance on a range of NLP tasks to assess the model's generalizability.

[More Information Needed] for any additional specific metrics or evaluation protocols that may have been used but are not mentioned in the provided references.

### Results

Evaluation Results of `bert-base-multilingual-uncased`:

### Factors:
1. **Language Coverage**: The model supports 102 languages, which makes it versatile for multilingual applications.
2. **Model Architecture**: It is a 12-layer transformer with 768-hidden units and 12-heads, totaling 110M parameters.
3. **Training Data**: The training set was machine-translated into various languages, which may affect the model's performance due to translation quality.
4. **Bidirectionality**: The model benefits from deep bidirectional training, although specific results related to the "next sentence prediction" (NSP) task are not provided here.

### Metrics:
1. **Performance on High-Resource Languages**: For English, the model performs worse than the English-only BERT baseline, indicating that for high-resource languages, a single-language model may be preferable.
2. **Chinese Language Results**:
   - XNLI Baseline: 67.0
   - BERT Multilingual Model: 74.2
   - BERT Chinese-only Model: 77.2
   This shows that the multilingual model underperforms compared to the Chinese-only model by about 3%.
3. **Training Methodology**:
   - **Translate Train**: The model was trained on machine-translated data, which could lead to lower accuracy due to translation quality issues.
   - **Zero Shot**: The model was fine-tuned on English MultiNLI and evaluated on foreign language XNLI tests without involving machine translation in pre-training or fine-tuning.
4. **Comparison with Other Models**: The model is not the recommended version for multilingual tasks as per the provided references. A cased version of the model is recommended instead.

[More Information Needed] on the specific evaluation metrics such as accuracy, F1 scores, or GLUE scores for the `bert-base-multilingual-uncased` model across different languages and tasks to provide a comprehensive evaluation.

#### Summary

The evaluation results for the `bert-base-multilingual-uncased` model can be summarized as follows:

1. When compared to the English-only BERT model, the Multilingual BERT model shows a decrease in performance for high-resource languages like English. This is evident from the English result being lower than the 84.2 MultiNLI baseline, indicating that for tasks in high-resource languages, a single-language model may perform better.

2. For Chinese, the `BERT Multilingual Model` achieved a score of 74.2, which is higher than the XNLI Baseline score of 67.0 but lower than the `BERT Chinese-only Model` score of 77.2. This suggests that while the multilingual model is effective, there is a performance trade-off compared to a monolingual model, with a difference of about 3%.

3. The evaluation was conducted using the XNLI dataset, which includes human-translated dev and test sets in 15 languages, while the training set was machine-translated.

4. Specific performance metrics for the `bert-base-multilingual-uncased` model on the XNLI dataset are as follows:
   - BERT - Translate Train Uncased: 81.4 (English), 74.2 (Chinese), 77.3, 75.2, 70.5, 61.7 (other languages not specified).
   - BERT - Translate Test Uncased: 81.4 (English), 70.1 (Chinese), 74.9, 74.4, 70.4, 62.1 (other languages not specified).
   - BERT - Zero Shot Uncased: 81.4 (English), 63.8 (Chinese), 74.3, 70.5, 62.1, 58.3 (other languages not specified).

5. The term "Zero Shot" refers to the scenario where the Multilingual BERT model was fine-tuned on English MultiNLI and then evaluated on the foreign language XNLI test without involving machine translation in pre-training or fine-tuning.

6. The `bert-base-multilingual-uncased` model is designed to support 102 languages with 12 layers, 768 hidden units, 12 attention heads, and a total of 110 million parameters.

7. BERT models, including the multilingual variant, are based on the Transformer architecture and are pre-trained using a bidirectional context, which is a departure from previous unidirectional language models.

8. The BERT model uses a "masked language model" (MLM) pre-training objective, where it predicts the original vocabulary id of masked words based on their context, allowing for deep bidirectional representations.

In summary, the `bert-base-multilingual-uncased` model is a versatile tool that supports a wide range of languages, but it may not always achieve the same level of performance as monolingual models in high-resource languages. It is particularly useful when a single model is needed to handle multiple languages, especially when resources for individual language models are limited.

## Model Examination

In the development of our model, bert-base-multilingual-uncased, we have prioritized understanding the inner workings and decision-making processes of the model to ensure transparency and interpretability. This aligns with the broader goals of the BERT framework, which emphasizes the importance of deep bidirectional representations for language understanding.

Our model, which is a variant of BERT BASE, does not include the "next sentence prediction" (NSP) task in its pretraining objectives, focusing solely on the "masked language model" (MLM) pre-training objective. This design choice was made to investigate the impact of the NSP task on the model's performance and to simplify the pretraining process. The MLM objective, inspired by the Cloze task, involves randomly masking some of the tokens from the input and predicting the original vocabulary id of the masked word based only on its context. This allows the model to learn rich contextual representations from unlabeled text.

As part of our commitment to explainability, we have conducted ablation experiments to better understand the relative importance of various facets of BERT, including the impact of bidirectionality and the MLM objective on the model's performance. These experiments help us to dissect the contributions of different components of the model to its overall effectiveness. Additional details on these ablation studies can be found in Appendix C of our documentation.

BERT's design allows for fine-tuning with just one additional output layer, which reduces the need for heavily-engineered task-specific architectures. This simplicity also aids in interpretability, as the model's behavior can be more easily attributed to its pre-trained representations rather than complex task-specific adjustments.

Our model advances the state of the art for multiple NLP tasks, demonstrating the effectiveness of pre-trained representations in reducing the complexity of NLP systems. By sharing our code and pre-trained models, we hope to contribute to the community's efforts in developing more interpretable and explainable NLP systems.

For further details on the interpretability and explainability aspects of bert-base-multilingual-uncased, we encourage users to refer to our documentation and the additional implementation and ablation details provided in the appendices.

## Environmental Impact

- **Hardware Type:** The model bert-base-multilingual-uncased was trained on 4 Cloud TPUs in Pod configuration (16 TPU chips total).
- **Software Type:** The software that the model bert-base-multilingual-uncased is trained on is not explicitly mentioned in the provided references. However, reference 8 mentions that BERT's model architecture is based on the original implementation described in Vaswani et al. (2017) and released in the tensor2tensor library. Therefore, it can be inferred that the software used for training the bert-base-multilingual-uncased model includes the tensor2tensor library, which utilizes TensorFlow as its backend.

If you require the specific software version or any additional details about the software environment, that information is not provided in the references, and I would need to say "[More Information Needed]".
- **Hours used:** The amount of time used to train the model bert-base-multilingual-uncased is not explicitly stated in the provided references. However, reference 1 mentions that training of BERT BASE was performed on 4 Cloud TPUs in Pod configuration and took 4 days to complete. Since bert-base-multilingual-uncased is a BERT BASE model, we can infer that it likely took approximately 4 days to train.
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture of bert-base-multilingual-uncased is based on a multi-layer bidirectional Transformer encoder, as described in Vaswani et al. (2017). This architecture is characterized by the use of bidirectional self-attention, which allows the model to jointly condition on both left and right context in all layers. The implementation of this architecture is almost identical to the original Transformer model and is detailed in the tensor2tensor library [Reference 3].

The objective of bert-base-multilingual-uncased is to pre-train deep bidirectional representations from unlabeled text. This is achieved through a "masked language model" (MLM) pre-training objective, which is inspired by the Cloze task. In this MLM task, a certain percentage of the input tokens are masked at random, and the model is trained to predict the original vocabulary id of the masked words based solely on their context. This approach enables the model to learn a deep understanding of language context and structure, as it must infer the missing information from the surrounding words [References 4, 5, 7, 10, 11].

Unlike traditional language models that are trained in a unidirectional manner, either left-to-right or right-to-left, the MLM approach allows for bidirectional pre-training, which has been shown to be important for capturing rich language representations [References 6, 11]. The bert-base-multilingual-uncased model, therefore, represents a significant advancement in the state of the art for natural language processing tasks, as it can be fine-tuned with just one additional output layer to achieve high performance across a variety of sentence-level and token-level tasks [References 8, 9].

### Compute Infrastructure

The compute infrastructure used for the model bert-base-multilingual-uncased involved training on 4 Cloud TPUs in Pod configuration, which equates to 16 TPU chips in total. The training duration is not specified for the base model, but as a reference, it is mentioned that each pretraining of BERT LARGE took 4 days to complete on 16 Cloud TPUs (64 TPU chips total). However, since the exact training time for the bert-base-multilingual-uncased model is not provided in the references, [More Information Needed] for that specific detail.

## Citation

```
@misc{jacob-bert,
    author = {Jacob Devlin and
              Ming-Wei Chang and
              Kenton Lee and
              Kristina Toutanova},
    title  = {BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
    url    = {https://arxiv.org/pdf/1810.04805.pdf}
}
```

