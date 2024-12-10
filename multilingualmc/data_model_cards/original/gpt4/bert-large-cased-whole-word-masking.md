# Model Card for bert-large-cased-whole-word-masking

The BERT-Large, Cased (Whole Word Masking) model is a 24-layer Transformer with 1024 hidden units per layer, 16 attention heads, and a total of 340 million parameters, pre-trained on a large corpus including Wikipedia and BookCorpus. It uses whole word masking during pre-training, where all of the tokens corresponding to a word are masked at once, allowing the model to better understand the context of each word in a bidirectional manner.

## Model Details

### Model Description

Model Card for BERT-Large, Cased (Whole Word Masking)

## Model Architecture
BERT-Large, Cased (Whole Word Masking) is a deep learning model based on the Transformer architecture. It consists of 24 layers of Transformer blocks, with 1024 hidden units in each layer and 16 attention heads. The model has a total of 340 million parameters.

## Training Procedure
The model was pre-trained on a large corpus comprising Wikipedia and the BookCorpus, amounting to a combined total of 3.3 billion words. The pre-training involved two main tasks: masked language modeling (MLM) and next sentence prediction (NSP). During MLM, 15% of the input tokens were masked at random, and the model was trained to predict these masked tokens. The masking strategy involved replacing the chosen token with the [MASK] token 80% of the time, with a random token 10% of the time, and leaving the original token unchanged 10% of the time.

The model was trained for 1,000,000 steps with a batch size of 256 sequences, each sequence containing 512 tokens. This equates to approximately 40 epochs over the training corpus. The Adam optimizer was used with a learning rate of 1e-4, β1 = 0.9, β2 = 0.999, and L2 weight decay of 0.01. The learning rate was warmed up over the first 10,000 steps and then linearly decayed.

## Parameters
- Number of layers: 24
- Hidden units per layer: 1024
- Attention heads: 16
- Total parameters: 340 million

## Important Disclaimers
- The model is deeply bidirectional, which is a significant departure from previous models that were unidirectional or shallowly bidirectional.
- The pre-training task of MLM creates a mismatch between pre-training and fine-tuning since the [MASK] token does not appear during fine-tuning. This is mitigated by the varied masking strategy during training.
- The model is cased, meaning it is sensitive to the case of the input tokens, which can affect its performance on downstream tasks.
- The whole word masking variant of BERT masks entire words instead of subword tokens, which may lead to better performance on tasks that require understanding of word-level semantics.

For any updates or further inquiries regarding the model, please contact the project organizer responsible for the model card.

- **Developed by:** Jacob Devlin; Ming-Wei Chang; Kenton Lee; Kristina Toutanova
- **Funded by:** [More Information Needed]
- **Shared by:** The contributors that made the model `bert-large-cased-whole-word-masking` available online as a GitHub repo are Jacob Devlin, Ming-Wei Chang, and Kenton Lee.
- **Model type:** The model bert-large-cased-whole-word-masking is a deep bidirectional transformer-based model trained using a masked language model (MLM) pre-training objective, which is a type of self-supervised learning, and it operates on textual modality with cased sensitivity.
- **Language(s):** The model bert-large-cased-whole-word-masking processes text in natural human language that preserves the original case and accent markers.
- **License:** The model `bert-large-cased-whole-word-masking` is released under the Apache 2.0 license. The link to the license is not directly provided in the references above, but it can typically be found in the `LICENSE` file in the repository where the code and models are hosted. Since all code and models are mentioned to be released under this license, you can refer to the standard Apache 2.0 license documentation for more details.

License Name: Apache 2.0 license
License Link: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
- **Finetuned from model:** The model `bert-large-cased-whole-word-masking` is fine-tuned from the base model `BERT-Large, Cased`. The link to the base model is [BERT-Large, Cased (Whole Word Masking)](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip).
### Model Sources

- **Repository:** https://github.com/google-research/bert
- **Paper:** https://arxiv.org/pdf/1810.04805.pdf
- **Demo:** The link to the demo of the model `BERT-Large, Cased (Whole Word Masking)` is:

https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip
## Uses

### Direct Use

The model `bert-large-cased-whole-word-masking` can be used without fine-tuning by extracting pre-trained contextual embeddings for each input token. These embeddings are fixed representations generated from the hidden layers of the pre-trained model. This approach can be beneficial in scenarios where fine-tuning the entire model is not feasible, such as when dealing with out-of-memory issues.

To use the model in this way, you can utilize the provided script `extract_features.py`. This script allows you to input text and obtain the contextual embeddings from the model. Here is an example of how to use the script:

```shell
python extract_features.py \
  --input_file=<path_to_input_text> \
  --output_file=<path_to_output_jsonl> \
  --vocab_file=<path_to_vocab_file> \
  --bert_config_file=<path_to_config_file> \
  --init_checkpoint=<path_to_model_checkpoint> \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
```

Please replace `<path_to_input_text>`, `<path_to_output_jsonl>`, `<path_to_vocab_file>`, `<path_to_config_file>`, and `<path_to_model_checkpoint>` with the actual paths to your input file, desired output file, vocabulary file, BERT configuration file, and model checkpoint, respectively.

This script will output the token representations from the top four hidden layers of the pre-trained Transformer, as mentioned in the references. These representations can then be used for various downstream tasks without the need for fine-tuning the entire model.

### Downstream Use

The `bert-large-cased-whole-word-masking` model is a variant of BERT that has been pre-trained on a large corpus of text and is designed to understand the context of each word in a sentence more accurately by masking entire words during training. When fine-tuned for a specific task, this model can be adapted to perform a wide range of natural language processing (NLP) tasks, such as paraphrasing, entailment, question answering, text classification, and sequence tagging.

To fine-tune the model for a specific task, you would start by initializing the model with the pre-trained parameters. Then, you would prepare your dataset for the task at hand, ensuring that the input data is formatted correctly (e.g., sentence pairs for paraphrasing, hypothesis-premise pairs for entailment, etc.). The output layer of the model would be adjusted according to the task—for token-level tasks, the token representations would be fed into an output layer designed for that specific task.

During fine-tuning, most hyperparameters remain the same as in pre-training, with the exception of batch size, learning rate, and the number of training epochs. It's important to note that the optimal hyperparameter values are task-specific, and an exhaustive search over these parameters is recommended to find the best-performing model on the development set.

For integration into a larger ecosystem or app, the fine-tuned model can be used to provide NLP capabilities such as understanding user queries, classifying text into categories, or providing answers to questions based on a given context. The model can be accessed via APIs or by embedding it directly into the application.

Here is an example of how you might use the `transformers` library to fine-tune the `bert-large-cased-whole-word-masking` model for a classification task:

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
model = BertForSequenceClassification.from_pretrained('bert-large-cased-whole-word-masking')

# Prepare the dataset (this is a placeholder, actual dataset preparation code is needed)
train_dataset = ...
valid_dataset = ...

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

# Train the model
trainer.train()
```

Please note that the above code snippet is a general example. The actual implementation would require a dataset and might involve additional steps such as preprocessing the data, evaluating the model, and fine-tuning the hyperparameters based on the specific task and dataset.

[More Information Needed] for the actual dataset preparation code and any task-specific adjustments that may be required.

### Out-of-Scope Use

As the sociotechnic on the deep learning model development team, it is my responsibility to consider the potential societal impacts and misuse of the bert-large-cased-whole-word-masking model. Here are some considerations for users regarding what they should not do with the model:

1. **Biased Output**: Given that the model has been trained on large datasets that may contain societal biases, users should not use the model in applications where biased outputs could lead to unfair treatment of individuals or groups, such as in hiring tools, loan approval systems, or criminal justice applications.

2. **Misinformation**: Users should not employ the model to generate or propagate misinformation, including but not limited to fake news, impersonation, and fraudulent activities. The model's advanced language capabilities could be misused to create convincing but false narratives that could harm individuals or society.

3. **Privacy Violations**: The model should not be used to analyze or generate text in a way that could infringe on individuals' privacy rights or reveal sensitive personal information without consent.

4. **Deepfakes and Manipulation**: Users should not use the model to create deepfakes or manipulate audio, video, or text content with the intent to deceive, harm, or defraud others.

5. **Toxic Content**: The model should not be used to create or amplify hate speech, abusive, or other forms of toxic content. While the model itself does not inherently generate such content, it could be misused to do so if not properly safeguarded.

6. **Unintended High-Stakes Use**: The model should not be used as a sole decision-making tool in high-stakes scenarios, such as medical diagnosis, legal judgments, or safety-critical systems, where the model's outputs could have significant consequences.

7. **Intellectual Property Violations**: Users should not use the model to infringe on copyrights or other intellectual property rights by generating text that is derivative of protected works without proper authorization.

8. **Security Risks**: Users should not use the model in a way that could pose a security risk, such as generating phishing content or other forms of security threats.

It is important to note that while the model has been pre-trained on a diverse set of language data, it is not immune to the biases present in the training data. Users should be transparent about the use of the Whole Word Masking variant of BERT-Large and should conduct thorough testing and bias mitigation before deploying the model in any application.

In conclusion, users of the bert-large-cased-whole-word-masking model should exercise caution and ethical judgment to prevent misuse that could harm individuals or society. It is the responsibility of the users to ensure that the model is used in a manner that is consistent with societal norms, legal standards, and ethical guidelines.

### Bias, Risks, and Limitations

The BERT-large-cased-whole-word-masking model, while a significant advancement in NLP, presents several known and foreseeable issues that span technical and sociotechnical domains:

1. **Resource Intensiveness**: As indicated in references 5 and 6, the BERT-Large model requires substantial memory due to its size and the nature of the training process. This can limit its accessibility to researchers and practitioners without access to high-end computational resources, potentially creating a divide in who can utilize this technology.

2. **Optimization and Efficiency**: The default optimizer, Adam, is mentioned in reference 7 as being memory-intensive. While alternative optimizers might reduce memory usage, they could also affect the model's performance. This presents a trade-off between efficiency and effectiveness that needs to be carefully considered.

3. **Batch Size Limitations**: Reference 6 highlights that the maximum batch sizes for BERT-Large can negatively impact model accuracy. This limitation can affect the training process and the model's final performance, especially when computational resources are constrained.

4. **Gradient Checkpointing**: As per reference 8, gradient checkpointing is a technique that can help manage memory usage, but it is not implemented in the current release. This means that users cannot take advantage of this technique to reduce memory consumption without additional development work.

5. **Model Complexity**: With 340M parameters as stated in references 9 and 10, the BERT-Large model is complex, which can lead to challenges in fine-tuning and adapting the model to specific tasks or datasets.

6. **Potential for Bias**: The model's pre-training on large corpora of text data can inadvertently encode societal biases present in the training data. This can lead to the perpetuation of stereotypes and unfair treatment of certain groups if not carefully mitigated.

7. **Misunderstandings of Model Capabilities**: Users may have misconceptions about the model's abilities, such as overestimating its understanding of context or nuance. This could lead to overreliance on the model's outputs without proper human oversight.

8. **Generalization to Low-Resource Languages**: While reference 2 suggests that pre-trained models like BERT can benefit low-resource tasks, the actual effectiveness for languages with limited training data may not be as robust as for high-resource languages.

9. **Ethical and Legal Considerations**: As a sociotechnic, it is important to consider the ethical implications of deploying such a model. For instance, the use of the model in surveillance, censorship, or other applications could have serious societal impacts. Additionally, the model's outputs could influence decision-making processes in critical areas such as healthcare, finance, or law enforcement, where errors or biases could have significant consequences.

10. **Non-Affiliation with Google**: As stated in reference 4, this model is not an official Google product. Users should be aware that support, updates, and accountability might differ from those provided by officially endorsed products.

In conclusion, while the BERT-large-cased-whole-word-masking model represents a powerful tool for NLP tasks, it is essential to be aware of its limitations and potential issues. Users should approach its application thoughtfully, considering the technical requirements, potential biases, and broader societal implications.

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model bert-large-cased-whole-word-masking:

1. Vocabulary Size: If you have created a custom vocabulary for a domain-specific application, ensure that the `vocab_size` parameter in `bert_config.json` is updated accordingly. Failure to do so could result in NaNs during training, particularly on GPU or TPU, due to unchecked out-of-bounds access. [Reference 1]

2. Domain-Specific Pre-training: For tasks with a large domain-specific corpus, it is beneficial to perform additional pre-training steps using your corpus, starting from the BERT checkpoint. This helps the model to better understand the domain-specific language and nuances. [Reference 1]

3. Learning Rate: When continuing pre-training from an existing BERT checkpoint, use a smaller learning rate (e.g., 2e-5) compared to the original 1e-4 used in the BERT paper. This helps in fine-tuning the model without causing drastic changes to the already learned representations. [Reference 2]

4. Sequence Length: Training with very long sequences is computationally expensive due to the quadratic cost of attention mechanisms. It is recommended to pre-train with shorter sequences (e.g., length 128) for the majority of steps and then switch to longer sequences (e.g., length 512) for additional steps to learn positional embeddings efficiently. [Reference 3, 5]

5. Computational Resources: Be aware that pre-training a model from scratch is computationally intensive, especially on GPUs. It is recommended to start with a `BERT-Base` model for pre-training if resources are limited. [Reference 4]

6. Pre-training Corpus and Duration: For the best results, a large model should be pre-trained on a large corpus (such as Wikipedia and BookCorpus) for an extended period (e.g., 1 million update steps). This extensive pre-training is what enables BERT to achieve its high performance. [Reference 6]

7. Task-Specific Architectures: BERT's pre-trained representations can reduce the need for complex task-specific architectures. It is designed to achieve state-of-the-art performance on a wide range of NLP tasks with minimal task-specific modifications. [Reference 7]

8. Adaptability: BERT can be adapted to various NLP tasks (sentence-level, sentence-pair-level, word-level, and span-level) with almost no task-specific modifications. This flexibility should be leveraged when fine-tuning BERT for different applications. [Reference 8]

In summary, when publishing the bert-large-cased-whole-word-masking model, it is important to communicate these recommendations to potential users to ensure they are aware of the best practices for using and further training the model. Additionally, users should be informed about the computational requirements and the adaptability of BERT to various tasks with minimal modifications.

## Training Details

### Training Data

The training data for the model bert-large-cased-whole-word-masking consists of the BooksCorpus (800M words) and the English Wikipedia (2,500M words), from which only text passages were used while lists, tables, and headers were excluded. This approach ensures the model learns from long contiguous sequences, which is crucial for capturing document-level context. [More Information Needed] on data pre-processing and additional filtering.

### Training Procedure

#### Preprocessing

The preprocessing for the data of the model bert-large-cased-whole-word-masking involves several steps to prepare the input text for training or inference. Here's a detailed description of the process:

1. **Tokenization**: We use the `FullTokenizer` from the BERT library to tokenize the raw text. This tokenizer handles the text by first normalizing it, which includes converting all whitespace characters to spaces. For the cased model, the text is not lowercased, preserving the case information. The tokenizer then applies WordPiece tokenization, splitting words into subword units that can be found in the predefined vocabulary.

2. **Whole Word Masking**: During pre-training, we apply a technique called Whole Word Masking, where all tokens corresponding to a single word are masked at once. This is different from the original BERT's masking strategy, where subword tokens could be masked independently. The overall masking rate is 15%, but the masking is applied after tokenization, ensuring that entire words are masked rather than individual subword tokens.

3. **Special Tokens**: We add special tokens to the tokenized input to provide the model with additional structural information. The `[CLS]` token is added at the beginning of every sequence, and the `[SEP]` token is used to separate sentence pairs or to mark the end of a single sentence.

4. **Sequence Length and Truncation**: The maximum sequence length for the model is 512 tokens. However, for efficiency, we often use shorter sequences if possible. During pre-training, we also use a strategy where we train with a sequence length of 128 for 90% of the steps and then switch to the full sequence length of 512 for the remaining 10% to learn the positional embeddings.

5. **Sentence Embeddings**: For tasks involving pairs of sentences, we differentiate between the two sentences by adding a learned embedding to every token to indicate whether it belongs to sentence A or sentence B.

6. **Resizing**: If the tokenized input exceeds the maximum sequence length, it is truncated to fit within the allowed number of tokens. Conversely, if the input is shorter than the maximum sequence length, padding tokens are added to reach the required length.

7. **Attention Mask**: An attention mask is created to inform the model which tokens are actual data and which are padding. This mask has the same length as the input sequence, with `1`s for real tokens and `0`s for padding tokens.

8. **Segment Embeddings**: For tasks that involve pairs of sentences, a segment embedding is added to each token to indicate whether it belongs to the first or the second sentence.

The above steps outline the preprocessing required for the bert-large-cased-whole-word-masking model. This process ensures that the input data is in the correct format for the model to process effectively.

#### Training Hyperparameters

The training hyperparameters for the model `bert-large-cased-whole-word-masking` are as follows:

- Learning Rate: Initially set to 1e-4, with a recommendation to use a smaller learning rate (e.g., 2e-5) if continuing pre-training from an existing BERT checkpoint. [Reference 1]
- Batch Size: 256 sequences per batch. [Reference 2]
- Sequence Length: 512 tokens. [Reference 2]
- Training Steps: 1,000,000 steps. [Reference 2]
- Optimizer: Adam with β1 = 0.9, β2 = 0.999. [Reference 2]
- L2 Weight Decay: 0.01. [Reference 2]
- Learning Rate Warmup: Over the first 10,000 steps. [Reference 2]
- Learning Rate Decay: Linear decay of the learning rate after warmup. [Reference 2]
- Masking: 15% of all WordPiece tokens are masked in each sequence. [Reference 3]
- Model Architecture: 24-layer, 1024-hidden, 16-heads, with a total of 340M parameters. [Reference 6]
- Whole Word Masking: Enabled during data generation with the flag `--do_whole_word_mask=True`. [Reference 8]

Please note that these hyperparameters are based on the provided references and are specific to the `bert-large-cased-whole-word-masking` model.

#### Speeds, Sizes, Times

The model `bert-large-cased-whole-word-masking` is a 24-layer Transformer with 1024 hidden units per layer and 16 attention heads, totaling approximately 340 million parameters. This model has been pre-trained on a large corpus, including Wikipedia and the BookCorpus, for a significant number of update steps, although the exact number of update steps for our training is not specified in the provided references.

During our pre-training phase, we used a small sample text file and trained the model for only 20 steps, which is far less than the recommended 10,000 steps or more for practical applications. This was done for demonstration purposes, and as a result, the model quickly overfitted the small dataset, achieving a masked language model (LM) accuracy of 98.5479% and a next sentence prediction accuracy of 100%. However, these results are not indicative of the model's performance on a diverse and large-scale dataset due to the overfitting.

The pre-training was conducted with a batch size of 32, a maximum sequence length of 128 tokens, and a maximum of 20 masked LM predictions per sequence. The learning rate was set to 2e-5, with 10 warmup steps. The checkpoint size is not explicitly mentioned in the provided references, so [More Information Needed] for that detail.

As for the throughput, start or end time of the model training, these details are not provided in the references, so [More Information Needed] for those specifics as well.

In terms of the model's performance compared to other works, it has been noted that BERT Large models significantly outperform BERT Base models across various tasks, particularly in scenarios with limited training data. The model's effectiveness has been demonstrated for both fine-tuning and feature-based approaches.

To summarize, while we have some details about the model's architecture and pre-training setup, we lack specific information on the throughput, start/end times, and checkpoint sizes for the `bert-large-cased-whole-word-masking` model. Additional information would be needed to provide a complete description of these aspects.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model bert-large-cased-whole-word-masking has been evaluated on the following benchmarks and datasets:

1. The General Language Understanding Evaluation (GLUE) benchmark, which is a collection of diverse natural language understanding tasks. Specific tasks within the GLUE benchmark are not listed in the provided references, but it is mentioned that BERT achieves a score of 80.5 on the GLUE leaderboard.

2. The Stanford Question Answering Dataset (SQuAD) v1.1, which consists of 100k crowdsourced question/answer pairs. The task involves predicting the answer text span in a passage from Wikipedia.

3. The SQuAD v2.0 task, which extends SQuAD 1.1 by allowing for the possibility that no short answer exists within the provided paragraph, making the problem more realistic and challenging.

These datasets are used to demonstrate the model's performance in various natural language processing tasks, including question answering and language inference. The model has shown state-of-the-art results on these benchmarks, outperforming previous systems and establishing new state-of-the-art results.

#### Factors

The model bert-large-cased-whole-word-masking is expected to exhibit several characteristics influenced by its design and training process, which will affect its behavior across different domains, contexts, and population subgroups. Here are some foreseeable characteristics:

1. **Domain and Context Sensitivity**: Since BERT is pre-trained on a large corpus of text, it is likely to perform well on tasks that are similar to the text it was trained on. However, its performance may degrade on tasks that are significantly different from its training data. For example, if BERT was trained primarily on formal text, it might struggle with slang or informal language that is common in social media text.

2. **Whole-Word Masking**: The whole-word masking strategy used in this model means that it is likely to have a better understanding of word boundaries and may perform better on tasks that require a strong understanding of whole words rather than subword pieces. This could influence its performance on tasks like Named Entity Recognition (NER) or other tasks where recognizing entire entities is important.

3. **Masking Strategies**: The ablation studies mentioned in the references suggest that different masking strategies can affect the model's performance. Since the model uses a mixed strategy for masking, it is designed to reduce the mismatch between pre-training and fine-tuning. However, the effectiveness of this approach may vary across different tasks and datasets.

4. **Bidirectionality**: BERT's bidirectional nature allows it to understand context from both left and right of a token, which should theoretically provide a more nuanced understanding of language. This characteristic is likely to influence its performance positively on a wide range of NLP tasks compared to unidirectional models.

5. **Population Subgroups**: The model's performance may vary across different population subgroups, especially if these subgroups use language in ways that were not well represented in the training data. For example, dialects, sociolects, or industry-specific jargon may pose challenges for the model if they were underrepresented in the pre-training phase.

6. **Performance Disparities**: Evaluation of the model should be disaggregated across factors such as language, dialect, genre, and formality to uncover any disparities in performance. It is foreseeable that the model may perform better on some subgroups than others, particularly those that are well-represented in the training data.

7. **Ethical Considerations**: As a sociotechnic, it is important to consider the ethical implications of deploying this model. For instance, if the model systematically underperforms on certain dialects or demographic groups, it could reinforce or exacerbate existing biases.

In conclusion, while bert-large-cased-whole-word-masking is a powerful model with a broad potential application range, its performance is likely to be influenced by the nature of the training data, the bidirectionality of the model, the whole-word masking strategy, and the specific characteristics of the tasks and populations it is applied to. Continuous evaluation and monitoring are necessary to ensure that it performs equitably across different domains and population subgroups.

#### Metrics

For the evaluation of the bert-large-cased-whole-word-masking model, we will use a variety of metrics that are appropriate for the different NLP tasks as mentioned in the references. Specifically:

1. For tasks like natural language inference (NLI), as performed on the MultiNLI dataset, we will use accuracy as a metric, as indicated by the improvement reported in reference 5.

2. For question answering tasks, such as those on the SQuAD v1.1 and SQuAD v2.0 datasets, we will use the F1 score as a primary metric, which combines precision and recall, as mentioned in reference 5.

3. For named entity recognition (NER), we will report both fine-tuning and Dev results, likely using precision, recall, and F1 score as metrics, as these are standard for evaluating NER systems. However, the exact metrics for NER are not explicitly stated in the provided references, so [More Information Needed] for confirmation.

4. The GLUE benchmark is a collection of diverse NLP tasks, and the overall GLUE score will be used as a composite metric to evaluate the model's performance across these tasks, as mentioned in reference 5.

It's important to note that the choice of metrics is also influenced by the need to reduce the mismatch between pre-training and fine-tuning stages, as discussed in reference 3. However, the specific tradeoffs between different errors (e.g., false positives vs. false negatives) are not detailed in the provided references, so [More Information Needed] to address that part of the question.

### Results

The BERT-Large, Cased (Whole Word Masking) model has demonstrated significant improvements across a variety of natural language processing tasks. Here are the evaluation results based on the provided references:

1. On the GLUE benchmark, BERT-Large with Whole Word Masking achieves a score of 80.5, which is a 7.7 percentage point absolute improvement over the previous state-of-the-art results.

2. For the MNLI (Multi-Genre Natural Language Inference) task, the model achieves an accuracy of 86.7%, which is a 4.6% absolute improvement over the previous best results.

3. In the SQuAD v1.1 question answering task, the model reaches a Test F1 score of 93.2, which is a 1.5 point absolute improvement.

4. For SQuAD v2.0, the model also shows an improvement, but the exact Test F1 score is not provided in the references. [More Information Needed]

5. The model size and architecture consist of 24 layers, 1024 hidden units, 16 attention heads, and a total of 340 million parameters.

6. During fine-tuning on GLUE tasks, the best learning rate was selected from among 5e-5, 4e-5, 3e-5, and 2e-5 based on performance on the development set.

7. Fine-tuning was performed with a batch size of 32 for 3 epochs.

8. Due to instability on small datasets, multiple random restarts were used during fine-tuning to select the best model based on the development set performance.

9. The model significantly outperforms BERT-Base across all tasks, especially on those with limited training data.

These results highlight the effectiveness of the BERT-Large, Cased (Whole Word Masking) model in improving the state-of-the-art for various NLP tasks without the need for substantial task-specific architectural modifications.

#### Summary

The evaluation results for the model BERT-Large, Cased (Whole Word Masking) indicate that it significantly outperforms BERT-Base across all tasks, with a particularly notable improvement in tasks with limited training data (Reference 1). Compared to the previous state of the art, BERT-Large achieves an average accuracy improvement of 7.0% (Reference 2). Specifically, for the MNLI task, BERT-Large obtains a 4.6% absolute accuracy improvement and achieves a score of 80.5 on the official GLUE leaderboard (Reference 2).

During fine-tuning on the GLUE tasks, the model uses a batch size of 32 and is fine-tuned for 3 epochs. The best fine-tuning learning rate is selected from among 5e-5, 4e-5, 3e-5, and 2e-5 based on performance on the Dev set (Reference 4). For BERT-Large, multiple random restarts were employed during fine-tuning on smaller datasets to ensure stability and the best model was selected based on the Dev set performance (Reference 5).

The BERT-Large, Cased (Whole Word Masking) model itself is a 24-layer network with 1024 hidden units per layer, 16 attention heads, and a total of 340 million parameters (Reference 6).

## Model Examination

The BERT-large-cased-whole-word-masking model is an extension of the original BERT model, which itself is a breakthrough in the field of natural language processing. The model is pre-trained on a large corpus, such as Wikipedia, to develop a deep understanding of language context and semantics before being fine-tuned for specific downstream tasks.

One of the key features of this model is the use of whole-word masking in its pre-training phase. Unlike the original BERT, which might mask individual subwords or tokens within a word, whole-word masking ensures that all tokens corresponding to a word are masked together. This approach is more aligned with how humans process language, as we typically consider words as the smallest units of semantic meaning.

In terms of explainability and interpretability, the whole-word masking strategy can potentially offer clearer insights into the model's understanding of language. Since the model is trained to predict the masked words based on their context, it develops an internal representation that captures the relationships between whole words, rather than fragmented subwords. This could lead to more coherent and interpretable predictions when the model is used for tasks such as question answering or text completion.

Furthermore, the ablation studies mentioned in the references provide additional insights into the model's behavior. By experimenting with different facets of BERT, including the masking strategies, we can better understand the relative importance of each component in the model's performance. These studies help in identifying which aspects of the model contribute most to its success and where there might be room for improvement.

In summary, the bert-large-cased-whole-word-masking model's use of whole-word masking during pre-training is a significant step towards improving the model's language understanding capabilities. The ablation studies conducted provide valuable information for interpreting the model's behavior and understanding the impact of different pre-training strategies on its performance.

## Environmental Impact

- **Hardware Type:** The model bert-large-cased-whole-word-masking is trained on GPUs. This is mentioned in reference 3, which states that pre-training is computationally expensive, especially on GPUs.
- **Software Type:** The software type that the model bert-large-cased-whole-word-masking is trained on is not explicitly mentioned in the provided references. However, given the context and the standard practices in the field, it is likely that the model was trained using machine learning frameworks such as TensorFlow or PyTorch, as these are commonly used for training deep learning models like BERT. Since the references include links to Google Storage URLs, which typically host TensorFlow checkpoints, it is reasonable to infer that TensorFlow might have been used. However, without explicit information provided in the references, the exact software type cannot be confirmed.

[More Information Needed]
- **Hours used:** The amount of time used to train the model bert-large-cased-whole-word-masking was 4 days. This information is found in reference 4, which states that "Training of BERT LARGE was performed on 16 Cloud TPUs (64 TPU chips total). Each pretraining took 4 days to complete."
- **Cloud Provider:** The cloud provider that the model bert-large-cased-whole-word-masking is trained on is Google Cloud Platform (GCP). This is indicated by the references to Google Cloud Storage (GCS) buckets and Cloud TPUs in the provided references.
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture of `bert-large-cased-whole-word-masking` is based on a multi-layer bidirectional Transformer encoder, as described in Vaswani et al. (2017). Specifically, it consists of 24 layers, with 1024 hidden units per layer, 16 attention heads, and a total of 340 million parameters. This architecture is almost identical to the original Transformer implementation released in the tensor2tensor library.

The objective of the `bert-large-cased-whole-word-masking` model is to train deep bidirectional representations by using a "masked language model" (MLM) pre-training objective. This is inspired by the Cloze task, where some percentage of the input tokens are masked at random, and the goal is to predict the original vocabulary id of the masked word based solely on its context. In the case of whole word masking, entire words are masked instead of individual subword tokens, which can lead to a better understanding of word context. For this model, 15% of all WordPiece tokens are masked during training.

The `cased` variant of the model preserves the true case and accent markers of the text, which can be crucial for certain tasks such as Named Entity Recognition or Part-of-Speech tagging, where case information carries significant weight.

### Compute Infrastructure

The compute infrastructure used for training the model `bert-large-cased-whole-word-masking` includes both Cloud TPUs and GPUs. Specifically, the model can be trained on Google Cloud TPUs as indicated by the instructions for setting up the output directory on Google Cloud Storage (Reference 1) and the use of Cloud TPUs with the appropriate flags in the training scripts (Reference 3). Additionally, the model can be trained on local machines using GPUs, as mentioned for running training/evaluation on a GPU like a Titan X or GTX 1080 (Reference 3).

For Cloud TPU usage, the pre-trained model files are available in the Google Cloud Storage folder `gs://bert_models/2018_10_18` (Reference 1), and further guidance on using Cloud TPUs can be found in the Google Cloud TPU tutorial and the Google Colab notebook provided (Reference 2).

The model was fine-tuned for 2 epochs with a learning rate of 5e-5 and a batch size of 48, although it is not explicitly stated whether this fine-tuning occurred on TPUs or GPUs (Reference 4).

In summary, the compute infrastructure for the `bert-large-cased-whole-word-masking` model includes the option to train on Google Cloud TPUs or local GPUs, with specific configurations and storage solutions provided for both types of hardware.

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

