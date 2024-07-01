# Model Card for bert-base-multilingual-uncased

bert-base-multilingual-uncased is a pre-trained deep bidirectional representation model based on the BERT architecture. It is trained on 102 languages using the masked language modeling (MLM) objective, with 12 layers, 768 hidden units, and 12 attention heads, containing 110 million parameters in total.

## Model Details

### Model Description

Model Details for bert-base-multilingual-uncased:

Model Architecture:
- Multi-layer bidirectional Transformer encoder based on the original implementation in Vaswani et al. (2017)
- Uses bidirectional self-attention, allowing each token to attend to both left and right context
- 12-layer, 768-hidden, 12-heads, 110M parameters

Training Procedures: 
- Pre-trained on unlabeled text using two unsupervised tasks:
  1. Masked Language Model (MLM): Some input tokens are randomly masked and the model learns to predict the masked tokens
  2. Next Sentence Prediction (NSP): The model learns to predict if two sentences follow each other
- Fine-tuned for downstream tasks by plugging in task-specific inputs and outputs and fine-tuning all parameters end-to-end

Important Disclaimers:
- Covers 102 languages
- Uncased model
- This is the original multilingual uncased model released, but not recommended. A newer cased version is recommended instead: 
[`BERT-Base, Multilingual Cased (New, recommended)`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)

[More Information Needed] on specific pre-training data and fine-tuning procedures used for this model.

For more details, please refer to the BERT paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018).

- **Developed by:** Jacob Devlin; Ming-Wei Chang; Kenton Lee; Kristina Toutanova
- **Funded by:** [More Information Needed]

The provided references do not contain any information about the funding sources for the BERT-Base Multilingual Uncased model. More details would be needed from the model developers or publishing organization to determine what people or organizations, if any, provided funding for this specific project.
- **Shared by:** Based on the provided references, it appears that BERT (Bidirectional Encoder Representations from Transformers) was developed by researchers at Google. The paper mentions that "The code and pre-trained models are available at https://github.com/google-research/bert."

However, the references do not specify the exact individuals who contributed to making the specific model bert-base-multilingual-uncased available online. To confidently name the contributors, [More Information Needed].
- **Model type:** The bert-base-multilingual-uncased model is a deep bidirectional Transformer encoder pre-trained using the masked language modeling (MLM) objective on multilingual uncased text data.
- **Language(s):** The BERT-Base, Multilingual Uncased model supports 102 languages, which were chosen because they are the top 100 languages with the largest Wikipedias.
- **License:** [More Information Needed]

The references provided do not contain any information about the specific license being used for the BERT-Base Multilingual Uncased model. More details would be needed from the model creators or documentation to determine the license that applies to this particular model.
- **Finetuned from model:** The bert-base-multilingual-uncased model is pre-trained, not fine-tuned from another model. It is a BERT model pre-trained on 102 languages using a masked language modeling (MLM) objective, as mentioned in Reference 7:

"BERT-Base, Multilingual Uncased (Orig, not recommended): 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters"

The model can be fine-tuned on downstream tasks by adding task-specific inputs and outputs, as described in References 2 and 3. However, the base model itself is not fine-tuned from another model.
### Model Sources

- **Repository:** https://github.com/google-research/bert/blob/master/multilingual.md
- **Paper:** https://arxiv.org/pdf/1810.04805.pdf
- **Demo:** [More Information Needed]

The references provided do not contain a link to a demo of the BERT-Base Multilingual Uncased model. The references discuss the BERT model architecture, pre-training tasks, and performance on NLP benchmarks, but do not mention a specific demo for this particular pre-trained model variant.
## Uses

### Direct Use

The BERT model bert-base-multilingual-uncased can be used without fine-tuning for feature extraction, where fixed features are extracted from the pre-trained model. This has certain advantages over fine-tuning the entire model:

1. It allows using BERT with task-specific model architectures that cannot be easily represented by a Transformer encoder architecture. 

2. There are certain computational benefits to pre-compute an expensive representation of the training data once and then run many experiments with cheaper models on top of this representation.

To use BERT for feature extraction:

1. Feed the input text into the pre-trained BERT model.

2. Extract the token representations from one or more of the top hidden layers of the model. 

3. Feed these extracted representations into a task-specific model architecture for the downstream task.

[More Information Needed] for a specific code snippet demonstrating feature extraction with bert-base-multilingual-uncased.

The pre-trained BERT model provides a powerful generic language understanding that can then be leveraged for specific tasks without the need for full fine-tuning. However, [More Information Needed] on comparative performance of feature extraction vs fine-tuning for bert-base-multilingual-uncased on specific tasks.

### Downstream Use

The bert-base-multilingual-uncased model can be fine-tuned for a variety of downstream NLP tasks across multiple languages. Fine-tuning is straightforward, as the self-attention mechanism in BERT allows it to model tasks involving single text or text pairs by providing the appropriate inputs and outputs.

For example, the model can be fine-tuned for the XNLI dataset, which is a 15-language version of MultiNLI. The fine-tuning process involves modifying the `run_classifier.py` script to support the XNLI dataset, as shown in the code snippet:

```
python run_classifier.py \
  --task_name=XNLI \
  --do_train=true \
  --do_eval=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=/tmp/xnli_output/
```

The multilingual model does not require any special consideration or API changes. However, the `BasicTokenizer` in `tokenization.py` should be updated to support Chinese character tokenization.

When fine-tuned, the bert-base-multilingual-uncased model can be used for various sentence-level and token-level tasks across multiple languages, often outperforming task-specific architectures. The pre-trained representations reduce the need for heavily-engineered task-specific architectures.

[More Information Needed] on how the model can be plugged into a larger ecosystem or app.

For more information and pre-trained models, please visit https://github.com/google-research/bert.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the BERT-base-multilingual-uncased model could be misused and what users should avoid doing:

- The model was trained on 102 languages in an uncased (case-insensitive) manner. Users should not expect it to handle casing or capitalization in a meaningful way. For better cased language handling, the cased multilingual model is recommended instead.

- As a multilingual model, its performance on high-resource languages like English may be somewhat worse than a dedicated single-language model. Users seeking maximum performance for a specific language should consider using a model trained solely on that language rather than this multilingual one.

- [More Information Needed] on other specific ways the model architecture or training data could lead to misuse or unintended consequences if applied incorrectly. The references do not provide enough details to speculate further.

In general, users should be cautious about applying the model to tasks or languages it was not designed or evaluated for. They should carefully test its suitability and performance before deploying it in any system that could negatively impact people if the model behaves unexpectedly. More targeted testing and analysis would be prudent to uncover additional potential misuses to warn users about.

### Bias, Risks, and Limitations

Based on the provided references about the BERT-base multilingual uncased model, here are some potential issues and limitations:

Technical limitations:
- For high-resource languages like English, the multilingual model performs somewhat worse than a single-language model. However, maintaining many single-language models is not feasible. (Reference 10)
- [More Information Needed] on detailed performance metrics for the 102 languages covered by this model. It's unclear how well it works for lower-resource languages.

Potential misunderstandings:
- This is an older version of the multilingual model that is no longer recommended. A newer cased version covering 104 languages is preferred. (Reference 9) 
- As a bidirectional model, BERT is designed for language understanding tasks, not generation. Using it for generation may lead to poor results.

Foreseeable sociotechnical issues:
- [More Information Needed] With 102 languages, the model will have varying quality across languages based on training data availability. This could disadvantage some language communities.
- Bias in the training data, especially for lower-resource languages, may get amplified by the model and harm underrepresented groups. More analysis is needed.
- The model may struggle with language and dialect variations not well-represented in training. This could limit usability for some language communities.

To summarize, the main issues are the model being an outdated version, having technical limitations especially for generation and non-English languages, and the potential to amplify data bias against certain languages and dialects. More information and analysis would help better characterize these risks. Moving to the newer recommended version is advised.

### Recommendations

Based on the provided references, here are some recommendations regarding foreseeable issues with the BERT-base-multilingual-uncased model:

1. The BERT-base-multilingual-uncased model is the original version covering 102 languages, but reference 5 indicates it is not recommended compared to the newer cased version covering 104 languages. Consider using the newer recommended version instead.

2. The model uses bidirectional pre-training which is important for performance, as shown in references 1 and 7. However, [More Information Needed] on any potential issues or limitations of the bidirectional approach for this specific model.

3. [More Information Needed] regarding any societal biases that may be captured in the pre-training data spanning 102 languages. Analyzing the training data for potentially problematic content would be prudent.

4. [More Information Needed] on how well the model performs for lower-resource languages out of the 102 covered. Additional testing may be warranted especially for languages with less pre-training data.

5. As an uncased model, it will not handle casing information which may be important in some languages or applications. The cased version may be preferable for such cases.

In summary, while a groundbreaking multilingual model, I would recommend using the newer cased version, analyzing biases in the pre-training data, and evaluating performance carefully on lower-resource languages and tasks requiring case information. More information is needed to draw firm conclusions in several areas.

## Training Details

### Training Data

The training data for bert-base-multilingual-uncased consists of the top 100 languages with the largest Wikipedias, with the entire Wikipedia dump for each language (excluding user and talk pages) taken as the training data. The data was weighted during pre-training data creation using exponentially smoothed weighting, with high-resource languages like English under-sampled and low-resource languages over-sampled.

### Training Procedure

#### Preprocessing

For the BERT base multilingual uncased model, the following preprocessing steps were applied to the training data:

Tokenization:
- A shared 110k WordPiece vocabulary was used across all languages
- For all languages except Chinese, the following recipe was applied:
  a) Lowercasing and accent removal
  b) Punctuation splitting
  c) Whitespace tokenization
- The word counts for the vocabulary were weighted the same way as the training data, so low-resource languages were upweighted by some factor
- No language-specific markers were used to denote the input language, to enable zero-shot learning

Masking:
- 15% of all WordPiece tokens in each sequence were randomly selected for prediction
- For the selected tokens:
  - 80% of the time, the token was replaced with the [MASK] token
  - 10% of the time, the token was replaced with a random token
  - 10% of the time, the token was left unchanged
- This masking allows the model to learn a bidirectional representation, but creates a mismatch with fine-tuning where no [MASK] tokens are used

[More Information Needed] on any resizing or rewriting steps for the input data.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the bert-base-multilingual-uncased model:

- Trained on 102 languages (Reference 9)
- 12-layer, 768-hidden, 12-heads, 110M parameters (Reference 9)
- Trained with a batch size of 256 sequences (256 sequences * 512 tokens = 128,000 tokens/batch) for 1,000,000 steps, which is approximately 40 epochs over the 3.3 billion word corpus (Reference 5)
- Used Adam optimizer with learning rate of 1e-4, β1 = 0.9, β2 = 0.999, L2 weight decay of 0.01, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate (Reference 5)
- Dropout probability of 0.1 on all layers (Reference 3)
- Used gelu activation rather than the standard relu (Reference 3)
- [More Information Needed] on the exact pre-training data and corpus
- Trained using the "masked LM" (MLM) and "next sentence prediction" (NSP) tasks (Reference 10)
- [More Information Needed] on the exact hardware setup and training time

#### Speeds, Sizes, Times

Based on the provided references, here are the details I can provide about the BERT-Base Multilingual Uncased model:

Model Architecture:
- 12-layer bidirectional Transformer encoder
- 768 hidden units
- 12 attention heads
- 110M parameters

Languages:
- Covers 102 languages

Performance:
- Performs somewhat worse than single-language models for high-resource languages like English and Chinese
- English XNLI result is lower than the 84.2 MultiNLI baseline due to using the multilingual model instead of English-only
- Chinese XNLI result is 74.2, compared to 77.2 for a Chinese-only BERT-Base model

[More Information Needed] for:
- Throughput
- Start or end training time
- Checkpoint sizes

The references do not provide specific information about these aspects of the model training process. More details would be needed from the model development logs or records to include this information in the model card.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the BERT-base-multilingual-uncased model likely evaluates on the following benchmarks and datasets:

- GLUE (General Language Understanding Evaluation) benchmark, which is a collection of diverse natural language understanding tasks. (Reference 5)

- SQuAD v1.1 and SQuAD v2.0 question answering datasets. BERT models were fine-tuned for these tasks and achieved state-of-the-art performance. (References 3, 4, 9)

However, the references do not provide specific details on the exact performance of the BERT-base-multilingual-uncased model on these benchmarks. The results mentioned are for BERT models in general, such as BERT-base and BERT-large.

[More Information Needed] on the specific evaluation results and metrics for the BERT-base-multilingual-uncased model on the mentioned benchmarks and datasets.

#### Factors

Based on the provided references, here are some key characteristics that may influence the behavior of the bert-base-multilingual-uncased model:

Language Representation:
- The model covers 102 languages, with the training data coming from the largest Wikipedias for each language (Reference 10). 
- However, the size of Wikipedia varies greatly by language, so low-resource languages may be under-represented in the model (Reference 1).
- To balance this, exponentially smoothed weighting was used during pre-training data creation to under-sample high-resource languages like English and over-sample low-resource languages (References 2-4).

Tokenization and Vocabulary:
- For languages other than English, lower casing, accent removal, punctuation splitting, and whitespace tokenization was applied (Reference 11).
- Removing accent markers provides benefits by reducing vocabulary size, but may introduce some ambiguity. The strong contextual models of BERT should compensate for this (Reference 11).

Model Size and Architecture: 
- The model has 12 layers, 768 hidden units, 12 attention heads, and 110M parameters total (Reference 10).
- [More Information Needed] on how model size impacts downstream task performance for this specific model.

Evaluation across Subgroups:
- [More Information Needed] The references do not provide details on evaluating the model's performance disaggregated across different languages or subgroups. This would be important to assess to uncover any performance disparities.

In summary, the key factors that may influence the model's behavior are the representation of different languages in the training data, the tokenization/vocabulary approach used, and the model architecture. More information would be needed to comment on performance across different subgroups.

#### Metrics

Based on the provided references, the key evaluation metrics for the bert-base-multilingual-uncased model appear to be:

- F1 score on SQuAD v1.1 and v2.0 question answering tasks
- Accuracy on GLUE benchmark tasks like MNLI 
- Accuracy on XNLI cross-lingual natural language inference dataset

The references mention these metrics being used to compare BERT's performance to prior state-of-the-art models. However, more specific information would be needed on the tradeoffs and error analysis for the multilingual uncased variant of BERT. The references focus more on the general BERT architecture rather than specifics of this multilingual model.

[More Information Needed] on the particular strengths, weaknesses and tradeoffs of using the multilingual uncased model compared to other BERT variants or models. Details on its performance across different languages and error patterns would help round out the evaluation metrics section.

### Results

Based on the provided references, here is the evaluation information I could find for the bert-base-multilingual-uncased model:

Evaluation Dataset: XNLI (dev and test sets translated into 15 languages)

Evaluation Metric: Accuracy

Results:
- English accuracy is lower than the 84.2 MultiNLI baseline because the multilingual model performs somewhat worse than an English-only model for high-resource languages.
- Chinese accuracy is 74.2, which is 3% lower than a Chinese-only BERT-Base model at 77.2.

Other Notes:
- The model has 102 languages, 12 layers, 768 hidden units, 12 attention heads, and 110M parameters. 
- A newer recommended version is BERT-Base Multilingual Cased which covers 104 languages.
- The training data for non-English languages was machine translated, so it's unclear how much lower accuracy is due to translation quality vs pre-trained model quality.

[More Information Needed] on specific accuracy results for the other 13 languages in XNLI besides English and Chinese.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the bert-base-multilingual-uncased model:

The model was evaluated on the XNLI dataset, which is a version of MultiNLI translated into 15 languages. The training set was machine translated.

Results on 6 languages:
- English: 81.4
- Chinese: 74.2 
- [More Information Needed for other 4 languages]

The multilingual uncased model performs slightly worse than a Chinese-only BERT-Base model on Chinese (74.2 vs 77.2).

Different fine-tuning approaches were compared:
- Translate Train Uncased: 81.4 (en), 74.2 (zh), 77.3 (fr), 75.2 (es), 70.5 (de), 61.7 (ar)  
- Translate Test Uncased: 81.4 (en), 70.1 (zh), 74.9 (fr), 74.4 (es), 70.4 (de), 62.1 (ar)
- Zero Shot Uncased: 81.4 (en), 63.8 (zh), 74.3 (fr), 70.5 (es), 62.1 (de), 58.3 (ar)

Zero Shot means fine-tuning only on English and evaluating on the other languages without translation.

The original multilingual uncased model covers 102 languages with 12-layers, 768-hidden, 12-heads, and 110M parameters. A newer version covers 104 languages.

[More Information Needed] on other downstream tasks like question answering.

## Model Examination

Based on the provided references, here is what I can say about the explainability and interpretability of the BERT base multilingual uncased model that our team has worked on:

The BERT model uses a bidirectional masked language modeling objective during pretraining, which allows it to jointly condition on both left and right context in all layers. This bidirectionality is important for the model's performance, as demonstrated by evaluating pretraining objectives with and without the "next sentence prediction" task [1].

By pretraining deep bidirectional representations from unlabeled text, BERT can be fine-tuned with just one additional output layer for various downstream tasks [6]. This reduces the need for heavily-engineered task-specific architectures [8].

Increasing the model size (layers, hidden units, attention heads) generally leads to improvements on downstream tasks, even for small-scale tasks, provided the model has been sufficiently pre-trained [2, 3, 4]. Our BERT base model contains 110M parameters [5].

[More Information Needed] on specific techniques used for explaining or interpreting the model's predictions and internal representations. The provided references do not cover this aspect in detail.

In summary, the bidirectional pretraining and the ability to scale to large model sizes contribute to BERT's strong performance across various NLP tasks, but more information is needed to discuss its explainability and interpretability in depth.

## Environmental Impact

- **Hardware Type:** Based on the provided references, the model bert-base-multilingual-uncased was trained on Cloud TPUs. Specifically, the training was performed on 4 Cloud TPUs in Pod configuration, which amounts to a total of 16 TPU chips.

Reference:
"Training of BERT BASE was performed on 4 Cloud TPUs in Pod configuration (16 TPU chips total)."
- **Software Type:** Based on the provided references, the model bert-base-multilingual-uncased is likely trained using the tensor2tensor library, as mentioned in this excerpt:

"BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017) and released in the tensor2tensor library."

However, more specific information about the exact software and version used for training bert-base-multilingual-uncased would be needed to provide a definitive answer.

[More Information Needed]
- **Hours used:** Based on the information provided in the references, the training time for BERT-Base, Multilingual Uncased is not explicitly stated. The references mention that pre-training BERT-Base was performed on 4 Cloud TPUs in Pod configuration (16 TPU chips total) and each pre-training took 4 days to complete. However, it is not clear if this refers specifically to the Multilingual Uncased model or the English-only BERT-Base model.

Therefore, the answer to the question "Provide the amount of time used to train the model bert-base-multilingual-uncased" is:

[More Information Needed]
- **Cloud Provider:** [More Information Needed]

The provided references do not contain any information about the cloud provider used to train the BERT base multilingual uncased model. The references discuss the BERT model architecture, pre-training approach, and performance on various NLP tasks, but do not specify the computing infrastructure or cloud platform utilized for training.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the bert-base-multilingual-uncased model. To answer this question, more specific details would be needed, such as the hardware used for training, the duration of the training process, and the energy consumption of the training setup.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017). The bert-base-multilingual-uncased model has the following architecture hyperparameters:
- Number of layers (Transformer blocks): 12
- Hidden size: 768
- Number of self-attention heads: 12
- Total parameters: 110M

Pre-training Objective:
BERT is pre-trained using two unsupervised tasks:
1. Masked Language Model (MLM): Some of the input tokens are randomly masked, and the objective is to predict the original vocabulary id of the masked word based only on its bidirectional context.
2. [More Information Needed]

The pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of downstream tasks.

### Compute Infrastructure

Based on the provided references, here is the information about the compute infrastructure used for training the bert-base-multilingual-uncased model:

Training of BERT BASE was performed on 4 Cloud TPUs in Pod configuration (16 TPU chips total). Each pretraining took 4 days to complete.

[More Information Needed] on the specific compute infrastructure details for the multilingual uncased variant of BERT base model, as the provided reference only mentions the compute used for BERT base model in general.

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

