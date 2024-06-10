# Model Card for bert-large-cased-whole-word-masking

bert-large-cased-whole-word-masking is a pre-trained BERT model that uses whole word masking, where all of the tokens corresponding to a word are masked at once during pre-training. It is a large model with 24 layers, 1024 hidden units, 16 attention heads, and 340M parameters, and is cased, meaning it is case-sensitive.

## Model Details

### Model Description

Model Card for bert-large-cased-whole-word-masking

Model Details:
- Architecture: 24-layer Transformer encoder, bidirectional self-attention
- Pre-training Objective: Masked Language Model (MLM), predicting masked tokens based on bidirectional context
- Masking Approach: Whole Word Masking (WWM), masking out 15% of the words at random
- Casing: Cased, retaining the original case of the input text
- Pre-training Data: Wikipedia + BookCorpus
- Number of Parameters: [More Information Needed]

Training Procedure:
- Unsupervised pre-training on a large corpus for 1M update steps
- Fine-tuning on downstream NLP tasks with an additional output layer

[More Information Needed] on specific training hyperparameters and hardware.

Important Disclaimers:
- BERT is a general-purpose language understanding model and its outputs should be carefully analyzed for specific use cases.
- The model may reflect biases present in the pre-training data.
- Fine-tuning is required for optimal performance on downstream tasks.

For more details, please refer to the original BERT paper: https://arxiv.org/abs/1810.04805

Contact Person: [Your Name]
For any questions or updates regarding this model card, please contact [Your Email Address].

- **Developed by:** Jacob Devlin; Ming-Wei Chang; Kenton Lee; Kristina Toutanova
- **Funded by:** Based on the provided references, there is no explicit information about the people or organizations that funded the BERT-Large, Cased (Whole Word Masking) model. The references mention the model and link to pre-trained model files, but do not specify the funding sources.

[More Information Needed]
- **Shared by:** Based on the references provided, the key contributors to BERT who made the model available are:

Jacob Devlin (jacobdevlin@google.com)
Ming-Wei Chang (mingweichang@google.com)  
Kenton Lee (kentonl@google.com)
Kristina Toutanova

The references mention these individuals as points of contact for personal communication related to BERT. However, there is no explicit information on who specifically made the bert-large-cased-whole-word-masking variant available as a GitHub repo. 

[More Information Needed] on the specific individuals or team that published this particular pre-trained BERT model on GitHub or Hugging Face. The references focus more on the core BERT contributors and the research behind the model architecture in general.
- **Model type:** bert-large-cased-whole-word-masking is a deep bidirectional pre-trained language model that uses masked language modeling (MLM) with whole word masking as the training objective, which is an unsupervised learning method for the text modality.
- **Language(s):** The BERT-Large, Cased (Whole Word Masking) model uses English text from Wikipedia and BooksCorpus, preserving the true case and accent markers in the text.
- **License:** The BERT-Large, Cased (Whole Word Masking) model is released under the Apache 2.0 license, as stated in the references:

"All code *and* models are released under the Apache 2.0 license. See the `LICENSE` file for more information."

The link to the Apache 2.0 license file is not provided in the given references, so [More Information Needed] for the direct link to the license file.
- **Finetuned from model:** The bert-large-cased-whole-word-masking model is not fine-tuned from another model. It is a pre-trained model that uses the Whole Word Masking (WWM) training approach.

According to the references:

"*   **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters"

This indicates that bert-large-cased-whole-word-masking is a standalone pre-trained model and not fine-tuned from another base model. The model architecture consists of 24 layers, 1024 hidden units, 16 attention heads, and has 340M parameters in total.
### Model Sources

- **Repository:** https://github.com/google-research/bert
- **Paper:** https://arxiv.org/pdf/1810.04805.pdf
- **Demo:** [More Information Needed]

The references provided do not contain a link to a demo of the BERT-Large, Cased (Whole Word Masking) model. The references only provide download links for the pre-trained model weights and some general information about BERT, but no specific demo link for this particular model variant.

To provide a demo link in the model card, more information would be needed, such as whether a live demo of this specific model has been set up somewhere that could be linked to.
## Uses

### Direct Use

Based on the provided references, the BERT model bert-large-cased-whole-word-masking can be used without fine-tuning using the feature-based approach, as mentioned in points 2 and 3:

The contextual embeddings from one or more layers of the pre-trained BERT model can be extracted without fine-tuning any parameters. These embeddings can then be used as input to a randomly initialized two-layer BiLSTM before the classification layer.

Specifically, the best performing method concatenates the token representations from the top four hidden layers of the pre-trained Transformer. This approach performs only 0.3 F1 behind fine-tuning the entire model on the CoNLL-2003 Named Entity Recognition task.

[More Information Needed] for the specific code snippet to extract the embeddings and use them in a BiLSTM classifier.

In summary, bert-large-cased-whole-word-masking can be used without fine-tuning by extracting its contextual embeddings and using them as features in a separate classifier. This approach is competitive with fine-tuning the entire model.

For more details on how to use the model in a pipeline or with post-processing, [More Information Needed].

### Downstream Use

The bert-large-cased-whole-word-masking model can be fine-tuned for a variety of downstream NLP tasks, such as:

- Text classification or sequence tagging, by feeding the token representations into an output layer (Reference 1)
- Sentence pair tasks like paraphrasing, entailment, and question answering, by encoding the sentence pairs with self-attention (References 1, 2, 3) 
- Extracting fixed contextual embeddings for each input token to mitigate out-of-memory issues (Reference 6)

To fine-tune the model, you can plug in the task-specific inputs and outputs and train all parameters end-to-end (Reference 3). The process is straightforward using the example code provided, such as `run_classifier.py` for sentence classification tasks (Reference 4).

For example, to fine-tune on the MRPC paraphrasing dataset, you can run (Reference 5):

[More Information Needed]

This will load the pre-trained BERT model and fine-tune it for the specific task.

The fine-tuned model can then be integrated into larger applications for tasks like text classification, entailment, question answering, sentiment analysis, etc. (Reference 7, 8, 9, 10). Using a pre-trained model like BERT reduces the need for task-specific architectures and engineering (Reference 10).

The model and code are available on GitHub at https://github.com/google-research/bert (Reference 10).

Please let me know if you need any other information to complete the model card. I'm happy to help clarify or expand on any part of the model's usage and capabilities.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the BERT-large-cased-whole-word-masking model could be misused and what users should not do:

The model should not be used for tasks or purposes that promote discrimination, bias, hate speech, or other unethical applications. Since BERT is trained on a large corpus like Wikipedia, it may pick up and amplify certain societal biases present in the training data.

Users should not assume the model's outputs are always factual, unbiased, or authoritative. More information is needed on the specific pre-training data used, but in general large language models can generate convincing text that is incorrect or reflects problematic biases.

The model should not be used as a sole decision making tool for high-stakes applications like healthcare, finance, criminal justice, etc. without extensive testing, human oversight, and alignment with domain-specific guidelines. Relying on it in sensitive domains could lead to harm.

Users should clearly disclose they are using the whole-word masking variant of BERT-large in any papers or applications, to avoid confusion with the original model. Lack of transparency around which specific model is used could be seen as misleading.

[More Information Needed] on other potential misuse cases specific to the whole-word masking pre-training approach used in this variant.

In general, users have an ethical obligation to carefully evaluate the impacts and limitations of using this powerful model, be transparent about its usage, and proactively mitigate risks of misuse that could cause societal harm or amplify discrimination against marginalized groups. More specific usage guidelines would require additional context on the intended use case.

### Bias, Risks, and Limitations

Based on the provided references, here are some potential issues and limitations with the BERT-Large Cased model using Whole Word Masking:

Foreseeable harms and misunderstandings:
- The model was trained on a large text corpus like Wikipedia, which may contain biases. This could lead to the model exhibiting biased behavior if applied to certain downstream tasks without proper debiasing techniques.
- [More Information Needed] on specific foreseeable harms or misunderstandings, as the references do not go into detail on this aspect.

Technical limitations: 
- The model is very large with 340M parameters, 24 layers, 1024 hidden units, and 16 attention heads. This makes it computationally expensive to run and may be challenging to deploy in resource-constrained environments.
- [More Information Needed] on specific performance limitations, as the references do not include evaluation metrics on downstream tasks.

Sociotechnical limitations:
- Deploying such a large pre-trained model may exacerbate inequalities by benefiting those with more computing resources. Smaller organizations or under-resourced languages may not be able to take full advantage.
- There is a risk of over-reliance or blind trust in the model's outputs, especially by non-expert users. Limitations and potential biases need to be clearly communicated.
- [More Information Needed] on the specific training data used and any sensitive data issues, as the references do not cover this.

To summarize, the key limitations based on the given references are the potential for biased outputs, high computational requirements, and sociotechnical factors around responsible deployment and clear communication of the model's abilities and limitations to end users. More specific information would be needed to do a fuller analysis of foreseeable harms and misunderstandings.

### Recommendations

Based on the provided references about the BERT-large-cased-whole-word-masking model, here are some recommendations regarding foreseeable issues:

1. When using this model in a paper, clearly state that it is the Whole Word Masking variant of BERT-Large, as the training data and process are identical to the original model except for the whole word masking flag. This ensures transparency and reproducibility.

2. Be aware that BERT is a bidirectional model, which is an improvement over previous unidirectional or shallowly bidirectional models. However, [More Information Needed] on any specific issues that may arise from this bidirectional nature.

3. Note that using BERT involves two stages: pre-training and fine-tuning. [More Information Needed] on potential issues that could occur during these stages.

4. If using a custom vocabulary:
```
[More Information Needed]
```

5. If your task has a large domain-specific corpus available, [More Information Needed] on the benefits and potential issues of running additional pre-training steps starting from the BERT checkpoint.

Overall, more information is needed to provide comprehensive recommendations on foreseeable issues specific to the BERT-large-cased-whole-word-masking model. The references provide some general context about BERT but lack details on potential problems and their mitigation strategies.

## Training Details

### Training Data

The training data for the BERT-Large Cased model with Whole Word Masking consists of the BooksCorpus (800M words) and English Wikipedia (2,500M words), where only the text passages are extracted from Wikipedia while ignoring lists, tables, and headers. [More Information Needed] on any additional data pre-processing or filtering steps.

### Training Procedure

#### Preprocessing

Here are the details on tokenization and preprocessing for the BERT-large-cased-whole-word-masking model, based on the provided references:

Tokenization:
- The tokenizer performs text normalization by converting all whitespace characters to spaces. Since this is the cased model, it does not lowercase the input or strip accent markers.
- For sentence-level tasks, the tokenizer is instantiated as tokenizer = tokenization.FullTokenizer. Raw text is tokenized with tokens = tokenizer.tokenize(raw_text). 
- The tokenized sequence is truncated to the maximum sequence length (up to 512, but shorter if possible for memory/speed reasons).
- [CLS] and [SEP] tokens are added in the appropriate places.

Preprocessing:
- The pre-training corpus consists of the BooksCorpus (800M words) and English Wikipedia (2,500M words). For Wikipedia, only text passages are extracted while lists, tables and headers are ignored.
- A document-level corpus is used rather than a shuffled sentence-level corpus in order to extract long contiguous sequences.
- [More Information Needed] on any resizing or rewriting of the input data.

Input Representation: 
- The input representation can unambiguously represent both a single sentence and a pair of sentences (e.g. Question, Answer) in one token sequence.
- [More Information Needed] on further details of the input representation.

Masked Language Model Pre-training:
- Pre-training largely follows existing literature on language model pre-training. 
- The "masked language model" (MLM) pre-training objective is used, inspired by the Cloze task. MLM randomly masks some tokens from the input, and the objective is to predict the original vocabulary ID of the masked word based only on its context.

Let me know if you need any clarification or have additional questions!

#### Training Hyperparameters

Here are the training hyperparameters for the BERT-large-cased-whole-word-masking model, based on the provided references:

Model Architecture:
- 24-layer Transformer (Reference 11)

Pre-training:
- Corpus: Wikipedia (2,500M words) + BooksCorpus (800M words) (Reference 2, 6)
- Sequence Length: 512 tokens (Reference 3)
- Masking: WordPiece tokenization with 15% uniform masking rate (Reference 3)
- Batch Size: 256 sequences (128,000 tokens/batch) (Reference 3)
- Training Steps: 1,000,000 steps (~40 epochs) (Reference 3)
- Optimizer: Adam (Reference 3)
  - Learning Rate: 1e-4 (Reference 3, 9)
  - β1 = 0.9, β2 = 0.999 (Reference 3)
  - L2 weight decay: 0.01 (Reference 3)
  - Learning rate warmup: First 10,000 steps (Reference 3)
  - Learning rate decay: Linear (Reference 3)
- Dropout: 0.1 on all layers (Reference 8)
- Activation: GELU (Reference 8)

Fine-tuning:
- Batch Size: 16, 32 (Reference 1, 10)
- Learning Rate: 2e-5 (Reference 10)
- Epochs: 3 (Reference 10)
- Dropout: 0.1 (Reference 4)

[More Information Needed] for the following:
- Specific learning rate schedule for fine-tuning
- Fine-tuning hyperparameters for tasks other than mentioned in Reference 10

#### Speeds, Sizes, Times

Here are the details I can provide about the BERT-Large, Cased (Whole Word Masking) model based on the given references:

Model Specifications:
- 24-layer, 1024-hidden, 16-heads, 340M parameters

Training Details:
- Pre-trained on a large corpus (Wikipedia + BookCorpus) 
- Trained for a long time (1M update steps)
- [More Information Needed] on specific throughput, start/end times, and checkpoint sizes during pre-training

Fine-tuning:
- Fine-tuned for 2 epochs with a learning rate of 5e-5 and a batch size of 48 on downstream tasks
- [More Information Needed] on specific fine-tuning throughput, times, and checkpoint sizes for each downstream task

The references do not contain the specific technical details requested about throughput, timings, and checkpoint sizes. More information would be needed from the model training logs and results to include those details in the model card description.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the BERT-Large, Cased (Whole Word Masking) model evaluates on the following benchmarks and datasets:

1. GLUE (General Language Understanding Evaluation) benchmark, which includes various natural language understanding tasks like MNLI (MultiNLI), SST-2, and others.

2. SQuAD v1.1 (Stanford Question Answering Dataset) for question answering.

3. SQuAD v2.0, an extension of SQuAD v1.1 that includes unanswerable questions.

The references suggest that BERT-Large outperforms BERT-Base and achieves state-of-the-art results on these benchmarks and datasets. However, the exact performance metrics for the Whole Word Masking variant are [More Information Needed].

#### Factors

Based on the provided references, here are some key characteristics that may influence the behavior of the BERT-large-cased-whole-word-masking model:

Domain and Context:
- The model was pre-trained on the BooksCorpus (800M words) and English Wikipedia (2,500M words). Performance may vary when applied to domains or contexts that differ significantly from these sources. [More Information Needed] on specific domain/context performance.

- The model uses a document-level corpus for pre-training to extract long contiguous sequences. Behavior may be impacted when applied to shorter, non-contiguous text.

Population Subgroups:
[More Information Needed]
The references do not provide information to assess performance disparities across population subgroups. Disaggregated evaluation would be needed to uncover any such differences.

Other Influential Factors:
- The model uses whole word masking during pre-training (`--do_whole_word_mask=True`). This may impact performance compared to models using subword/wordpiece masking.

- BERT uses bidirectional self-attention, allowing it to incorporate context from both directions. This is a key architectural difference from unidirectional or shallowly bidirectional models that may affect behavior.

To fully characterize model behavior, additional disaggregated evaluation across relevant factors would be beneficial, as many details are not covered in the provided references. The model card would be strengthened by including such evaluation results.

#### Metrics

Based on the provided references about the BERT model, the key evaluation metrics mentioned are:

1. GLUE benchmark scores - BERT LARGE obtained a GLUE score of 80.5, which was a 7.0% absolute improvement over the previous state-of-the-art. GLUE includes various natural language understanding tasks.

2. Accuracy on specific GLUE tasks like MNLI (MultiNLI) - BERT achieved 86.7% accuracy, a 4.6% absolute improvement. 

3. F1 scores on SQuAD v1.1 and v2.0 question answering tasks - BERT LARGE achieved 93.2 F1 on SQuAD v1.1 (1.5 point absolute improvement) and pushed the SQuAD v2.0 Test F1 score higher (exact number not provided in the references).

The references do not provide detailed information about tradeoffs between different error types for BERT. More specific information would be needed to comment on how different errors are prioritized or balanced.

In summary, GLUE benchmark scores, accuracy on GLUE tasks, and F1 scores on SQuAD are the main evaluation metrics discussed for assessing BERT's performance, but more details would be required to analyze tradeoffs between different error types.

### Results

Based on the provided references, here are the evaluation results for the BERT-Large, Cased (Whole Word Masking) model:

Model Size and Architecture:
- 24-layer, 1024-hidden, 16-heads, 340M parameters (Reference 8)
- BERT LARGE significantly outperforms BERT BASE across all tasks, especially those with very little training data. (Reference 1)

Performance on SQuAD:
- Outperforms the top leaderboard system by +1.5 F1 in ensembling and +1.3 F1 as a single system on SQuAD v1.1. (Reference 2)
- Extends well to SQuAD v2.0 by treating questions without an answer as having an answer span with start and end at the [CLS] token. (Reference 3)

Performance on GLUE Benchmark:
- BERT LARGE outperforms all systems on all tasks by a substantial margin, obtaining a 7.0% average accuracy improvement over the prior state of the art. (Reference 4)
- Obtains a score of 80.5 on the official GLUE leaderboard. (Reference 4)

[More Information Needed] on specific evaluation metrics and results for Whole Word Masking on the BERT-Large, Cased model.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the BERT-large-cased-whole-word-masking model:

- BERT LARGE significantly outperforms BERT BASE across all tasks, especially those with very little training data. (Reference 1)

- On the GLUE benchmark, BERT LARGE obtains a score of 80.5, outperforming the previous state-of-the-art models by a substantial margin. BERT LARGE achieves a 7.0% average accuracy improvement over the prior state of the art. (Reference 3)

- On the SQuAD question answering task, the best performing BERT system outperforms the top leaderboard system by +1.5 F1 in ensembling and +1.3 F1 as a single system. The single BERT model outperforms the top ensemble system in terms of F1 score. (Reference 5)

- Increasing the model size leads to continual improvements on large-scale tasks. BERT LARGE, with 340M parameters, demonstrates convincingly that scaling to extreme model sizes also leads to large improvements on very small scale tasks, provided that the model has been sufficiently pre-trained. (References 6, 8)

[More Information Needed] on specific fine-tuning hyperparameters and additional evaluation metrics.

## Model Examination

Here is my attempt at an experimental section on explainability/interpretability for the BERT-large-cased-whole-word-masking model:

BERT (Bidirectional Encoder Representations from Transformers) is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. This allows the pre-trained BERT model to be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks.

In terms of explainability and interpretability, BERT's bidirectional architecture allows it to build rich contextual word representations that can help shed light on how the model derives meaning from text. By masking tokens during pre-training, BERT learns to predict words based on their surrounding context from both directions. This could potentially be leveraged to explain the most informative or influential context for a given prediction.

[More Information Needed] on specific techniques that could be applied to the BERT-large-cased-whole-word-masking model to improve explainability, such as:
- Visualizing attention weights to see which tokens the model attends to 
- Analyzing the learned word embeddings 
- Probing the model's representations at different layers
- Perturbing inputs and examining the model's sensitivity to input changes

Overall, BERT's deeply bidirectional architecture presents opportunities to better interpret and explain the model's behavior and predictions compared to shallower or unidirectional models. However, more research is needed to develop and apply explainability techniques specifically to this pre-trained model.

## Environmental Impact

- **Hardware Type:** [More Information Needed]

The provided references do not contain specific information about the hardware used to train the bert-large-cased-whole-word-masking model. More details would be needed from the model training logs or system specifications to determine the exact hardware type, such as GPU models or TPU versions, that were utilized during the pre-training process.
- **Software Type:** Based on the provided references, the model bert-large-cased-whole-word-masking is likely trained using the BERT (Bidirectional Encoder Representations from Transformers) method. BERT is a pre-training approach for language representations that learns from unlabeled text data.

The references mention that BERT can be fine-tuned using tools like Colab and TensorFlow (Reference 1), and it outperforms previous methods by being an unsupervised and deeply bidirectional system for pre-training NLP (Reference 7).

However, the specific software or framework used for training the bert-large-cased-whole-word-masking model is not explicitly stated in the given references. More information would be needed to determine the exact software type used for training this particular model.

[More Information Needed]
- **Hours used:** According to the references provided, the training of BERT LARGE was performed on 16 Cloud TPUs (64 TPU chips total), and each pretraining took 4 days to complete (Reference 4). 

However, there is no specific information on the training time for the BERT-Large, Cased (Whole Word Masking) variant. The references only mention that pre-trained models with Whole Word Masking have identical structure and vocab to the original BERT-Large models, and the data and training were otherwise identical (Reference 6).

Therefore, for the specific training time of the bert-large-cased-whole-word-masking model, [More Information Needed].
- **Cloud Provider:** Based on the provided references, there is no explicit mention of the cloud provider used for training the BERT-Large, Cased (Whole Word Masking) model. The references only provide information about the model architecture, training data, and pre-trained model download links hosted on Google Cloud Storage. To definitively answer the question about the specific cloud provider used for training, [More Information Needed].
- **Carbon Emitted:** [More Information Needed]

Based on the provided references, there is no specific information about the amount of carbon emitted during the training of the bert-large-cased-whole-word-masking model. The references discuss various aspects of BERT, such as the whole word masking strategy, vocabulary size considerations, pre-training and fine-tuning stages, and generating BERT activations. However, they do not provide any details about the carbon footprint or environmental impact of training this specific model.

To accurately report the carbon emissions, more information would be needed, such as the hardware used for training, the duration of the training process, the energy consumption of the training infrastructure, and the carbon intensity of the electricity grid used. Without these details, it is not possible to provide a reliable estimate of the carbon emissions associated with training the bert-large-cased-whole-word-masking model.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
- 24-layer, 1024-hidden, 16-heads, 340M parameters
- Based on the original Transformer encoder architecture described in Vaswani et al. (2017)
- Uses bidirectional self-attention, allowing each token to attend to both left and right context

Training Objective:
- Masked Language Model (MLM): Some percentage of input tokens are randomly masked, and the objective is to predict the original vocabulary id of the masked word based only on its bidirectional context
- During training, 15% of all tokens are masked in each sequence

[More Information Needed] on the specific details of the Whole Word Masking (WWM) technique used in this model variant compared to the original BERT.

### Compute Infrastructure

Based on the provided references, here is the information I can provide about the compute infrastructure used for the BERT-large-cased-whole-word-masking model:

The BERT-large model was trained on a large corpus consisting of Wikipedia and BookCorpus data for 1M update steps (Reference 4). 

Whole Word Masking was enabled during data generation by passing the flag `--do_whole_word_mask=True` to `create_pretraining_data.py` (Reference 5).

[More Information Needed] on the specific hardware (GPU/TPU types and counts) and distributed training setup used.

The model was fine-tuned for 2 epochs with a learning rate of 5e-5 and a batch size of 48 (Reference 1). [More Information Needed] on the hardware used for fine-tuning.

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

