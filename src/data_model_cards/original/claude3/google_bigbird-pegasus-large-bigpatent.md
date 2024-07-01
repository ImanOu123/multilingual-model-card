# Model Card for google/bigbird-pegasus-large-bigpatent

The google/bigbird-pegasus-large-bigpatent model is a large-sized pre-trained Pegasus Encoder-Decoder Transformer model that uses the BigBird sparse attention mechanism to handle longer sequences for improved performance on various NLP tasks such as question answering and summarization, specifically fine-tuned on the BigPatent dataset for long document summarization.

## Model Details

### Model Description

Model Card for google/bigbird-pegasus-large-bigpatent

Model Details:
- Architecture: BigBird-Pegasus, a sparse-attention based transformer encoder-decoder model that extends Pegasus to much longer sequences. It uses a combination of random, local, and global attention to achieve linear complexity with respect to sequence length, while preserving the properties of the quadratic full attention model. (References 1, 2, 4, 5, 9, 11)
- Pretraining Data: Books, CC-News, Stories, and Wikipedia datasets (Reference 6)
- Pretraining Objective: Masked Language Modeling (MLM), following the original BERT training by masking 15% of tokens (References 6, 8)
- Fine-tuning: The model has been fine-tuned for long document summarization on the BigPatent dataset (Reference 9)
- Model Size: Large (Reference 9)

Training Procedure:
- Initialization: Warm-started from the public RoBERTa checkpoint (Reference 6)  
- Sequence Length: Trained on sequences up to 4096 tokens long, which is 8x longer than standard transformer models (References 6, 11)
- Other training details like batch size, learning rate, hardware used: [More Information Needed]

Parameters:
[More Information Needed]

Important Disclaimers:
[More Information Needed]

For any further questions or information, please contact the model development team at [More Information Needed].

- **Developed by:** Manzil Zaheer; Guru Guruganesh; Avinava Dubey; Joshua Ainslie; Chris Alberti; Santiago Ontanon; Philip Pham; Anirudh Ravula; Qifan Wang; Li Yang; Amr Ahmed
- **Funded by:** Based on the provided references, there is no explicit mention of the people or organizations that funded the google/bigbird-pegasus-large-bigpatent project. The references discuss technical details about the model architecture, training data, and performance, but do not specify the funding sources.

[More Information Needed]
- **Shared by:** Based on the provided references, the main contributors to the BigBird model are:

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed

However, there is no specific information provided about who made the model google/bigbird-pegasus-large-bigpatent available online as a GitHub repo. [More Information Needed]
- **Model type:** The google/bigbird-pegasus-large-bigpatent model is a large-sized sparse-attention based encoder-decoder transformer model, pretrained using the masked language modeling (MLM) objective on long document datasets, and fine-tuned for long document summarization tasks.
- **Language(s):** The model google/bigbird-pegasus-large-bigpatent processes natural English language text for long document summarization tasks.
- **License:** [More Information Needed]

The provided references do not contain any information about the license being used for the specific model google/bigbird-pegasus-large-bigpatent. More details would be needed from the model developers or documentation to determine the applicable license.
- **Finetuned from model:** The model google/bigbird-pegasus-large-bigpatent is fine-tuned from the BigBird model, which is a sparse-attention based transformer that extends Transformer based models like BERT to much longer sequences.

Specifically, it uses the pretrained Pegasus Encoder-Decoder Transformer in large size (bigbp_large) as mentioned in reference 6:

"pretrained Pegasus Encoder-Decoder Transformer in large size(`bigbp_large`). Again following original implementation of Pegasus, they are transformers with pre-normalization. They have full set of separate encoder-decoder weights."

However, no direct link to the base Pegasus large model is provided in the references. [More Information Needed] on the exact link to the base model used.
### Model Sources

- **Repository:** https://github.com/google-research/bigbird
- **Paper:** https://arxiv.org/pdf/2007.14062.pdf
- **Demo:** [More Information Needed]

The provided references do not contain a direct link to a demo of the model google/bigbird-pegasus-large-bigpatent. The references discuss the BigBird architecture and training process in general, but do not mention this specific pre-trained model or provide a demo link for it.

To include the demo link in the model card, more specific information about the google/bigbird-pegasus-large-bigpatent model would need to be provided, such as where it is hosted and if a interactive demo is available. Without that, I do not have enough context to definitively answer the question.
## Uses

### Direct Use

The model google/bigbird-pegasus-large-bigpatent can be used without fine-tuning by directly utilizing the pretrained BigBird encoder. This allows it to replace BERT's encoder for tasks requiring longer sequence lengths.

To use the BigBird encoder directly instead of BERT:

```python
from bigbird.core import modeling

bigb_encoder = modeling.BertModel(...)
```

Alternatively, you can experiment with just the layers of the BigBird encoder:

```python
from bigbird.core import encoder

only_layers = encoder.EncoderStack(...)
```

[More Information Needed] on using the model without any post-processing or in a pipeline.

The pretrained and fine-tuned checkpoints for the model are available in a Google Cloud Storage Bucket and can optionally be downloaded using gsutil:

```bash 
mkdir -p bigbird/ckpt
gsutil cp -r gs://bigbird-transformer/ bigbird/ckpt/
```

### Downstream Use

The google/bigbird-pegasus-large-bigpatent model can be used for long document summarization tasks, especially on patent datasets. It is a pretrained Pegasus Encoder-Decoder Transformer in large size (bigbp_large) that has been fine-tuned on longer documents.

To directly use the BigBird encoder instead of the BERT model, you can use the following code:

```python
from bigbird.core import modeling

bigb_encoder = modeling.BertModel(...)
```

It can easily replace BERT's encoder.

Alternatively, you can also try playing with layers of the BigBird encoder:

```python
from bigbird.core import encoder

only_layers = encoder.EncoderStack(...)
```

The model comes with fine-tuned checkpoints (model.ckpt-300000) that work on longer documents, which can be plugged into a larger ecosystem or app for patent summarization.

[More Information Needed] on specific fine-tuning steps or integration into a larger app ecosystem.

### Out-of-Scope Use

Based on the provided references, there is limited information to comprehensively address potential misuse cases and what users ought not do with the google/bigbird-pegasus-large-bigpatent model. The references focus mainly on technical aspects like the model's ability to handle long sequences, comparisons to full attention mechanisms, pretraining datasets used, and fine-tuning for specific tasks.

To properly answer the question, more information would be needed on:

- The specific capabilities and limitations of the model
- The intended use cases it was designed and trained for
- Analysis of how the model's outputs could potentially be misinterpreted or misapplied
- Evaluation of risks if the model is used on inappropriate data or for unintended purposes
- Guidance from the developers on what constitutes proper and improper usage

Without these additional details, it is difficult to provide an informed response on foreseeable misuse and what users should avoid doing with this particular model. A more thorough model card covering these aspects would be required to fully address the question from the perspective of analyzing the interaction of this technology and society.

[More Information Needed]

### Bias, Risks, and Limitations

Based on the provided references about the BIGBIRD model, here are some potential issues and limitations:

Technical limitations:
- The sparse attention mechanism used in BIGBIRD, while more efficient than full attention for long sequences, does have theoretical limitations compared to full attention. Reference 7 proves there are tasks that can be solved by full attention in O(1) layers that would require Ω(n) layers for sparse attention with Õ(n) edges. So sparse attention is not a universal replacement for full attention.

Foreseeable harms and misunderstandings: 
[More Information Needed] The provided references focus on technical details of the BIGBIRD architecture and do not directly discuss potential harms or misunderstandings from applying the model.

Sociotechnical limitations:
[More Information Needed] The references do not cover sociotechnical aspects of deploying the BIGBIRD model. More information would be needed on the intended use cases, users, and social impact to analyze potential sociotechnical limitations.

To summarize, the key known limitation based on the references is that the sparse attention mechanism, while efficient, has provable limitations compared to full attention for certain problems. More information would be needed to assess foreseeable harms, misunderstandings, and sociotechnical limitations of applying the BIGBIRD model in real-world contexts. The provided technical details alone are not sufficient to fully analyze those aspects.

### Recommendations

Based on the provided references about the BigBird model, here are some recommendations regarding foreseeable issues with the google/bigbird-pegasus-large-bigpatent model:

1. Performance-memory tradeoff: While BigBird reduces memory consumption compared to other long-range transformers without sacrificing performance (Ref 1), it's important to assess if this tradeoff is suitable for the specific use case of the bigbird-pegasus-large-bigpatent model. 

2. Limitations on certain tasks: The references show that sparse attention mechanisms like BigBird's cannot universally replace dense attention for all tasks. There exist problems solvable in O(1) layers by full attention that would require Ω(n) layers for sparse attention with Õ(n) edges (Ref 2-5). It's crucial to evaluate if the bigbird-pegasus-large-bigpatent model's intended applications fall under this limitation.

3. Need for regularization: Reference 8 mentions that additional regularization was needed when training BigBird for certain competitive tasks. [More Information Needed] on whether such regularization was used for bigbird-pegasus-large-bigpatent and its impact.

4. Pretraining data: The bigbird-pegasus-large-bigpatent model should document what pretraining data was used (Ref 9 mentions Books, CC-News, Stories, Wikipedia for the original BigBird). Any potential biases or limitations from the pretraining data sources should be analyzed and communicated to users.

5. Evaluation on long sequences: As BigBird is designed for longer sequences, the bigbird-pegasus-large-bigpatent model should be thoroughly evaluated on patent documents of various lengths to understand its performance characteristics and limitations.

In summary, while BigBird enables handling longer sequences, the bigbird-pegasus-large-bigpatent model card should clearly communicate its performance tradeoffs, potential limitations on certain tasks, any important training details, pretraining data used, and extensive evaluation results to help users determine if it suits their use case. [More Information Needed] for some of these aspects based on the given references alone.

## Training Details

### Training Data

The training data for the model google/bigbird-pegasus-large-bigpatent consists of publicly available datasets including Books, CC-News, Stories, and Wikipedia. The documents longer than 4096 tokens were split into multiple documents, while much smaller documents were joined together, and 15% of tokens in these datasets were masked for the model to predict during training. [More Information Needed] on any additional data pre-processing or filtering steps specific to the bigpatent model.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the google/bigbird-pegasus-large-bigpatent model:

Tokenization:
The model borrows the SentencePiece vocabulary from RoBERTa, which in turn borrowed it from GPT-2 (Reference 8).

Resizing/Rewriting:
Documents longer than 4096 tokens are split into multiple documents, while documents much smaller than 4096 tokens are joined together (Reference 8).

Each example is prefixed and appended with [CLS] and [SEP] tokens respectively (Reference 9).

15% of tokens in the training datasets are masked, and the model is trained to predict these masked tokens (Reference 8).

[More Information Needed] on other specific preprocessing steps like normalization, filtering, etc.

#### Training Hyperparameters

Based on the provided references, there is not enough specific information to determine the exact training hyperparameters used for the google/bigbird-pegasus-large-bigpatent model. The references mention some general hyperparameter settings and regularization techniques used for certain BIGBIRD models, but they do not appear to be directly related to the bigbird-pegasus-large-bigpatent model in question.

To provide the detailed training hyperparameters, more specific information would be needed, such as:
- Learning rate and learning rate schedule 
- Batch size
- Number of training epochs or steps
- Optimizer used (e.g. Adam, AdaFactor)
- Dropout rate
- Weight decay
- Gradient clipping threshold
- Any data augmentation techniques employed

Without direct references to the hyperparameters for the bigbird-pegasus-large-bigpatent model, the most appropriate response is:

[More Information Needed]

#### Speeds, Sizes, Times

Here are the details I can provide about the google/bigbird-pegasus-large-bigpatent model based on the given references:

Model Architecture:
- It is a pretrained Pegasus Encoder-Decoder Transformer in large size (bigbp_large). 
- Following the original Pegasus implementation, the transformers use pre-normalization.
- It has a full set of separate encoder-decoder weights.

Checkpoints:
- For long document summarization datasets, Pegasus checkpoints (model.ckpt-0) have been converted for each dataset.
- Fine-tuned checkpoints (model.ckpt-300000) are also provided which work on longer documents.
- Checkpoints are available in a Google Cloud Storage Bucket at gs://bigbird-transformer/

[More Information Needed] on the following:
- Throughput 
- Start or end training time
- Exact checkpoint sizes

The references do not contain specific details on those aspects of the model. More information would be needed from the model developers or training logs to include those details in the model card.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the google/bigbird-pegasus-large-bigpatent model evaluates on the following benchmarks and datasets:

1. Long document classification tasks, such as the Arxiv and Patents datasets (References 1, 8, 10)
2. DNA sequence analysis tasks (References 3, 5)
3. Question answering tasks, such as the Natural Questions dataset (Reference 7)
4. GLUE (General Language Understanding Evaluation) benchmark, which includes 8 different natural language understanding tasks (Reference 11)

[More Information Needed] on the specific metrics and results for each benchmark or dataset, as the provided references do not include detailed performance numbers for the google/bigbird-pegasus-large-bigpatent model.

#### Factors

Based on the provided references about the google/bigbird-pegasus-large-bigpatent model, here are some key characteristics that may influence the model's behavior:

Domain and Context:
- The model is designed to handle long document classification tasks where discriminating information may not be located in the first 512 tokens (Ref 1). This suggests the model is well-suited for domains with lengthy text inputs.
- The model has been applied to genomics data, such as DNA sequence analysis for tasks like promoter site prediction and predicting effects of non-coding variants (Ref 5, 7). This indicates potential applicability in the bioinformatics domain.

Population Subgroups:
- [More Information Needed] The references do not provide clear information about performance across different population subgroups.

Other Factors:
- The model uses an efficient sparse attention mechanism that allows it to process longer input sequences (up to 4096 tokens) while maintaining memory efficiency (Ref 4, 6). This architectural choice influences the model's capabilities.
- Pretraining was done using masked language modeling (MLM) on large datasets like Books, CC-News, Stories and Wikipedia (Ref 3, 11). The pretraining data characteristics can impact the model's knowledge and biases.

Evaluation:
- The model achieves state-of-the-art performance on various NLP tasks like question answering and long document classification (Ref 6, 9). However, more disaggregated evaluation across different factors would help uncover potential performance disparities.

In summary, the google/bigbird-pegasus-large-bigpatent model is designed to handle long text sequences and has shown applicability in domains like NLP and genomics. The sparse attention mechanism and pretraining approach influence its capabilities. However, more information is needed to assess performance across population subgroups and conduct disaggregated evaluation.

#### Metrics

Based on the provided references about the BIGBIRD model, there is no direct information on what specific metrics will be used to evaluate the google/bigbird-pegasus-large-bigpatent model, especially in light of tradeoffs between different error types.

The references mention evaluating BIGBIRD on various NLP tasks and datasets, using metrics like:
- Bits per character for evaluating MLM pretraining performance 
- F1 score on a DNA sequence classification task
- Leaderboard performance on the Natural Questions dataset
- Accuracy on GLUE benchmark tasks

However, none of these directly address the tradeoffs between different error types for the specific google/bigbird-pegasus-large-bigpatent model.

[More Information Needed] on the precise evaluation metrics used for this model and how they account for tradeoffs between different errors. Details on the types of errors considered, metrics chosen, and rationale for those choices would be needed to fully answer the question.

### Results

Based on the provided references, here is the evaluation information I could find for the google/bigbird-pegasus-large-bigpatent model:

Performance:
- BIGBIRD outperforms models with smaller context lengths on tasks like Natural Questions, showing the importance of using longer context. However, the BIGBIRD submission was a single model whereas other top entries used ensembles, which may explain slightly lower accuracy in exact answer phrase selection. [Reference 6]
- On GLUE benchmark, even the BIGBIRD-ITC base model (with specific configuration) is competitive with models like BERT, XLNet and RoBERTa. [Reference 8]

Efficiency: 
- The models were trained on reasonable 16GB memory/chip with batch sizes of 32-64. The memory efficiency is due to the efficient blocking and sparsity structure of the sparse attention mechanism. [Reference 2]

[More Information Needed] for other specific performance metrics, efficiency measurements, and model architecture details for the google/bigbird-pegasus-large-bigpatent model. The references mostly discuss BIGBIRD models in general but do not provide evaluation results for this specific pretrained model.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the google/bigbird-pegasus-large-bigpatent model:

The BIGBIRD model was pretrained using the MLM objective on standard datasets, warm-starting from the public RoBERTa checkpoint. BIGBIRD performed better than limited length RoBERTa in predicting masked out tokens.

For downstream tasks:
- On a DNA task, finetuned BIGBIRD achieved nearly perfect accuracy, with a 5% jump from the previous best reported method. 
- BIGBIRD-ETC (with expanded context) performed the best, due to the efficient blocking and sparsity structure of the sparse attention mechanism. Models were trained on 16GB memory/chip.
- On very competitive tasks, BIGBIRD required some additional regularization during training. The best configuration was selected on the dev set for each dataset.
- Comparing to top leaderboard entries, the importance of longer context is clear, as both Longformer and BIGBIRD outperformed models with smaller contexts. BIGBIRD had slightly lower accuracy than some top ensembles on exact answer phrase selection for Natural Questions.

[More Information Needed] on the specific datasets, metrics and scores for the downstream evaluation beyond what is provided in the references.

## Model Examination

Based on the provided references, here is the information I could gather about the google/bigbird-pegasus-large-bigpatent model in terms of explainability and interpretability:

The model is a pretrained Pegasus Encoder-Decoder Transformer in large size (bigbp_large). It follows the original implementation of Pegasus, using transformers with pre-normalization and a full set of separate encoder-decoder weights.

For long document summarization datasets, the model checkpoints (model.ckpt-0) have been converted for each dataset. Fine-tuned checkpoints (model.ckpt-300000) that work on longer documents have also been provided.

[More Information Needed] regarding specific explainability or interpretability techniques applied to this model, such as analyzing attention weights, generating explanations for predictions, or using visualization tools to understand the model's behavior.

To improve the explainability and interpretability section, more details could be provided on:
- Any techniques used to analyze or visualize the model's decision-making process
- Examples or case studies demonstrating how the model's predictions can be interpreted
- Discussion on the limitations or challenges in explaining the model's behavior
- Plans for future work on enhancing the model's explainability and interpretability

## Environmental Impact

- **Hardware Type:** Based on the provided references, the model google/bigbird-pegasus-large-bigpatent was trained on GPUs with 16GB memory per chip, using a batch size of 32-64. Specifically, reference 3 states:

"We note that we trained our models on a reasonable 16GB memory/chip with batch size of 32-64. Our memory efficiency is due to efficient blocking and sparsity structure of the sparse attention mechanism described in Sec. 2."

The references also mention TPUs as a potential hardware accelerator, but do not explicitly state that TPUs were used for training this specific model.
- **Software Type:** Based on the provided references, the google/bigbird-pegasus-large-bigpatent model is trained using the following software:

- Pegasus: The model is a pretrained Pegasus Encoder-Decoder Transformer in large size, utilizing state-of-the-art summarization specific pretraining from Pegasus (Reference 1 and 2).

- BigBird: The model incorporates the BigBird sparse-attention based transformer, which extends Transformer based models to handle longer sequences (Reference 3).

The references do not explicitly mention the specific software versions or libraries used for training the model. [More Information Needed] on the exact software stack and versions employed in the training process.
- **Hours used:** [More Information Needed]

The provided references do not contain specific information about the training time for the google/bigbird-pegasus-large-bigpatent model. The references discuss pretraining datasets, hyperparameters, and performance of BIGBIRD models in general, but do not provide the exact training time for this particular model.
- **Cloud Provider:** Based on the provided references, the model google/bigbird-pegasus-large-bigpatent is likely trained on Google Cloud Platform (GCP). This can be inferred from the following reference:

```
gcloud compute ssh --zone "europe-west4-a" "bigbird"
```

This command shows an example of connecting to a Google Cloud instance named "bigbird" in the "europe-west4-a" zone using the `gcloud` command-line tool, which is specific to Google Cloud Platform.

However, more specific information about the exact cloud provider and instance type used for training this particular model is not provided in the given references. To confirm the cloud provider and provide more details, additional information would be needed from the model developers or documentation.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the google/bigbird-pegasus-large-bigpatent model. To accurately report the carbon emissions, more details would be needed, such as the computational resources used, training duration, and energy consumption.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The google/bigbird-pegasus-large-bigpatent model is based on the BigBird sparse attention mechanism combined with the Pegasus large encoder-decoder architecture. Key points:

- Uses BigBird sparse attention which reduces the quadratic dependency on sequence length to linear
- BigBird attention has each query attend to r random keys, w/2 tokens to the left and right of its location, and g global tokens
- The encoder and decoder are transformer models with pre-normalization and shared query/key/value matrices and feedforward layers
- Initialized from the pretrained Pegasus large checkpoint

Model Objective:
The model aims to enable handling much longer sequence lengths (up to 8x previous limits) while preserving the expressive power and capabilities of full attention transformers. This allows it to significantly improve performance on NLP tasks like question answering and document summarization that benefit from longer context.

[More Information Needed] on the specific pretraining objective used for this model.

### Compute Infrastructure

Based on the provided references, here is the information about the compute infrastructure used for the google/bigbird-pegasus-large-bigpatent model:

The model was trained on Google Cloud TPU instances. An example command to create the TPU instance is:

```
gcloud compute ssh --zone "europe-west4-a" "bigbird"
```

The specific TPU instance name (e.g. "bigbird") and zone (e.g. "europe-west4-a") may vary.

The model training was memory-efficient due to the efficient blocking and sparsity structure of the sparse attention mechanism. The models were trained on 16GB memory/chip TPUs with batch sizes of 32-64.

[More Information Needed] on the exact TPU type, number of cores, and other infrastructure details.

The code was optimized for modern hardware like GPUs and TPUs by "blockifying" the lookups to enable coalesced memory operations that load blocks of contiguous bytes efficiently. Sparse matrix multiplications are not efficiently implemented on GPUs.

## Citation

```
@misc{manzil-big,
    author = {Manzil Zaheer and
              Guru Guruganesh and
              Avinava Dubey and
              Joshua Ainslie and
              Chris Alberti and
              Santiago Ontanon and
              Philip Pham and
              Anirudh Ravula and
              Qifan Wang and
              Li Yang and
              Amr Ahmed},
    title  = {Big Bird: Transformers for Longer Sequences},
    url    = {https://arxiv.org/pdf/2007.14062.pdf}
}
```

