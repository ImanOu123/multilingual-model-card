# Model Card for google/bigbird-roberta-large

google/bigbird-roberta-large is a large pretrained BigBird model based on RoBERTa. BigBird is a sparse-attention based transformer which extends transformer-based models like BERT to much longer sequences, drastically improving performance on various NLP tasks such as question answering and summarization.

## Model Details

### Model Description

Model Card for google/bigbird-roberta-large

Model Details:
- Architecture: BigBird, a sparse-attention based transformer that extends BERT to handle much longer sequences. It includes additional "global" tokens that attend to all existing tokens. (References 1, 4, 5, 9)
- Training Data: Books, CC-News, Stories, and Wikipedia datasets. Documents longer than 4096 tokens were split, and smaller documents were joined. (References 6, 10) 
- Training Objective: Masked Language Modeling (MLM), predicting 15% of randomly masked out tokens. (References 6, 7, 10)
- Parameters: [More Information Needed]
- Vocabulary: Sentencepiece vocabulary borrowed from RoBERTa, which was borrowed from GPT-2. (Reference 10)

Training Procedure:
- Initialization: Warm-started from the public RoBERTa checkpoint. (References 6, 10)
- Batch Size: 32-64 (Reference 8)
- Hardware: 16GB memory/chip (Reference 8)
- Two model versions, base and large, were trained. (Reference 7)

Important Disclaimers:
[More Information Needed]

Contact:
For questions about this model, please contact the model development team at [More Information Needed].

- **Developed by:** Manzil Zaheer; Guru Guruganesh; Avinava Dubey; Joshua Ainslie; Chris Alberti; Santiago Ontanon; Philip Pham; Anirudh Ravula; Qifan Wang; Li Yang; Amr Ahmed
- **Funded by:** Based on the provided references about the google/bigbird-roberta-large model, there is no explicit mention of the people or organizations that funded this project. The references discuss technical details about the model's training data, architecture, and performance, but do not specify the funding sources.

[More Information Needed]

To fill out this part of the model card, the project organizer would need to gather additional information about the funding and support for developing the google/bigbird-roberta-large model.
- **Shared by:** The contributors that made the model google/bigbird-roberta-large available online as a GitHub repo are:

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed.

This information is directly stated in the references provided.
- **Model type:** google/bigbird-roberta-large is a large version of the BigBird model, which is a sparse-attention based transformer model pretrained using the masked language modeling (MLM) objective on textual data, extending the capabilities of BERT-like models to handle much longer sequences.
- **Language(s):** The google/bigbird-roberta-large model uses English natural language for processing and understanding various NLP tasks such as question answering, summarization, and long document classification.
- **License:** [More Information Needed]

The provided references do not contain any information about the specific license being used for the google/bigbird-roberta-large model. More details would be needed from the model developers or official documentation to determine the license under which this model is being released.
- **Finetuned from model:** Based on the provided references, the google/bigbird-roberta-large model is fine-tuned from the RoBERTa model checkpoint:

"We warm start from RoBERTa's checkpoint." (Reference 7)

However, a direct link to the specific RoBERTa model used as the base is not provided in the given references. 

[More Information Needed] on the exact RoBERTa model and link used as the base for fine-tuning google/bigbird-roberta-large.
### Model Sources

- **Repository:** https://github.com/google-research/bigbird
- **Paper:** https://arxiv.org/pdf/2007.14062.pdf
- **Demo:** Based on the provided references, there is no direct link to a demo of the google/bigbird-roberta-large model. The references mention a quick fine-tuning demonstration for text classification in [imdb.ipynb](bigbird/classifier/imdb.ipynb), but this appears to be an example notebook rather than a hosted demo of the specific google/bigbird-roberta-large model.

[More Information Needed] to provide a link to an actual hosted demo of the google/bigbird-roberta-large model, if one exists. The model card would benefit from including such a demo link to allow users to easily test out the model's capabilities, but more details are required from the team.
## Uses

### Direct Use

The model google/bigbird-roberta-large can be used without fine-tuning by directly utilizing the BigBird encoder instead of the BERT encoder. This can be done using the following code snippet:

```python
from bigbird.core import modeling

bigb_encoder = modeling.BertModel(...)
```

Alternatively, one can also try playing with layers of the BigBird encoder:

```python
from bigbird.core import encoder

only_layers = encoder.EncoderStack(...)
```

[More Information Needed] on how to use the model without post-processing or plugging into a pipeline.

### Downstream Use

The google/bigbird-roberta-large model can be used for various NLP tasks that benefit from longer input sequences, such as question answering with supporting evidence and long document classification. 

To directly use the BigBird encoder instead of the BERT model, you can use the following code:

```python
from bigbird.core import modeling

bigb_encoder = modeling.BertModel(...)
```

It can easily replace BERT's encoder.

Alternatively, you can also try using just the layers of the BigBird encoder:

```python
from bigbird.core import encoder

only_layers = encoder.EncoderStack(...)
```

For a quick demonstration of fine-tuning BigBird for text classification, refer to the provided notebook:
[imdb.ipynb](bigbird/classifier/imdb.ipynb)

To get started with running a classification experiment using BigBird, execute the following commands:

```shell
export GCP_PROJECT_NAME=bigbird-project  # Replace by your project name 
export GCP_EXP_BUCKET=gs://bigbird-transformer-training/  # Replace
sh -x bigbird/classifier/base_size.sh
```

[More Information Needed] on integrating BigBird into a larger ecosystem or app.

### Out-of-Scope Use

Based on the provided references, there is limited information to comprehensively assess how the google/bigbird-roberta-large model may be misused. The references focus mainly on technical details of the model architecture, training datasets, and benchmark performance. They do not directly address potential misuse cases or provide guidance on what users should not do with the model.

To properly answer the question, more information would be needed, such as:

[More Information Needed]
- Details on the intended use cases and limitations of the model 
- Analysis of potential harmful outputs the model could generate if misused
- Ethical considerations around deploying the model, especially for high-stakes applications
- Guidelines and restrictions on using the model in ways that could cause societal harm

Without further details from the model developers on these aspects, it is difficult to provide a complete answer on foreseeable misuse and what users ought not do. The team should carefully consider and document these points to promote responsible use of the google/bigbird-roberta-large model.

### Bias, Risks, and Limitations

Based on the provided references about the BigBird model (google/bigbird-roberta-large), here are some potential issues and limitations:

Foreseeable harms and misunderstandings:
- The model was pretrained on large web corpora like Books, CC-News, Stories and Wikipedia. This data may contain biases, misinformation, and problematic content that could be reflected in the model's outputs. [More Information Needed] on specific risks.
- As a large language model, BigBird may be misused to generate fake news, spam, hateful/offensive text if prompted. Appropriate safeguards are needed.

Technical limitations: 
- BigBird uses sparse attention to scale to longer sequences (4096 tokens), but theoretical analysis shows sparse attention cannot universally replace dense attention. Some tasks provably require Ω(n) layers with sparse attention vs O(1) with dense (full) attention.
- Scaling limitations still exist beyond 4096 tokens. BigBird enables 8x longer context than previous models on similar hardware, but further efficiency improvements are needed.

Sociotechnical limitations:
- BigBird was evaluated on academic NLP benchmarks like GLUE, but real-world use cases may reveal additional robustness and fairness issues. [More Information Needed] 
- As a large pretrained model, BigBird has high compute costs for finetuning and inference. This could lead to inequitable access. Responsible deployment practices are important to consider.

In summary, key limitations seem to be: scaling limits of sparse attention, potential biases from pretraining data, robustness to misuse, and compute-related access barriers. More specific information is needed to fully characterize harms and sociotechnical impacts. Thoughtful deployment and monitoring are recommended.

### Recommendations

Based on the provided references about the BIGBIRD model, here are some recommendations regarding foreseeable issues with google/bigbird-roberta-large:

1. Sparse attention mechanisms like those used in BIGBIRD cannot universally replace dense attention mechanisms without tradeoffs. There are certain tasks that require Ω(n) layers for sparse attention with Õ(n) edges to solve, while full attention can solve in O(1) layers. This suggests BIGBIRD may require more layers and computation for certain problem types compared to dense attention. [References 1, 2, 4, 5]

2. [More Information Needed] on the specific downstream tasks and datasets google/bigbird-roberta-large was evaluated on. The references mention GLUE benchmark results for a BIGBIRD-base model, but not the -large variant. More information is needed on the large model's performance. [Reference 8]

3. The ability for BIGBIRD to handle longer sequence lengths up to 8x previous limits is noted as a benefit that improves various NLP task performance. However, [More Information Needed] on any potential negative consequences or failure modes that may arise from significantly increasing the context window. [Reference 10]

4. [More Information Needed] on the pretraining data and methodology used for google/bigbird-roberta-large specifically. The references mention pretraining datasets and procedures for BIGBIRD models in general, but not the exact setup for this particular large variant. [Reference 9]

In summary, while BIGBIRD enables longer context and shows promising results, the use of sparse attention may require more layers for certain tasks compared to dense attention. More specific information is needed on the large model variant's pretraining setup, downstream evaluation, and potential issues that may arise from its extended sequence length. I recommend conducting further analyses and benchmarks to better understand the model's capabilities and limitations.

## Training Details

### Training Data

The model google/bigbird-roberta-large was pretrained on four publicly available datasets: Books, CC-News, Stories, and Wikipedia. Documents longer than 4096 tokens were split into multiple documents, while much smaller documents were joined together.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the google/bigbird-roberta-large model:

Tokenization:
- The model uses the same sentencepiece vocabulary as RoBERTa, which was borrowed from GPT-2. (Reference 7)
- Each example is prefixed and appended with [CLS] and [SEP] tokens respectively. (Reference 6)

Resizing/Rewriting:
- Documents longer than 4096 tokens are split into multiple documents. (Reference 7)
- Documents much smaller than 4096 tokens are joined together. (Reference 7)

Other Preprocessing:
- 15% of tokens in the pretraining datasets are masked out for the MLM (masked language modeling) objective. (Reference 7)

[More Information Needed] on further specifics of the tokenization process and any additional preprocessing steps.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the google/bigbird-roberta-large model:

- Pretrained on four datasets: Books, CC-News, Stories, and Wikipedia (Reference 7)
- Split documents longer than 4096 into multiple documents, and joined documents much smaller than 4096 (Reference 7) 
- Masked 15% of tokens in the datasets, following original BERT training (Reference 7)
- Warm started from RoBERTa's checkpoint (Reference 7)
- Used a variant of contrastive predictive coding as a dual encoder model for regularization (Reference 8)
- Used position embedding for ITC and relative position encoding for ETC (Reference 8)

[More Information Needed] on the exact values used for:
- Number of attention heads 
- Hidden size
- Intermediate size
- Number of hidden layers
- Activation function
- Attention dropout
- Hidden dropout
- Learning rate
- Batch size
- Number of training steps/epochs

The detailed hyperparameters are likely in Tables 12 and 13 mentioned in Reference 8, but the contents of those tables were not provided. More details from those tables would be needed to specify the exact hyperparameter values used.

#### Speeds, Sizes, Times

Here are the details about the google/bigbird-roberta-large model based on the provided references:

Throughput: [More Information Needed]

Start/End Time: [More Information Needed]

Checkpoint Sizes: 
- Pretrained BERT model checkpoints are available in base (bigbr_base) and large (bigbr_large) sizes, corresponding to BERT/RoBERTa-like encoder only models. (Reference 6)
- Pretrained Pegasus Encoder-Decoder Transformer checkpoints are available in large size (bigbp_large). For long document summarization datasets, both Pegasus checkpoints (model.ckpt-0) and fine-tuned checkpoints (model.ckpt-300000) are provided. (Reference 7)

Other Details:
- BigBird uses a sparse attention mechanism that reduces the quadratic dependency on sequence length to linear, allowing it to handle sequences up to 8x longer than previously possible using similar hardware. (References 8, 10) 
- BigBird outperforms limited length RoBERTa models, with BigBird-ETC performing the best. The models were trained on 16GB memory/chip with batch sizes of 32-64. (Reference 1)
- In the Long Range Arena benchmark, BigBird reduces memory consumption without sacrificing performance compared to other long-range transformers. (Reference 2)
- BigBird was pretrained using the MLM objective on four standard datasets, warm-starting from the public RoBERTa checkpoint. (Reference 4)
- On a DNA sequence classification task, BigBird achieved nearly perfect accuracy, a 5% jump from the previous best reported method. (Reference 5)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the google/bigbird-roberta-large model evaluates on the following benchmarks and datasets:

1. Long Range Arena benchmark, which includes six tasks that require longer context (Reference 1)

2. GLUE (General Language Understanding Evaluation) benchmark, which includes 8 different natural language understanding tasks (Reference 9)

3. Question answering datasets:
   - Natural Questions (Reference 6, 10)
   - [More Information Needed] on other specific question answering datasets

4. Document summarization datasets:
   - [More Information Needed] on specific document summarization datasets

5. Document classification datasets:
   - Arxiv dataset (Reference 8)
   - Patents dataset (Reference 8)

6. DNA sequence analysis tasks:
   - Predicting masked out tokens in DNA sequences, using datasets like human reference genome and Saccharomyces Genome Database (Reference 5)
   - Two downstream tasks related to DNA sequence analysis (Reference 5), but [More Information Needed] on the specific tasks and datasets

The model achieves state-of-the-art or competitive performance on various tasks, such as improving accuracy on the Arxiv dataset by about 5% points compared to previous methods (Reference 8, 11).

#### Factors

Based on the provided references about the google/bigbird-roberta-large model, here are some foreseeable characteristics that may influence the model's behavior:

Domain and Context:
- The model was pretrained on large datasets including Books, CC-News, Stories and Wikipedia (Ref 9). This broad pretraining corpus suggests the model should have strong general language understanding capabilities across various domains.
- The model shows strong performance on several downstream NLP tasks like question answering with supporting evidence retrieval and long document classification (Ref 3, 4). This indicates the model can handle tasks requiring reasoning over longer contexts.
- The model was also applied to DNA sequence analysis by tokenizing base pairs, allowing it to operate on very long genetic sequences (Ref 7, 8). This demonstrates the model's flexibility to adapt to specialized domains beyond natural language.

Population Subgroups:
[More Information Needed] The provided references do not contain information on model performance disaggregated across different population subgroups.

Other Notable Characteristics:
- The model uses an efficient sparse attention mechanism that allows it to process sequences up to length 4096 while maintaining memory efficiency (Ref 1, 2). 
- Despite its capability to handle long sequences, theoretical analysis shows there are still some tasks that require full attention and cannot be efficiently solved by sparse attention models like BigBird (Ref 5, 6).
- Evaluation on the GLUE benchmark shows BigBird is competitive with other top models like BERT, XLNet and RoBERTa (Ref 11).

To fully characterize the model's performance across different subgroups, more disaggregated evaluation would be needed. The current references primarily focus on overall model capabilities and efficiency.

#### Metrics

Based on the provided references about the google/bigbird-roberta-large model, the following metrics are mentioned for evaluation:

1. Bits per character for evaluating performance on the masked language modeling (MLM) pretraining task, following the approach in [8].

2. F1 score for evaluating performance on a DNA sequence classification task after finetuning the pretrained BIGBIRD model. 

3. [More Information Needed] on specific metrics used for the GLUE benchmark tasks, though the model's performance is compared to BERT, XLNet, and RoBERTa on the 8 different natural language understanding tasks in GLUE.

4. Accuracy in exact answer phrase selection on the Natural Questions dataset, where BIGBIRD's performance as a single model is compared to other top leaderboard entries which are ensembles.

5. Accuracy on document classification tasks like the Arxiv and Patents datasets, where BIGBIRD improved state-of-the-art, especially for longer documents with fewer training examples.

To summarize, bits per character, F1 score, accuracy, and presumably standard metrics for the GLUE tasks are used to evaluate the BIGBIRD model, showcasing tradeoffs and gains in performance on different tasks involving long sequences. However, more specific information would be needed to fully describe the evaluation metrics and tradeoffs.

### Results

Based on the provided references about the google/bigbird-roberta-large model, here are the evaluation results I could find:

Performance:
- BIGBIRD outperforms models with smaller context lengths like BERT, XLNet and RoBERTa on long document tasks where discriminating information may not be in the first 512 tokens. (Reference 4)
- On the Natural Questions leaderboard, BIGBIRD-ETC outperforms the top 3 entries (excluding itself). The slightly lower accuracy compared to the top entries may be because they use model ensembles while the BIGBIRD submission is a single model. (Reference 6)
- On a DNA sequence modeling task, BIGBIRD achieves nearly perfect accuracy, a 5% jump from the previous best reported accuracy. (Reference 2)

Efficiency:
- BIGBIRD models were trained on 16GB memory per chip with batch sizes of 32-64. The memory efficiency is due to the efficient blocking and sparse attention mechanism used. (Reference 1) 

Fairness:
[More Information Needed]

Robustness:
[More Information Needed]

Intended Use:
- BIGBIRD's longer input sequence handling capability would be beneficial for genomics tasks using deep learning, as many functional effects in DNA are highly non-local. (Reference 5)

Limitations:
[More Information Needed]

Risks & Potential Misuse:
[More Information Needed]

#### Summary

Here is a summary of the evaluation results for the google/bigbird-roberta-large model based on the provided references:

- BigBird outperforms limited length RoBERTa models, with BigBird-ETC performing the best, while being memory efficient due to its sparse attention mechanism. (Reference 1)

- On a DNA sequence modeling task, BigBird achieves nearly perfect accuracy, a 5% jump over the previous best reported method. (Reference 2) 

- On the Long Range Arena benchmark of tasks requiring longer context, BigBird reduces memory consumption without sacrificing performance compared to other long-range Transformer models. (Reference 3)

- BigBird with MLM pretraining significantly improves performance on two downstream genomics tasks. (Reference 4)

- [More Information Needed] on the specific bits per character results for BigBird's MLM pretraining. (Reference 5)

- On long document classification with discriminating information beyond the first 512 tokens, BigBird uses a sequence length of 4096. (Reference 6)

- [More Information Needed] on BigBird's performance on the 8 GLUE natural language understanding tasks compared to BERT, XLNet and RoBERTa. (Reference 8)

- On the Natural Questions leaderboard, BigBird outperforms models with smaller contexts, though the top 3 entries are ensembles while the BigBird submission is a single model. (Reference 9)

In summary, BigBird demonstrates strong performance on tasks requiring long context while being memory efficient, though more specific results are needed in some areas.

## Model Examination

Based on the provided references, here is the experimental section on explainability/interpretability for the google/bigbird-roberta-large model:

The BigBird model extends the capabilities of Transformer-based models like BERT to handle much longer sequences while preserving the expressive power through the use of sparse attention and extra global tokens. The theoretical analysis reveals some benefits of having O(1) global tokens (such as CLS) that attend to the entire sequence as part of the sparse attention mechanism. 

[More Information Needed] on specific techniques or experiments conducted to explain or interpret the inner workings and decision making process of the BigBird model. Further research into explainability methods tailored for the unique sparse attention architecture could yield valuable insights.

By enabling longer context modeling, BigBird has demonstrated state-of-the-art performance improvements on various downstream NLP tasks like question answering and document summarization. [More Information Needed] regarding analysis of these results to better understand how the extended context contributes to the model's strong performance.

In summary, while the theoretical underpinnings and empirical results of BigBird have been established, there is opportunity for additional explainability and interpretability work to provide a clearer understanding of the model's capabilities and behaviors.

## Environmental Impact

- **Hardware Type:** Based on the provided references, the google/bigbird-roberta-large model was trained on GPUs with 16GB memory per chip, using a batch size of 32-64. The references mention that the memory efficiency of the model is due to the efficient blocking and sparsity structure of the sparse attention mechanism described in the BigBird paper.
- **Software Type:** Based on the provided references, the google/bigbird-roberta-large model is trained on the following software/datasets:

- Books dataset
- CC-News dataset
- Stories dataset
- Wikipedia

The model is pretrained on these four publicly available datasets using a masked language modeling (MLM) objective, similar to the pretraining of BERT and RoBERTa models. The references do not provide specific information about the software or libraries used for training the model.
- **Hours used:** [More Information Needed]

The provided references do not contain specific information about the amount of time used to train the google/bigbird-roberta-large model. More details would be needed from the model training logs or records to determine the exact training time.
- **Cloud Provider:** Based on the provided references, the model google/bigbird-roberta-large was trained on Google Cloud Platform (GCP). Specifically, it was trained using Google Cloud TPUs, as evidenced by the following excerpts:

1. "gcloud compute ssh --zone "europe-west4-a" "bigbird""
2. "More details about creating Google Cloud TPU can be found in [online documentations](https://cloud.google.com/tpu/docs/creating-deleting-tpus#setup_TPU_only)."
3. "Please create a project first and create an instance in a zone which has quota as follows"
   ```bash
   gcloud compute instances create \
     bigbird \
     --zone=europe-west4-a \
     --machine-type=n1-standard-16 \
     --boot-disk-size=50GB \
     --image-project=ml-images \
     --image-family=tf-2-3-1 \
     --maintenance-policy TERMINATE \
     --restart-on-failure \
     --scopes=cloud-platform

   gcloud compute tpus create \
     bigbird \
     --zone=europe-west4-a \
     --accelerator-type=v3-32 \
     --version=2.3.1
   ```

These references clearly indicate the usage of Google Cloud Platform and Google Cloud TPUs for training the google/bigbird-roberta-large model.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the google/bigbird-roberta-large model. To accurately report the carbon emissions, specific details about the computing resources used, training duration, and energy consumption would be necessary.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
BigBird is a sparse-attention based transformer which extends Transformer based models, such as BERT to much longer sequences. The attention mechanism in BigBird has three properties:
1. Queries attend to r random keys
2. Each query attends to w/2 tokens to the left and w/2 tokens to the right of its location
3. They contain g global tokens that can attend to the entire sequence

This reduces the quadratic dependency on sequence length to linear. BigBird-RoBERTa-Large likely follows a 24-layer, 1024-hidden, 16-heads architecture similar to RoBERTa-Large, but with the BigBird sparse attention mechanism.

Objective:
The objective of BigBird-RoBERTa-Large is likely Masked Language Modeling (MLM), similar to BERT and RoBERTa. [More Information Needed] on the exact pretraining details and corpus used.

BigBird drastically improves performance on various NLP tasks such as question answering and summarization as a consequence of its capability to handle longer context. It also enables novel applications like extracting contextual representations of long genomics sequences like DNA.

### Compute Infrastructure

Based on the provided references, here is the information about the compute infrastructure used for the google/bigbird-roberta-large model:

The model was trained on a reasonable 16GB memory/chip with batch size of 32-64, as mentioned in reference 1. The memory efficiency is due to the efficient blocking and sparsity structure of the sparse attention mechanism.

To set up the training infrastructure, a Google Cloud TPU instance named "bigbird" in the "europe-west4-a" zone was used, as shown in the example command in reference 4:

```
gcloud compute ssh --zone "europe-west4-a" "bigbird"
```

However, the specific TPU type (e.g., v2, v3) and the number of cores used are not mentioned in the provided references. [More Information Needed]

The exact architecture description and additional regularization techniques used for training BIGBIRD on specific datasets are provided in Appendix E.2, as mentioned in reference 8. [More Information Needed]

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

