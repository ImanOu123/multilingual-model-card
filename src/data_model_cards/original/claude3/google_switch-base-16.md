# Model Card for google/switch-base-16

The google/switch-base-16 model is a Switch Transformer model with 223M parameters, designed to be a FLOP-matched counterpart to the T5-Base model. It simplifies the Mixture of Experts architecture to be more scalable, sample-efficient, and effective across diverse natural language tasks in pre-training, fine-tuning and multi-task training settings.

## Model Details

### Model Description

Model Card for google/switch-base-16

Model Details:
- Architecture: Switch Transformer, a sparse-activation Mixture-of-Experts (MoE) model. It routes each token to the best determined top-k experts selected from a set of N experts. The number of parameters increases with the number of experts, while maintaining manageable memory and computation on each device.
- Parameters: [More Information Needed]

Training Procedure: 
- Pre-trained on a large corpus, then fine-tuned on smaller downstream tasks
- Auxiliary load balancing loss added to encourage balanced load across experts:
[More Information Needed]
- Initialized weights from truncated normal distribution with mean 0 and std dev s/n
- Used bfloat16 for weights, float32 for gradients and optimizer states to balance speed and stability
- Regularized with dropout during fine-tuning to prevent overfitting

Important Disclaimers:
- Sparsely-activated expert models like Switch Transformers may introduce training instabilities compared to standard dense Transformers
- Appropriate initialization is critical for successful training
- [More Information Needed]

For further questions or to update this model card, please contact the project organizer of the google/switch-base-16 model development team.

- **Developed by:** William Fedus; Noam Shazeer
- **Funded by:** Based on the provided references, there is no clear information about the specific people or organizations that funded the google/switch-base-16 model project. The references mention some individuals like Margaret Li, Hugo Larochelle, William Fedus, and Noam Shazeer who provided insights or worked on related Switch Transformer models, but it is not explicitly stated if they funded this particular project. To confidently answer this question, [More Information Needed] about the funding sources for the google/switch-base-16 model development.
- **Shared by:** Based on the provided references, there is not enough information to determine the specific contributors who made the model google/switch-base-16 available online as a GitHub repo. The references mention the Switch Transformer architecture and comparisons to T5 models, but do not specify the individual contributors for publishing the google/switch-base-16 model.

[More Information Needed]
- **Model type:** The google/switch-base-16 model is a sparse expert model based on the Switch Transformer architecture, which is a type of Transformer model that uses a mixture-of-experts approach for pre-training on a large natural language corpus, making it a language model trained using self-supervised learning on text data.
- **Language(s):** The google/switch-base-16 model is pre-trained on a multilingual variant of the Common Crawl data set (mC4) spanning 101 languages, and it demonstrates strong multi-task learning capabilities across all these languages.
- **License:** [More Information Needed]

The provided references do not contain information about the specific license being used for the google/switch-base-16 model. More details would be needed from the model developers or documentation to determine the applicable license.
- **Finetuned from model:** The model google/switch-base-16 is fine-tuned from the T5-Base model. This can be inferred from the following references:

1. "Baseline and Switch models used for fine-tuning. Our baselines are the highly-tuned 223M parameter T5-Base model and the 739M parameter T5-Large model (Raffel et al., 2019). For both versions, we design a FLOP-matched Switch Transformer, with many more parameters, which is summarized in Table 9."

9. "Negative log perplexity comparing Switch Transformers to T5 (Raffel et al., 2019) models using the same compute budget."

The link to the T5-Base model is not directly provided in the given references. [More Information Needed]
### Model Sources

- **Repository:** https://github.com/google-research/t5x
- **Paper:** https://arxiv.org/pdf/2101.03961.pdf
- **Demo:** [More Information Needed]

The provided references do not contain a link to a demo of the google/switch-base-16 model. More information would be needed to determine if a public demo of this specific model is available.
## Uses

### Direct Use

The google/switch-base-16 model can be used for inference without fine-tuning using the `t5x/infer.py` script from the T5X library. Here's a code snippet showing how to run inference:

```sh
INFER_OUTPUT_DIR="..."  # directory to write infer output
T5X_DIR="..."  # directory where the t5x is cloned, e.g., ${HOME}"/t5x".
TFDS_DATA_DIR="..."
CHECKPOINT_PATH="..."

python3 ${T5X_DIR}/t5x/infer.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_infer.gin" \
  --gin.CHECKPOINT_PATH=\"${CHECKPOINT_PATH}\" \
  --gin.INFER_OUTPUT_DIR=\"${INFER_OUTPUT_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```

This script uses a `seqio.Task` for inference, but does not use the target features other than logging them alongside the prediction in a JSON file.

[More Information Needed] on details about post-processing or plugging the model into a pipeline.

The pretrained model can also be exported as a TensorFlow Saved Model and deployed to the Vertex AI Prediction service using Optimized TensorFlow Runtime. However, the exported model won't work with the open-source TensorFlow Model Server.

### Downstream Use

The google/switch-base-16 model can be fine-tuned on a diverse set of downstream NLP tasks such as question answering, summarization, and benchmarks like GLUE and SuperGLUE. Significant improvements were observed when fine-tuning Switch models compared to T5 baselines, especially on SuperGLUE, Winogrande, closed book Trivia QA, and XSum.

To run inference with the fine-tuned model, you can use the `t5x/infer.py` script:

```sh
INFER_OUTPUT_DIR="..."  # directory to write infer output
T5X_DIR="..."  # directory where the t5x is cloned, e.g., ${HOME}"/t5x".
TFDS_DATA_DIR="..."
CHECKPOINT_PATH="..."
```

The pretrained model can also be exported as a TensorFlow Saved Model and deployed to the Vertex AI Prediction service using Optimized TensorFlow Runtime. However, the exported model won't work with the open-source TensorFlow Model Server.

[More Information Needed] on specific details of plugging the model into a larger ecosystem or app.

The model demonstrates strong multi-task learning capabilities, improving over the multilingual T5-base model across all 101 languages. Fine-tuned sparse models can also be distilled into smaller dense models like T5-Base while preserving some of the performance gains.

### Out-of-Scope Use

Based on the provided references about the Switch Transformer model google/switch-base-16, here are some potential ways the model may foreseeably be misused and what users ought not do:

The model should not be used to generate hateful, biased, or discriminatory language, as the training data (like the C4 corpus mentioned) may contain such content that gets learned by the model. Users should be cautious about potential biases.

Users should not rely on outputs from the model as factual or true, as language models can generate plausible sounding but incorrect information. The model is not intended to serve as a knowledge base.

[More Information Needed] on the specific downstream tasks and fine-tuning datasets used for this google/switch-base-16 model. Depending on the tasks and data, there may be additional misuse risks to call out.

The model should not be deployed for high-stakes real-world decision making, as the long-term societal impacts of powerful language models are not yet fully understood. More testing is needed before using it in sensitive applications.

In general, users should be aware this is a research model and not a production-ready system. Outputs may be unreliable or inconsistent. Users should critically examine model generations and not blindly trust them.

[More Information Needed] on the environmental impact and computational cost of this large Switch model. Users should consider the energy usage implications, especially for non-essential applications of the model.

### Bias, Risks, and Limitations

Based on the provided references about the Switch Transformer model google/switch-base-16, here are some potential issues and limitations:

Foreseeable harms and misunderstandings:
- The model was trained on a large web corpus (C4) which may contain biased, offensive, or factually incorrect content that could be reflected in the model's outputs. More analysis is needed on potential harms from the training data. [More Information Needed]

- As a large language model, it may be misused to generate misinformation, spam, or other harmful content at scale if released publicly without safeguards. [More Information Needed]

Technical limitations: 
- The references mention training instability challenges for the largest Switch-XXL models. Stability techniques that worked for Switch-Base were not sufficient at larger scales.

- Fine-tuning performance anomalies were observed where a larger 1.6T parameter Switch-C model performed worse than the smaller Switch-XXL model on some downstream tasks like SQuAD, despite similar pre-training perplexity. The cause is not fully understood.

- Very large sparse models like this still require supercomputer-scale computational resources to train and serve, limiting accessibility.

Sociotechnical limitations:
- The focus on ever-larger models requiring massive compute has environmental and economic costs in terms of energy usage and hardware. This may concentrate power in the hands of a few large actors who can afford to develop and deploy such models.

- There are open questions around how the model's sparsity and scale impact interpretability and our ability to understand its capabilities, biases and failure modes. More interdisciplinary research is needed here.

- Releasing such a capable model may have complex societal impacts in terms of jobs, education, information ecosystems, and more that are difficult to predict. Thoughtful policies and norms around responsible development and deployment are needed.

In summary, while Switch Transformers demonstrate impressive capabilities, more research is required to fully understand and mitigate potential harms and limitations, especially as they are scaled up further. Addressing these challenges likely requires ongoing collaboration between technical and sociotechnical experts.

### Recommendations

Based on the provided references about the Switch Transformer model, here are some recommendations regarding foreseeable issues with the google/switch-base-16 model:

1. Training stability: The references mention that training stability was a challenge for the largest Switch-XXL model, even though stability techniques were effective for smaller models like Switch-Base. More information is needed on the specific training stability of the Switch-Base-16 model.

2. Computational efficiency vs performance: The Switch Transformer aims to improve computational efficiency while scaling up parameters. However, the references note some anomalies, like the larger Switch-C model underperforming the smaller Switch-XXL on certain tasks, possibly due to differences in FLOPs per token. More analysis would be needed to understand the efficiency-performance tradeoffs specific to Switch-Base-16.

3. Applicability to new domains and modalities: The references suggest exploring the Switch Transformer architecture in new modalities and multi-modal networks as a promising direction. More information is needed on if and how Switch-Base-16 was tested on other domains beyond language.

4. Fine-tuning results: While the references show significant improvements on downstream tasks via fine-tuning in general, specific fine-tuning results for the Switch-Base-16 model are not provided. More information on its fine-tuning performance across various NLP tasks would help assess its practical utility.

5. Comparisons to dense models: As the references note, the success of sparse models like the Switch Transformer should be contextualized against the strong performance of large dense models. Direct comparisons of Switch-Base-16 to FLOP-matched dense models across key metrics would help users understand the relative advantages.

In summary, key open questions relate to the model's training stability, efficiency-performance tradeoffs, applicability to other domains, fine-tuning results, and comparative performance against dense models. Addressing these points in the model card would enable more informed decision making by potential users.

## Training Details

### Training Data

The training data of the model google/switch-base-16 is the multilingual variant of the Common Crawl data set (mC4) spanning 101 languages, which contains 107 tasks due to script variants within certain languages. [More Information Needed] for links to documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the google/switch-base-16 model:

Tokenization:
[More Information Needed]

Text Sequence Length:
[More Information Needed]

Masking:
During pre-training, 15% of tokens were dropped out and replaced with a single sentinel token, as determined to be optimal in Raffel et al. (2019). Specifically, from Reference 2:

"In our pre-training setting, as determined in Raffel et al. (2019) to be optimal, we drop out 15% of tokens and then replace the masked sequence with a single sentinel token."

Other Preprocessing Details:
The model was pre-trained on an improved version of the "Colossal Clean Crawled Corpus" (C4) which removed intra-example text duplication to increase efficacy as a pre-training task (Reference 2).

[More Information Needed] on other specific preprocessing steps like text normalization, special token handling, etc.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the google/switch-base-16 model:

- Trained using bfloat16 precision, with selective casting to float32 precision in localized parts of the model for stability (Reference 5)
- Weight matrices initialized from a truncated normal distribution with mean 0 and standard deviation s/(fan-in), where s is a scale hyperparameter reduced by a factor of 10 from the default Transformer initialization scale of 1.0 (References 2, 3, 6)
- Trained on the C4 corpus with over 180B target tokens (Reference 8)
- Trained with the same amount of computation (32 cores) and on the same hardware (TPUv3) as the baseline models (Reference 1)
- Pre-trained for more than 100k steps to achieve a perplexity threshold of -1.50 (Reference 1)
- [More Information Needed] on the exact number of pre-training steps
- [More Information Needed] on the batch size
- [More Information Needed] on the learning rate schedule
- Dropout used during fine-tuning to prevent overfitting (Reference 4)
- [More Information Needed] on the dropout rate

The model is a FLOP-matched Switch Transformer version of the T5-Base architecture, with many more parameters than the 223M parameter T5-Base while using the same FLOPS per token (References 7, 9).

#### Speeds, Sizes, Times

Here are the details about the google/switch-base-16 model based on the provided references:

Throughput and Training Speed:
- The Switch-Base 64 expert model achieves the same performance as the T5-Base model in 7.5x fewer training steps (60k steps vs 450k steps). This represents a significant speedup in training time. (Reference 2, 5)

Model Size:
- The Switch-Base model has many more parameters compared to the FLOP-matched T5-Base model (223M parameters). (Reference 9)

[More Information Needed] for the following:
- Exact number of parameters in Switch-Base-16 model
- Start and end training time 
- Checkpoint sizes

The references do not provide specific code blocks relevant to answering the question, so no code is included in the response.

In summary, the Switch-Base models demonstrate superior scaling properties and sample efficiency compared to the T5 dense models, enabling faster training with the same compute budget. However, more specific details like exact parameter counts, training times, and checkpoint sizes for the Switch-Base-16 variant are not available in the provided references.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the references provided, the Switch-Base model (which I assume is the google/switch-base-16 model) was fine-tuned and evaluated on the following benchmarks and datasets:

- GLUE benchmark, which includes tasks like SST-2, MRPC, STS-B, QQP, MNLI, etc.
- SuperGLUE benchmark 
- Winogrande dataset
- Trivia QA dataset (closed book)
- XSum dataset
- AI2 Reasoning Challenge (ARC) datasets

The model showed significant improvements over the T5-Base baseline on most tasks, with notable gains on SuperGLUE (+4.4 percentage points), Winogrande, Trivia QA, and XSum. The only tasks where gains were not observed were on the ARC datasets.

[More Information Needed] on the exact scores and metrics for each dataset/benchmark.

#### Factors

Based on the provided references about the Switch Transformer model google/switch-base-16, here are some key characteristics that may influence the model's behavior:

Domain and Context:
- The model was pre-trained on the large C4 corpus, which covers a wide range of domains and contexts. However, the references do not provide specific details on the breakdown of domains or potential domain gaps.
[More Information Needed] on specific domain coverage and any known domain gaps or biases.

- The model's performance was evaluated on downstream NLP tasks like SQuAD and SuperGLUE. Results show strong performance, but there may be task-specific nuances.
[More Information Needed] on comprehensive evaluation across a diverse set of tasks and domains to uncover potential disparities.

Population Subgroups:
- The references do not discuss disaggregated evaluation across different population subgroups (e.g. by demographics, dialects, etc.). 
[More Information Needed] on model performance broken down by relevant population subgroups to assess fairness and identify potential biases or disparities.

Other Factors:
- Model size and computational efficiency are key focuses. Larger sparse models with more experts tend to be more sample efficient and perform better than dense models.
- Fine-tuning dynamics with expert models can be complicated and warrant further study to fully realize sparse models' potential, especially for very large models.

In summary, while the Switch Transformer shows promising performance, more disaggregated evaluation is needed across domains, tasks, and population subgroups to comprehensively characterize model behavior and uncover any disparities or biases. The provided references lack this granular information.

#### Metrics

Based on the provided references about the Switch Transformer model google/switch-base-16, the following evaluation metrics will likely be used, considering the tradeoffs between different types of errors:

- Average scores across all subtasks for GLUE and SuperGLUE benchmarks (Reference 3)
- Rouge-2 metric for CNNDM and XSum tasks (Reference 3)
- Exact match percentage for SQuAD and closed book tasks (Web, Natural, and Trivia Questions) (Reference 3)
- Accuracy for ARC Easy, ARC Challenge, ANLI, and Winogrande tasks (Reference 3)

The model's performance will be compared against the T5-Base and T5-Large baselines, which are FLOP-matched (Reference 1, 2, 4). 

Additionally, the model's sample efficiency and scaling properties will be evaluated by comparing the negative log perplexity on the C4 corpus at different training steps (Reference 8, 11).

[More Information Needed] on specific tradeoffs between different errors for the google/switch-base-16 model.

### Results

Based on the provided references about the Switch Transformer model google/switch-base-16, here is the evaluation information I can provide:

Model Performance:
The Switch-Base model, which is FLOP-matched to the T5-Base model but has more parameters, achieves significant improvements on various downstream NLP tasks after fine-tuning. Notable gains are observed on SuperGLUE (+4.4 percentage points over T5-Base), Winogrande, closed book Trivia QA, and XSum. 

The Switch-Base model outperforms the T5-Base model on most tasks considered, demonstrating the effectiveness of the Switch Transformer architecture in translating pre-training quality improvements to downstream task performance via fine-tuning.

Model Efficiency:
[More Information Needed]

Limitations and Biases:
The only tasks where Switch-Base does not show gains compared to T5-Base are on the AI2 Reasoning Challenge (ARC) datasets. The T5-Base model outperforms Switch-Base on this challenge dataset.

Intended Use:
The Switch-Base model is intended to be fine-tuned on a variety of downstream NLP tasks, including reasoning and knowledge-heavy language tasks, to achieve improved performance compared to the T5-Base model baseline.

Training Details:
[More Information Needed]

#### Summary

Based on the provided references, here is a summary of the evaluation results for the google/switch-base-16 model:

The Switch-Base model, which is a FLOP-matched variant of the T5-Base model but with many more parameters, showed significant improvements over T5-Base on most downstream tasks evaluated, including reasoning and knowledge-heavy language tasks. 

Notable gains were observed on the SuperGLUE benchmark, where Switch-Base outperformed T5-Base by 4.4 percentage points. Large improvements were also seen on the Winogrande, closed-book Trivia QA, and XSum tasks.

The only tasks where Switch-Base did not show gains over T5-Base were on the AI2 Reasoning Challenge (ARC) datasets, where T5-Base performed better on the challenge set.

Despite T5-Large applying 3.5x more FLOPs per token, Switch-Base was still more sample efficient and yielded a 2.5x speedup in training time.

[More Information Needed] on the specific hyperparameters, model architecture details, and code used for the Switch-Base model.

In summary, the Switch-Base model demonstrated superior performance over the FLOP-matched T5-Base model on most natural language tasks evaluated, validating the effectiveness of the Switch Transformer architecture in enabling quality improvements via fine-tuning.

## Model Examination

Based on the provided references about the Switch Transformer model google/switch-base-16, here is what I can say about the model's explainability and interpretability:

The Switch Transformer introduces sparsely-activated expert layers to improve computational efficiency while scaling up model size. This sparsity comes from only activating a subset of the model weights for each input example. 

The references discuss adding these Switch layers in place of the dense feedforward layers and potentially the self-attention layers in the Transformer architecture. However, more details would be needed to fully explain the inner workings and interpretability of these sparse expert layers in the specific google/switch-base-16 model.

The performance of the Switch Transformer on downstream tasks provides some insight into its capabilities. The references note it performs well on knowledge-based tasks after fine-tuning, but may need further improvements for reasoning tasks. Examining its strengths and weaknesses on different task types can aid interpretability.

Overall though, the provided information focuses more on the high-level Switch Transformer architecture than an interpretability deep-dive into google/switch-base-16. [More Information Needed] for a truly comprehensive explainability analysis of this specific model and the role of its sparse expert layers. Code-level details, visualization techniques, or probing studies could help shed more light.

## Environmental Impact

- **Hardware Type:** Based on the provided references, the google/switch-base-16 model was likely trained on TPU architectures. Reference 8 states:

"Our work here focuses on TPU architectures, but these class of models may be similarly trained on GPU clusters."

While GPUs are mentioned as a possibility, the focus seems to be on TPU training for the Switch Transformer models discussed in the references.
- **Software Type:** Based on the provided references, the google/switch-base-16 model is likely trained using the T5X framework, which is implemented in JAX and Flax. T5X is described as "a new and improved implementation of the T5 codebase in JAX and Flax."

However, to definitively answer the question, [More Information Needed] on the specific software and libraries used to train the google/switch-base-16 model.
- **Hours used:** Based on the provided references, there is no specific information about the amount of time used to train the google/switch-base-16 model. The references mention training times and speedups for various Switch Transformer models compared to T5 models, but do not provide the exact training time for the specific google/switch-base-16 model.

[More Information Needed]
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider that the model google/switch-base-16 was trained on. More information would be needed to determine the cloud provider used for training this particular model.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the google/switch-base-16 model. To accurately report the carbon emissions, details such as the hardware used for training, energy consumption, duration of training, and the energy sources would be required.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The Switch Transformer is a sparsely-activated expert model that simplifies the Mixture of Experts (MoE) architecture. It activates a subset of the neural network weights for each incoming example to achieve greater computational efficiency compared to dense models.

The key details about the Switch Transformer architecture:
- Designed to maximize parameter count in a simple and computationally efficient way
- Uses a masked language modeling pre-training objective, where 15% of tokens are dropped out and replaced with a single sentinel token
- Combines expert, model and data parallelism for scalability
- Specific architectural hyper-parameters like d_model, d_ff, d_kv, number of heads and layers are [More Information Needed] for the google/switch-base-16 model

Model Objective:
The Switch Transformer aims to be a scalable and effective natural language learner. It is pre-trained on the "Colossal Clean Crawled Corpus" (C4) dataset using a masked language modeling objective to predict missing tokens. 

The model is designed to excel across diverse natural language tasks and different training regimes, including pre-training, fine-tuning and multi-task training. The goal is to enable training models with hundreds of billions to trillions of parameters that are vastly more sample efficient than equivalently-sized dense models.

### Compute Infrastructure

Based on the provided references, there is not enough information to comprehensively describe the compute infrastructure used for training the google/switch-base-16 model. The references discuss some general concepts related to distributed training and parallelism strategies, such as data parallelism, model parallelism, and expert parallelism. However, they do not provide specific details about the compute infrastructure used for the google/switch-base-16 model.

To provide a complete answer, more information is needed, such as:
- The number and type of GPUs or TPUs used
- The distributed training framework and configuration (e.g., Mesh TensorFlow)
- The specific parallelism strategy employed (data, model, expert, or a combination)
- The batch size and other training hyperparameters
- Any other relevant details about the compute infrastructure and training setup

[More Information Needed]

## Citation

```
@misc{william-switch,
    author = {William Fedus and
              Noam Shazeer},
    title  = {Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity},
    url    = {https://arxiv.org/pdf/2101.03961.pdf}
}
```

