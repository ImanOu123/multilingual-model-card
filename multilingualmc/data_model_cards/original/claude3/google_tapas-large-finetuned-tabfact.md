# Model Card for google/tapas-large-finetuned-tabfact

The model google/tapas-large-finetuned-tabfact is a TAPAS (Table Parsing via Pre-training) model based on BERT-large that has been fine-tuned on the TABFACT dataset for table entailment. It achieves state-of-the-art performance on the TABFACT benchmark, outperforming previous models by 9 points.

## Model Details

### Model Description

Here are the basic details about the google/tapas-large-finetuned-tabfact model:

Model Architecture:
- Uses a model architecture derived from BERT 
- Adds additional embeddings to encode the table structure, following the approach of Herzig et al. (2020)
- Uses token embeddings, position embeddings, segment embeddings, column embeddings, row embeddings, and rank embeddings for numeric columns
- Passes the input sequence through a transformer to obtain contextual representations for each token
- Models the probability of entailment using an MLP on the output of the [CLS] token

Training Procedures: 
- Starts with the public TAPAS checkpoint
- Trains an entailment model on counterfactual and synthetic data 
- Fine-tunes on the TABFACT end task
- Reports median accuracy over 9 runs (3 pre-training x 3 fine-tuning)
- Optimizes hyperparameters using Bayesian optimization, looking at validation accuracy after 8,000 steps
- Hyperparameter ranges: learning rate 10^-6 to 3x10^-4, dropout 0 to 0.2, warmup ratio 0 to 0.05

Parameters:
- Provides pre-trained models for different model sizes
- [More Information Needed] on the exact model sizes and number of parameters

Important Disclaimers:
- Some TABFACT input examples can be too long for BERT-based models
- Evaluates table pruning techniques as a pre-processing step to select relevant columns that respect input length limits

The model achieves over 9 points higher accuracy than the previous TABFACT state-of-the-art. Training on both counterfactual and synthetic data yields the best performance at 79.6% accuracy.

For more details or questions, please contact [More Information Needed].

- **Developed by:** Julian Martin Eisenschlos; Syrine Krichene; Thomas Müller; Greg Norman; Steve Elkington; Billy Mayfair
- **Funded by:** Based on the provided references, it appears that Google funded the development of the TAPAS model and its fine-tuning on the TabFact dataset. Specifically, reference 5 states "This is not an official Google product.", implying that Google was involved in the model's development even though it is not an official product.

The other references do not clearly mention the funding sources. They mainly acknowledge individuals who provided feedback (reference 1), list some of the authors (reference 2), describe updates to the code and model releases (reference 3), and explain the evaluation dataset (reference 4).

In summary, while it's not stated explicitly, the information suggests Google likely funded the google/tapas-large-finetuned-tabfact model, but [More Information Needed] to say conclusively. Reaching out to the authors listed in reference 2 may help clarify the funding sources if needed for the model card.
- **Shared by:** Based on the references provided, the contributors who made the model google/tapas-large-finetuned-tabfact available online as a GitHub repo are:

Julian Martin Eisenschlos, Syrine Krichene, Thomas Müller

The references mention that they released code and models to run TAPAS on TabFact for table entailment, as a companion for their EMNLP 2020 Findings paper "Understanding tables with intermediate pre-training". They also added a colab to try predictions on TabFact.

[More Information Needed] on the specific roles of Greg Norman, Steve Elkington, and Billy Mayfair in making this model available, as they are listed in the references but their contributions are not specified.
- **Model type:** The google/tapas-large-finetuned-tabfact model is a transformer-based table entailment model that uses intermediate pre-training on counterfactual and synthetic data, and is fine-tuned on the TABFACT dataset for the task of table entailment in the natural language processing modality.
- **Language(s):** The model google/tapas-large-finetuned-tabfact processes statements written in natural language by crowd workers that are either entailed or refuted by tables extracted from Wikipedia, as part of the TABFACT dataset.
- **License:** [More Information Needed]

The provided references do not contain any information about the license being used for the model google/tapas-large-finetuned-tabfact. To fill out this part of the model card, more information would need to be gathered about the licensing terms under which this model is being released.
- **Finetuned from model:** The model google/tapas-large-finetuned-tabfact is fine-tuned from the TAPAS (Table Parser) model, which is based on BERT. This is evident from the following references:

1. "We use a model architecture derived from BERT and add additional embeddings to encode the table structure, following the approach of Herzig et al. (2020) to encode the input."

7. "We adapt TAPAS (Herzig et al., 2020), a table-based BERT model, to recognize entailment."

8. "TAPAS is essentialy a BERT model and thus has the same requirements."

The link to the base TAPAS model is not directly provided in the given references. [More Information Needed]
### Model Sources

- **Repository:** https://github.com/google-research/tapas
- **Paper:** https://arxiv.org/pdf/2010.00571.pdf
- **Demo:** Here is the link to the demo of the model google/tapas-large-finetuned-tabfact:

http://tiny.cc/tapas-tabfact-colab

This colab was added to try predictions on the TabFact dataset, as mentioned in the first reference point.
## Uses

### Direct Use

The google/tapas-large-finetuned-tabfact model can be used for table entailment tasks without requiring additional fine-tuning, post-processing, or being part of a pipeline. As mentioned in the references:

1. The pre-trained models are already capable table entailment predictors, achieving an accuracy close to the fully fine-tuned TABLE-BERT model on the TABFACT evaluation set (63.8% vs 66.1%).

To use the model directly:

[More Information Needed] (No specific code snippet provided in the references)

The model takes as input a statement and a table, which are tokenized into word pieces and concatenated using standard [CLS] and [SEP] tokens. The table is flattened row by row without additional separators between cells or rows.

The model architecture is derived from BERT with additional embeddings to encode the table structure, following the approach of Herzig et al. (2020).

A colab notebook is available to try out predictions on TabFact using the released model: http://tiny.cc/tapas-tabfact-colab

For more details on the model's performance and efficiency:

- Pruning the input length to 256 tokens using heuristic exact match (HEM) results in a model that is twice as fast to train and apply, while only being 0.8 points behind the best full-length model. 
- Pre-training with shorter input sizes of 512 and 256 tokens gives similar accuracy, while 128 tokens results in about 1 point lower performance.

### Downstream Use

The google/tapas-large-finetuned-tabfact model can be used for table entailment tasks, to predict whether a given statement is entailed or refuted by the information in a table. 

Some key use cases and capabilities:

- It can reason over tabular data to determine if textual statements are supported or contradicted by the tables. This requires sophisticated reasoning and operations like filtering, combining, comparing and aggregating information across table rows and columns.

- The model sets a new state-of-the-art on the TABFACT dataset, outperforming previous models by 9 points. So it represents the current best performance on this task.

- Fine-tuning the model is data-efficient, achieving comparable accuracy to previous approaches with only 10% of the training data. This makes it more practical to adapt to new table entailment datasets and domains.

- Table entailment has many real-world applications wherever information is stored in structured formats. Some examples are health data, stock markets, sports statistics, etc. The model can enable surprising user queries to be answered from such data.

To try out the model's predictions on the TABFACT task, a Colab notebook is available:

[More Information Needed]

In terms of plugging into a larger system, the model could serve as a module that takes in a table and statement, and returns a prediction of entailed or refuted. This could be integrated with a user interface for querying data, or a pipeline for mining and fact-checking claims against a knowledge base of tables.

### Out-of-Scope Use

Based on the provided references about the google/tapas-large-finetuned-tabfact model, here are some potential ways the model could be misused and what users should not do:

The model should not be used to make definitive claims about information in tables without carefully fact-checking the model's outputs. While the model achieves state-of-the-art performance on the TABFACT entailment task, it is not 100% accurate and may make mistakes or produce false statements.

Users should not assume the model can reason about tables in the same nuanced way humans can. The references note that human accuracy remains substantially higher than the model. The model's reasoning abilities are limited.

The model should not be applied to tables significantly different from those it was trained and evaluated on (Wikipedia tables). Performance on other types of tables is unknown. 

Users should not employ the model for high-stakes decisions or applications where generating false or misleading statements from tabular data could cause harm.

[More Information Needed] on the specific pre-training data and if there are any problematic biases or privacy concerns with that data that could be reflected in the model's outputs.

In general, the model should be used cautiously and its limitations properly understood. More testing is needed on its robustness, failure modes, and potential to produce false or biased outputs when misused outside its intended domain. Users should critically examine model statements, not take them as ground truth.

### Bias, Risks, and Limitations

Based on the provided references about the google/tapas-large-finetuned-tabfact model, here are some known or foreseeable issues:

Limitations:
- The model may struggle with examples requiring multi-hop reasoning or complex numerical operations, as mentioned in reference 3. More complex pre-training tasks may be needed to fully handle these cases.
- Input examples that are too long for BERT-based models like TAPAS can be an issue. Table pruning techniques to select relevant columns are evaluated as a pre-processing step to respect input length limits (reference 9).

Potential Misunderstandings:
- For complex binary classification tasks like TABFACT, there is a question of whether models are just guessing the right answer. Analysis of model agreement across multiple runs provides some insight, but more analysis may be needed to fully understand model reasoning (reference 1).

[More Information Needed] on other specific foreseeable harms or sociotechnical limitations, as the provided references focus mainly on model performance and technical limitations. A broader impact analysis considering potential misuse and societal effects would help identify additional issues to document.

### Recommendations

Based on the provided references about the google/tapas-large-finetuned-tabfact model, here are some recommendations regarding foreseeable issues:

1. Model agreement and guessing: As mentioned in reference 1, there are questions around whether the model is just guessing the right answer for complex binary classification tasks like TABFACT. While counterfactual and synthetic pre-training improved agreement across model runs, only 55.5% agreed on the right answer in all cases. The model card should note this limitation and the potential for inconsistent or guessed outputs.

2. Struggles with complex reasoning: Reference 2 notes that even after pre-training improvements, the model still produces errors on examples requiring multi-hop reasoning or complex numerical operations. The model card should clearly state these limitations so users understand the types of queries the model may struggle with.

3. Speedup techniques impact accuracy: Reference 7 discusses using column selection methods to speed up training and inference by 2-4x. However, this comes with a moderate to large drop in accuracy. If these speedup techniques are used, the tradeoffs should be made clear in the model card.

4. [More Information Needed] on model biases and failure modes across different types of tables and queries. The references focus mainly on overall accuracy and some specific linguistic phenomena, but a more thorough error analysis across a range of tabular data would help identify additional issues to document.

5. [More Information Needed] on potential misuse cases and harms from a model designed to classify factual statements about tabular data. Depending on the intended use cases, there could be risks around using the model to spread misinformation or make decisions based on incorrect judgments.

In summary, the key issues to highlight are the model's inconsistency and potential to guess, struggles with complex reasoning, and accuracy tradeoffs when using speedup techniques. More analysis is needed on broader error patterns and misuse risks. Being transparent about these limitations in the model card is important for responsible open-sourcing.

## Training Details

### Training Data

The model google/tapas-large-finetuned-tabfact was trained on the TABFACT dataset, which consists of 118,000 statements that are either entailed or refuted by tables extracted from Wikipedia. The dataset was created by crowd workers who wrote simple and complex sentences referring to the information in the tables, and the model was also pre-trained on synthetic and counterfactual data to improve its performance.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the data of the model google/tapas-large-finetuned-tabfact:

Tokenization:
The statement and table in a pair are tokenized into word pieces and concatenated using the standard [CLS] and [SEP] tokens in between. The table is flattened row by row and no additional separator is added between the cells or rows. (Reference 9)

Table Resizing:
Some input examples from TABFACT can be too long for BERT-based models. Table pruning techniques are evaluated as a pre-processing step to select relevant columns that respect the input example length limits. (Reference 11)

The TAPAS model handles this by shrinking the text in cells. A token selection algorithm loops over the cells. For each cell it starts by selecting the first token, then the second and so on until the maximal length is reached. Unless stated otherwise, the same approach is used. (Reference 10)

[More Information Needed] on specific code for tokenization and table resizing.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters used for the google/tapas-large-finetuned-tabfact model:

- Started with the public TAPAS checkpoint
- Pre-trained an entailment model on the intermediate pre-training data 
- Fine-tuned on the TABFACT end task
- Used the same hyper-parameters for pre-training and fine-tuning:
  - Input length: 256 for pre-training, 512 for fine-tuning
  - Training steps: 80,000
  - Learning rate: 2e-5 
  - Warm-up ratio: 0.05
  - Disabled attention dropout in BERT
  - Hidden dropout probability: 0.07
  - Adam optimizer with weight decay, same configuration as BERT
- [More Information Needed] on specific hardware used for training
- Hyper-parameters were optimized using black box Bayesian optimization similar to Google Vizier, looking at validation accuracy after 8,000 steps
- Reported median accuracy over 3 pre-training and 3 fine-tuning runs (9 runs total)

#### Speeds, Sizes, Times

Here are the details about the google/tapas-large-finetuned-tabfact model based on the provided references:

Model size:
- The model has 340M parameters, the same as BERT-Large models. [5]

Training:
- The model was first initialized with the public TAPAS checkpoint, then trained on an entailment task using the counterfactual and synthetic data from Section 3, and finally fine-tuned on TABFACT. [1,7]
- Training was done on Cloud TPUs V3. [3] 
- [More Information Needed] on exact training times and throughput.

Results:
- This model set a new state-of-the-art on TABFACT, outperforming previous models by 9 points. [8]
- It achieved 78.6% median test accuracy over 9 runs. [7]

Released artifacts:
- Code and model were released to run TAPAS on TabFact [6]
- A Colab notebook was provided to try predictions on TabFact [6]

[More Information Needed] on specific checkpoint sizes, start/end training times, and more detailed throughput metrics. The references focus more on the model architecture, training procedure and final accuracy results.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the google/tapas-large-finetuned-tabfact model evaluates on the following benchmarks and datasets:

1. TABFACT dataset: A dataset of statements that are either entailed or refuted by tables from Wikipedia. It consists of 118,000 examples, with crowd workers creating positive (entailed) and negative (refuted) sentence pairs. (References 1, 3, 6, 7, 9)

2. SQA (Sequential Question Answering) dataset: Used for pre-training and testing if the pre-training is useful for related tasks. SQA is a question answering dataset created by splitting a subset of WikiTableQuestions into multiple referential questions. It consists of 6,066 sequences with an average of 2.9 questions per sequence. (Reference 4)

The model's performance on TABFACT is reported to outperform the previous state-of-the-art by more than 9 points (for the Large version), achieving 78.5% accuracy. (References 2, 5)

#### Factors

Based on the provided references about the google/tapas-large-finetuned-tabfact model, here are some key characteristics that may influence the model's behavior:

Domain and Context:
- The model is trained and evaluated on the TABFACT dataset, which consists of statements that are either entailed or refuted by tables extracted from Wikipedia. So the model's performance is tied to this specific domain of Wikipedia tables and crowd-sourced statements about them.

Population Subgroups:
- The TABFACT dataset contains two batches of statements - simple sentences referring to a single table row, and complex sentences using information from multiple rows. Model performance may vary between these two subgroups.
- Specific types of reasoning required, such as aggregations, superlatives, comparatives, negations, and multi-operation reasoning, can impact model accuracy. Disaggregated evaluation across these groups would help uncover performance disparities.
- [More Information Needed] on other relevant population subgroups the model may perform differently on, such as based on table size, topic domain, numerical vs textual data, etc.

Other Factors:
- Table pruning techniques used as pre-processing to handle input length limits may affect model behavior on larger tables.
- The pre-training tasks of generating synthetic and counterfactual statements have a significant impact on model accuracy. More analysis is needed on what specific capabilities these impart.
- Model size (BERT-base vs BERT-large) influences performance on TABFACT.

In summary, key factors include the Wikipedia domain, simple vs complex query types, specific reasoning operations required, table pre-processing, pre-training tasks, and model size. More disaggregated evaluation across relevant subgroups would help further characterize model behavior and uncover potential performance disparities. Additional information on the makeup of the TABFACT dataset and further testing on other datasets would also be beneficial.

#### Metrics

Based on the references provided, the main evaluation metric used for the google/tapas-large-finetuned-tabfact model is accuracy on the TABFACT dataset. Specifically:

- The model outperforms the previous state-of-the-art on TABFACT by more than 9 points (Reference 1). 
- Median accuracy values over multiple runs are reported, with the error margin estimated as half the interquartile range between the 25th and 75th percentiles (Reference 2).
- Switching from BERT-Base to BERT-Large improves accuracy by 2.5 points, with improvements consistent across all test sets (Reference 3).

The references do not provide details on tradeoffs between different error types. To fully answer that part of the question, [More Information Needed] on false positives vs false negatives and the relative costs of each error type for the intended use case of the model.

In summary, accuracy on the TABFACT benchmark appears to be the primary evaluation metric, but more details would be needed to analyze tradeoffs between different kinds of errors the model may make.

### Results

Based on the references provided, here are the evaluation results for the google/tapas-large-finetuned-tabfact model:

Factors:
- Model size: LARGE
- Reset option: Two variants - noreset and reset

Metrics:
- Dev Accuracy: 
  - noreset: 0.8101
  - reset: 0.8159

The model outperforms the previous state-of-the-art on TABFACT by more than 9 points. Switching from BERT-Base to LARGE improves the accuracy by another 2.5 points. The improvements are consistent across all test sets.

[More Information Needed] on the exact training data and hyperparameters used. The references mention using counterfactual data and synthetic data, but more specifics would be needed to fully document the model.

#### Summary

Based on the references provided, here is a summary of the evaluation results for the model google/tapas-large-finetuned-tabfact:

- The TAPAS model outperforms the previous state-of-the-art on TABFACT by more than 9 points when using the Large size (Reference 1). 

- Using both counterfactual data and synthetic data for pre-training achieves the best test accuracy of 77.9% on TABFACT (Reference 1).

- Reducing the input length to 256 while using heuristic exact match (HEM) pruning gives similar accuracy as full length, while being twice as fast to train and apply (References 2 and 3).

- The best pre-trained TAPAS model before fine-tuning on TABFACT is only 2 points behind the fully trained TABLE-BERT (63.8% vs 66.1%) (Reference 4).

- Using counterfactual and synthetic pre-training data increases the percentage of examples where all 9 model runs agree on the right answer from 24.2% to 55.5% (Reference 5).

- The intermediate pre-training data improves accuracy the most on superlatives (39% error reduction) and negations (31%) (Reference 8).

- The Large TAPAS model fine-tuned on TABFACT achieves 81.59% dev accuracy when using the "reset" option (Reference 9).

[More Information Needed] on specific code blocks or hyperparameters used.

In summary, the google/tapas-large-finetuned-tabfact model achieves strong results on the TABFACT dataset, outperforming previous SOTA, by leveraging intermediate pre-training on counterfactual and synthetic data. Pruning techniques can be used to improve efficiency while maintaining accuracy.

## Model Examination

Based on the provided references about the google/tapas-large-finetuned-tabfact model, here is what I can say about the model's explainability and interpretability:

The references provide some insights into understanding the model's behavior and performance improvements:

- The model agreement analysis (Reference 2) shows that using counterfactual and synthetic pre-training data increases the percentage of examples where multiple model runs agree on the correct answer, compared to just using MASK-LM pre-training. This suggests the intermediate pre-training helps the model be more consistent and confident in its predictions.

- The salient group analysis (References 3-5) looked at model performance on specific linguistic/reasoning phenomena like aggregations, superlatives, comparatives and negations. The intermediate pre-training data especially helped improve accuracy on superlatives and negations. Examining model outputs on these groups provides some interpretability of model strengths and weaknesses.

- Reference 4 shows some specific examples where the model with intermediate pre-training is correct while the MASK-LM only model is not, as well as examples where the full model still makes errors, often on examples requiring multi-hop reasoning or complex numerical operations. Analyzing these individual examples provides insight into model capabilities and limitations.

However, to make stronger claims about the model's explainability and interpretability, [More Information Needed] on topics like:

- Visualizing model attention to see what table cells and question words the model focuses on for its predictions
- Generating counterfactual examples (with small changes to the table or question) to test model robustness and error boundaries 
- Extracting learned representations to examine what features and concepts the model has captured
- Applying attribution methods to determine what input features were important for individual predictions

So in summary, the intermediate pre-training analysis provides some useful insights, but additional explainability techniques could be applied to better understand the inner workings and decision making of the google/tapas-large-finetuned-tabfact model.

## Environmental Impact

- **Hardware Type:** Based on the provided references, the model google/tapas-large-finetuned-tabfact was trained on Cloud TPUs V3.

Specifically, reference 2 states:
"We train all our models on Cloud TPUs V3."
- **Software Type:** Based on the references provided, the model google/tapas-large-finetuned-tabfact is trained using TPUs (Tensor Processing Units). Specifically, reference 6 states:

"This means that training the large model with 512 sequence length will require a TPU."

So the software type used for training this TAPAS model is TPU hardware from Google.
- **Hours used:** Based on the provided references, the exact training time for the google/tapas-large-finetuned-tabfact model is not explicitly stated. However, we can infer some relevant information:

- The training time depends on the sequence length used. For a BERT-Base model, it takes around 78 minutes using 128 tokens and scales almost linearly up to 512 tokens (Reference 2).
- The TAPAS model is essentially a BERT model (Reference 7).
- Pre-training with shorter input sizes can improve training time. Sequence lengths of 512 and 256 give similar accuracy, while results for 128 are about 1 point lower (Reference 8).

Given that TAPAS is a large model and was likely trained with a sequence length between 256 and 512 tokens, the training time would be longer than 78 minutes. However, without more specific information about the exact sequence length and hardware used for this particular model, we cannot provide a precise training time.

[More Information Needed] on the specific training setup and sequence length used for the google/tapas-large-finetuned-tabfact model to determine the exact training time.
- **Cloud Provider:** Based on the provided references, the model google/tapas-large-finetuned-tabfact was trained on TPUs (Tensor Processing Units). Specifically, reference 2 states:

"This means that training the large model with 512 sequence length will require a TPU."

However, the specific cloud provider is not explicitly mentioned. Google Cloud Platform offers TPUs, but it's not confirmed in the given information that Google Cloud was used for this particular model.

[More Information Needed] on the exact cloud provider used for training the model.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the google/tapas-large-finetuned-tabfact model. To answer this question, more specific details would be needed, such as:

- The hardware used for training (CPUs, GPUs, TPUs, etc.)
- The duration of the training process 
- The energy consumption of the hardware during training
- The carbon intensity of the electricity used

Without these additional details, it is not possible to accurately estimate the carbon emissions from the model training process based solely on the given references.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
- The model architecture is derived from BERT, with additional embeddings to encode the table structure (Reference 1).
- Six types of learnable input embeddings are added together: token embeddings, position embeddings, segment embeddings, column embeddings, row embeddings, and rank embeddings for numeric columns (Reference 2).
- The bi-directional self-attention mechanism in transformers is used, which is unaware of order and motivates the usage of positional and segment embeddings for text, and column and row embeddings when processing tables (Reference 3).
- The sequence of embeddings is passed through a transformer to obtain a contextual representation for every token (Reference 4).
- The probability of entailment is modeled with a single hidden layer neural network computed from the output of the [CLS] token (Reference 4).

Model Objective:
- The model is pre-trained and fine-tuned on the TABFACT dataset, which consists of statements that are either entailed or refuted by tables from Wikipedia (Reference 6).
- The model directly predicts the entailment decision by encoding the tables with BERT-based models (Reference 7).
- Two intermediate pre-training tasks are introduced: one based on synthetic statements and the other on counterfactual statements (Reference 8).
- The synthetic statements are generated by sampling from a set of logical expressions that filter, combine, and compare the information in the table (Reference 8).
- [More Information Needed] on the specific details of the counterfactual statements pre-training task.
- The model outperforms the previous state-of-the-art on TABFACT by more than 6 points (Base) or more than 9 points (Large) (Reference 10).

### Compute Infrastructure

Based on the provided references, here is the compute infrastructure information for the model google/tapas-large-finetuned-tabfact:

- All models were trained on Cloud TPUs V3. (Reference 3)
- Training time depends on the sequence length used. For a BERT-Base model, it takes around 78 minutes using 128 tokens and scales almost linearly up to 512. (Reference 2)
- TAPAS, being essentially a BERT model, has the same requirements. Training the large model with 512 sequence length will require a TPU. (Reference 7)
- The `max_seq_length` option can be used to create shorter sequences to make the model trainable on GPUs, but this will reduce accuracy. (Reference 7)
- Reducing the batch size (`train_batch_size`) is another option, but it will likely also affect accuracy. (Reference 7)

[More Information Needed] on the specific hardware configuration and training time for the google/tapas-large-finetuned-tabfact model.

## Citation

```
@misc{julian-understanding,
    author = {Julian Martin Eisenschlos and
              Syrine Krichene and
              Thomas Müller and
              Greg Norman and
              Steve Elkington and
              Billy Mayfair},
    title  = {Understanding tables with intermediate pre-training},
    url    = {https://arxiv.org/pdf/2010.00571.pdf}
}
```

