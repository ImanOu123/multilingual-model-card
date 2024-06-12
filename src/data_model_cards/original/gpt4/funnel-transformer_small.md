# Model Card for funnel-transformer/small

The Funnel-Transformer/small is an efficient self-attention model that reduces computational cost by compressing the sequence of hidden states and reallocates saved FLOPs to enhance model capacity, with an optional decoder for tasks requiring token-level predictions. It is designed for scalability and efficiency in language pretraining, capable of handling both tasks that need token-level representation and those that require a single-vector representation of the sequence.

## Model Details

### Model Description

Model Card for Funnel-Transformer/Small

## Model Architecture
The Funnel-Transformer (F-TFM) is a novel self-attention model designed to reduce computational cost by compressing the sequence of hidden states to a shorter one. The architecture consists of an encoder with several blocks of consecutive Transformer layers. Each block maintains the same sequence length of hidden states, but the sequence is pooled and reduced when transitioning to a higher-level block. The model also includes a decoder that is used for token-level prediction tasks, such as standard pretraining or sequence labeling. For sequence-level tasks like classification, the decoder is not used during finetuning. The specific architecture details for the "small" variant are not provided in the references, but it likely follows a smaller scale compared to larger models, with fewer layers and/or hidden units.

## Training Procedures
The F-TFM model is pretrained and then finetuned for downstream tasks. Pretraining is conducted on datasets such as Wikipedia and Book Corpus, with two common settings mentioned: Base scale (1M steps, batch size 256) and Large scale (500K steps, batch size 8K, additional datasets like ClueWeb, Gigaword, and Common Crawl). The model is then finetuned on sequence-level tasks, with a focus on tasks that require a single vector representation of the input sequence, such as the GLUE benchmark.

## Parameters
The references mention a model with layers of hidden size 768 (L12H768) and a Funnel-Transformer variant with 3 blocks of 6 layers each (B6-6-6H768). However, the exact parameters for the "small" variant are not specified. [More Information Needed]

## Important Disclaimers
The Funnel-Transformer is designed to be efficient by reducing the sequence length during encoding, which allows for a deeper or wider model given the same computational budget. It is important to note that the decoder is only used for tasks that require token-level predictions. For tasks requiring only a single vector representation, such as classification, the decoder is discarded after pretraining. The model's performance improvements over standard Transformers are empirically demonstrated, but the exact performance metrics for the "small" variant are not provided. [More Information Needed]

For further inquiries or updates to the model card, please contact the project organizer.

- **Developed by:** Zihang Dai; Guokun Lai; Yiming Yang; Quoc V Le
- **Funded by:** [More Information Needed]
- **Shared by:** The contributors who made the model funnel-transformer/small available online as a GitHub repo are Zihang Dai, Guokun Lai, Yiming Yang, and Quoc V. Le. These individuals are listed as authors of the paper detailing the technical aspects of the Funnel-Transformer and are likely responsible for the development and publication of the model code.
- **Model type:** The Funnel-Transformer/small is a self-attention-based neural network model trained with a large-scale pretraining method on diverse datasets, designed for sequence-level tasks and token-level predictions, falling under the category of supervised learning, and it operates on textual modality.
- **Language(s):** The model funnel-transformer/small processes text in English, as it adapts the "uncased" word piece model originally used by BERT, which is designed for English language tasks.
- **License:** [More Information Needed]
- **Finetuned from model:** [More Information Needed]

The provided references do not contain explicit information about a specific "funnel-transformer/small" model being fine-tuned from another base model. The references discuss the Funnel-Transformer architecture in general, its components, and some implementation details, but they do not mention a "small" variant or its base model. To provide the name and link to the base model from which "funnel-transformer/small" was fine-tuned, additional information about this specific model variant would be required.
### Model Sources

- **Repository:** https://github.com/laiguokun/Funnel-Transformer
- **Paper:** https://arxiv.org/pdf/2006.03236.pdf
- **Demo:** [More Information Needed]
## Uses

### Direct Use

The Funnel-Transformer model, specifically the `funnel-transformer/small` variant, is designed to be a high-capacity model with reduced computational cost due to its self-attention mechanism that compresses the sequence of hidden states. However, the question asks about using the model without fine-tuning, post-processing, or plugging it into a pipeline.

Based on the provided references, the model is intended to be fine-tuned for specific tasks such as the GLUE benchmark, text classification, SQuAD, and RACE, especially on GPUs or TPUs. The references do not provide explicit information on using the model without any fine-tuning or further processing.

Given the nature of deep learning models and the context provided, it is generally expected that a model like `funnel-transformer/small` would need to be fine-tuned on a specific task to be effectively used. Without fine-tuning, the model would only be able to provide generic representations that might not be optimized for any particular task.

Since there is no direct reference to using the model without fine-tuning or post-processing, and no code snippet is provided for such usage, the answer to the question is:

[More Information Needed]

### Downstream Use

The Funnel-Transformer (F-TFM) model, specifically the `funnel-transformer/small` variant, is designed to be efficient in both pretraining and finetuning stages, making it suitable for a variety of downstream Natural Language Processing (NLP) tasks. When fine-tuned, the model can be adapted to specific tasks such as text classification, sentiment analysis, and question answering.

For tasks that require a single vector representation of the input sequence, such as classification tasks, the decoder part of the Funnel-Transformer is discarded after pretraining, and only the encoder is fine-tuned. This is because the encoder of the Funnel-Transformer has a compression functionality that reduces the sequence length while retaining the essential information, which is sufficient for such tasks.

On the other hand, for tasks that require token-level predictions, such as question answering where the model needs to select a token span from the context as the answer, the full encoder-decoder structure of the Funnel-Transformer is utilized.

When integrating `funnel-transformer/small` into a larger ecosystem or app, the model can serve as a powerful feature extractor or as a standalone component for NLP-related functionalities. For example, in a sentiment analysis app, the model can be fine-tuned on a sentiment dataset and then used to predict the sentiment of user input text.

Here's a conceptual example of how you might fine-tune the `funnel-transformer/small` model on a text classification task using the Huggingface Transformers library. Note that actual code would require a dataset, a tokenizer, and additional setup, which are not provided in the references:

```python
from transformers import FunnelTokenizer, FunnelForSequenceClassification, Trainer, TrainingArguments

# Load the tokenizer and model
tokenizer = FunnelTokenizer.from_pretrained('funnel-transformer/small')
model = FunnelForSequenceClassification.from_pretrained('funnel-transformer/small', num_labels=num_labels)

# Tokenize the input dataset
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

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
    train_dataset=train_dataset,  # provide your training dataset
    eval_dataset=eval_dataset,    # provide your evaluation dataset
)

# Train the model
trainer.train()
```

Please note that the above code is a simplified example and does not include the actual dataset or the specifics of the training and evaluation datasets. For a real-world application, you would need to prepare the data accordingly and potentially perform additional steps such as hyperparameter tuning and evaluation.

### Out-of-Scope Use

As a sociotechnic on the deep learning model development team, it is my responsibility to consider the potential misuses of the Funnel-Transformer/small model and provide guidance to users on what they should avoid doing with the model. Here are some foreseeable misuses and recommendations:

1. **Biased or Unethical Applications**: Given that the Funnel-Transformer has the capability to process language efficiently, it could be misused to propagate biased, discriminatory, or unethical content at scale. Users should not use the model to generate or amplify content that promotes hate speech, misinformation, or any form of discrimination.

2. **Privacy Violations**: Users should not use the model to process sensitive or private data without proper consent and data protection measures in place. The model's ability to process sequential data could be misused to analyze personal text data, such as private messages or confidential documents, in a way that violates privacy rights.

3. **Deepfakes and Misrepresentation**: The model's language processing capabilities could be misused to create convincing fake content, such as deepfake texts that could mislead readers. Users should not use the model to impersonate individuals or entities, or to create content that is intended to deceive.

4. **Security Risks**: The model could be used in malicious ways, such as generating phishing emails or crafting sophisticated social engineering attacks. Users should avoid using the model for any form of cybercrime or to compromise the security of individuals or organizations.

5. **Intellectual Property Infringement**: Users should not use the model to generate content that infringes on the intellectual property rights of others, such as automatically generating text that copies the style or substance of copyrighted works without permission.

6. **Over-reliance on Automation**: While the Funnel-Transformer is designed to be efficient, users should not over-rely on the model for critical decision-making without human oversight. Automated language processing should be used as a tool to assist humans, not replace them, especially in contexts where nuanced understanding and ethical judgments are required.

7. **Lack of Transparency**: Users should not use the model in applications where the lack of transparency could lead to harm. For example, using the model in legal or healthcare settings without clear explanations of how decisions are made could have serious consequences.

In conclusion, users of the Funnel-Transformer/small model should ensure that their applications are ethical, respect privacy, do not spread misinformation, do not pose security risks, respect intellectual property rights, maintain human oversight, and are transparent in their use of AI technology. It is important to consider the broader societal impacts and to use the model responsibly.

### Bias, Risks, and Limitations

The Funnel-Transformer model, particularly the `funnel-transformer/small` variant, introduces several advancements in the field of natural language processing by compressing the sequence of hidden states and re-investing the saved FLOPs into constructing a model with higher capacity. However, there are known and foreseeable issues that stem from this model, which can be categorized into technical and sociotechnical limitations:

1. **Compression Scheme Optimization**: The current model utilizes a compression scheme to reduce the sequence length and save computation. While this is beneficial, there is room for improvement in the compression mechanism. Future research is needed to optimize this aspect, which could lead to better performance or efficiency. [Reference 1]

2. **Block Layout Design**: The model uses a 3-block design by default, and while experiments have been conducted with 2-block and 4-block designs, there may be an optimal block layout that has not yet been discovered. This could affect the model's performance on various tasks. [Reference 2]

3. **Positional Encoding**: The model compares relative attention parameterization with learned absolute position embedding. The choice between these two methods of positional encoding could have implications for the model's ability to handle sequence order information, which is crucial for many NLP tasks. [Reference 2]

4. **Pretraining and Finetuning**: The model has been pretrained and finetuned on specific datasets and tasks. There may be limitations in generalizing to other datasets or tasks that were not part of the pretraining or finetuning process. This could lead to performance degradation or biases when applied to different contexts. [Reference 3, 4]

5. **Sequence-Level Task Focus**: The Funnel-Transformer is designed with a focus on sequence-level tasks that require a single vectorial representation of the input sequence. This design choice may limit its effectiveness on token-level tasks or tasks that require fine-grained token-level representations. [Reference 4]

6. **Model Extensions and Complexity**: While there are discussions on potential model extensions, implementing these could introduce additional complexity. This might make the model more difficult to understand, adapt, or maintain. [Reference 7]

7. **Sociotechnical Considerations**: As a sociotechnic, it is important to consider the broader implications of deploying this model. For instance, if the model is trained on biased data, it may perpetuate or amplify these biases. Additionally, the model's complexity and the need for large-scale pretraining may limit its accessibility to researchers with fewer computational resources, potentially exacerbating inequalities in the field.

8. **Misunderstandings and Misuse**: Given the complexity of the model, there is a risk of misuse or misunderstanding by practitioners who may not fully grasp the intricacies of the compression and decompression mechanisms. This could lead to incorrect applications or interpretations of the model's outputs.

9. **Combination with Other Techniques**: The potential for combining the Funnel-Transformer with other model compression techniques like knowledge distillation and quantization is mentioned. While this could enhance practical impact, it also introduces additional layers of complexity that could result in unforeseen issues or trade-offs in model performance. [Reference 5]

In conclusion, while the `funnel-transformer/small` model presents significant advancements, it also comes with a set of technical and sociotechnical challenges that need to be addressed. Continuous research and careful consideration of the model's impact on society are required to mitigate potential harms and limitations.

### Recommendations

Based on the provided references, here are some recommendations with respect to the foreseeable issues about the model funnel-transformer/small:

1. **Model Extensions and Generalization**: As the Funnel-Transformer can be applied to any tasks dealing with sequential data, it is recommended to explore its application beyond NLP tasks, such as in time series and video stream analysis. This could help in identifying any limitations or issues when the model is applied to different types of data.

2. **Model Compression Techniques**: Combining the Funnel-Transformer with model compression techniques like knowledge distillation and quantization is an important direction. This could enhance the practical impact of the model, especially for deployment in resource-constrained environments. However, it is necessary to monitor the impact of these techniques on the model's performance and fairness.

3. **Optimization of Block Layout Design**: Future research should focus on optimizing the block layout design of the Funnel-Transformer. This could potentially improve the compression scheme and the model's efficiency. It is important to consider the trade-offs between model complexity and performance during this optimization.

4. **Complex Attention Mechanisms**: For tasks that require complex control of the attention mechanism, it is recommended to investigate how additional input signals can be effectively incorporated into the Funnel-Transformer. This could help in maintaining or improving the model's performance on tasks that are sensitive to the order of input data or require additional context.

5. **Token-Level Predictions**: Since the Funnel-Transformer is capable of performing token-level predictions, it is important to ensure that the recovery of deep representations for each token is accurate and does not introduce biases or errors, especially when dealing with diverse datasets.

6. **Ethical and Societal Considerations**: As a sociotechnic, it is crucial to consider the ethical and societal implications of deploying the Funnel-Transformer. This includes evaluating the model for biases, ensuring that it does not perpetuate or amplify harmful stereotypes, and being transparent about the model's capabilities and limitations.

7. **Performance Benchmarking**: Continuous benchmarking of the Funnel-Transformer against standard models is recommended to ensure that it maintains a competitive edge in terms of performance and efficiency. This should include running time comparisons on different hardware platforms.

8. **Documentation and Transparency**: A comprehensive model card should be created, documenting the model's intended use cases, performance benchmarks, limitations, and ethical considerations. This transparency will help users understand the model's capabilities and make informed decisions about its deployment.

In conclusion, while the Funnel-Transformer shows promise in terms of computational efficiency and performance, it is important to continuously evaluate and extend the model in various directions, keeping in mind the ethical and societal impacts of its deployment.

## Training Details

### Training Data

The training data for the model funnel-transformer/small consists of Wikipedia and Book Corpus, following the base scale pretraining setting used by the original BERT, with a pretraining duration of 1 million steps and a batch size of 256. Additional preprocessing included the removal of certain HTTP symbols for text classification tasks. [More Information Needed] on data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

For the preprocessing of the data used with the funnel-transformer/small model, we adapted the "uncased" word piece tokenizer originally used by BERT, which has a vocabulary size of about 30,000 tokens. This tokenizer converts text to lowercase and includes default preprocessing steps inherent to the word piece tokenization process. Additionally, we performed some extra preprocessing to remove certain HTTP symbols (e.g., `<b>`) for the text classification tasks.

During pretraining, we used different mask sampling strategies for the Masked Language Modeling (MLM) and ELECTRA-style training. For MLM, we followed previous work and used word span sampling, which allows for up to 5 complete words to be masked and predicted. However, for ELECTRA training under the base-scale setting, we observed a drop in performance when using word span sampling, so we opted for single-token sampling instead. This issue did not occur under the large-scale setting, where word span sampling worked fine.

For finetuning experiments, we inherited the hyperparameters used by XLNet, which suggests that no significant changes to the preprocessing steps were made for this stage.

It is important to note that the preprocessing steps mentioned are specific to text data, and the model card should reflect any additional preprocessing steps if the model is adapted for different modalities. If there were any other modality-specific preprocessing steps taken, such as resizing or rewriting, this information is not provided in the references, and therefore, [More Information Needed].

#### Training Hyperparameters

The training hyperparameters for the model `funnel-transformer/small` are as follows:

- **Layer-wise Decay**: We implemented a layer-wise decay strategy for learning rates, as described in reference 1. If the top layer (24-th layer) has a learning rate of `l`, and the decay rate is `α`, then the learning rate for layer `m` is calculated as `lα^(24−m)`. This approach helps in fine-tuning the model on datasets like RACE and SQuAD.

- **Learning Rate Search**: For GLUE and text classification datasets, we fixed most hyper-parameters as shown in a referenced table (Table 8, which is not provided here). We then searched for the optimal learning rate from the set `[1e-5, 2e-5, 3e-5]`, selecting the best one based on performance on the validation set, as mentioned in reference 2.

- **Runs and Seeds**: The GLUE benchmark results are based on the median of 5 runs with different random seeds for the base setting. For large settings, it's the median of 15 runs. For text classification tasks, the base-scale results are the median of 5 runs with different random seeds, and for large-scale experiments, the best performance among these runs is reported, in line with previous work (reference 3).

- **Attention Heads**: The number of attention heads is determined by dividing the hidden size by 64, as stated in reference 6.

- **Mask Sampling Strategy**: For Masked Language Model (MLM) training, we followed previous work and used complete word span sampling, allowing up to 5 complete words. However, for ELECTRA training, we observed a drop in performance with word span sampling under the base-scale setting, so we opted for single-token sampling instead (reference 6).

Unfortunately, the exact values for some hyperparameters such as the learning rate `l`, the decay rate `α`, and the hidden size are not provided in the references. Therefore, for those specific values, [More Information Needed].

#### Speeds, Sizes, Times

The model card description for the `funnel-transformer/small` could include the following details based on the provided references:

```
# Model Card for Funnel-Transformer/Small

## Model Description

The Funnel-Transformer (F-TFM) small model is a variant of the standard Transformer architecture that aims to provide similar or improved performance with fewer computational resources. The small model configuration is designed with fewer layers and hidden units compared to larger variants, specifically with a layout of L6H768, indicating 6 layers and a hidden size of 768.

## Technical Details

- **Throughput**: [More Information Needed]
- **Start or End Time**: [More Information Needed]
- **Checkpoint Sizes**: The checkpoint sizes for similar models with different block layouts are available, but the exact size for the `funnel-transformer/small` is not provided in the references. [More Information Needed]

## Performance

- The F-TFM small model, with its reduced sequence length and increased depth, outperforms the standard Transformer in most tasks on the GLUE benchmark, except for STS-B, particularly for smaller models.
- The model shows good scalability and performs well on text classification tasks, although specific results are referred to in Appendix C.1 due to page constraints.
- In token-level tasks such as SQuAD, the F-TFM small model outperforms previous models in the base group by a significant margin.

## Efficiency

- The F-TFM small model enjoys a super-linear complexity drop when the sequence length is reduced by half in the encoder, as the complexity of processing a length-T sequence is O(T^2D + TD^2).
- By reinvesting the saved FLOPs from length reduction into constructing a deeper or wider model, the F-TFM improves model capacity while maintaining or reducing computational costs.

## Usage

- The model can be applied to token-level tasks by finetuning the decoder.
- For tasks that utilize additional input signals, such as permutation order, this information can be injected into the Funnel-Transformer via the decoder input to recover more complex control of the attention mechanism.

## Additional Information

- The model speed in the finetuning stage and running time comparisons on GPUs and TPUs are summarized in the references, but specific figures are not provided here. [More Information Needed]
- Checkpoints for similar models with different block layouts and hidden sizes are available at the provided links.

## Conclusion

The Funnel-Transformer small model is a highly efficient and scalable architecture that provides competitive performance with fewer computational resources. It is suitable for a wide range of NLP tasks and can be adapted for token-level predictions and tasks requiring additional input signals.

```

Please note that some specific details such as throughput, start/end time, and checkpoint sizes for the `funnel-transformer/small` model are not provided in the references and are marked as "[More Information Needed]". Additional information may be available in the full paper or supplementary materials.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model funnel-transformer/small has been evaluated on the following benchmarks and datasets:

1. GLUE benchmark for language understanding.
2. Text classification tasks on the following datasets:
   - IMDB (sentiment classification)
   - AD (sentiment classification) [More Information Needed for the full form of AD]
   - DBpedia (topic classification)
   - Yelp-2 (sentiment classification)
   - Yelp-5 (sentiment classification)
   - Amazon-2 (sentiment classification)
   - Amazon-5 (sentiment classification)
3. RACE reading comprehension dataset.
4. SQuAD question answering task.

#### Factors

The foreseeable characteristics that will influence how the model funnel-transformer/small behaves can be inferred from the pretraining and finetuning settings, the design choices made during development, and the intended use cases. Here's an analysis based on the provided references:

1. **Pretraining Data and Scale**: The model has been pretrained on a combination of Wikipedia and Book Corpus at the base scale (1M steps, batch size 256) and on an extended dataset including ClueWeb, Gigaword, and Common Crawl at a large scale (500K steps, batch size 8K). This suggests that the model's behavior will be influenced by the textual characteristics of these datasets, which are predominantly English and may contain biases present in the source material. The model may perform better on tasks related to the domains covered by these datasets and may not generalize as well to domains not represented in the training data.

2. **Downstream Tasks**: The model is fine-tuned for sequence-level tasks that require a single vectorial representation of the input sequence. This indicates that the model is likely to perform well on tasks like those in the GLUE benchmark but may not be as effective for tasks requiring fine-grained token-level predictions or those that involve complex input structures beyond a single sequence representation.

3. **Model Design Choices**: The model utilizes a 3-block design and relative attention parameterization, which may affect its ability to capture long-range dependencies differently than models using absolute position embeddings. The pooling strategy, which selects a subset of "hub" states, could influence the model's behavior by focusing on certain parts of the input sequence deemed more critical, potentially affecting performance on tasks where this assumption does not hold.

4. **Population Subgroups**: Since the pretraining data is sourced from datasets that are not explicitly designed to be demographically balanced, there may be disparities in performance across different population subgroups, especially those not well-represented in the training data. [More Information Needed] to determine the exact nature of these disparities, as the provided references do not include disaggregated evaluations across factors such as gender, ethnicity, or language variety.

5. **Domain and Context**: The model's performance is likely to be influenced by the domain and context of the input data. Given its pretraining on diverse datasets, it may be more adaptable to a range of topics and styles. However, the specific characteristics of the "funnel" approach, which compresses the sequence length for efficiency, may lead to varying performance depending on the complexity and length of the input sequences.

6. **Efficiency and Scalability**: The model is designed to be more efficient than the standard Transformer, which may influence its behavior in resource-constrained environments. It is expected to scale well in certain domains, but [More Information Needed] to determine the full extent of its scalability across various applications.

In summary, the funnel-transformer/small model's behavior will be influenced by its pretraining data, design choices tailored for sequence-level tasks, and the efficiency-oriented architecture. Disparities in performance may exist across different population subgroups and domains, but further evaluation is needed to uncover these in detail.

#### Metrics

Based on the provided references, the evaluation metrics for the model funnel-transformer/small will include:

1. Performance on the GLUE benchmark: This is a collection of diverse natural language understanding tasks, and the model's performance on these tasks will be a key metric. The references indicate that the model is evaluated on the GLUE benchmark, comparing it to the standard Transformer models.

2. Performance on text classification tasks: Although specific metrics are not mentioned, it is common to use accuracy, F1 score, precision, and recall for text classification tasks. The references suggest that the model's performance on 7 text classification tasks is also considered, with details available in Appendix C.1.

3. Pretraining and finetuning performance: The model is pretrained and then finetuned on downstream tasks, following the common practice in the field. The quality of the pretraining can be assessed by how well the model finetunes to specific tasks.

4. Median result of multiple runs: For the GLUE benchmark, the median result of 5 runs from different random seeds in the base setting and 15 runs in the large setting is used to assess the model's performance. This suggests that stability and consistency of the model's performance across different initializations are also considered important.

5. Learning rate search: The best learning rate is chosen according to the validation set performance, indicating that validation set performance is a metric for hyperparameter tuning.

6. Comparison of FLOPs: The model's efficiency is measured by comparing the number of floating-point operations (FLOPs) to the standard Transformer models, ensuring that the Funnel-Transformer has fewer or similar FLOPs.

7. Running time: Reference 6 mentions running time comparisons on TPU v2-8, which suggests that computational efficiency is also a metric of interest.

In summary, the evaluation metrics for the funnel-transformer/small model include GLUE benchmark performance, text classification task performance, pretraining and finetuning effectiveness, stability across multiple runs, validation set performance for hyperparameter tuning, FLOPs comparison for computational efficiency, and running time.

### Results

Based on the provided references, here are the evaluation results for the model `funnel-transformer/small`:

1. **Comparison to Standard Transformer**: The `funnel-transformer/small` (F-TFM/small) has been compared to the standard Transformer of similar size (small, with L6H768 configuration) in terms of FLOPs. The F-TFM/small is designed to have fewer or similar FLOPs compared to its standard Transformer counterpart.

2. **Performance on GLUE Benchmark**: The F-TFM/small outperforms the standard Transformer in most tasks on the GLUE benchmark, with the exception of STS-B. This performance gain is attributed to the trade-off between sequential resolution and the number of layers, which is more beneficial for smaller models like the F-TFM/small.

3. **Performance Loss in Certain Settings**: When the sequence length is compressed without increasing the depth or the number of parameters, the F-TFM/small may experience some performance loss on certain GLUE datasets. However, this gap diminishes or disappears as the model size increases.

4. **Pretraining Settings**: The F-TFM/small has been pretrained under the base scale setting, which involves 1 million steps with a batch size of 256 on Wikipedia and Book Corpus, following the original BERT pretraining setup.

5. **Generality of F-TFM**: The generality of the F-TFM has been tested with ELECTRA pretraining, and a similar trend of performance gain is observed, although slightly smaller on the GLUE benchmark. This could be due to the reuse of hyperparameters tuned for the standard Transformer without adjustment for F-TFMs.

6. **Model Design for Efficiency**: The F-TFM/small is designed to be more efficient by reducing the sequence length and reinvesting the saved FLOPs into constructing a deeper or wider model, thereby improving model capacity.

7. **Token-Level Predictions**: Despite the sequence length reduction, the F-TFM/small can recover a deep representation for each token via a decoder, which is necessary for common pretraining objectives.

In summary, the `funnel-transformer/small` model demonstrates improved efficiency and performance on most GLUE benchmark tasks compared to the standard Transformer of similar size, with some exceptions and potential performance loss in specific configurations. Pretraining has been conducted following established protocols, and the model's design emphasizes efficiency and capacity for sequence-level tasks.

#### Summary

The evaluation results for the model `funnel-transformer/small` can be summarized as follows:

- When compared to the standard Transformer model of similar size (small, with L6H768 configuration), the Funnel-Transformer (F-TFM) is designed to have fewer or similar FLOPs (floating-point operations per second) to ensure a fair comparison.
- The F-TFM demonstrates improved performance on the GLUE benchmark tasks compared to the standard Transformer, particularly for smaller models. This is achieved by trading sequential resolution for more layers, which enhances the model's capacity.
- However, there are some exceptions where the F-TFM does not outperform the standard Transformer, such as on the STS-B (Semantic Textual Similarity Benchmark) task.
- In cases where the sequence length is compressed without increasing the depth or the number of parameters, the F-TFM may experience some performance loss on certain GLUE datasets. This performance gap tends to diminish or disappear as the model size increases.
- The F-TFM incorporates the relative positional attention parameterization from Transformer-XL, which is used across all models implemented in this work, including the Transformer baselines.
- The F-TFM is capable of handling additional input signals through its decoder input, which allows for more complex control of the attention mechanism for specific tasks.
- Overall, the F-TFM, with its ability to compress the sequence of hidden states and reinvest the saved FLOPs into a deeper or wider model structure, tends to outperform the standard Transformer in terms of capacity and efficiency, given the same or fewer FLOPs.

[More Information Needed] for any additional specific quantitative results or comparisons not covered in the provided references.

## Model Examination

Explainability/Interpretability Section for Model Card: Funnel-Transformer/Small

The Funnel-Transformer, as introduced in our work, is a novel architecture that aims to reduce computational complexity by compressing the sequence of hidden states. This compression is achieved through an encoder-decoder framework where the encoder reduces the length of the input sequence, and the decoder recovers token-level representations. Our small variant of the Funnel-Transformer model maintains the essence of this architecture while being designed to be more computationally efficient.

For explainability and interpretability, it is important to note that the Funnel-Transformer/small model incorporates several design choices that impact its behavior:

1. Pool-query-only design: This design choice affects how information is pooled in the model. By focusing on query tokens, the model aims to retain the most relevant information during the compression phase.

2. Separating [CLS] in the pooling operation: The [CLS] token, often used for classification tasks, is treated separately in the pooling operation to preserve its role in aggregating sequence-level representations.

3. Block layout design: Our experiments with the Funnel-Transformer/small model utilize a 3-block design. This design was chosen after comparing it with 2-blocks and 4-blocks designs, aiming to find a balance between computational efficiency and model performance.

4. Relative attention parameterization: We also explored the use of relative attention parameterization, as proposed in Transformer-XL, and compared it with learned absolute position embedding, following the BERT model. This impacts how the model understands and utilizes positional information.

5. Model compression techniques: While not explicitly implemented in the small variant, combining the Funnel-Transformer with techniques like knowledge distillation and quantization is a potential direction for enhancing the model's practical impact and efficiency.

In terms of interpretability, the Funnel-Transformer/small model's ability to recover token-level deep representations from a compressed sequence can be leveraged to understand how the model processes and compresses information. This is particularly relevant for tasks that require token-level predictions, as the model demonstrates its capability to maintain performance even with reduced sequence length.

Future work on explainability may involve developing methods to visualize and interpret the compressed representations and the information flow within the model. This could provide insights into the decision-making process of the model and the importance of different tokens and sequences in the final predictions.

In summary, the Funnel-Transformer/small model is designed to be efficient without sacrificing performance. Its architecture choices are geared towards maintaining a balance between computational savings and the ability to perform complex NLP tasks effectively. Further research into explainability and interpretability will be crucial in making the model more transparent and trustworthy for users.

## Environmental Impact

- **Hardware Type:** The model funnel-transformer/small was trained on TPU v3-16 (16 cores x 16Gb) for pretraining, as mentioned in reference 2. For finetuning, it was trained on TPU v2-8 (8 cores x 8Gb) with TensorFlow and on Nvidia-V100 (16Gb) GPU with PyTorch, as also indicated in reference 2.
- **Software Type:** The model `funnel-transformer/small` is trained on TPUs, as indicated by the reference stating that the corresponding source code in the `tensorflow` folder was developed and used for TPU pretraining & finetuning as presented in the paper.
- **Hours used:** [More Information Needed]
- **Cloud Provider:** The cloud provider that the model funnel-transformer/small is trained on is Google Cloud Platform.
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The Funnel-Transformer/small model is a novel self-attention model architecture that introduces a sequence compression mechanism to reduce computational cost while maintaining or improving model capacity. The architecture is designed with the following key features:

1. **Sequence Compression**: The encoder of the Funnel-Transformer gradually compresses the sequence of hidden states to a shorter one. This is achieved by employing pooling operations between blocks of consecutive Transformer layers within the encoder. The sequence length remains constant within each block but is reduced when transitioning to a higher-level block.

2. **Model Architecture**: The small variant of the Funnel-Transformer, although not explicitly detailed in the provided references, would likely follow the general design principles of the Funnel-Transformer. It would consist of several blocks with a reduced number of layers and hidden size compared to larger variants. For instance, a larger variant with a configuration of B6-6-6H768 is mentioned, which has 18 layers in total. The small variant would have fewer layers and a smaller hidden size.

3. **Efficiency and Capacity**: By compressing the sequence length, the Funnel-Transformer enjoys a super-linear drop in complexity, allowing for a deeper or wider model given the same or fewer floating-point operations per second (FLOPs). This results in a model that is more efficient and potentially has a higher capacity for the same computational budget.

4. **Decoder Usage**: The decoder in the Funnel-Transformer is used for tasks that require token-level predictions, such as pretraining or sequence labeling. It is capable of recovering a deep representation for each token from the compressed hidden sequence. However, for tasks that require a single vector representation of the sequence, such as classification, the decoder is not used during fine-tuning.

5. **Objective**: The primary objective of the Funnel-Transformer is to provide a more efficient architecture that can scale well and maintain high performance on various tasks. It aims to reduce the resource-to-performance ratio while still being able to perform token-level predictions effectively.

In summary, the Funnel-Transformer/small model is designed to be a more computationally efficient version of the Transformer that achieves high performance with fewer resources by compressing the sequence of hidden states and re-investing the saved FLOPs into a deeper or wider model structure. The exact configuration details of the small variant, such as the number of layers and hidden size, are not provided in the references and would require [More Information Needed].

### Compute Infrastructure

The compute infrastructure used for the model funnel-transformer/small involved the following:

- For pretraining speed evaluation, the model was tested on a TPU v3-16, which consists of 16 cores with 16GB of memory each, using TensorFlow.
- For finetuning speed evaluation, the model was tested on a TPU v2-8, which has 8 cores with 8GB of memory each, using TensorFlow, and on an Nvidia-V100 GPU with 16GB of memory using PyTorch.
- The TensorFlow version used was 2.2.0, and the PyTorch version was 1.5.0.
- For the GPU experiments, an 8-GPU node on the Google Cloud Platform was utilized.
- All running speeds were reported with the FP16 optimizer. In the PyTorch implementation, the "O2" options of the AMP manager in the apex package were used for FP16 optimization.
- The maximum possible batch sizes that could be accommodated by the memory size of the devices were chosen for both pretraining and finetuning.
- The actual model running time was measured by performing 1000 steps of gradient descent with random input sequences of fixed length.

For more specific details regarding the compute infrastructure for the funnel-transformer/small model, such as the exact batch sizes used or the running times, [More Information Needed] as they are not provided in the given references.

## Citation

```
@misc{zihang-funneltransformer,
    author = {Zihang Dai and
              Guokun Lai and
              Yiming Yang and
              Quoc V Le},
    title  = {Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing},
    url    = {https://arxiv.org/pdf/2006.03236.pdf}
}
```

