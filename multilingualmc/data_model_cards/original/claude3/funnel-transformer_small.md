# Model Card for funnel-transformer/small

Funnel-Transformer (F-TFM) is a new self-attention model that gradually compresses the sequence of hidden states to a shorter one, reducing computation cost. By re-investing the saved FLOPs from length reduction in constructing a deeper or wider model, Funnel-Transformer achieves improved performance over the standard Transformer on various sequence-level prediction tasks, with comparable or fewer FLOPs.

## Model Details

### Model Description

Model Card for funnel-transformer/small

Model Details:
- Architecture: Funnel-Transformer, which compresses the sequence of hidden states to a shorter one to reduce computation cost. It consists of an encoder that performs compression and a decoder that can recover token-level representations from the reduced hidden sequence.
- Hidden size: 768
- Number of layers: [More Information Needed]
- Number of attention heads: hidden size / 64
- Training objective: [More Information Needed] (Possibly MLM or ELECTRA-style)
- Pretraining data: [More Information Needed]
- Pretraining steps: [More Information Needed] 
- Batch size: [More Information Needed]

[More Information Needed] on the specific layer configuration, training objective, pretraining data, number of pretraining steps and batch size used for the small variant.

Intended Use:
The model is primarily intended for sequence-level tasks that require a single vector representation of the input, such as text classification. The decoder allows recovering token-level representations for pretraining or token-level prediction tasks.

Limitations and Bias:
[More Information Needed] 

Additional Notes:
Funnel-Transformer is more efficient than a standard Transformer with the same FLOPs due to its compression mechanism. The saved computation from length reduction can be reinvested to make the model deeper or wider, improving model capacity.

Contact:
For questions or issues regarding this model, please contact [More Information Needed].

- **Developed by:** Zihang Dai; Guokun Lai; Yiming Yang; Quoc V Le
- **Funded by:** [More Information Needed]

The provided references do not contain any information about the funding sources for the funnel-transformer/small model project. More details would be needed from the model developers or associated documentation to determine what people or organizations, if any, provided funding for this specific model's development.
- **Shared by:** Based on the provided references, the main contributors to the Funnel-Transformer model are:

Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le

The paper that introduces Funnel-Transformer is authored by them, with Zihang Dai and Guokun Lai marked as equal contribution.

[More Information Needed] The references do not specify additional details about who exactly made the specific model variant funnel-transformer/small available online as a GitHub repo. More information would be needed to confidently determine that.
- **Model type:** The funnel-transformer/small model is a self-attention based Transformer model that gradually compresses the sequence of hidden states to reduce computation cost while maintaining high capacity, and it is pretrained on text data to perform sequence-level prediction tasks.
- **Language(s):** The funnel-transformer/small model processes English text, as evidenced by its pretraining on the Wikipedia and BookCorpus datasets and finetuning on English language understanding tasks like IMDB, AD, DBpedia, Yelp, Amazon, RACE, and SQuAD.
- **License:** [More Information Needed]

The provided references do not contain information about the specific license being used for the funnel-transformer/small model. More details would be needed from the model developers or documentation to determine the applicable license.
- **Finetuned from model:** Based on the provided references, there is no explicit mention of the funnel-transformer/small model being fine-tuned from another pre-existing model. The references discuss the general Funnel-Transformer architecture and its advantages, but do not specify the particular funnel-transformer/small model or its potential fine-tuning from a base model.

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/laiguokun/Funnel-Transformer
- **Paper:** https://arxiv.org/pdf/2006.03236.pdf
- **Demo:** [More Information Needed]

The provided references do not contain a link to a demo of the funnel-transformer/small model. More specific information would be needed to provide a demo link for this particular model.
## Uses

### Direct Use

Based on the provided references, the funnel-transformer/small model is primarily designed to be fine-tuned on downstream tasks, especially sequence-level tasks that only require a single vectorial representation of the input sequence (References 2, 3, 4). The references do not provide clear information on how to use the model without fine-tuning, post-processing, or in a pipeline.

The model compresses the input sequence into a reduced-length hidden representation using the encoder (References 1, 6, 10, 11). For token-level predictions, the decoder can recover token-level representations from the reduced hidden sequence (References 1, 10). However, the exact process and code for using the model directly without fine-tuning or in a pipeline are not provided.

In summary, [More Information Needed] on how to use the funnel-transformer/small model without fine-tuning, post-processing, or plugging into a pipeline based on the given references. The references focus more on the model architecture and fine-tuning performance on downstream tasks.

### Downstream Use

The funnel-transformer/small model can be fine-tuned for various sequence-level tasks that require a single vectorial representation of the input sequence, such as text classification tasks in the GLUE benchmark.

With an additional decoder, the model can also be applied to token-level tasks like question answering on SQuAD datasets, by fine-tuning the decoder to recover token-level representations from the compressed hidden states. However, the performance may be lower compared to standard Transformers that maintain full-length token-level representations.

[More Information Needed] for a specific code snippet, as the provided references do not contain a direct code block.

Overall, the funnel-transformer/small model offers computational efficiency by compressing the sequence length, while the saved FLOPs can be reinvested in increasing model depth or width to improve performance. It can be plugged into NLP applications dealing with sequential data, but further exploration is needed to optimize the compression scheme and block layout design for specific tasks.

### Out-of-Scope Use

Based on the provided references about the Funnel-Transformer model, here are some potential ways the model funnel-transformer/small may foreseeably be misused and what users ought not do with it:

The model should not be used for tasks or applications where preserving the full sequence length and attention to all tokens is critical, as the key feature of Funnel-Transformer is that it compresses the sequence to reduce computation. For tasks requiring fine-grained token-level analysis, a different model may be more appropriate.

Users should be cautious about applying Funnel-Transformer to sensitive tasks like hate speech detection or misinformation identification without thorough testing, as the sequence compression could potentially lose important details. Responsible testing and monitoring for unintended biases or blind spots is advised.

[More Information Needed] on specific risks if Funnel-Transformer is applied to domains outside NLP like time series and video stream analysis, as alluded to in Reference 1. More research is needed on Funnel-Transformer's suitability and failure modes in those areas.

The ability to inject additional decoder input to control attention, mentioned in Reference 2, should be used judiciously. Users should not exploit this to manipulate the model's attention in unintended or deceptive ways.

In general, as with any compressed representation, users should not over-rely on or place excess trust in the model's outputs, especially for high-stakes decisions. The model's limitations and potential information loss from compression should be kept in mind. Human oversight and critical analysis of the model's fitness for a given task remains necessary.

### Bias, Risks, and Limitations

Based on the provided references about the Funnel-Transformer model, here are some potential issues and limitations:

Technical limitations:
- The optimal block layout design for trading off sequence length and model depth/width is an open challenge. The paper only considered regular layouts. More systematic studies are needed. (Reference 7)
- Increasing model depth/width to compensate for reduced sequence length leads to more parameters, which may increase communication costs in distributed training and memory consumption for serving. (Reference 8)

[More Information Needed] on other specific technical limitations of the small variant.

Potential misunderstandings:
- The Funnel-Transformer is designed for more efficient language processing by compressing sequence length, not necessarily for superior performance to a standard Transformer with the same compute budget. Performance gains come from reinvesting the saved compute into a deeper/wider model. (References 1, 7)

[More Information Needed] on other potential misunderstandings.

Foreseeable harms and sociotechnical issues:
[More Information Needed] The provided references focus on the technical aspects of the Funnel-Transformer and do not discuss societal harms or sociotechnical issues. More information would be needed from the model developers on these aspects, such as potential misuse cases, bias/fairness considerations, environmental impact, etc.

In summary, key limitations seem to be the need for further studies on optimal model configurations, increased parameters from the depth-length tradeoff, and a lack of information on broader societal implications. More details would be needed especially on potential harms and sociotechnical issues.

### Recommendations

Based on the provided references about the Funnel-Transformer model, here are some recommendations regarding foreseeable issues:

1. Further research is needed to improve the compression scheme used in the Funnel-Transformer to make it more efficient and effective. [More Information Needed] on specific techniques that could be explored.

2. Optimizing the block layout design is an open challenge. The references mention comparing 2-block, 3-block and 4-block designs, but [More Information Needed] on the optimal configuration for different tasks and datasets.

3. Combining Funnel-Transformer with model compression techniques like knowledge distillation and quantization is an important direction to enhance its practical impact. [More Information Needed] on specific experiments and results in this area.

4. While the focus has been on NLP tasks, Funnel-Transformer could potentially be applied to other sequential data tasks like time series and video stream analysis. Further research and experiments are recommended to explore its applicability and performance in these domains.

5. [More Information Needed] on how the Funnel-Transformer handles additional decoder inputs for tasks that utilize extra signals, and how this impacts its attention mechanism and overall performance compared to standard approaches.

In summary, while the Funnel-Transformer shows promise in improving efficiency by compressing sequence length and reinvesting computation, more research is needed to optimize its architecture, combine it with other compression techniques, and expand its application to additional domains. Providing more detailed experimental results and comparisons would help in understanding and addressing potential issues.

## Training Details

### Training Data

The model funnel-transformer/small was pretrained on the Wikipedia + Book Corpus datasets for 1M steps with a batch size of 256, which is the same setting used by the original BERT model. The text was preprocessed by lowercasing and tokenizing with the BERT uncased wordpiece tokenizer, and some additional preprocessing was done to remove certain HTML symbols in downstream text classification tasks.

### Training Procedure

#### Preprocessing

For the model funnel-transformer/small, the preprocessing details are as follows:

Tokenization:
- Adapted the "uncased" word piece model originally used by BERT, with a vocabulary size of about 30K.
- Performed lower casing and the default preprocessing included in the word piece tokenizer.
- Additionally, removed some http symbols (e.g., <b>) in the 7 text classification tasks.

[More Information Needed] on resizing/rewriting, as no specific details were provided in the given references.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the funnel-transformer/small model:

Vocabulary:
- Used the "uncased" word piece model originally used by BERT, with a vocabulary size of about 30K. [Reference 1]

Preprocessing:
- Lowercased text and applied default preprocessing included in the word piece tokenizer.
- Removed some http symbols (e.g. <b>) in the 7 text classification tasks. [Reference 1]

Pretraining:
- [More Information Needed] on the specific pretraining hyperparameters used for the small model.
- In general, for base-scale models, reduced learning rate to 8e-5 for the B10-10-10H1024 F-TFM to handle training instability beyond 24 layers. [Reference 7]

Finetuning:
- For GLUE and text classification datasets:
  - Fixed most hyperparameters as shown in Table 8 [Reference not provided].
  - Searched learning rates from the set [1e-5, 2e-5, 3e-5], and chose the best based on validation set. [Reference 4]
- For RACE and SQuAD datasets:
  - [More Information Needed] on the specific hyperparameters used.
  - In general, reused hyperparameters from XLNet as summarized in Table 9 [Reference not provided]. [Reference 2]
- Reported GLUE performance is the median result of 5 runs from different random seeds in the base setting. [Reference 4]
- Reported text classification performance is the median of [More Information Needed] runs in the base setting. [Reference 4]

Hardware:
- All performance numbers reported are obtained on TPUs with TensorFlow 2.2. [Reference 6]

[More Information Needed] on additional hyperparameters and training details specific to the funnel-transformer/small model.

#### Speeds, Sizes, Times

Based on the provided references, here are the details I can provide about the funnel-transformer/small model:

Throughput and Running Time:
- The funnel-transformer/small model has fewer or similar FLOPs compared to the standard Transformer small model (L6H768). (Reference 1)
- [More Information Needed] on specific throughput numbers.
- [More Information Needed] on start or end training times.

Checkpoint Sizes:
- [More Information Needed] No specific information provided about checkpoint sizes for the small model.

Other Details:
- The funnel-transformer/small model outperforms the standard Transformer small model on most GLUE benchmark tasks, except STS-B, given similar or fewer FLOPs. (Reference 1)
- Trading sequential resolution for more layers allows the funnel-transformer/small to outperform the standard Transformer small model. (Reference 2)
- Compressing sequence length without increasing depth may lead to some performance loss on GLUE datasets for the small model, but gaps become smaller as model size increases. (Reference 2)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the funnel-transformer/small model was evaluated on the following benchmarks and datasets:

1. GLUE benchmark for language understanding
2. 7 text classification tasks: IMDB, AD, DBpedia, Yelp-2, Yelp-5, Amazon-2, Amazon-5
3. RACE reading comprehension dataset
4. SQuAD question answering task

The model was first pretrained and then finetuned on these downstream tasks. The base-scale pretraining setting involved 1M steps with batch size 256 on Wikipedia + Book Corpus.

[More Information Needed] on the specific performance metrics and scores achieved by the funnel-transformer/small model on each of these benchmarks and datasets.

#### Factors

Based on the provided references about the Funnel-Transformer model, here are some foreseeable characteristics that may influence how the funnel-transformer/small model behaves:

Domain and Context:
- The model is pretrained on Wikipedia + Book Corpus data, so it may perform better on tasks and domains similar to that pretraining data. Performance on very different domains like social media, scientific papers, etc. is less certain. [More Information Needed] on evaluation across diverse domains.

Population Subgroups:
- [More Information Needed] The references do not discuss model performance disaggregated by demographic subgroups. Evaluation should ideally analyze performance across different genders, races, ages, etc. to uncover any disparities or biases.

Model Architecture Choices:
- Using mean/max pooling performs significantly better than attention-based pooling for selecting "hub" states (Reference 7)
- The 3-block model design was used, but [More Information Needed] on how this compares to 2-block and 4-block designs (Reference 4)
- Relative positional attention from Transformer-XL was used instead of absolute position embeddings like BERT (References 4, 8)

These model architecture choices may impact performance, but more disaggregated evaluation is needed across different task types, domains, and subgroups to fully characterize the model's strengths and weaknesses. The references focus more on technical details than on thorough behavioral testing.

#### Metrics

Based on the provided references about the Funnel-Transformer model, the following metrics are mentioned for evaluation:

1. Performance on GLUE benchmark tasks for language understanding, compared to standard Transformer baselines of similar FLOPs (Reference 1, 4).

2. Performance on 7 text classification tasks (IMDB, AD, DBpedia, Yelp-2, Yelp-5, Amazon-2, Amazon-5), with results referred to in Appendix C.1 (Reference 3).

3. Performance on SQuAD question answering task to evaluate token-level prediction (Reference 2).

4. FLOPs (floating point operations) to measure and compare the computation cost between Funnel-Transformer and standard Transformer (Reference 1, 8, 10).

5. Running time comparison between Funnel-Transformer and standard Transformer on TPU v2-8 (Reference 9).

[More Information Needed] on the specific tradeoffs and error analysis for the funnel-transformer/small model, as the references do not provide direct information about this particular variant.

### Results

Based on the provided references, here are the evaluation results for the funnel-transformer/small model:

Model Size and Computation:
- The funnel-transformer/small model is compared against the standard Transformer small model (L6H768). 
- The funnel-transformer/small model is constructed to have fewer or similar FLOPs (floating point operations) compared to the Transformer small baseline.

Pretraining:
- The model is pretrained using the MLM (masked language modeling) objective for 1M steps with batch size 256 on the Wikipedia + Book Corpus dataset, following the original BERT base scale setting.

Downstream Tasks:
- GLUE benchmark: The funnel-transformer/small model outperforms the Transformer small model on most GLUE tasks, except for STS-B, given similar or fewer FLOPs. 
- Text classification: [More Information Needed]
- RACE reading comprehension: [More Information Needed]
- SQuAD question answering: [More Information Needed]

Additional Notes:
- Trading sequential resolution for more layers in the funnel-transformer/small model leads to performance gains, especially for smaller models.
- Only compressing sequence length without increasing depth could lead to some performance loss on GLUE, but the gap becomes smaller as model size increases.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the funnel-transformer/small model:

The funnel-transformer/small model was compared against standard Transformer models of similar computational cost (FLOPs). Specifically, it was evaluated against the Transformer small model (L6H768).

Key findings:
- Given similar or fewer FLOPs, by trading sequential resolution for more layers, the funnel-transformer/small model outperformed the standard Transformer small on most GLUE benchmark tasks, except for STS-B.
- When only compressing sequence length without increasing depth (and parameters), the funnel-transformer/small model suffered some performance loss on certain GLUE tasks compared to Transformer. However, this gap became smaller or disappeared as model size increased.

[More Information Needed] on exact performance metrics, as the references do not provide specific numbers for the small model size.

In summary, the funnel-transformer/small demonstrates that, with comparable computation, trading sequential resolution for increased depth can yield performance improvements over the standard Transformer architecture on many natural language understanding tasks.

## Model Examination

Based on the provided references about the Funnel-Transformer model, here is a possible answer to the question about explainability/interpretability for the funnel-transformer/small model:

The funnel-transformer/small model utilizes the Funnel-Transformer architecture, which compresses the sequence of hidden states to a shorter one to reduce computation cost. By re-investing the saved computation from length reduction, the model can be made deeper or wider to improve capacity while maintaining comparable or fewer FLOPs compared to standard models.

Some aspects that could contribute to the explainability and interpretability of the funnel-transformer/small model include:

1. Analyzing the attention maps from the self-attention layers to understand which positions the model attends to most and carries critical information about the sequence. This could provide insights into what the model focuses on for a given input.

2. Examining the behavior of the decoder in recovering token-level deep representations from the reduced hidden sequence. Understanding how the decoder reconstructs the full sequence could shed light on how the model preserves and utilizes information.

3. [More Information Needed] on specific techniques or experiments conducted to assess the explainability and interpretability of the funnel-transformer/small model.

4. [More Information Needed] on any visualizations, case studies, or examples that demonstrate the model's decision-making process or highlight the most influential input features.

Overall, while the provided references give some high-level insights into the Funnel-Transformer architecture, more specific information would be needed to comprehensively address the explainability and interpretability aspects of the funnel-transformer/small model.

## Environmental Impact

- **Hardware Type:** Based on the provided references, the funnel-transformer/small model was likely trained on TPU v3-16 (16 cores x 16Gb) with TensorFlow for pretraining. For finetuning, it was likely trained on TPU v2-8 (8 cores x 8Gb) with TensorFlow and on Nvidia-V100 (16Gb) GPU with PyTorch.

However, the exact hardware used for the funnel-transformer/small model is not explicitly stated in the given references. More specific information about the hardware configuration for this particular model would be needed to provide a definitive answer.
- **Software Type:** Based on the provided references, the Funnel-Transformer model was developed and trained using both PyTorch and TensorFlow:

1. The PyTorch code in the `pytorch` folder only supports GPU finetuning for the GLUE benchmark & text classification.

2. The TensorFlow code in the `tensorflow` folder was used for TPU pretraining & finetuning as presented in the paper. It mainly supports TPU finetuning on GLUE benchmark, text classification, SQuAD and RACE.

To definitively answer what software type the specific model funnel-transformer/small was trained on, [More Information Needed].
- **Hours used:** [More Information Needed]

The provided references do not contain specific information about the training time for the funnel-transformer/small model. More details would be needed, such as the exact pretraining settings used (base scale or large scale), hardware specifications, and the measured training time for this particular model size.
- **Cloud Provider:** Based on the provided references, the model funnel-transformer/small was trained on Google Cloud Platform. Specifically, reference 1 mentions:

"Google Cloud Platform. All running speeds are reported with the FP16 optimizer."

So the cloud provider used for training the model is Google Cloud Platform.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the carbon emissions from training the funnel-transformer/small model. To accurately report the carbon footprint, details such as the hardware used for training, training duration, energy consumption, and carbon intensity of the energy source would be needed.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
Funnel-Transformer (F-TFM) is a new self-attention model that gradually compresses the sequence of hidden states to a shorter one as the layer gets deeper, reducing computation cost. It keeps the same overall skeleton of interleaved multi-head self-attention (S-Attn) and position-wise feed-forward network (P-FFN) sub-modules wrapped by residual connection and layer normalization as the standard Transformer. For tasks involving per-token predictions like pretraining, a simple decoder is used to reconstruct a full sequence of token-level representations from the compressed encoder output.

Model Objective:
The objective of Funnel-Transformer is to improve efficiency by compressing the sequential resolution and re-investing the saved FLOPs in constructing a deeper or wider model to improve model capacity and performance. The pretraining objective is likely to be masked language modeling, which can be expressed as:

[More Information Needed]

After pretraining, the entire model is finetuned on downstream tasks. For tasks that only require a single vectorial representation of the sequence like classification, the decoder is discarded after pretraining and only the encoder is finetuned.

### Compute Infrastructure

Based on the provided references, here is the compute infrastructure information I can provide about the funnel-transformer/small model:

- The model was pretrained using the base scale setting of 1M steps with batch size 256 on the Wikipedia + Book Corpus dataset. This is the same setting used by the original BERT model. [Reference 1]

- For finetuning, the model running time was measured on both GPUs and TPUs by performing 1000 steps of gradient descent with random input sequences of fixed lengths (128, 256, 512). The maximum possible batch size was used based on the memory size of the device(s). [References 6, 7]

- The finetuning experiments were run on Google Cloud Platform using FP16 optimization. For PyTorch, the "O2" options of the AMP manager in the apex package were used to handle FP16. [References 7, 8]

[More Information Needed] on the specific GPU and TPU models used, as well as the memory sizes and batch sizes for each sequence length setting during finetuning.

The PyTorch source code in the `pytorch` folder only supports GPU finetuning for the GLUE benchmark & text classification tasks. [Reference 9]

[More Information Needed] on the compute infrastructure used for the large scale pretraining setting and any TPU pretraining.

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

