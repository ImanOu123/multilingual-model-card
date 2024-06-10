# Model Card for SCUT-DLVCLab/lilt-roberta-en-base

The SCUT-DLVCLab/lilt-roberta-en-base model is a language-independent layout Transformer (LiLT) pre-trained on visually-rich documents of a single language (English) using the LayoutLMv2 framework, which can be directly fine-tuned on other languages with the corresponding off-the-shelf monolingual/multilingual pre-trained textual models like RoBERTa for structured document understanding tasks.

## Model Details

### Model Description

Model Architecture:
- LiLT is a parallel dual-stream Transformer model, consisting of a text flow and a layout flow.
- The text flow is initialized from the pre-trained English RoBERTa BASE model.
- The layout flow shares a similar structure as the text flow, but with reduced hidden size and intermediate size for computational efficiency.
- Special tokens [CLS], [SEP] and [PAD] are attached with fixed position embeddings.
- Text and layout features are combined using layer normalization and summation.

Training Procedures:
- LiLT BASE is pre-trained using the Adam optimizer with learning rate 2×10^-5, weight decay 1×10^-2, and (β1, β2) = (0.9, 0.999). 
- The learning rate is linearly warmed up over the first 10% steps and then linearly decayed.
- Batch size is set to 96.
- LiLT BASE is trained for 5 epochs on the IIT-CDIP dataset using 4 NVIDIA A40 48GB GPUs.
- Pre-training tasks include masked visual-language modeling (MVLM), key point location (KPL), and cross-modal alignment identification (CAI).

Parameters:
- The maximum sequence length N is set to 512.
- The layout flow has 6.1M parameters.
- [More Information Needed] on the total number of parameters.

Important Disclaimers:
- LiLT assumes that the OCR results provide text bounding boxes at the text string level, not the token level.
- LiLT is designed to be language-independent. It can be pre-trained on English data and adapted to other languages without additional pre-training.
- During fine-tuning, the layout flow (LiLT) can be separated and combined with off-the-shelf pre-trained textual models to handle downstream tasks.

- **Developed by:** Jiapeng Wang; Lianwen Jin; Kai Ding
- **Funded by:** Based on the provided references, there is no direct information about the funding sources for the SCUT-DLVCLab/lilt-roberta-en-base model. The references mainly discuss the datasets (XFUND, EPHOIE, RVL-CDIP, CORD, FUNSD) and provide some code snippets for model training, but do not mention the specific organizations or people that funded the development of the lilt-roberta-en-base model.

[More Information Needed]
- **Shared by:** Based on the references provided, the contributors who made the model SCUT-DLVCLab/lilt-roberta-en-base available online as a GitHub repo are:

Jiapeng Wang, Lianwen Jin, and Kai Ding

This can be seen from reference 4 which lists the authors of the paper describing LiLT (Jiapeng Wang; Lianwen Jin; Kai Ding), and reference 5 which links to the GitHub repo for LiLT maintained by jpWang (presumably Jiapeng Wang).
- **Model type:** SCUT-DLVCLab/lilt-roberta-en-base is a language-independent layout transformer model that is pre-trained on English visually-rich documents using self-supervised learning tasks and can be fine-tuned on other languages for structured document understanding tasks involving both text and layout modalities.
- **Language(s):** The SCUT-DLVCLab/lilt-roberta-en-base model is pre-trained on 11 million monolingual English documents from the IIT-CDIP dataset and can be adapted to process structured documents in other languages.
- **License:** Based on the provided references, there is no explicit mention of the license being used for the SCUT-DLVCLab/lilt-roberta-en-base model. The references discuss the model architecture, datasets used for evaluation, and links to download the model weights, but do not specify the license.

[More Information Needed] regarding the license for the SCUT-DLVCLab/lilt-roberta-en-base model.
- **Finetuned from model:** Based on the provided references, the SCUT-DLVCLab/lilt-roberta-en-base model is fine-tuned from the RoBERTa base model. This can be inferred from the following statements:

From reference 9:
"During fine-tuning, the layout flow (LiLT) can be separated and combined with the off-the-shelf pre-trained textual models (such as RoBERTa (Liu et al., 2019b), XLM-R (Conneau et al., 2020), InfoXLM (Chi et al., 2021), etc) to deal with the downstream tasks."

The model name "lilt-roberta-en-base" also suggests it is based on RoBERTa.

However, no direct link to the base RoBERTa model is provided in the references. [More Information Needed] for the link to the base model.
### Model Sources

- **Repository:** https://github.com/jpwang/lilt
- **Paper:** https://arxiv.org/pdf/2202.13669.pdf
- **Demo:** [More Information Needed]

The provided references do not contain a link to a demo of the SCUT-DLVCLab/lilt-roberta-en-base model. The references mainly discuss how to generate the model weights by combining the LiLT pre-trained model with an English RoBERTa base model, as well as some training details. However, no specific demo link for this particular model is provided.

To include the demo link in the model card, more information would need to be gathered from the model development team or other relevant sources.
## Uses

### Direct Use

The SCUT-DLVCLab/lilt-roberta-en-base model is designed to be fine-tuned on downstream structured document understanding tasks like FUNSD, as shown in Reference 7:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 examples/run_funsd.py \
        --model_name_or_path lilt-roberta-en-base \
        --tokenizer_name roberta-base \
        --output_dir ser_funsd_lilt-roberta-en-base \
        --do_train \
        --do_predict \
        --max_steps 2000 \
        --per_device_train_batch_size 8 \
        --warmup_ratio 0.1 \
        --fp16
```

The references do not provide information on how to use the model without fine-tuning, post-processing, or in a pipeline. [More Information Needed] on these aspects of using the SCUT-DLVCLab/lilt-roberta-en-base model directly.

### Downstream Use

The SCUT-DLVCLab/lilt-roberta-en-base model can be fine-tuned on various structured document understanding tasks in English, such as:

- Form understanding on the FUNSD dataset (Ref 1, 3, 5)
- Receipt understanding on the CORD dataset (Ref 3) 
- Document classification on the RVL-CDIP dataset (Ref 3, 7)
- Key information extraction on the EPHOIE dataset (Ref 3, 6)

When fine-tuned, the LiLT layout model is combined with a pre-trained RoBERTa textual model to jointly learn the layout and textual information for the specific task (Ref 10).

LiLT can also enable multitask learning by simultaneously fine-tuning on datasets in multiple languages, which can further boost performance by leveraging commonalities in document layouts across languages (Ref 2, 8).

To use the model in a larger document processing pipeline, the fine-tuned model can be applied to extract key information (entities and their relationships) from scanned or digital documents (Ref 8, 9). The extracted structured data can then be integrated into downstream applications.

[More Information Needed] for code snippets.

### Out-of-Scope Use

Based on the provided references about the SCUT-DLVCLab/lilt-roberta-en-base model, here are some potential ways the model could foreseeably be misused and what users ought not do:

The model is designed for cross-lingual document understanding tasks like form understanding, receipt parsing, document classification, etc. Users should not attempt to use it for tasks it was not designed or evaluated for, such as open-ended language generation, question answering, or language translation, as the performance and outputs would be unpredictable.

Since the model was only pre-trained on monolingual English documents, users should be cautious about relying on it too heavily for processing multilingual documents, especially for high-stakes use cases. More testing is needed to establish its multilingual transfer capabilities and limitations.

[More Information Needed] on the details of the training data. If the pre-training data contained any sensitive personal information, copyrighted material, or biased content, the model could potentially leak or amplify those issues in unintended ways when applied to new data.

In general, users should not treat the model outputs as authoritative or deploy the model in production use cases without extensive testing, monitoring, and human oversight. Responsible and limited use within the intended scope of document AI research is advised.

### Bias, Risks, and Limitations

Based on the provided references about the LiLT model (SCUT-DLVCLab/lilt-roberta-en-base), here are some potential issues and limitations:

Foreseeable harms and misunderstandings:
- The model is trained on the IIT-CDIP dataset which contains English documents. If applied to documents in other languages or domains without proper fine-tuning, the model may produce inaccurate or biased results. More testing is needed to understand cross-lingual and cross-domain generalization.
- [More Information Needed] on potential misuse cases or harms from applying the model to sensitive documents.

Technical limitations: 
- The complete pre-training process is very time-consuming. The ablation experiments only used a subset of data (2M documents for 5 epochs). Full pre-training likely requires significant compute resources.
- [More Information Needed] on model performance metrics, failure modes, and areas for improvement.

Sociotechnical considerations:
- Deploying the model for real-world applications would require carefully curating multilingual training data. Reference 7 notes this is a "heavy process". Responsible data sourcing, cleaning and governance is important.
- As an early research project, the model needs further development and empirical validation before use in high-stakes scenarios. Reference 6 welcomes feedback from the community.
- [More Information Needed] on fairness across languages, document types, and downstream use cases. Transparency on intended use and limitations is advised.

In summary, while LiLT presents a promising approach for language-independent document understanding, more information is needed to fully characterize its potential harms, limitations and sociotechnical implications. Careful testing and incremental deployment is recommended.

### Recommendations

Based on the provided references about the SCUT-DLVCLab/lilt-roberta-en-base model, here are some recommendations regarding foreseeable issues:

1. Language Limitations: While LiLT is designed to transfer knowledge from monolingual to multilingual structured document understanding, it is still primarily trained on English documents (IIT-CDIP dataset). More diverse language pre-training data may be needed to ensure robust performance across a wider range of languages. [More Information Needed] on specific language performance and limitations.

2. Bias and Fairness: [More Information Needed] The references do not provide information about potential biases in the training data or model outputs. It's important to assess and mitigate any biases related to language, document structure, or content that could lead to unfair or discriminatory results.

3. Privacy and Security: Structured documents often contain sensitive information. [More Information Needed] on how the model handles private data during training and inference, and what security measures are in place to protect user privacy.

4. Explainability and Transparency: The BiACM mechanism and other architectural choices should be clearly explained to users. [More Information Needed] on providing more interpretable outputs and explanations for the model's predictions, especially for high-stakes applications.

5. Generalization and Robustness: While LiLT shows promising results on benchmark datasets, its performance on real-world, diverse structured documents should be thoroughly evaluated. [More Information Needed] on testing the model's robustness to variations in document layout, noise, and adversarial examples.

6. Ethical Considerations: [More Information Needed] The references do not discuss potential misuse cases or ethical implications of the model. It's crucial to consider and address any foreseeable negative impacts, such as using the model for malicious purposes or perpetuating societal biases.

## Training Details

### Training Data

The SCUT-DLVCLab/lilt-roberta-en-base model is pre-trained on the IIT-CDIP Test Collection 1.0 dataset, which contains more than 6 million documents with over 11 million scanned document images. The dataset is pre-processed using the TextIn API to obtain text bounding boxes and strings.

[More Information Needed] for links to documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

For the SCUT-DLVCLab/lilt-roberta-en-base model, the preprocessing steps for the input data are as follows:

Tokenization:
- All text strings in the OCR results are first tokenized and concatenated as a sequence S_t by sorting the corresponding text bounding boxes from the top-left to bottom-right. 
- Special tokens [CLS] and [SEP] are added at the beginning and end of the sequence respectively.
- The sequence S_t is truncated or padded with extra [PAD] tokens until its length equals the maximum sequence length N, which is set to 512.

Resizing/Rewriting:
[More Information Needed]

Other preprocessing details:
- The text flow is initialized from the existing pre-trained English RoBERTa BASE model.
- For each token, the bounding box of the text string it belongs to is directly utilized, because the fine-grained token-level information is not always included in the results of some OCR engines.
- Special tokens [CLS], [SEP] and [PAD] are attached with (0,0,0,0,0,0), (1000,1000,1000,1000,0,0) and (0,0,0,0,0,0) respectively for their bounding box coordinates.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the SCUT-DLVCLab/lilt-roberta-en-base model:

- Optimizer: [More Information Needed]
- Learning rate: [More Information Needed] 
- Weight decay: [More Information Needed]
- Beta values (β1, β2): [More Information Needed]
- Learning rate schedule: [More Information Needed]
- Batch size: [More Information Needed] 
- Number of training epochs: [More Information Needed]
- Hardware: [More Information Needed]

Fine-tuning on FUNSD dataset:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 examples/run_funsd.py \
        --model_name_or_path lilt-roberta-en-base \
        --tokenizer_name roberta-base \
        --output_dir ser_funsd_lilt-roberta-en-base \
        --do_train \
        --do_predict \
        --max_steps 2000 \
        --per_device_train_batch_size 8 \
        --warmup_ratio 0.1 \
        --fp16
```
- Max steps: 2000
- Per device train batch size: 8
- Warmup ratio: 0.1
- Mixed precision (FP16) enabled

The model is initialized from the pre-trained RoBERTa-base model. The maximum sequence length is set to 512.

#### Speeds, Sizes, Times

Here are the details about the SCUT-DLVCLab/lilt-roberta-en-base model based on the provided references:

Model size: 293MB

Training data: Pre-trained on the IIT-CDIP Test Collection 1.0 dataset, which contains more than 6 million documents with more than 11 million scanned document images.

Training details:
- Optimizer: Adam 
- Learning rate: 2×10^-5
- Weight decay: 1×10^-2
- (β1, β2) = (0.9, 0.999)
- Batch size: 96
- Trained for 5 epochs on 4 NVIDIA A40 48GB GPUs

[More Information Needed] for:
- Throughput 
- Start and end training time
- Checkpoint sizes

The text flow is initialized from the pre-trained English RoBERTa BASE model. The maximum sequence length is set to 512.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the SCUT-DLVCLab/lilt-roberta-en-base model was evaluated on the following benchmarks and datasets:

1. FUNSD (Jaume et al., 2019)
2. CORD (Park et al., 2019) 
3. EPHOIE
4. RVL-CDIP (Lewis et al., 2006)
5. Multilingual XFUND (Xu et al., 2021b)

The model was evaluated in language-specific fine-tuning, cross-lingual zero-shot transfer, and multitask learning settings across eight languages.

#### Factors

Based on the provided references about the SCUT-DLVCLab/lilt-roberta-en-base model, here are some key characteristics that may influence the model's behavior:

Domain and Context:
- The model is designed for structured document understanding (SDU) tasks across multiple languages. It can handle monolingual and multilingual documents.
- It was pre-trained on the IIT-CDIP dataset containing 11 million English documents, allowing it to learn language-independent layout information.
- The model can be fine-tuned on various downstream SDU tasks such as key information extraction, document classification, etc.

Population Subgroups:
- The model's performance was evaluated on datasets in eight different languages, demonstrating its ability to transfer knowledge from English to other languages.
- [More Information Needed] regarding specific performance disparities across different language subgroups or document types.

Evaluation and Disaggregation:
- The model was evaluated on several widely-used monolingual datasets (FUNSD, CORD, EPHOIE, RVL-CDIP) and the multilingual XFUND benchmark.
- Multitask learning, where the model is simultaneously fine-tuned on all eight languages, further improved performance compared to language-specific fine-tuning.
- [More Information Needed] on detailed disaggregation of evaluation results across different factors like language, document type, layout complexity, etc. to uncover potential performance disparities.

In summary, while the provided references highlight the model's language-independent capabilities and strong performance across multiple languages and tasks, more specific information is needed to fully assess potential biases or disparities in performance across different subgroups or contexts.

#### Metrics

Based on the provided references, the main evaluation metrics used for the SCUT-DLVCLab/lilt-roberta-en-base model are:

1. F1 score: Used to evaluate the model's performance on both Semantic Entity Recognition (SER) and Relation Extraction (RE) tasks across different languages (Reference 9).

2. [More Information Needed] on specific tradeoffs between different errors and how they are balanced in the evaluation metrics.

The model is evaluated on various datasets, including:
- Monolingual datasets: FUNSD, CORD, EPHOIE, and RVL-CDIP (Reference 10)
- Multilingual dataset: XFUND (Reference 9)

The model's performance is compared against other models, such as XLM-R, InfoXLM, and LayoutXLM, in different settings:
- Language-specific fine-tuning (Reference 9)
- Zero-shot transfer learning (Reference 1)
- Multitask fine-tuning (Reference 11)

### Results

Based on the provided references, here are the evaluation results for the SCUT-DLVCLab/lilt-roberta-en-base model:

Datasets:
- The model was evaluated on several widely-used monolingual datasets, including FUNSD, CORD, EPHOIE, and RVL-CDIP (References 2, 8).

Performance:
- LiLT achieves competitive or even superior performance compared to existing language-specific SDU models such as LayoutLMv2 and DocFormer, especially on the FUNSD dataset with limited training samples (Reference 8).
- On the FUNSD dataset, the model achieves high F1 scores for both the SER (Semantic Entity Recognition) and RE (Relation Extraction) tasks (Reference 6).

Training Details:
- The model was pre-trained on 2M documents randomly sampled from IIT-CDIP for 5 epochs for ablation experiments (Reference 1).
- The model uses the proposed KPL (Key Point Localization) and CAI (Cross-modal Alignment and Interaction) tasks during pre-training, which improve the model performance substantially (Reference 3).

Evaluation Command for FUNSD:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 examples/run_funsd.py \
        --model_name_or_path lilt-roberta-en-base \
        --tokenizer_name roberta-base \
        --output_dir ser_funsd_lilt-roberta-en-base \
        --do_train \
        --do_predict \
        --max_steps 2000 \
        --per_device_train_batch_size 8 \
        --warmup_ratio 0.1 \
        --fp16
```

[More Information Needed] for specific evaluation metrics and results on CORD, EPHOIE, and RVL-CDIP datasets.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the SCUT-DLVCLab/lilt-roberta-en-base model:

- LiLT was evaluated on several widely-used monolingual datasets (FUNSD, CORD, EPHOIE, RVL-CDIP) and the multilingual XFUND benchmark. It achieved the highest F1 scores on both the SER and RE tasks for each language in XFUND while using only 11M monolingual data, outperforming models like XLM-R, InfoXLM and LayoutXLM.

- Ablation experiments showed that the Bi-ACM module significantly improved performance when transferring from monolingual to multilingual settings. The KPL and CAI pre-training tasks also substantially improved model performance, with CAI being more beneficial. 

- LiLT is flexible and can work with monolingual or multilingual plain text models for downstream tasks. Although designed for monolingual to multilingual transfer, it can also cooperate with monolingual textual models to achieve strong results.

- [More Information Needed] on specific quantitative results from the monolingual dataset evaluations and ablation studies.

In summary, LiLT demonstrates strong cross-lingual transfer capabilities and flexibility in working with different textual models while using less training data compared to prior approaches. The novel Bi-ACM module and pre-training tasks are key to its performance.

## Model Examination

Based on the provided references about the LiLT (Language-independent Layout Transformer) model, here is what I can infer about the explainability and interpretability of the SCUT-DLVCLab/lilt-roberta-en-base model:

The LiLT model is designed to be language-independent and can be pre-trained on monolingual data (e.g., IIT-CDIP dataset) and then adapted to other languages for structured document understanding tasks. This allows for better interpretability of the model's performance across different languages.

The layout flow (LiLT) component can be separated and combined with pre-trained textual models like RoBERTa during fine-tuning. This modular approach may help in understanding the contributions of the layout and textual components to the model's predictions.

However, the provided references do not go into detail about specific explainability techniques or tools used to interpret the model's decisions. [More Information Needed] on how the model's attention weights, feature importance, or other interpretability methods can be used to understand its behavior.

Additionally, [More Information Needed] on any visualizations or case studies that demonstrate how the model makes predictions on specific examples from the datasets mentioned (FUNSD, CORD, EPHOIE, RVL-CDIP, XFUND).

In summary, while the language-independent and modular design of LiLT-RoBERTa-en-base may contribute to its interpretability, more information is needed on specific explainability techniques and examples to fully address this aspect of the model.

## Environmental Impact

- **Hardware Type:** According to the reference, the LiLT BASE model is trained on 4 NVIDIA A40 48GB GPUs, as stated in the following excerpt:

"We set the batch size as 96 and train LiLT BASE for 5 epochs on the IIT-CDIP dataset using 4 NVIDIA A40 48GB GPUs."

Therefore, the hardware type that the model SCUT-DLVCLab/lilt-roberta-en-base is trained on is 4 NVIDIA A40 48GB GPUs.
- **Software Type:** Based on the references provided, the model SCUT-DLVCLab/lilt-roberta-en-base is pre-trained on the IIT-CDIP Test Collection 1.0 dataset, which contains over 11 million scanned document images. Specifically, reference 3 states:

"We pre-train LiLT on the IIT-CDIP Test Collection 1.0 (Lewis et al., 2006), which is a large-scale scanned document image dataset and contains more than 6 million documents with more than 11 million scanned document images."

So the software type that the model is trained on is scanned document images.
- **Hours used:** Based on the references provided, the LiLT BASE model was pre-trained for 5 epochs on the IIT-CDIP dataset using 4 NVIDIA A40 48GB GPUs (Reference 1). However, the total amount of time used for this training process is not explicitly mentioned.

For the ablation experiments (Reference 6), LiLT BASE was pre-trained with 2M documents randomly sampled from IIT-CDIP for 5 epochs, but the training time is not specified.

Therefore, to provide the exact amount of time used to train the SCUT-DLVCLab/lilt-roberta-en-base model, [More Information Needed].
- **Cloud Provider:** Based on the provided references, the model SCUT-DLVCLab/lilt-roberta-en-base was trained using 4 NVIDIA A40 48GB GPUs. However, there is no explicit mention of the cloud provider used for training this model.

[More Information Needed] on the specific cloud provider used for training SCUT-DLVCLab/lilt-roberta-en-base.
- **Carbon Emitted:** [More Information Needed]

Based on the provided references, there is no specific information about the amount of carbon emitted during the training of the SCUT-DLVCLab/lilt-roberta-en-base model. The references mention some training details like batch sizes, learning rates, and the number of training steps, but do not provide any data related to carbon emissions. To accurately report the carbon footprint, more details would be needed, such as the hardware used for training, total training time, and energy consumption.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The model SCUT-DLVCLab/lilt-roberta-en-base has a parallel dual-stream Transformer architecture, consisting of a text flow and a layout flow. The text flow processes the tokenized text sequence, while the layout flow processes the 2D position sequence constructed from the text bounding boxes. The two flows share a similar structure, but the layout flow has reduced hidden size and intermediate size for computational efficiency. The text and layout embeddings are summed with their respective 1D and 2D positional embeddings, and then fed into the Transformer layers for cross-modality interaction.

Objectives:
The model is pre-trained with the following objectives:
1. Masked Visual-Language Modeling (MVLM): Randomly masks some input tokens and asks the model to recover them using the encoded features from both text and layout flows.
2. Key Point Location (KPL): [More Information Needed]
3. Cross-modal Alignment Identification (CAI): [More Information Needed]

During fine-tuning, the layout flow (LiLT) can be separated and combined with off-the-shelf pre-trained textual models like RoBERTa to tackle downstream tasks.

### Compute Infrastructure

Based on the provided references, the compute infrastructure used for training the SCUT-DLVCLab/lilt-roberta-en-base model is:

4 NVIDIA A40 48GB GPUs

The model was trained for 5 epochs on the IIT-CDIP dataset with a batch size of 96.

For fine-tuning on the FUNSD dataset, the following compute setup was used:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 examples/run_funsd.py \
        --model_name_or_path lilt-roberta-en-base \
        --tokenizer_name roberta-base \
        --output_dir ser_funsd_lilt-roberta-en-base \
        --do_train \
        --do_predict \
        --max_steps 2000 \
        --per_device_train_batch_size 8 \
        --warmup_ratio 0.1 \
        --fp16
```

[More Information Needed] on the exact compute infrastructure used for fine-tuning.

## Citation

```
@misc{jiapeng-lilt,
    author = {Jiapeng Wang and
              Lianwen Jin and
              Kai Ding},
    title  = {LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding},
    url    = {https://arxiv.org/pdf/2202.13669.pdf}
}
```

