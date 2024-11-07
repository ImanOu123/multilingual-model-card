# Model Card for Salesforce/codet5-base

CodeT5 is a pre-trained encoder-decoder model for code understanding and generation tasks. It builds on the T5 architecture and leverages token type information in code, such as developer-assigned identifiers, to derive generic representations for both programming language (PL) and natural language (NL).

## Model Details

### Model Description

Model Details:
- Architecture: CodeT5 is built on an encoder-decoder framework with the same architecture as T5, aiming to derive generic representations for programming language (PL) and natural language (NL). It extends the de-noising Seq2Seq objective in T5 with two identifier tagging and prediction tasks.
- Training Data: CodeT5 is pre-trained on the CodeSearchNet data consisting of both unimodal (PL-only) and bimodal (PL-NL) data on six programming languages. Extra C/C# data is collected from open-source Github repositories. 
- Training Procedure: The model is pre-trained with the denoising objective for 100 epochs and bimodal dual training for a further 50 epochs. A span masking objective similar to T5 is used that randomly masks spans with arbitrary lengths.
- Model Parameters: CodeT5-base has 220M parameters.
- Important Disclaimers: [More Information Needed]

- **Developed by:** Yue Wang; Weishi Wang; Shafiq Joty; Steven C H Hoi; Salesforce Research
- **Funded by:** Based on the references provided, the model Salesforce/codet5-base was funded by Salesforce Research. Specifically, the model was introduced in papers by researchers from Salesforce Research, including Yue Wang, Weishi Wang, Shafiq Joty, and Steven C H Hoi (Reference 1).
- **Shared by:** Based on the references provided, the main contributors who made the model Salesforce/codet5-base available online are:

Yue Wang, Weishi Wang, Shafiq Joty, and Steven C H Hoi from Salesforce Research.

The references also thank Akhilesh Deepak Gotmare, Amrita Saha, Junnan Li, Chen Xing, and Kathy Baxter for valuable discussions and ethical review, as well as anonymous reviewers for their feedback. However, it's not clearly stated if they directly contributed to making the model available online.

[More Information Needed] on the specific GitHub repo where the model was made available.
- **Model type:** CodeT5 is a pre-trained encoder-decoder Transformer model that employs denoising sequence-to-sequence pre-training on both unimodal (programming language-only) and bimodal (programming language-natural language) data to support code understanding and generation tasks.
- **Language(s):** The CodeT5 model processes both natural language (NL) and programming language (PL) to enable tasks such as text-to-code generation, code autocompletion, and code summarization.
- **License:** The code for the Salesforce/codet5-base model is released under the BSD-3 License. This is mentioned in the references:

The code is released under the BSD-3 License (see `LICENSE.txt` for details)

However, a direct link to the license file is not provided in the given references, so for the link, I would say:

[More Information Needed]
- **Finetuned from model:** The Salesforce/codet5-base model is fine-tuned from the T5 model architecture developed by Google. Specifically, it is based on the Huggingface PyTorch implementation of T5.

Relevant references from the provided information:

"Our CodeT5 builds on an encoder-decoder framework with the same architecture as T5 (Raffel et al., 2020)." (Reference 1)

"We build CodeT5 based on Huggingface's T5 (Raffel et al., 2020) PyTorch implementation..." (Reference 5)

Link to the T5 base model: [More Information Needed]
### Model Sources

- **Repository:** https://github.com/huggingface/tokenizers
- **Paper:** https://arxiv.org/pdf/2109.00859.pdf
- **Demo:** Based on the provided references, the link to the demo of the model Salesforce/codet5-base is not directly mentioned. The references mention that Salesforce built an AI coding assistant demo using CodeT5 as a VS Code plugin, but no specific link is provided for the Salesforce/codet5-base model demo.

[More Information Needed]
## Uses

### Direct Use

Based on the provided references, CodeT5 is a pre-trained encoder-decoder model that can be fine-tuned on various code-related tasks such as code summarization, generation, translation, and refinement. The references do not provide specific information on how to use the Salesforce/codet5-base model without fine-tuning, post-processing, or plugging into a pipeline.

[More Information Needed]

The references mention that CodeT5 models can be deployed as an AI-powered coding assistant, such as a VS Code plugin, to boost developer productivity. However, no code snippets or details are provided on how to use the model directly without additional steps.

To provide a complete answer, more information would be needed on the specific API or interface provided by the Salesforce/codet5-base model for direct usage without fine-tuning or additional processing.

### Downstream Use

The Salesforce/codet5-base model can be used for various code intelligence tasks when fine-tuned or integrated into a larger ecosystem or app:

1. Code defect detection: It can predict whether a given code snippet is vulnerable or not, outperforming baselines like GraphCodeBERT.

2. Code clone detection: It can measure the similarity between two code snippets to determine if they have the same functionality, achieving comparable results to SOTA models. 

3. Code translation: It can migrate code from one programming language to another, such as translating functions from Java to C# and vice versa.

4. Code refinement: It can automatically fix bugs in code, significantly outperforming baselines especially on more challenging medium-length functions.

5. AI coding assistant: At Salesforce, CodeT5 powers an AI coding assistant VS Code plugin that provides capabilities like:
[More Information Needed]

By leveraging CodeT5 for these tasks, software developers can boost their productivity, reduce working time, and potentially lower computation and operational costs by catching bugs early that could degrade system performance or cause crashes.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the Salesforce/codet5-base model may be misused and what users should not do with the model:

1. Users should not rely solely on the model's code generation outputs without further correctness and security checking by domain experts. The model's outputs should only be taken as references, as they may produce vulnerable code that could enable more advanced malware development if misused. (Reference 1, 3)

2. The model should not be used in high-stakes applications without appropriate documentation and consideration of ethical implications. Users are encouraged to inform Salesforce about their applications and use cases. (Reference 6, 9)

3. Users should be aware of potential automation bias, especially developers who may over-rely on the model's code suggestions. The generated code may superficially appear correct but not align with the developer's intent, leading to safety issues and increased debugging time if adopted without proper review. (Reference 4)

4. [More Information Needed] regarding specific examples of how the model might encode sensitive information or stereotypes from the training data, and how this could lead to misuse.

In summary, the Salesforce/codet5-base model should be used responsibly, with its outputs treated as references rather than fully trusted code. Appropriate documentation, human oversight, and consideration of ethical implications are necessary, especially in high-stakes applications. Misuse could lead to the generation of vulnerable or biased code.

### Bias, Risks, and Limitations

Based on the provided references about the Salesforce/codet5-base model, here are some known or foreseeable issues:

Potential Harms and Misuse:
- The model could potentially produce vulnerable code that may enable more advanced malware development if deliberately misused. (Reference 1)
- The model's outputs should only be taken as references and require domain experts to further check for correctness and security, as incorrect code suggestions adopted by developers could lead to significant safety issues. (Reference 4)

Dataset and Automation Bias:
- The training datasets may encode stereotypes like race and gender from text comments or source code. (Reference 2) 
- Developers may over-rely on the model's outputs due to automation bias, even if the generated code does not fully align with their intent. (Reference 5)

Privacy and Security:
- The model may have encoded sensitive information like personal addresses or identification numbers from the training data, despite efforts to clean the data. (Reference 3)

Computational Cost and Environmental Impact: 
- Training the model required non-trivial computational resources, though efforts were made to offset the carbon footprint. (References 6, 7)

[More Information Needed] on the specific technical limitations of the model and how the multi-task pre-training setup impacts downstream performance. The provided references do not go into enough detail to draw clear conclusions there.

Overall, key issues to consider are potential misuse, dataset biases, privacy concerns, environmental impact, and the need for human oversight of the model's outputs. More research is needed to fully characterize the model's limitations.

### Recommendations

Based on the provided references about the Salesforce/codet5-base model, here are some recommendations regarding foreseeable issues:

1. Computational cost: While the model size is much smaller than some recent large language models, pre-training still requires non-trivial computational resources. Be mindful of the computational costs when deploying and using the model.

2. Correctness and security: Model outputs should be treated as references only and reviewed by domain experts to ensure correctness and security before use, as incorrect code suggestions could lead to bugs or safety issues.

3. Automation bias: Developers should be cautious not to over-rely on the model's code suggestions, as they may superficially appear correct but not align with the developer's true intent. Carefully review generated code before adopting it.

4. Security implications: Although data cleaning was performed, the model may have encoded sensitive information from the training data sourced from public GitHub repositories. Be aware of potential data leakage risks.

5. Dataset bias: The training datasets could encode stereotypes from text comments or source code. [More Information Needed] on specific biases present and mitigation strategies.

6. Responsible deployment: When deploying CodeT5 as an AI coding assistant, ensure appropriate human oversight, communicate the model's limitations to users, and have processes to monitor for and address unintended consequences.

In summary, while CodeT5 shows promising results for aiding software development, care should be taken in its deployment to manage computational costs, ensure output correctness and security, mitigate automation bias and potential dataset biases, and use it responsibly as an assistive tool with human oversight.

## Training Details

### Training Data

The training data of the model Salesforce/codet5-base consists of the CodeSearchNet dataset (Husain et al., 2019) which includes both unimodal (programming language-only) and bimodal (programming language-natural language) data on six programming languages, as well as additional C/C# data collected from open-source Github repositories. [More Information Needed]

### Training Procedure

#### Preprocessing

Based on the references provided, here are the details about preprocessing for the data of the Salesforce/codet5-base model:

Tokenization:
The model employs a Byte-Pair Encoding (BPE) tokenizer, as mentioned in reference 2: "Tokenization is a key ingredient for the success of pre-trained language models like BERT and GPT. They often employ a Byte-Pair Encoding (BPE) tokenizer (Sennrich et al., 2016)".

Input Format:
For NL-PL bimodal inputs, the natural language (NL) and programming language (PL) sequences are concatenated with a delimiter token [SEP] and represented in the following format (reference 5):
x = ([CLS], w1, ..., wn, [SEP], c1, ..., cm, [SEP])

Identifier Extraction:
To obtain the identifier labels from code, the tree-sitter library is used to convert the PL into an abstract syntax tree and extract its node type information. Reserved keywords for each PL are filtered out from the identifier list (reference 8).

[More Information Needed] on specific details of resizing/rewriting of the input data, as it is not clearly mentioned in the given references.

#### Training Hyperparameters

Here are the training hyperparameters for the Salesforce/codet5-base model based on the provided references:

- Model size: 220M parameters (CodeT5-base)
- Maximum source sequence length: 512 
- Maximum target sequence length: 256
- Mixed precision: FP16
- Batch size: 1024
- Peak learning rate: 2e-4 with linear decay
- Pre-training:
  - Denoising objective for 100 epochs
  - Bimodal dual training for further 50 epochs 
- Hardware: 16 NVIDIA A100 GPUs with 40G memory
- Total training time: 12 days

[More Information Needed] on the following:
- Optimizer 
- Weight decay
- Warmup steps
- Validation frequency
- Checkpoint saving frequency
- Vocabulary size
- Dropout

The model was fine-tuned on downstream tasks, but [More Information Needed] on the exact hyperparameters used for each task.

#### Speeds, Sizes, Times

Here are the details about the Salesforce/codet5-base model based on the provided references:

Model size: CodeT5-base has 220M parameters.

Training time: The total training time for CodeT5-base is 12 days on a cluster of 16 NVIDIA A100 GPUs with 40G memory. It was pre-trained with the denoising objective for 100 epochs and bimodal dual training for a further 50 epochs.

[More Information Needed] on the following:
- Throughput 
- Start and end time of training
- Checkpoint sizes

The model significantly outperforms prior methods on various code understanding and generation tasks. Key results include:
- CodeT5-base achieves around 4.7 points improvement on CodeBLEU over PLBART on code generation, indicating better comprehension of code syntax and semantics.
- On the code defect detection task, CodeT5-base yields a 2.6 accuracy score improvement over PLBART. 
- For clone detection, CodeT5-base achieves comparable results to state-of-the-art models like GraphCodeBERT and PLBART.

The pre-trained CodeT5 models have been released at https://github.com/salesforce/CodeT5. They can be deployed as an AI-powered coding assistant to boost developer productivity, such as in a VS Code plugin demo at Salesforce providing capabilities like [More Information Needed].

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the Salesforce/codet5-base model was evaluated on the following benchmarks and datasets:

1. CodeXGLUE benchmark, which includes various code understanding and generation tasks:
   - Code summarization: Dataset consists of six programming languages (Ruby, JavaScript, Go, Python, Java, and PHP) from CodeSearchNet.
   - Code generation
   - Code translation: Translating between Java and C#.
   - Code refinement
   - Defect detection: C dataset provided by [More Information Needed].
   - Clone detection: Java dataset provided by Wang et al. (2020).

2. The model was pre-trained on the CodeSearchNet dataset, which contains both unimodal (programming language-only) and bimodal (programming language-natural language) data on six programming languages. Additional C/C# data was collected from open-source GitHub repositories.

[More Information Needed] on the specific metrics used for each task, except for the smoothed BLEU-4 metric mentioned for the code summarization task.

#### Factors

Based on the provided references about the Salesforce/codet5-base model, here are some foreseeable characteristics that may influence the model's behavior:

Domain and Context:
- The model is pre-trained on existing code corpora including CodeSearchNet and a small fraction of Google BigQuery, which were originally collected from public Github repositories. This means the model's performance may be influenced by the programming languages, coding styles, and problem domains represented in that training data.
- The model is designed to support both code understanding tasks (e.g. code defect detection, clone detection) and code generation tasks (e.g. PL-NL, NL-PL, PL-PL). Its performance may vary depending on the specific task it is applied to.

Population Subgroups:
- Developers who tend to over-rely on the model's code generation outputs may be susceptible to automation bias, accidentally adopting incorrect code suggestions that don't align with their intent. More cautious developers may fare better.
- [More Information Needed] on how the model's performance may vary across developers with different levels of coding experience, familiarity with specific programming languages, etc.

Other Factors:
- The model has a much smaller size (220M parameters) compared to some other large language models like Codex (12B parameters). While this makes it more computationally efficient, it may limit the model's knowledge and generation capabilities compared to larger models.
- [More Information Needed] on how the model's performance may be impacted by different decoding strategies, generation prompt formats, etc.

To uncover potential performance disparities, it would be valuable to disaggregate the model's evaluation across different programming languages, problem domains, developer subgroups, and task formats. More targeted testing and analysis is needed to fully characterize the model's strengths, weaknesses, and contextual influences.

#### Metrics

Based on the references provided, the following metrics will be used to evaluate the Salesforce/codet5-base model:

For code refinement tasks:
- Exact match (EM) score will be the main focus, as BLEU scores can be misleadingly high due to large overlap between source and target code. (Reference 1)

For code translation tasks:
- BLEU score
- CodeBLEU score (Reference 2 mentions CodeT5-base achieves ~4.7 point improvement over PLBART on this metric)
- [More Information Needed] on other specific metrics used

For code understanding tasks like defect detection and clone detection:
- F1 score 
- Accuracy
(Reference 5)

The references also suggest that BLEU score alone may not be a perfect metric for code generation tasks, as sometimes a higher BLEU score doesn't necessarily indicate better code functionality and readability (Reference 4). So a combination of metrics, as well as qualitative analysis of model outputs, will be important for a comprehensive evaluation.

### Results

Based on the references provided, here are the evaluation results for the Salesforce/codet5-base model:

Factors:
- Uses identifier-aware denoising pre-training objectives like masked span prediction (MSP), identifier tagging (IT), and masked identifier prediction (MIP) to better capture syntactic and semantic information from code
- Pre-trained on a large amount of Python and Java data (over 100 times more than some baselines)
- Employs a bimodal dual generation approach for better natural language-programming language alignment

Metrics:
- Significantly outperforms prior state-of-the-art models like PLBART on a broad set of CodeXGLUE downstream tasks
- Achieves around 4.7 points improvement on CodeBLEU over PLBART on code generation tasks, indicating better comprehension of code syntax and semantics
- Outperforms encoder-only models (RoBERTa, CodeBERT, DOBF) and the encoder-decoder PLBART on code-to-code generation tasks like code translation and refinement
- Boosts overall performance by over 1.2 absolute points compared to PLBART when scaling up to the base model size

[More Information Needed] for the specific scores on each downstream task and the exact model sizes used in the comparisons.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the Salesforce/codet5-base model:

- CodeT5-base significantly outperforms prior SOTA models like PLBART across various code-related tasks, achieving over 1.2 absolute points improvement in overall performance. (Reference 1)

- On code generation tasks, CodeT5-base pushes SOTA results across three metrics, with around 4.7 points improvement on CodeBLEU over PLBART. This indicates CodeT5 better comprehends code syntax and semantics. (Reference 2) 

- For code-to-code generation tasks like code translation, CodeT5-base achieves consistent improvements over PLBART across various metrics for translating between Java and C#. (Reference 5)

- On code understanding tasks, CodeT5-base yields 2.6 accuracy score improvement over PLBART on the defect detection task. For clone detection, it achieves comparable results to SOTA models like GraphCodeBERT and PLBART. (Reference 4)

- The identifier-aware pre-training enables CodeT5 to better distinguish and leverage identifier information. An example shows CodeT5 can correctly generate a function while a variant without this pre-training fails to recover some identifiers. (Reference 6)

[More Information Needed] for code examples demonstrating the model's performance, as no direct code blocks are provided in the references.

In summary, the Salesforce/codet5-base model achieves state-of-the-art performance on a broad range of code understanding and generation tasks, outperforming strong baselines. The identifier-aware pre-training is shown to be beneficial.

## Model Examination

Based on the provided references about the Salesforce/codet5-base model, here is the experimental section on explainability/interpretability:

The CodeT5 model, including codet5-base, demonstrates improved code understanding and generation capabilities compared to prior methods. This can be attributed to the identifier-aware denoising pre-training and effective utilization of bimodal training data. By capturing both syntactic and semantic information from code, CodeT5 exhibits better alignment between natural language and programming language.

Experiments reveal that CodeT5 can correctly generate exact functions, while variants without identifier-aware pre-training fail to recover certain identifiers. This indicates that the identifier-aware denoising pre-training enables CodeT5 to better distinguish and leverage identifier information.

[More Information Needed] on specific techniques or metrics used for explainability and interpretability of the codet5-base model.

[More Information Needed] on any visualizations or examples demonstrating the model's decision-making process or attention mechanisms.

In summary, the identifier-aware pre-training and bimodal training data contribute to CodeT5's improved code understanding and generation abilities. However, more detailed information is needed regarding the specific explainability and interpretability techniques applied to the codet5-base model.

## Environmental Impact

- **Hardware Type:** [More Information Needed]

The provided references do not contain information about the specific hardware type that the Salesforce/codet5-base model was trained on. More details would be needed from the model developers or associated research papers to determine the training hardware.
- **Software Type:** Based on the references provided, the CodeT5 model is trained on various programming languages including:

- Ruby, JavaScript, Go, Python, Java, and PHP (from reference 4)
- C (from reference 1) 
- Java (from reference 1)

So in summary, the Salesforce/codet5-base model is trained on code from multiple programming language types including Ruby, JavaScript, Go, Python, Java, PHP and C. The references don't specify the exact percentage breakdown of each language in the training data.
- **Hours used:** According to the references provided, the total training time for CodeT5-base is 12 days on a cluster of 16 NVIDIA A100 GPUs with 40G memory.
- **Cloud Provider:** The model Salesforce/codet5-base was trained on Google Cloud Platform, which purchases carbon credits to offset the carbon footprint of the training process. Specifically, training CodeT5-base produced around 49.25 kg CO2, which was totally offset by the cloud provider.
- **Carbon Emitted:** According to the references provided, training the CodeT5-base model produced around 49.25 kg of CO2 emissions. Specifically, reference 1 states:

"training CodeT5-base produced around 49.25 kg CO 2 which was totally off-set by the provider."

So the amount of carbon emitted when training the Salesforce/codet5-base model was approximately 49.25 kg of CO2.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
CodeT5 is built on an encoder-decoder framework with the same architecture as T5. It employs a denoising sequence-to-sequence (Seq2Seq) pre-training objective.

Pre-training Objective:
CodeT5 extends the de-noising Seq2Seq objective in T5 by proposing two identifier tagging and prediction tasks to enable the model to better leverage the token type information from programming languages, which are the identifiers assigned by developers. It also leverages the natural language-programming language pairs that are naturally available in source code to learn a better cross-modal alignment.

[More Information Needed] on specific model sizes and hyperparameters used for pre-training Salesforce/codet5-base.

### Compute Infrastructure

Based on the provided references, here is the compute infrastructure information for the Salesforce/codet5-base model:

- Model was pre-trained on a cluster of 16 NVIDIA A100 GPUs with 40G memory each
- Used mixed precision FP16 to accelerate pre-training 
- Batch size set to 1024
- Peak learning rate of 2e-4 with linear decay
- Pre-trained for 100 epochs with denoising objective and further 50 epochs with bimodal dual training
- Total pre-training time was 12 days

[More Information Needed] on the exact compute infrastructure used for fine-tuning and inference.

## Citation

```
@misc{yue-codet,
    author = {Yue Wang and
              Weishi Wang and
              Shafiq Joty and
              Steven C H Hoi and
              Salesforce Research},
    title  = {CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation},
    url    = {https://arxiv.org/pdf/2109.00859.pdf}
}
```

