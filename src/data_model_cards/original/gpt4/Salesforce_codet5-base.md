# Model Card for Salesforce/codet5-base

Salesforce/codet5-base is a pre-trained encoder-decoder model designed for code understanding and generation tasks, leveraging token type information and developer-assigned identifiers to enhance semantic comprehension of programming languages. It builds on the T5 architecture and introduces novel pre-training objectives tailored for programming languages, demonstrating significant improvements over prior methods in both code understanding and generation tasks.

## Model Details

### Model Description

Model Name: Salesforce/codet5-base

Model Architecture:
Salesforce/codet5-base is built on an encoder-decoder framework similar to T5 (Raffel et al., 2020). It is designed to understand and generate programming language (PL) and natural language (NL) by leveraging pre-training on unlabeled source code. The model introduces identifier tagging and prediction tasks to improve its understanding of token type information, particularly identifiers in programming languages.

Training Procedures:
- Pre-training was conducted on the CodeSearchNet dataset, which includes unimodal (PL-only) and bimodal (PL-NL) data across six programming languages, with additional C/C# data from open-source GitHub repositories.
- The model was fine-tuned on the CodeXGLUE benchmark, covering tasks like code defect detection, clone detection, code summarization, generation, translation, and refinement.
- A mixed precision of FP16 was used to accelerate pre-training, with a batch size of 1024 and a peak learning rate of 2e-4 with linear decay.
- The model was pre-trained with a denoising objective for 100 epochs and bimodal dual training for an additional 50 epochs.
- For multi-task learning, balanced sampling was used to mitigate bias towards high-resource tasks, and language IDs were added for bimodal data points to improve NL-PL alignment.

Parameters:
- CodeT5-base has 220 million parameters.
- The maximum source and target sequence lengths are set to 512 and 256, respectively.

Important Disclaimers:
- Extensive experiments have shown that CodeT5 achieves state-of-the-art results on fourteen sub-tasks in the CodeXGLUE benchmark.
- The model demonstrates improved code semantics capture and benefits NL↔PL tasks with its identifier-aware pre-training and bimodal dual generation.
- The model's performance on understanding and generation tasks may vary, and specific transfer learning strategies are employed for each task type.
- The effectiveness of the model in real-world applications, such as an AI-powered coding assistant, may depend on the specific use case and integration.

[More Information Needed]: For any additional disclaimers or limitations that may not have been explicitly mentioned in the provided references.

- **Developed by:** Yue Wang; Weishi Wang; Shafiq Joty; Steven C H Hoi; Salesforce Research
- **Funded by:** The references provided do not explicitly mention the people or organizations that fund the project for the model Salesforce/codet5-base. However, since the model and research are attributed to Salesforce Research, it can be inferred that Salesforce is the organization behind the funding and development of the project.

If there are other specific funders or contributors to the project's funding, that information is not included in the provided references, and therefore, for those details, the answer would be "[More Information Needed]".
- **Shared by:** The contributors that made the model Salesforce/codet5-base available online as a GitHub repo include Yue Wang, Weishi Wang, Shafiq Joty, and Steven C H Hoi from Salesforce Research. Additionally, the team thanks Akhilesh Deepak Gotmare, Amrita Saha, Junnan Li, and Chen Xing for valuable discussions and Kathy Baxter for the ethical review. They also acknowledge their anonymous reviewers for insightful feedback on their paper.
- **Model type:** The Salesforce/codet5-base is a unified encoder-decoder model pre-trained on unimodal and bimodal data using a novel identifier-aware pre-training method, supporting both code-related understanding and generation tasks in a multi-task learning setting.
- **Language(s):** The model Salesforce/codet5-base processes natural human language (NL) in conjunction with programming languages (PL) for various code-related tasks, including understanding and generation.
- **License:** The license used for the model Salesforce/codet5-base is the BSD-3 License. However, there is no direct link provided to the license in the provided references. Typically, the `LICENSE.txt` file would be included in the repository where the code is hosted. Since the code is mentioned to be hosted on GitHub, you would normally find the license file there. 

For the exact link to the license, you would need to visit the GitHub repository for Salesforce/codet5-base. If the standard naming convention is followed, the license file would be accessible at a URL similar to this: `https://github.com/salesforce/CodeT5/blob/main/LICENSE.txt`. However, without a direct link in the provided references, I must say [More Information Needed] for the exact URL.
- **Finetuned from model:** The model Salesforce/codet5-base is fine-tuned from the T5 (Text-to-Text Transfer Transformer) model. The specific reference to the base model is found in point 4, which states that CodeT5 is built based on Huggingface's T5 PyTorch implementation. Here is the link to the T5 model on Huggingface: [T5 model](https://huggingface.co/t5-base).
### Model Sources

- **Repository:** https://github.com/huggingface/tokenizers
- **Paper:** https://arxiv.org/pdf/2109.00859.pdf
- **Demo:** The demo of the model Salesforce/codet5-base is not directly linked in the provided references. However, reference 6 mentions that at Salesforce, an AI coding assistant demo using CodeT5 was built as a VS Code plugin. To access the demo, one might need to look for the VS Code plugin marketplace or Salesforce's official channels for such tools. Since the exact link is not provided in the references, the answer is "[More Information Needed]".
## Uses

### Direct Use

Salesforce/codet5-base is a pre-trained model that has been designed to understand and generate code across various programming languages. It can be used without fine-tuning for tasks where the pre-training objectives align closely with the desired application. For example, since the model has been pre-trained with an identifier-aware denoising objective and bimodal dual generation, it can be used for tasks such as code summarization, code generation, and natural language to code translation, as long as the tasks do not deviate significantly from the pre-training setup.

However, without fine-tuning, the model's performance might not be optimal for specific downstream tasks that require domain-specific knowledge or a particular style of code. The pre-trained model is likely to perform best on tasks that are similar to the pre-training conditions, such as recovering masked code spans or converting between comments and code snippets.

To use Salesforce/codet5-base without fine-tuning, you would typically load the model and tokenizer from Huggingface's Transformers library and then use it to generate predictions for your input data. Here's a conceptual Python code snippet to demonstrate this (note that actual code execution requires a suitable environment with the necessary libraries installed):

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("Salesforce/codet5-base")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")

# Example input text
input_text = "def hello_world(): # Write a function that prints 'Hello, World!'"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate the output
outputs = model.generate(input_ids)

# Decode the generated text
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(decoded_output)
```

This code snippet is a general example of how to use a pre-trained model from Huggingface's Transformers library. It does not require fine-tuning, post-processing, or plugging into a pipeline. However, for specific use cases or to achieve the best performance on a particular task, fine-tuning and post-processing might be necessary.

### Downstream Use

The Salesforce/codet5-base model is a versatile deep learning model that can be fine-tuned for a variety of code-related tasks, both for generation and understanding. When fine-tuned for a specific task, the model adapts to the nuances of that task, leveraging its pre-trained knowledge to perform at a high level.

For generation tasks, such as code translation and code refinement, the model can be fine-tuned to translate code from one programming language (PL) to another or to fix bugs in a given code snippet. For example, it can translate functions from Java to CSharp and vice versa, or convert a buggy Java function into a correct one. The model's performance on these tasks can be evaluated using metrics like BLEU-4 and exact match.

For understanding tasks, such as defect detection and clone detection, the model can be fine-tuned to predict whether a code is vulnerable or to measure the similarity between two code snippets, respectively. These tasks can be evaluated using F1 score and accuracy.

In a larger ecosystem or app, CodeT5 can be integrated as an AI-powered coding assistant to boost the productivity of software developers. For instance, at Salesforce, CodeT5 has been deployed as a Visual Studio Code plugin to provide capabilities such as code summarization, defect detection, and code translation.

To notify the model of the task it is dealing with, task control codes are prepended to the source inputs. For example, for code-to-code translation from Java to CSharp, the source prompt "Translate Java to CSharp:" is used.

Here is a conceptual example of how the model could be used when fine-tuned for the task of code translation from Java to CSharp:

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("Salesforce/codet5-base-finetuned-java-to-csharp")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base-finetuned-java-to-csharp")

# Example Java code to translate
java_code = "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello, World!\"); } }"

# Encode the Java code along with the task prompt
input_ids = tokenizer.encode("Translate Java to CSharp: " + java_code, return_tensors="pt")

# Generate the CSharp code
outputs = model.generate(input_ids)
translated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translated_code)
```

Please note that the above code snippet is a conceptual example and assumes the existence of a fine-tuned model specifically for Java to CSharp translation. The actual implementation may require additional steps and considerations, such as handling of the model's input and output formats, and integration with the development environment.

[More Information Needed] for any specific code snippets directly related to the tasks mentioned in the references, as none were provided.

### Out-of-Scope Use

The Salesforce/codet5-base model, as a powerful tool for code generation and understanding, has the potential to be misused in various ways. Users should be aware of the following foreseeable misuses and refrain from engaging in such activities:

1. **Sensitive Information Leakage**: Despite efforts to clean the training data, there is a possibility that the model may have encoded sensitive information. Users should not use the model to generate code that may inadvertently expose or exploit personal data or confidential information.

2. **Malware Development**: The model's ability to generate code could be misused to create or enhance malware. Users must not use Salesforce/codet5-base for developing harmful software or for any activities that compromise the security of individuals or organizations.

3. **Bias Propagation**: Given that the training datasets may contain biases, users should not use the model in contexts where these biases could lead to discriminatory practices or reinforce stereotypes, particularly in relation to race, gender, or other protected characteristics.

4. **Automation Bias**: Users should avoid overreliance on the model's output without critical evaluation. The model may produce code that appears correct but does not align with the intended functionality, which could lead to errors if used without proper verification.

5. **High-Stakes Applications**: For applications where incorrect code could lead to significant harm, such as in medical devices, transportation systems, or financial services, users should exercise extreme caution and not solely depend on the model's output without rigorous testing and human oversight.

In summary, users of Salesforce/codet5-base should ensure that they are using the model responsibly, with an awareness of its limitations and potential for misuse. They should adhere to ethical guidelines, validate the model's output thoroughly, and consider the societal implications of their use cases. Additionally, users developing high-stakes applications should follow appropriate documentation and best practices as suggested by the Partnership on AI and other relevant bodies.

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the model Salesforce/codet5-base include:

1. **Security Implications**: Despite multiple rounds of data cleaning, there is a possibility that the pre-trained model may encode sensitive information from the training data, such as personal addresses or identification numbers. This could lead to privacy breaches if such information is inadvertently included in the model's outputs.

2. **Automation Bias**: There is a risk of automation bias where developers might over-rely on the model-generated outputs. This could result in the adoption of incorrect code suggestions that appear superficially correct but do not align with the developer's intent, leading to longer debugging times and potential safety issues.

3. **Dataset Bias**: The training datasets, sourced from open-source GitHub repositories, could encode stereotypes related to race, gender, or other biases present in the user-written comments or the source code itself. This could perpetuate existing biases and stereotypes in the model's outputs.

4. **Misuse Potential**: The non-deterministic nature of generation models like CodeT5 means that it could produce vulnerable code that might be exploited for malicious purposes, such as malware development.

5. **Computational Cost**: While the model's computational cost for pre-training is significant, efforts have been made to design experiments to save unnecessary computation costs. However, the environmental and economic impact of such computational resources should be considered.

6. **Technical Limitations**: The model's performance is based on the quality of the training data and the pre-training objectives. While CodeT5 has shown to outperform prior methods, it is not infallible and may still fail to capture certain nuances, especially in the absence of identifier-aware denoising pre-training (MIP and IT).

7. **Sociotechnical Limitations**: The deployment of CodeT5 in real-world applications, especially high-stakes environments, requires careful consideration and appropriate documentation. Users should be aware of the model's limitations and ensure that domain experts review the model's outputs for correctness and security.

In summary, while CodeT5 presents significant advancements in code understanding and generation tasks, it is important to be mindful of the potential for sensitive information leakage, automation bias, dataset bias, misuse, computational costs, technical limitations, and the broader sociotechnical implications of its deployment.

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model Salesforce/codet5-base:

1. **Security Implications**: Users should be aware that despite multiple rounds of data cleaning, the pre-trained model may still encode some sensitive information. It is recommended that users apply additional data sanitization processes and review the generated code for any potential leaks of sensitive information.

2. **Debugging and Safety**: Practitioners should treat the outputs of CodeT5 as starting points rather than definitive solutions. It is crucial to involve domain experts to review and validate the generated code for correctness and security. Users should not blindly trust the model's output and should be prepared to invest time in debugging and verification to ensure safety.

3. **Computational Cost**: While the computational cost of pre-training CodeT5 is significant, it is designed to be more efficient than larger models like Codex. Users should be mindful of the computational resources required for training and deploying the model and plan accordingly.

4. **Automation Bias**: Users should be cautious of automation bias when using CodeT5 as a coding assistant. Developers should critically assess the model's suggestions and ensure they align with their intended outcomes to prevent the integration of incorrect or suboptimal code.

5. **Non-deterministic Nature**: Given the non-deterministic nature of generation models like CodeT5, there is a possibility of generating vulnerable code. Users should be vigilant and conduct thorough security checks to prevent the creation of software that could be exploited or used for malicious purposes.

6. **NL-PL Alignment**: CodeT5 has been shown to outperform prior methods in tasks such as code defect detection and clone detection. Users should leverage these capabilities to improve code quality and maintainability.

7. **AI-powered Coding Assistant**: When deploying CodeT5 or CodeT5+ models as AI-powered coding assistants, users should ensure that they are used to augment developer productivity responsibly and not as a replacement for human expertise and oversight.

In summary, while CodeT5 offers significant advancements in code-related tasks, users should be mindful of its limitations and potential risks, involving human expertise in the review process, and remaining vigilant against security vulnerabilities and automation bias.

## Training Details

### Training Data

The training data for the Salesforce/codet5-base model consists of code corpora from CodeSearchNet and a subset of Google BigQuery, which includes code from public GitHub repositories. Efforts have been made to clean the data and remove sensitive information through multiple rounds of data cleaning prior to training. [More Information Needed] on the specific data pre-processing or additional filtering steps.

### Training Procedure

#### Preprocessing

For the Salesforce/codet5-base model, the preprocessing of the data involves several steps to prepare both programming language (PL) and natural language (NL) inputs for the model. Here's a detailed description of the preprocessing steps based on the provided references:

1. **Tokenization**: The model employs a tokenizer that is likely based on Byte-Pair Encoding (BPE), as mentioned in reference 1. This tokenizer is responsible for breaking down the input text into tokens that the model can understand. BPE is a common choice for tokenization in pre-trained language models as it effectively handles a large vocabulary in a memory-efficient way.

2. **Identifier Tagging (IT)**: As part of the preprocessing, the model distinguishes between identifiers and other tokens in the code. This process is akin to syntax highlighting, which helps the model understand the role of each token in the code. The identifier tagging is treated as a sequence labeling problem, where the model predicts a sequence of probabilities indicating whether each token is an identifier or not (reference 2).

3. **Input Formatting**: For bimodal NL-PL inputs, the data is formatted by concatenating the NL and PL segments with a [SEP] delimiter token. The entire input sequence is represented in the format: ([CLS], w1, ..., wn, [SEP], c1, ..., cm, [SEP]), where w represents NL tokens and c represents PL tokens (reference 4).

4. **Language IDs**: To handle different programming languages and natural languages, language identifiers (e.g., <java>, <en>) are added to the training instances. This helps the model distinguish between different languages and optimize its performance on bimodal data points (reference 3).

5. **Denoising Pre-training**: The model uses a denoising sequence-to-sequence pre-training objective. This involves corrupting the source sequence with noising functions, such as span masking, and then requiring the decoder to recover the original texts. The span masking randomly masks spans of tokens, which the model then attempts to predict (reference 5).

6. **Additional Structural Information**: To incorporate code-specific structural information, the model preprocessing includes tasks like Identifier Tagging (IT) and Masked Identifier Prediction (MIP). These tasks help the model to understand the syntax and data flow in the code by leveraging the identifier node type information from Abstract Syntax Trees (AST) (reference 7).

In summary, the preprocessing for Salesforce/codet5-base involves tokenization using BPE, identifier tagging to highlight identifiers in the code, input formatting with delimiter tokens for bimodal data, addition of language IDs for different languages, denoising pre-training with span masking, and the inclusion of code-specific structural information through additional tasks. If there are any specific preprocessing steps not covered by the references provided, [More Information Needed].

#### Training Hyperparameters

The training hyperparameters for the Salesforce/codet5-base model are as follows:

- Model Size: CodeT5-base with 220 million parameters.
- Maximum Sequence Lengths: 512 for source sequences and 256 for target sequences.
- Precision: Mixed precision of FP16 to accelerate the pre-training.
- Batch Size: 1024.
- Learning Rate: Peak learning rate of 2e-4 with linear decay.
- Pre-training Epochs: 100 epochs for the denoising objective, followed by an additional 50 epochs for bimodal dual training.
- GPUs Used: Training was conducted on a cluster of 16 NVIDIA A100 GPUs with 40GB of memory each.
- Total Training Time: Approximately 12 days for the CodeT5-base model.
- Multi-task Learning: All downstream tasks in CodeXGLUE were covered except for clone detection.
- Sampling Strategy: Balanced sampling with α set to 0.7 to alleviate bias towards high-resource tasks.

For any additional specifics regarding the hyperparameters, such as the exact learning rate schedule, optimizer details, or weight initialization, [More Information Needed].

#### Speeds, Sizes, Times

The Salesforce/codet5-base model is an encoder-decoder framework based on the T5 architecture, specifically designed for programming language tasks. Here are the details regarding the model's throughput, start or end time, and checkpoint sizes:

- **Throughput and Training Time**: The model was pre-trained for 100 epochs with a denoising objective, followed by an additional 50 epochs of bimodal dual training. The total training time for the CodeT5-base model was 12 days. The training was conducted on a cluster of 16 NVIDIA A100 GPUs, each with 40G of memory. [More Information Needed] on the exact throughput in terms of examples per second.

- **Start or End Time**: The pre-training and fine-tuning of the model likely started before September 2021, as this is when the model's capabilities were being discussed. The fine-tuned checkpoints for downstream tasks were released in September 2022. The latest update mentioned is from May 2023, with the release of CodeT5+ paper and models.

- **Checkpoint Sizes**: The CodeT5-base model has a size of 220M parameters. The checkpoint size in terms of disk space is not explicitly mentioned in the provided references, so [More Information Needed] for the exact file size of the checkpoints.

- **Batch Size and Learning Rate**: The batch size was set to 1024, and the peak learning rate was 2e-4 with linear decay.

- **Sequence Lengths**: The maximum source and target sequence lengths were set to 512 and 256, respectively.

- **Precision**: The model used mixed precision of FP16 to accelerate the pre-training.

For more detailed information on the model's throughput or the exact sizes of the checkpoints, one would need to refer to the model's repository or associated documentation that is not provided in the references above.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model Salesforce/codet5-base evaluates on the following benchmarks or datasets:

1. CodeXGLUE benchmark, which includes a variety of tasks such as code summarization, generation, translation, and refinement.
2. CodeSearchNet dataset, which consists of six programming languages (PLs): Ruby, JavaScript, Go, Python, Java, and PHP for the code summarization task.
3. A dataset for the code translation task, which involves translating between Java and C#.
4. A dataset for defect detection, specifically using the C dataset provided by an unspecified source.
5. A dataset for clone detection, which uses Java data provided by Wang et al. (2020).

These datasets are used to evaluate the model's performance on both understanding and generation tasks in the software development domain.

#### Factors

The Salesforce/codet5-base model is designed to perform a variety of code-related tasks, including code generation, defect detection, and clone detection. The model's behavior and performance are influenced by several characteristics:

1. **Domain and Context**: The model has been trained on a code corpus that includes CodeSearchNet and a fraction of Google BigQuery, which are collections of public GitHub repositories. This suggests that the model is likely to perform better on code that is similar to open-source projects typically found on GitHub. It may not perform as well on code from proprietary or niche domains that are underrepresented in the training data.

2. **Population Subgroups**: The model has been pre-trained and evaluated on code written in programming languages such as Python and Java. Therefore, its performance may be optimized for these languages, and it might not generalize as well to other programming languages that were not included or were underrepresented in the training data.

3. **Performance Disparities**: The model has been shown to outperform state-of-the-art models on a broad set of CodeXGLUE downstream tasks, which suggests that it is robust across these tasks. However, the evaluation of performance across different factors such as programming languages, code complexity, and task types is not explicitly mentioned. [More Information Needed] to determine if there are disparities in performance when disaggregated across these factors.

4. **Social Biases**: Since the model is trained on existing code corpus, it may inherit social biases present in the data, such as biases in variable, function, and class names. While interventions may help mitigate these biases, the extent to which this has been done and its effectiveness are not detailed, so [More Information Needed] on the specific measures taken and their impact on bias mitigation.

5. **Security Implications**: The model has been trained on data that might contain sensitive information, despite efforts to clean the data. This means that there is a risk of the model inadvertently exposing sensitive information if it was not completely removed during the data cleaning process.

In summary, the Salesforce/codet5-base model's behavior is influenced by the domain and context of its training data, the programming languages it has been optimized for, and the potential for inherited social biases and security risks. Detailed evaluation across various factors is necessary to fully understand performance disparities and to ensure the model's responsible use across different contexts and populations.

#### Metrics

For the evaluation of the Salesforce/codet5-base model, the following metrics will be used across different tasks:

1. Code Summarization: Smoothed BLEU-4 metric will be used to evaluate the performance of code summarization across six programming languages (PLs) including Ruby, JavaScript, Go, Python, Java, and PHP.

2. Defect Detection: F1 score and accuracy will be employed to assess the model's ability to predict whether a code is vulnerable in the defect detection task.

3. Clone Detection: F1 score and accuracy will also be used to measure the similarity between two code snippets and predict whether they have the same functionality in the clone detection task.

4. Code Translation: BLEU-4 metric will be used to evaluate the code translation task, which involves translating functions from Java to CSharp and vice versa.

5. Code Refinement: Due to the high overlap between source and target code, the exact match (EM) metric is focused on for evaluating the code refinement task, especially since even a naive copy approach can yield high BLEU scores but zero exact matches.

These metrics are chosen to balance the tradeoffs between different types of errors and to provide a comprehensive evaluation of the model's performance across various code-related tasks.

### Results

Evaluation results of the Salesforce/codet5-base model are as follows:

1. **Code Generation (Code-to-Code Generation Tasks)**:
   - The model demonstrates significant improvements in generating code snippets, particularly showing a 4.7 points increase in CodeBLEU over PLBART, which suggests that CodeT5 has a better understanding of code syntax and semantics due to identifier-aware pre-training.
   - In code translation tasks, CodeT5-base consistently outperforms PLBART across various metrics for translating between Java and C#.
   - For code refinement tasks, the model shows superior performance compared to baselines, although specific metrics are not provided in the reference.

2. **Code Summarization**:
   - The model's performance is evaluated using smoothed BLEU-4 on six programming languages (PL), but specific results are not detailed in the reference provided.

3. **Understanding Tasks**:
   - CodeT5 significantly outperforms previous methods in code defect detection and clone detection tasks, indicating a strong capability in understanding code semantics.

4. **Size and Performance**:
   - CodeT5-base boosts overall performance by over 1.2 absolute points over PLBART, despite being pre-trained with much larger Python and Java data.
   - Even the smaller variant, CodeT5-small, outperforms all decoder-only models and the state-of-the-art PLBART, confirming the effectiveness of the encoder-decoder framework used in CodeT5.

5. **Comparison with Other Models**:
   - CodeT5 variants significantly outperform prior work with either encode-only frameworks (like RoBERTa, CodeBERT, DOBF) or encoder-decoder frameworks (like PLBART).
   - The performance gap between these two groups of models confirms that encode-only frameworks are suboptimal for generation tasks.

6. **Bimodal Training Data**:
   - The model benefits from identifier-aware denoising pre-training and better employment of bimodal training data, which includes both unimodal (PL-only) and bimodal (PL-NL) data.

7. **Multi-task Fine-tuning**:
   - The model is fine-tuned on most tasks in the CodeXGLUE benchmark, which includes understanding and generation tasks, but specific fine-tuning results are not provided in the reference.

In summary, Salesforce/codet5-base exhibits state-of-the-art performance in various code-related tasks, outperforming previous models and confirming the advantages of its pre-training and fine-tuning strategies. Specific numerical results for each task are not provided in the reference, so [More Information Needed] for detailed metrics.

#### Summary

The Salesforce/codet5-base model has demonstrated significant improvements in various code-related tasks over previous state-of-the-art (SOTA) models. In code generation tasks, it has shown a notable increase of around 4.7 points in CodeBLEU over PLBART, indicating a better understanding of code syntax and semantics due to identifier-aware pre-training. For code-to-code generation tasks, such as code translation and code refinement, CodeT5-base consistently outperforms PLBART across various metrics, translating between Java and C# effectively.

In understanding tasks, CodeT5-base has outperformed all baselines in defect detection, achieving a 2.6 accuracy score improvement over PLBART. It also performs comparably to SOTA models like GraphCodeBERT and PLBART in clone detection tasks. The model's strong code understanding capability is further highlighted by its performance on the more challenging medium task, where it boosts over 4.8 points on Exact Match (EM) compared to GraphCodeBERT.

The model has been evaluated using binary labels generated as a unigram sequence from the decoder for defect detection, and sequence embeddings from the last decoder state for clone detection, with subsequent label prediction based on similarity measures.

Overall, CodeT5-base has shown to significantly outperform prior methods on understanding tasks such as code defect detection and clone detection, as well as generation tasks across various directions including programming language to natural language (PL-NL), natural language to programming language (NL-PL), and programming language to programming language (PL-PL). The model's ability to capture semantic information from code has been a key factor in its performance.

Salesforce has also built an AI coding assistant demo using CodeT5 as a VS Code plugin, which showcases the practical deployment of the model to enhance software developer productivity.

## Model Examination

Explainability and Interpretability of Salesforce/codet5-base:

Our CodeT5 model is designed to enhance the productivity of software developers by providing assistance in various coding tasks. To ensure that users can trust and effectively interact with our model, we acknowledge the importance of explainability and interpretability in AI-powered tools.

While our model card does not currently include a dedicated section on explainability, we recognize its significance and are actively exploring methods to improve this aspect of CodeT5. For instance, we have conducted detailed analyses on the proposed identifier-aware pre-training (referenced in point 3), which helps the model distinguish and leverage identifier information, contributing to its understanding of code semantics.

Moreover, we are aware of the potential for automation bias (referenced in point 4), where developers might over-rely on model-generated outputs without critical evaluation. This underscores the need for explainability in our model's suggestions, ensuring that developers can understand the rationale behind the code generated by CodeT5 and verify its alignment with their intent.

In terms of interpretability, our experiments (referenced in point 6) have shown that CodeT5 can effectively capture semantic information from code, which is crucial for tasks such as defect detection and clone detection. By providing insights into how the model processes and understands code, we aim to make its decision-making process more transparent to the end-users.

We are committed to improving the explainability and interpretability of CodeT5 and will continue to update our model card with findings and methodologies that contribute to these objectives. Our goal is to provide a tool that not only aids in coding tasks but also does so in a manner that is understandable and trustworthy for software developers.

## Environmental Impact

- **Hardware Type:** The model Salesforce/codet5-base is trained on a cluster of 16 NVIDIA A100 GPUs with 40G memory.
- **Software Type:** The model Salesforce/codet5-base is trained on programming language (PL) and natural language (NL) data.
- **Hours used:** The total training time for CodeT5-base is 12 days.
- **Cloud Provider:** The model Salesforce/codet5-base was trained on Google Cloud Platform.
- **Carbon Emitted:** The amount of carbon emitted when training the model Salesforce/codet5-base is around 49.25 kg CO2.
## Technical Specification

### Model Architecture and Objective

The Salesforce/codet5-base model is built upon an encoder-decoder framework with the same architecture as T5 (Raffel et al., 2020). It is designed to derive generic representations for programming languages (PL) and natural language (NL) through pre-training on unlabeled source code. The model extends the de-noising sequence-to-sequence (Seq2Seq) objective of T5 by introducing two identifier tagging and prediction tasks, which help the model to better utilize token type information from programming languages, particularly the identifiers assigned by developers.

The primary objective of CodeT5 is to support both code-related understanding and generation tasks. It is one of the first unified encoder-decoder models that enables multi-task learning for a wide range of tasks. The novel identifier-aware pre-training objective is a key feature of CodeT5, which takes into account the crucial token type information from code. Additionally, the model leverages natural language-programming language (NL-PL) pairs that are naturally available in source code to learn better cross-modal alignment.

In summary, the architecture of Salesforce/codet5-base is an encoder-decoder model based on T5, and its objective is to handle both understanding and generation tasks related to code by leveraging identifier-aware pre-training and bimodal dual generation for improved NL-PL alignment.

### Compute Infrastructure

The compute infrastructure used for training the model Salesforce/codet5-base involved a cluster of 16 NVIDIA A100 GPUs, each with 40GB of memory. The model was pre-trained for a total of 150 epochs, with 100 epochs dedicated to the denoising objective and an additional 50 epochs for bimodal dual training. The total training time for the CodeT5-base model was 12 days.

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
