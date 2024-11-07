# Model Card for SCUT-DLVCLab/lilt-roberta-en-base

The SCUT-DLVCLab/lilt-roberta-en-base is a language-independent layout Transformer (LiLT) pre-trained on English structured documents, which can be fine-tuned on various languages for structured document understanding (SDU) tasks, achieving competitive or superior performance on multiple benchmarks.

## Model Details

### Model Description

Model Name: SCUT-DLVCLab/lilt-roberta-en-base

Model Architecture:
The SCUT-DLVCLab/lilt-roberta-en-base model is a parallel dual-stream Transformer-based architecture designed for structured document understanding. It incorporates both text and layout flows to process textual and visual information from documents. The text flow is based on the RoBERTa model, which is a robustly optimized BERT variant, while the layout flow is designed to capture the spatial layout of the text in the document. The model uses special tokens ([CLS], [SEP], [PAD]) with specific embeddings to handle the beginning, separation, and padding of sequences. The layout flow has a reduced hidden size and intermediate size compared to the text flow for computational efficiency.

Training Procedures:
The model is pre-trained using self-supervised tasks to learn joint representations with cross-modal cooperation. Three pre-training tasks are used: key point location (KPL), crossmodal alignment identification (CAI), and masked visual-language modeling (MVLM). During pre-training, different learning rate ratios are explored to prevent the layout flow from coupling too tightly with the text flow, which could be detrimental to the model's ability to cooperate with different textual models during fine-tuning. For fine-tuning, a unified learning rate is used, and the DETACH operation of BiACM is canceled.

Parameters:
- The model uses a unified learning rate for end-to-end optimization during fine-tuning.
- The bounding box coordinates are normalized and discretized to integers in the range [0, 1000].
- The 2D positional embedding is generated using four embedding layers for the x-axis, y-axis, height, and width features.
- The model's text feature dimension (d_T) and the layer normalization (LN) are applied to the token embeddings and 1D positional embeddings.
- [More Information Needed] for the exact number of parameters in the model.

Important Disclaimers:
- The model is pre-trained on the IIT-CDIP dataset and is designed to be language-independent, allowing it to be adapted to other languages.
- The fine-grained token-level information is not always included in the results of some OCR engines; the model utilizes the bounding box of the text string the token belongs to.
- The model's ability to transfer language-independent knowledge from monolingual data has been demonstrated, but it is important to note that the performance may vary based on the specific downstream tasks and datasets used for fine-tuning.
- [More Information Needed] for any additional disclaimers specific to the use or limitations of the model that are not covered in the provided references.

For further details or updates on the model, please feel free to reach out to the project organizer responsible for the model card.

- **Developed by:** Jiapeng Wang; Lianwen Jin; Kai Ding
- **Funded by:** The information provided does not explicitly mention the people or organizations that funded the project for the model SCUT-DLVCLab/lilt-roberta-en-base. Therefore, the answer is:

[More Information Needed]
- **Shared by:** The contributors who made the model SCUT-DLVCLab/lilt-roberta-en-base available online as a GitHub repo are Jiapeng Wang, Lianwen Jin, and Kai Ding. This information can be inferred from the citation provided in the references, which lists these individuals as the authors of the paper associated with the model. Additionally, the GitHub repository link provided in reference 7 (https://github.com/jpWang/LiLT) suggests that Jiapeng Wang, whose GitHub username is jpWang, is likely to have a significant role in making the model available online.
- **Model type:** The SCUT-DLVCLab/lilt-roberta-en-base is a dual-stream Transformer-based model that employs self-supervised pre-training with a differentiated learning rate strategy for its layout and text streams, fine-tuned end-to-end on language-specific and multilingual structured document understanding tasks, demonstrating cross-modal and language-independent capabilities.
- **Language(s):** The model SCUT-DLVCLab/lilt-roberta-en-base processes English as its primary natural human language for pre-training and is capable of transferring knowledge to other languages for structured document understanding tasks.
- **License:** [More Information Needed]
- **Finetuned from model:** The model SCUT-DLVCLab/lilt-roberta-en-base is fine-tuned from the English RoBERTa BASE model. The reference to the base model is:

Liu et al., 2019b - RoBERTa: A Robustly Optimized BERT Pretraining Approach.

Unfortunately, a direct link to the base model is not provided in the references you've given. However, the RoBERTa BASE model is widely known and can typically be found on the Huggingface Model Hub.

[More Information Needed] for the direct link to the English RoBERTa BASE model on Huggingface or another source.
### Model Sources

- **Repository:** https://github.com/jpwang/lilt
- **Paper:** https://arxiv.org/pdf/2202.13669.pdf
- **Demo:** [More Information Needed]
## Uses

### Direct Use

The model SCUT-DLVCLab/lilt-roberta-en-base is a pre-trained model that combines the layout flow (LiLT) with the RoBERTa English base model. It is designed to understand both textual and non-textual (layout) information in documents. While the model is primarily intended to be fine-tuned on a specific downstream task to achieve the best performance, it can still be used without fine-tuning to extract features from text and layout data.

Without fine-tuning, the model can be used in a feature extraction mode, where it can provide embeddings for input sequences that can be used for various unsupervised tasks such as clustering, similarity search, or as input features for other machine learning models.

Here is a code snippet to use the model for feature extraction:

```python
from transformers import AutoModel, AutoTokenizer

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
model = AutoModel.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")

# Example text input
text = "Hello, this is an example to extract features."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Get the embeddings from the model
with torch.no_grad():
    outputs = model(**inputs)

# Extract the last hidden states
embeddings = outputs.last_hidden_state

# The embeddings can now be used for various unsupervised tasks
```

Please note that this code snippet assumes that you have already installed the necessary libraries (`transformers` and `torch`) and that you are using a Python environment. The embeddings obtained from the model can be used as is, without any further fine-tuning, post-processing, or integration into a pipeline for downstream tasks.

If you need to use the model for a specific task that requires fine-tuning or post-processing, or if you want to integrate it into a pipeline for end-to-end processing, [More Information Needed] to provide guidance on those aspects.

### Downstream Use

The model `SCUT-DLVCLab/lilt-roberta-en-base` is a fine-tuned version of the English RoBERTa BASE model, which has been adapted for tasks that require understanding of both textual and layout information in documents. This model can be particularly useful for a variety of document understanding tasks such as relation extraction, document classification, and semantic entity recognition.

When fine-tuned for relation extraction, the model can identify and classify relationships between entities within a document. This is achieved by constructing relation candidates from pairs of semantic entities and using a specific type embedding layer to enhance the representation of these entities.

For document classification tasks, the model leverages high-level visual features by incorporating image features from a ResNeXt101-FPN backbone. These visual features are pooled into a global feature and concatenated with the [CLS] output feature, which is then used for classification.

In semantic entity recognition, the model predicts BIO tags for each entity field at the token level, which is essential for extracting structured information from unstructured text in documents.

The model can be plugged into a larger ecosystem or app where document understanding is crucial. For instance, it can be integrated into a document management system to automatically categorize documents, extract key information for indexing, or facilitate information retrieval by understanding the relationships between different entities within the documents.

To use the model, you would typically follow these steps:

1. Install the necessary dependencies, including PyTorch and Detectron2, as specified in the provided installation commands.
2. Clone the LiLT repository and install the required Python packages.
3. Fine-tune the model on your specific task using the appropriate dataset.

Here is a code snippet for setting up the environment, which is derived from the provided references:

```bash
conda create -n liltfinetune python=3.7
conda activate liltfinetune
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
python -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
git clone https://github.com/jpWang/LiLT
cd LiLT
pip install -r requirements.txt
pip install -e .
```

Please note that the actual fine-tuning code and usage of the model for specific tasks are not provided in the references, so [More Information Needed] for those parts. However, the above setup is necessary before you can proceed with fine-tuning or integrating the model into an application.

### Out-of-Scope Use

As the sociotechnic on the deep learning model development team, it is my responsibility to consider the potential misuses of the SCUT-DLVCLab/lilt-roberta-en-base model and to advise users on what they should not do with the model. The model is designed to understand and process multilingual structured documents by leveraging layout and language information. However, there are several ways in which it could be misused:

1. **Privacy Violations**: Given the model's ability to extract information from structured documents, it could be used to process sensitive or private documents without the consent of the individuals involved. Users should not use the model to extract or analyze personal data unless they have explicit permission to do so and are in compliance with relevant data protection laws.

2. **Misrepresentation**: The model could be used to create or spread misinformation by altering the layout and textual content of documents to mislead readers. Users should not use the model to intentionally manipulate documents for fraudulent or deceptive purposes.

3. **Intellectual Property Theft**: The model's ability to understand and replicate document layouts and content could be used to infringe on intellectual property rights by copying or mimicking proprietary document designs. Users should respect copyright and intellectual property laws and not use the model to infringe on the rights of others.

4. **Bias Propagation**: While the references do not explicitly mention biases, any language model can potentially propagate or amplify biases present in the training data. Users should be cautious when using the model in applications where biased outputs could lead to discrimination or unfair treatment of individuals or groups.

5. **Security Risks**: The model could be used to analyze and extract information from confidential documents, posing a security risk if the information is not handled properly. Users should ensure that they have robust security measures in place to prevent unauthorized access to sensitive information processed by the model.

6. **Unintended Use Cases**: The model is designed for structured document understanding. Using it for tasks it was not designed for, such as unstructured text analysis or image recognition without textual content, may lead to poor performance and should be avoided.

In summary, users of the SCUT-DLVCLab/lilt-roberta-en-base model should not use it to violate privacy, create or spread misinformation, infringe on intellectual property, propagate biases, compromise security, or apply it to unintended use cases. It is important for users to consider the ethical and legal implications of their use of the model and to use it responsibly.

### Bias, Risks, and Limitations

The SCUT-DLVCLab/lilt-roberta-en-base model, as described in the references, is a multilingual structured document understanding (SDU) model that has shown significant improvements in handling multilingual documents. However, there are several known or foreseeable issues and limitations that stem from this model:

1. **Language and Cultural Bias**: The model is pre-trained on the monolingual IIT-CDIP Test Collection 1.0, which may have a bias towards English or the cultural context within which the dataset was created (Reference 6). This could limit the model's effectiveness in understanding documents from diverse languages and cultures.

2. **Data Collection and Cleaning**: The model requires a heavy process of data collection and cleaning for multilingual data, which can be resource-intensive and may introduce biases if not done carefully (Reference 6).

3. **Generalization Limitations**: While the model aims to handle multilingual documents, it may still struggle with language-specific visual information. Future research is needed to explore the generalized rather than language-specific visual information contained in multilingual structured documents (Reference 5).

4. **Technical Complexity and Efficiency**: The model introduces novel components like Bi-ACM and DETACH, which, while improving performance, also add complexity to the model. This could impact the efficiency and scalability of the model, especially when deployed in real-world applications (References 1, 3).

5. **Cross-Modal Interaction**: The model uses a Bi-ACM mechanism for cross-modal interaction, which is crucial for its performance. However, replacing it with other mechanisms like co-attention has shown to severely drop performance, indicating a potential fragility in the model's design (Reference 4).

6. **Misunderstandings and Misuse**: Users may misunderstand the capabilities of the model, expecting it to perform equally well across all languages and document types. This could lead to misuse or overreliance on the model in scenarios where it is not the best fit (Reference 9).

7. **Harms and Ethical Considerations**: There is a potential for harm if the model is used in sensitive applications without proper understanding of its limitations. For example, if used in legal or medical document processing, inaccuracies due to language or cultural biases could have serious consequences (Reference 8).

8. **Multitask Learning**: While multitask learning improves performance, it also increases the complexity of the model and the data required for training. This could lead to challenges in maintaining and updating the model, as well as ensuring that it remains fair and unbiased across tasks (Reference 11).

In conclusion, while the SCUT-DLVCLab/lilt-roberta-en-base model represents a significant advancement in multilingual SDU, it is important to be aware of its limitations and potential issues. Continuous evaluation and updates will be necessary to address these challenges and ensure the model remains effective and ethical in its applications.

### Recommendations

Based on the provided references, here are some recommendations with respect to the foreseeable issues about the model SCUT-DLVCLab/lilt-roberta-en-base:

1. **Cross-Modal and Cross-Lingual Generalization**: Future research should continue to explore the transfer from "monolingual" to "multilingual" to further unlock the power of LiLT (Reference 4). This includes enhancing the model's ability to generalize across different languages and document structures without compromising the consistency of text flow, as observed with the co-attention mechanism (Reference 5).

2. **Optimization of Pre-training and Fine-tuning**: The necessity of the DETACH strategy in pre-training is highlighted, and its removal in fine-tuning leads to better performance (Reference 3). It is recommended to continue refining these strategies to optimize the model's performance across various tasks.

3. **Language-Independent Knowledge Transfer**: LiLT has demonstrated the capability to transfer language-independent knowledge effectively using a smaller dataset (Reference 6). It is advisable to leverage this strength to create more efficient models that require less data and computational resources.

4. **Collaboration with Monolingual Models**: Despite being designed for multilingual transfer, LiLT has shown surprising effectiveness in cooperating with monolingual textual models (Reference 7). This suggests that the model could be further developed to enhance its compatibility and performance with monolingual datasets.

5. **Ethical and Societal Considerations**: As a sociotechnic, it is crucial to consider the ethical implications of deploying such a model. This includes ensuring that the model does not inadvertently perpetuate biases present in the training data, maintaining privacy and security of the data used, and ensuring that the model is accessible and fair to users across different languages and regions.

6. **Documentation and Transparency**: A comprehensive model card should be created to document the model's capabilities, limitations, training data, intended use cases, and ethical considerations. This transparency is essential for users to understand the model and for fostering trust in its applications.

7. **Monitoring and Evaluation**: After deployment, continuous monitoring of the model's performance and impact on society should be conducted. This includes evaluating the model for fairness and unintended biases across different demographics and use cases.

8. **Collaboration with Multidisciplinary Teams**: It is recommended to involve a multidisciplinary team, including ethicists, sociologists, and rights advocates, in the development and deployment process to address the broader societal impacts of the technology.

In conclusion, while the model SCUT-DLVCLab/lilt-roberta-en-base shows promising results in structured document understanding, it is important to consider the broader implications of its deployment and continue refining its capabilities in a responsible and ethical manner.

## Training Details

### Training Data

The training data for the model SCUT-DLVCLab/lilt-roberta-en-base consists of the IIT-CDIP Test Collection 1.0 dataset, which includes 11 million English documents. This dataset was used to pre-train the model with the LayoutLMv2 framework, enabling the model to understand the layout and structure of documents in a language-independent manner. [More Information Needed] on data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

The preprocessing for the data of the model SCUT-DLVCLab/lilt-roberta-en-base involves several steps to prepare the text data for training. Here's a detailed description of the preprocessing steps based on the provided references:

1. **Tokenization and Sequence Formation**: As per the common practice outlined in reference 1, text strings from OCR results are tokenized and concatenated into a sequence `S t`. The text bounding boxes are sorted from top-left to bottom-right to determine the order of the text strings in the sequence. Special tokens `[CLS]` and `[SEP]` are added at the beginning and end of the sequence, respectively.

2. **Sequence Length Adjustment**: The sequence `S t` is then either truncated or padded with `[PAD]` tokens to ensure that its length equals the maximum sequence length `N`, which is set to 512 as mentioned in reference 2.

3. **Text Embedding**: The token embedding `E token` of `S t` and the 1D positional embedding `P 1D` are combined to obtain the text embedding `E T` as shown in references 3 and 5. The text embedding is normalized using layer normalization (LN).

4. **Pre-training Data**: The model is pre-trained on the IIT-CDIP Test Collection 1.0, which contains a large number of scanned document images. The TextIn API is used to obtain text bounding boxes and strings for this dataset, as mentioned in reference 4.

5. **Masked Visual-Language Modeling (MVLM)**: As part of the pre-training objectives, MVLM is used where 15% of the text tokens are randomly masked, and the model is trained to recover them using the output encoded features, as described in reference 9.

6. **Special Tokens and Bounding Boxes**: Special tokens `[CLS]`, `[SEP]`, and `[PAD]` are assigned specific bounding box values as mentioned in reference 10. For each token, the bounding box of the text string it belongs to is used directly.

7. **Fine-tuning Data and Code**: The model is fine-tuned on the FUNSD and XFUND datasets, and the fine-tuning codes are provided in the repository, as stated in reference 7. Pre-processed data for these datasets can be downloaded and placed under the `LiLT/` directory.

8. **Optimization Strategy**: A unified learning rate is not used for all model parameters. Instead, multiple ratios are explored to slow down the pre-training optimization of the text stream to prevent harmful updates to the layout flow, as discussed in references 6 and 8.

For any additional specific details regarding the preprocessing steps that are not covered in the provided references, [More Information Needed] would be the appropriate response.

#### Training Hyperparameters

The training hyperparameters for the model SCUT-DLVCLab/lilt-roberta-en-base are as follows:

- Optimizer: Adam optimizer
- Learning rate: 2×10^-5
- Weight decay: 1×10^-2
- Beta parameters: (β1, β2) = (0.9, 0.999)
- Learning rate schedule: Linear warmup over the first 10% of steps, followed by linear decay
- Batch size: 96
- Number of epochs: 5
- Training dataset: IIT-CDIP dataset
- Hardware: 4 NVIDIA A40 48GB GPUs

For the specific combination of `lilt-only-base` with `roberta-en-base`, the code provided in reference 7 is used to generate the weights for `lilt-roberta-en-base`. However, the exact training hyperparameters for this combination process are not explicitly stated in the provided references, so [More Information Needed] for any additional hyperparameters specific to the combination process.

#### Speeds, Sizes, Times

The model `SCUT-DLVCLab/lilt-roberta-en-base` is an English language model that has been pre-trained on the IIT-CDIP Test Collection 1.0 dataset, which contains more than 11 million scanned document images. The model utilizes the RoBERTa BASE architecture as the starting point for the text flow initialization and is fine-tuned on datasets such as FUNSD and XFUND.

Here are the details regarding the model's training and specifications:

- **Optimizer**: The model was pre-trained using the Adam optimizer with a learning rate of 2×10^-5, weight decay of 1×10^-2, and beta parameters (β1, β2) set to (0.9, 0.999).
- **Learning Rate Schedule**: The learning rate was linearly warmed up over the first 10% of the steps and then linearly decayed.
- **Batch Size**: The batch size was set to 96.
- **Training Duration**: The model was trained for 5 epochs.
- **Hardware Used**: Training was conducted on 4 NVIDIA A40 48GB GPUs.
- **Checkpoint Size**: The size of the model is 293MB, as indicated in the provided download link.
- **Throughput and Timing**: [More Information Needed]
- **Start or End Time of Training**: [More Information Needed]
- **Pre-training Objectives**: The model was pre-trained with objectives including key point location (KPL) and crossmodal alignment identification (CAI), combined with masked visual-language modeling (MVLM).
- **Fine-tuning**: During fine-tuning, a unified learning rate was adopted, and the DETACH operation of BiACM was canceled.

For more specific details such as the exact start or end time of the training process, throughput, and the sizes of intermediate checkpoints, [More Information Needed] as these details are not provided in the references.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model SCUT-DLVCLab/lilt-roberta-en-base evaluates on the following benchmarks or datasets:

1. IIT-CDIP Test Collection 1.0 (Lewis et al., 2006) - A large-scale scanned document image dataset used for pre-training.
2. FUNSD (Jaume et al., 2019) - A dataset used for language-specific fine-tuning tasks.
3. XFUND (Xu et al., 2021b) - A multilingual benchmark used for both language-specific fine-tuning and multitask fine-tuning.
4. CORD (Park et al., 2019) - A dataset used for evaluating the model on monolingual tasks.
5. EPHOIE - A dataset used for evaluating the model on monolingual tasks.
6. RVL-CDIP (Lewis et al., 2006) - A dataset used for evaluating the model on monolingual tasks.

These datasets cover a range of tasks including semantic entity recognition (SER) and relationship extraction (RE), and they are used to demonstrate the model's performance in both monolingual and multilingual settings.

#### Factors

The model SCUT-DLVCLab/lilt-roberta-en-base is designed to handle multilingual structured documents, leveraging the capabilities of transfer learning and multitask fine-tuning. Here are the foreseeable characteristics that will influence its behavior:

1. **Domain and Context**: 
   - The model has been evaluated on monolingual datasets as well as the multilingual XFUND benchmark, which suggests that it is tailored for structured document understanding tasks. Its performance may be optimized for the types of documents included in these datasets, such as forms and other structured documents that have both text and layout features.
   - The model incorporates Bi-ACM (Bidirectional Affinity Computation Module) to successfully transfer from monolingual to multilingual settings, indicating that it is designed to work across different languages while considering the layout of documents.

2. **Language and Transfer Learning**:
   - The model has been fine-tuned on English data and evaluated on multiple languages, indicating that it is capable of zero-shot transfer learning. This means that while it is fine-tuned on English, it can generalize to other languages without requiring fine-tuning on those languages specifically.
   - Multitask fine-tuning has shown to improve the model's performance, suggesting that the model benefits from learning commonalities across different languages' layouts.

3. **Population Subgroups**:
   - The model's performance on language-specific tasks and its ability to transfer language-independent knowledge suggests that it should work well across different linguistic subgroups. However, the actual performance may vary depending on the representation of different languages in the training data.
   - Since the model outperforms its counterparts on the XFUND benchmark, it is likely to be more effective for populations that require processing of multilingual structured documents.

4. **Disparities in Performance**:
   - The references do not provide specific disaggregated evaluation results across different factors such as language, domain, or demographic subgroups. Therefore, to fully understand disparities in performance, further evaluation would be needed.
   - Given that the model is pre-trained on the monolingual IIT-CDIP Test Collection 1.0, which may have its own biases, the model's performance could be influenced by the characteristics of this pre-training dataset.

In summary, SCUT-DLVCLab/lilt-roberta-en-base is expected to perform well in multilingual structured document understanding tasks, with the ability to generalize across languages through zero-shot learning and multitask fine-tuning. However, to ensure fairness and effectiveness across all potential user groups, additional evaluations would be necessary to uncover any disparities in performance.

#### Metrics

For the evaluation of the model SCUT-DLVCLab/lilt-roberta-en-base, we will primarily use the F1 score as our metric, as indicated in reference 5. The F1 score is a harmonic mean of precision and recall, and it is particularly useful in scenarios where we need to balance the tradeoff between false positives and false negatives. This metric is suitable for evaluating the model's performance on both the Semantic Entity Recognition (SER) and Relation Extraction (RE) tasks across different languages.

Additionally, as per reference 7, the model's performance will be evaluated in a multitask learning setting, where it is fine-tuned and assessed on data from all eight languages. This evaluation will help us understand how well the model can leverage commonalities in the layout of multilingual structured documents.

In summary, the F1 score will be the primary metric used for evaluating the SCUT-DLVCLab/lilt-roberta-en-base model, considering the tradeoffs between different types of errors. This metric will be applied to both monolingual and multilingual datasets, as well as under different fine-tuning paradigms such as language-specific fine-tuning, zero-shot transfer learning, and multitask fine-tuning.

### Results

The SCUT-DLVCLab/lilt-roberta-en-base model has been evaluated on several tasks and benchmarks to demonstrate its performance. Here are the evaluation results based on the factors and metrics mentioned in the references:

1. **Zero-shot Transfer Learning and Multitask Fine-tuning**: The model has been fine-tuned on English data and evaluated on multiple target languages, showing its ability to transfer knowledge across different languages effectively (Reference 1).

2. **Performance on Monolingual and Multilingual Benchmarks**: The model has been tested on widely-used monolingual datasets and the multilingual XFUND benchmark, achieving high performance. This includes language-specific fine-tuning and the two settings designed in (Dauphinee et al., 2019) (Reference 2).

3. **Introduction of Bi-ACM**: The Bi-ACM (Bidirectional Attentional Co-Modulation) mechanism has significantly improved the model's ability to transfer from "monolingual" to "multilingual" settings, as evidenced by the performance improvement from setting (a)#1 to (a)#3 (Reference 3).

4. **Optimization Slow-down Ratio**: A slow-down ratio of 1000 for the pre-training optimization of the text flow has been found to be the most suitable, as it leads to the highest F1 scores before they begin to fall (Reference 4).

5. **Comparison of KPL and CAI Tasks**: Both the KPL (Knowledge-Powered Learning) and CAI (Contextualized Attention Interaction) tasks have substantially improved the model's performance, with CAI providing more benefits than KPL. Using both tasks together yields the most effective results (Reference 5).

6. **Language-specific Fine-tuning Tasks**: On the FUNSD and multilingual XFUND tasks, LiLT has achieved the highest F1 scores on both the SER (Semantic Entity Recognition) and RE (Relation Extraction) tasks of each language, outperforming plain text models and LayoutXLM models pre-trained with more multilingual structured documents (Reference 6).

7. **Multitask Learning Results**: When fine-tuned simultaneously with all eight languages, the pre-trained LiLT model shows improved performance compared to language-specific fine-tuning. This indicates that the model can leverage commonalities in the layout of multilingual structured documents (Reference 7).

8. **Flexibility and Cooperation with Monolingual Models**: LiLT has been evaluated on four monolingual datasets (FUNSD, CORD, EPHOIE, and RVL-CDIP) and has demonstrated flexibility in working with both monolingual and multilingual plain text models for downstream tasks. It has also shown surprising cooperation with monolingual textual models to achieve high performance (Reference 8).

In summary, the SCUT-DLVCLab/lilt-roberta-en-base model exhibits strong performance across a range of tasks and languages, benefiting from novel mechanisms like Bi-ACM, and optimization strategies like the slow-down ratio. It also demonstrates the ability to work well in both monolingual and multilingual settings, and to improve through multitask learning.

#### Summary

The evaluation results for the model SCUT-DLVCLab/lilt-roberta-en-base, referred to as LiLT, indicate that the model demonstrates significant improvements and capabilities in various aspects:

1. Introduction of Bi-ACM: The novel Bi-ACM (Bidirectional Attention Cross-Modal) mechanism is crucial for the model's success in transferring from "monolingual" to "multilingual" settings. It significantly outperforms a plain design where text and layout features are simply concatenated (Chi et al., 2021).

2. Performance on Monolingual and Multilingual Datasets: LiLT has been tested on several monolingual datasets and the multilingual XFUND benchmark, showing superior performance compared to plain text models like XLM-R/InfoXLM and the LayoutXLM model. It achieves the highest F1 scores on both the SER and RE tasks of each language, despite using only 11M monolingual data (Xu et al., 2021b).

3. Zero-shot Transfer Learning and Multitask Fine-tuning: The model exhibits the ability to transfer knowledge among different languages through zero-shot transfer learning, where it is fine-tuned on English data only and then evaluated on other languages. Additionally, multitask fine-tuning, where the model is fine-tuned with all eight languages simultaneously, further improves performance, suggesting that the model benefits from commonalities in the layout of multilingual structured documents (Xu et al., 2021b).

4. Flexibility and Cooperation with Monolingual Models: LiLT is flexible and can work with both monolingual and multilingual plain text models for downstream tasks. It can also cooperate with monolingual textual models to achieve significant performance improvements (Jaume et al., 2019; Park et al., 2019; Lewis et al., 2006).

5. Comparison with Co-attention Mechanism: Replacing BiACM with a co-attention mechanism results in a severe performance drop, indicating that the deeper interaction of the co-attention mechanism may disrupt the text flow consistency in pre-training optimization. In contrast, BiACM maintains cross-modal cooperation while providing cross-modal information (Lu et al., 2019).

6. Necessity of DETACH in Pre-training: The use of DETACH in pre-training is proven to be necessary for the model's performance, and removing DETACH in fine-tuning leads to better results.

In summary, the SCUT-DLVCLab/lilt-roberta-en-base model, with its Bi-ACM mechanism and DETACH strategy, demonstrates strong performance across monolingual and multilingual tasks, showcasing its ability to effectively transfer knowledge and handle structured document understanding tasks.

## Model Examination

### Model Card - Experimental Section: Explainability/Interpretability

#### SCUT-DLVCLab/lilt-roberta-en-base

In the development of the SCUT-DLVCLab/lilt-roberta-en-base model, we have focused on enhancing the model's ability to understand structured documents by incorporating both textual and layout features. The model leverages a novel Bi-directional Adaptive Cross-Modal (Bi-ACM) mechanism, which has been shown to significantly improve performance over concatenating text and layout features without interaction (Chi et al., 2021).

The Bi-ACM mechanism enables the model to maintain cross-model cooperation while providing essential cross-modal information. This is particularly important for tasks such as semantic entity recognition (SER) and relation extraction (RE), where understanding the interplay between text and its spatial arrangement is crucial.

For explainability and interpretability, we have introduced two pre-training tasks: Key Point Location (KPL) and Crossmodal Alignment Identification (CAI). These tasks are designed to help the model learn to identify and align textual and visual elements within documents, which is a step towards making the model's predictions more interpretable.

The model has been evaluated on the FUNSD and XFUND datasets, which include a variety of languages and document types. The results demonstrate that our model, pre-trained with only 11 million monolingual documents, can achieve high F1 scores on SER and RE tasks across different languages, showcasing its ability to transfer language-independent knowledge (Xu et al., 2021b).

In terms of interpretability, while we have made strides in understanding the model's internal representations through the pre-training tasks, further work is needed to fully explain how the model makes its predictions. This could involve techniques such as attention visualization or probing tasks to better understand the role of layout features in the model's decision-making process.

[More Information Needed] on specific methods or results related to the explainability and interpretability of the SCUT-DLVCLab/lilt-roberta-en-base model beyond what has been described here. Our team is committed to advancing this aspect of the model in future updates.

---

This section provides a brief overview of the efforts towards explainability and interpretability in the SCUT-DLVCLab/lilt-roberta-en-base model. As the field evolves, we anticipate incorporating more sophisticated methods to enhance our understanding of the model's inner workings and decision-making processes.

## Environmental Impact

- **Hardware Type:** The model SCUT-DLVCLab/lilt-roberta-en-base was trained on 4 NVIDIA A40 48GB GPUs.
- **Software Type:** The model SCUT-DLVCLab/lilt-roberta-en-base is trained on NVIDIA A40 48GB GPUs.
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model SCUT-DLVCLab/lilt-roberta-en-base is based on a parallel dual-stream Transformer architecture, which includes a text flow and a layout flow. The text flow is initialized from the existing pre-trained English RoBERTa BASE model, which processes the textual content of documents. The layout flow, on the other hand, is designed to handle the spatial layout information of the structured documents.

For the text flow, the model tokenizes the text strings from OCR results and sorts them from top-left to bottom-right, adding special tokens [CLS] and [SEP] at the beginning and end of the sequence, respectively. The sequence is then truncated or padded with [PAD] tokens to match the maximum sequence length N, which is set to 512. The text flow utilizes layer normalization (LN) to normalize the sum of token embeddings and 1D positional embeddings, as indicated by the equation E_T = LN(E_token + P_1D).

The layout flow constructs a 2D position sequence using the corresponding text bounding boxes, which are normalized and discretized. Four embedding layers generate features for the x-axis, y-axis, height, and width of the bounding boxes. The layout flow has a reduced hidden size and intermediate size compared to the text flow for computational efficiency.

The model's objectives include masked visual-language modeling (MVLM), key point location (KPL), and crossmodal alignment identification (CAI). MVLM masks some input tokens and asks the model to predict them using the output encoded features, while KPL and CAI tasks help the model learn the spatial layout and alignment between text and non-textual elements.

LiLT, the underlying method of SCUT-DLVCLab/lilt-roberta-en-base, is language-independent and can be pre-trained on structured documents of a single language and then fine-tuned on other languages. It has been shown to achieve competitive or superior performance on various downstream benchmarks, making it a versatile tool for structured document understanding.

### Compute Infrastructure

The compute infrastructure used for training the model SCUT-DLVCLab/lilt-roberta-en-base involved 4 NVIDIA A40 48GB GPUs. The model was trained using a batch size of 96 for 5 epochs on the IIT-CDIP dataset.

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

