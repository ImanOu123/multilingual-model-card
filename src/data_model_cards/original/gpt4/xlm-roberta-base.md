# Model Card for xlm-roberta-base

XLM-RoBERTa (XLM-R) is a state-of-the-art multilingual masked language model trained on 2.5TB of CommonCrawl data across 100 languages, designed to excel in a wide range of cross-lingual understanding tasks while also being competitive with monolingual models on language-specific benchmarks.

## Model Details

### Model Description

The `xlm-roberta-base` model, also known as `XLM-R`, is a powerful multilingual model that follows the architecture of BERT-base. It is designed to understand and encode sentences across a diverse set of languages, achieving state-of-the-art results on various cross-lingual understanding benchmarks.

### Model Architecture:
- The model is based on the Transformer architecture (Vaswani et al., 2017) and uses a multilingual Masked Language Model (MLM) objective for training, similar to the approach used by Devlin et al. (2018) in BERT and Lample and Conneau (2019) in XLM.
- It does not use language embeddings, which helps it handle code-switching scenarios more effectively.
- The base model consists of 12 layers (L = 12), with a hidden size of 768 (H = 768), and uses 12 attention heads (A = 12).

### Training Procedures:
- The model is trained on 2.5 terabytes of filtered CommonCrawl data in 100 languages, ensuring a broad linguistic coverage.
- It uses SentencePiece tokenization with a unigram language model and a large vocabulary size of 250,000 tokens.
- The training data is sampled with a temperature of α = 0.3 to balance the representation of languages in the training process.
- The model is trained for 1.5 million updates on five-hundred 32GB Nvidia V100 GPUs with a batch size of 8192.

### Parameters:
- The `xlm-roberta-base` model has approximately 250 million parameters, which allows it to capture complex patterns and nuances across different languages.

### Important Disclaimers:
- While the model aims to be inclusive by supporting 100 languages, the quality of the model's performance may vary across languages due to differences in the amount of training data available for each language.
- The model does not use language-specific embeddings, which may affect its performance on tasks that are highly sensitive to language-specific features.
- The model's large size and computational requirements for training may limit its accessibility for researchers or practitioners with limited resources.

For further details or updates on the `xlm-roberta-base` model, please refer to the official model card or contact the project organizer.

- **Developed by:** Alexis Conneau; Kartikay Khandelwal; Naman Goyal; Vishrav Chaudhary; Guillaume Wenzek; Francisco Guzmán; Edouard Grave; Myle Ott; Luke Zettlemoyer; Veselin Stoyanov; Facebook Ai
- **Funded by:** The funding for the project of the model `xlm-roberta-base` is not explicitly mentioned in the provided references. However, given that the authors listed in reference 4 (Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov) are associated with Facebook AI, it is reasonable to infer that Facebook AI provided funding or resources for the development of the `xlm-roberta-base` model. Without additional explicit information on funding sources, we would need to state "[More Information Needed]" for a definitive answer.
- **Shared by:** The contributors who made the model `xlm-roberta-base` available online as a GitHub repo include Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. These individuals are listed as authors in the referenced articles and are associated with Facebook AI, which is the team behind the development of the XLM-RoBERTa model.
- **Model type:** The XLM-RoBERTa-base model is a transformer-based multilingual masked language model trained on monolingual data using a multilingual MLM objective, falling under the category of unsupervised learning, and it processes textual modality.
- **Language(s):** The model xlm-roberta-base processes text in 100 different languages, including commonly used ones such as romanized Hindi and traditional Chinese.
- **License:** [More Information Needed]
- **Finetuned from model:** The model `xlm-roberta-base`, also referred to as `xlmr.base`, is not fine-tuned from another model but is trained from scratch using the BERT-base architecture. It is a part of the XLM-R (XLM-RoBERTa) family of models, which are generic cross-lingual sentence encoders. The base model itself is trained on `2.5T` of filtered CommonCrawl data in 100 languages. There is no indication in the provided references that `xlm-roberta-base` is fine-tuned from a pre-existing model; instead, it is an original model trained by the team.

For more information or to download the `xlm-roberta-base` model, you can use the following link: [xlm.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz).
### Model Sources

- **Repository:** https://github.com/pytorch/fairseq/tree/master/examples/xlmr
- **Paper:** https://arxiv.org/pdf/1911.02116.pdf
- **Demo:** [More Information Needed]
## Uses

### Direct Use

The `xlm-roberta-base` model can be used without fine-tuning, post-processing, or plugging into a pipeline for feature extraction tasks. This means that you can directly pass input data to the model to obtain the last layer hidden states, which can then be used for various downstream tasks such as clustering, similarity search, or as input features to other machine learning models.

Here's an example of how you can use the `xlm-roberta-base` model for feature extraction:

```python
from transformers import XLMRobertaModel, XLMRobertaTokenizer

# Load pre-trained model tokenizer (vocabulary)
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Encode some text in a sequence of tokens (for example in Chinese)
text = "你好世界"  # "Hello, world" in Chinese
zh_tokens = tokenizer(text, return_tensors='pt')

# Load pre-trained model (weights)
model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

# Get the last layer hidden states from the model
with torch.no_grad():
    last_layer_features = model(**zh_tokens).last_hidden_state

# Verify the size of the output
assert last_layer_features.size() == torch.Size([1, 6, 768])  # Note: The size here should match the model's output, which is [batch size, sequence length, hidden size]
```

In this code snippet, we first tokenize some input text using the `XLMRobertaTokenizer`. Then, we load the `XLMRobertaModel` and pass the tokenized input to it. The model returns the last layer hidden states, which we can use for our desired application. Note that the size of the last layer features should match the expected output shape, which is `[batch size, sequence length, hidden size]`. However, the hidden size in the reference is 1024, which corresponds to the larger `XLM-R` model, while the `xlm-roberta-base` model has a hidden size of 768. Therefore, the assertion should check for the correct hidden size of the base model.

Please note that the above code snippet assumes that you have the `transformers` library installed and that you are using the `xlm-roberta-base` model. If you are using a different setup or require a different model, the code may need to be adjusted accordingly.

### Downstream Use

The `xlm-roberta-base` model is a multilingual variant of the RoBERTa model pre-trained on a large corpus of text in 100 languages. It is designed to understand and generate text across multiple languages, making it particularly useful for cross-lingual tasks. When fine-tuned on a specific task, `xlm-roberta-base` can be adapted to perform a wide range of natural language processing (NLP) tasks such as text classification, named entity recognition (NER), and question answering in different languages.

For instance, if you want to fine-tune `xlm-roberta-base` for a NER task, you would start by loading the pre-trained model and then continue training it on a labeled NER dataset in the target language or languages. The fine-tuning process allows the model to specialize in identifying named entities within the context of the provided training data.

Similarly, for a question answering task, you would fine-tune the model on a dataset containing question-answer pairs, enabling the model to learn how to predict answers to questions based on the context provided in a passage of text.

Once fine-tuned, the model can be integrated into a larger ecosystem or application, such as a multilingual chatbot, a customer service automation system, or a cross-lingual information retrieval system. The model can be used to understand user queries in different languages and provide appropriate responses or information.

Here is a code snippet for loading the `xlm-roberta-base` model using PyTorch, as referenced in the provided materials (reference 8). However, please note that this code only loads the model and sets it to evaluation mode; it does not include the fine-tuning process, which would require additional code and a suitable dataset:

```python
import torch
xlmr = torch.hub.load('pytorch/fairseq:main', 'xlmr.base')
xlmr.eval()  # disable dropout (or leave in train mode to finetune)
```

To fine-tune the model, you would need to add a task-specific head on top of the pre-trained model, define a loss function, and train the model on your task-specific dataset. The exact code for fine-tuning would depend on the task and the dataset, and thus is not provided in the references. If you need assistance with the fine-tuning process, you would need to consult additional documentation or tutorials specific to your task.

### Out-of-Scope Use

As a sociotechnic on the deep learning model development team, it is my responsibility to consider the potential misuses of the `xlm-roberta-base` model and provide guidance to users on what they should avoid doing with the model. Here are some foreseeable misuses and recommendations:

1. **Bias Amplification**: Given that `xlm-roberta-base` is trained on CommonCrawl data, which is a reflection of the content available on the internet, there is a risk that the model may inadvertently amplify existing biases present in the training data. Users should not use the model in applications where such biases could lead to unfair or discriminatory outcomes without implementing appropriate bias mitigation strategies.

2. **Misrepresentation of Low-Resource Languages**: The model has been trained on 100 languages, but there is a trade-off between high-resource and low-resource languages (as mentioned in the references). Users should not assume equal performance across all languages, especially for low-resource ones, and should be cautious when using the model for tasks involving such languages to avoid misrepresentation or inaccurate results.

3. **Inappropriate Content Generation**: The model's ability to generate text can be misused to create inappropriate or harmful content. Users should not use `xlm-roberta-base` to generate or disseminate content that is offensive, illegal, or violates the rights of others.

4. **Over-reliance on Model Predictions**: The model, while state-of-the-art, is not infallible and its predictions should not be used as the sole basis for critical decision-making processes without human oversight and validation.

5. **Privacy Violations**: Users should not use the model to process sensitive or personal data without proper anonymization and consent, as the model's training does not specifically account for privacy-preserving measures.

6. **Language Homogenization**: The model's training involves adding similar higher-resource languages to improve low-resource language performance, which could lead to a homogenization of linguistic features. Users should not use the model in a way that disregards the unique linguistic and cultural aspects of individual languages.

7. **Misinterpretation of Model Limitations**: Users should not ignore the limitations of the model, such as the fixed vocabulary size and the potential for vocabulary dilution as the number of languages increases. It is important to understand these limitations when interpreting the model's outputs.

In summary, users of `xlm-roberta-base` should be mindful of the model's limitations, the potential for bias, and the ethical implications of its use. It is crucial to ensure that applications of the model are aligned with principles of fairness, accountability, and respect for privacy and cultural diversity.

### Bias, Risks, and Limitations

The known and foreseeable issues stemming from the `xlm-roberta-base` model can be categorized into technical and sociotechnical limitations:

1. **Curse of Multilinguality**: As indicated in references 2, 5, and 7, there is a trade-off between the number of languages supported by the model and the per-language capacity. For a fixed-sized model, the capacity allocated to each language decreases as more languages are added. This can lead to suboptimal performance, especially for low-resource languages, as the model may not have enough capacity to learn the nuances of all included languages effectively.

2. **Vocabulary Dilution**: Reference 2 also mentions the issue of vocabulary dilution. As the model supports more languages, the fixed-size vocabulary must be shared among them, which can lead to a dilution effect where the model has fewer tokens to represent each language, potentially reducing the model's effectiveness in understanding and generating language-specific content.

3. **High-Resource vs. Low-Resource Trade-off**: Reference 6 highlights the trade-off between high-resource and low-resource languages. While the performance of low-resource languages can be improved by adding similar high-resource languages during pretraining, this can negatively impact the overall downstream performance due to the limited model capacity.

4. **Transfer-Dilution Trade-off**: Reference 7 discusses the balance between positive transfer and capacity dilution. While adding more languages can initially lead to performance improvements due to positive transfer, beyond a certain point, the additional languages can degrade performance across all languages due to capacity dilution.

5. **Performance on Downstream Tasks**: Reference 4 and 9 suggest that multilingual models like `xlm-roberta-base` may not perform as well as monolingual models on certain downstream tasks. This is a critical consideration when deploying the model in real-world applications where performance on specific tasks is paramount.

6. **Sociotechnical Considerations**: The model's limitations in handling low-resource languages could lead to biases against speakers of these languages, potentially exacerbating existing inequalities in technology access and representation. Moreover, the model's performance may vary significantly across different languages, which could lead to misunderstandings or misinterpretations of the model's outputs by users who speak those languages.

7. **Data Source and Quality**: Reference 6 mentions that the model is trained on CommonCrawl data, which, while extensive, may contain biases or low-quality information that could be propagated through the model's outputs.

8. **Ethical and Legal Considerations**: [More Information Needed] - The references provided do not explicitly discuss ethical or legal considerations. However, it is important to note that any AI model, including `xlm-roberta-base`, may have ethical implications related to privacy, consent, and the potential for misuse in surveillance or censorship, which would require careful consideration by legal experts and ethicists.

In summary, `xlm-roberta-base` faces several technical challenges related to model capacity, vocabulary size, and the inherent trade-offs of multilingual support. Sociotechnical issues also arise from potential biases and inequalities that could be perpetuated or exacerbated by the model's limitations, particularly for low-resource languages.

### Recommendations

Based on the references provided, here are some recommendations with respect to the foreseeable issues about the model `xlm-roberta-base`:

1. **Performance on Low-Resource Languages**: While `xlm-roberta-base` shows significant improvements on low-resource languages such as Swahili and Urdu, it is important to continue monitoring and improving the model's performance on such languages. Future work could involve collecting more diverse and representative datasets for these languages to further enhance the model's capabilities.

2. **Cross-Lingual Transfer**: The model relies heavily on cross-lingual transfer. It is recommended to conduct further research to understand the limits of this transfer, especially for languages that are underrepresented in the training data. This could help in identifying any biases or weaknesses in the model's performance across different languages.

3. **Sampling Rate and Vocabulary Size**: The fixed vocabulary size of 150K and the sampling rate controlled by the α parameter have a significant impact on the model's performance. It is recommended to experiment with different vocabulary sizes and sampling rates, especially when adapting the model for specific languages or tasks, to find the optimal balance between high-resource and low-resource language performance.

4. **Trade-offs and Limitations**: The model presents trade-offs between positive transfer and capacity dilution, as well as between the performance of high and low-resource languages. It is important to communicate these trade-offs to users and to continue research on how to minimize capacity dilution while maximizing positive transfer.

5. **Comparison with Monolingual Models**: `xlm-roberta-base` should be regularly benchmarked against monolingual models to ensure that it maintains competitive performance. This is crucial as a recurrent criticism against multilingual models is their potential underperformance compared to monolingual counterparts.

6. **Ethical and Societal Considerations**: As a sociotechnic, it is important to consider the ethical and societal implications of deploying `xlm-roberta-base`. This includes ensuring that the model does not perpetuate or amplify biases present in the training data, and that it is used in a manner that respects the linguistic and cultural diversity of its users.

7. **Transparency and Documentation**: Comprehensive documentation should be provided, detailing the model's training data, methodology, performance across different languages, and any known limitations. This transparency is essential for users to make informed decisions about how to deploy the model effectively and ethically.

8. **Continuous Evaluation and Updates**: The model should undergo continuous evaluation to ensure its performance remains state-of-the-art. This includes updating the model with new data, addressing any discovered biases, and adapting to the evolving linguistic landscape.

In summary, while `xlm-roberta-base` has achieved state-of-the-art results, it is important to continue refining the model, to be transparent about its capabilities and limitations, and to consider the broader societal impact of its deployment.

## Training Details

### Training Data

The training data for the model `xlm-roberta-base` consists of over 2.5 terabytes of filtered CommonCrawl data across 100 languages, as detailed in references 1 and 6. This data was processed using a clean-up pipeline that involved language identification and filtering to ensure quality and relevance for language model training. The model does not use language embeddings, allowing it to handle code-switching scenarios more effectively, as mentioned in reference 4. [More Information Needed] on specific documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

For the preprocessing of the data used in training the `xlm-roberta-base` model, we employed the SentencePiece tokenization method. This approach allows us to apply tokenization directly on raw text data across all languages without the need for language-specific preprocessing tools. The SentencePiece model (SPM) we used is based on a unigram language model, which was chosen due to its simplicity and effectiveness, as indicated in reference 1 and 3.

We did not use language-specific byte-pair encoding (BPE) as the SentencePiece model was found to perform equally well without any loss in performance (reference 1). The model was trained using a large vocabulary size of 250,000 tokens, which is beneficial for capturing a wide range of linguistic phenomena across different languages (reference 3).

In terms of data sampling, we followed the approach of Lample and Conneau (2019), using a temperature-based sampling distribution with a temperature of α = 0.3. This sampling strategy ensures a balanced representation of languages in the training batches. Notably, we did not use language embeddings in our model, which allows `xlm-roberta-base` to handle code-switching scenarios more effectively (reference 3).

The training data itself was sourced from a clean CommonCrawl Corpus in 100 languages, following the methodology described by Wenzek et al. (2019). We utilized an internal language identification model in conjunction with the fastText model to ensure the quality of the data by filtering out documents that did not meet our criteria (reference 4).

In summary, the preprocessing for the `xlm-roberta-base` model involved the use of SentencePiece tokenization with a unigram language model on raw text data, a large vocabulary size, temperature-based sampling for language representation without language embeddings, and high-quality multilingual data sourced from CommonCrawl.

#### Training Hyperparameters

The training hyperparameters for the model `xlm-roberta-base` are as follows:

- **Architecture**: The model uses a Transformer architecture with the following specifications: 12 layers (L = 12), 768 hidden states (H = 768), and 12 attention heads (A = 12). This is inferred from the reference to "XLM-R Base (L = 12, H = 768, A = ..." in reference 4.

- **Vocabulary Size**: The model employs a large vocabulary size of 250,000 tokens, as mentioned in reference 4 where it states "We use a large vocabulary size of 250K with a full softmax..."

- **Training Objective**: The model is trained using the multilingual Masked Language Model (MLM) objective, which is a standard approach for pre-training language models as described in reference 2.

- **Sampling Distribution**: For sampling batches from different languages, the model uses a sampling distribution with α = 0.3, as specified in reference 4.

- **Language Coverage**: The model is trained on 100 languages, and while the full list of languages is not provided in the references, it is mentioned that the model includes commonly used languages such as romanized Hindi and traditional Chinese (reference 5).

- **Training Data**: The model is trained on a clean CommonCrawl Corpus in 100 languages, as mentioned in reference 3.

- **Tokenization**: SentencePiece tokenization with a unigram language model is used directly on raw text data, as stated in reference 4.

- **Language Embeddings**: Unlike previous models, `xlm-roberta-base` does not use language embeddings, which allows the model to better handle code-switching scenarios (reference 4).

- **Training Duration**: The exact number of training updates for the `xlm-roberta-base` model is not specified in the references provided. Therefore, [More Information Needed] regarding the number of updates or training steps.

- **Hardware and Batch Size**: The references do not provide specific information about the hardware used or the batch size for the `xlm-roberta-base` model. However, reference 8 mentions training a larger model (not the base model) on five-hundred 32GB Nvidia V100 GPUs with a batch size of 8192, but this does not directly apply to the `xlm-roberta-base` model. Therefore, [More Information Needed] regarding the hardware and batch size for the base model.

Please note that some details specific to the `xlm-roberta-base` model, such as the number of training updates, hardware used, and batch size, are not provided in the references and thus require further information.

#### Speeds, Sizes, Times

The model `xlm-roberta-base`, also referred to as `xlmr.base`, is a multilingual model that uses the BERT-base architecture. Here are the details based on the provided references:

- **Model Size**: The `xlm-roberta-base` model has approximately 250 million parameters, which is indicative of its capacity to handle complex language tasks across multiple languages.
  
- **Vocabulary Size**: The model has a vocabulary size of 250,000 tokens, which allows it to process text in a wide range of languages effectively.

- **Training Data**: It is trained on 2.5 terabytes of filtered CommonCrawl data, which includes text in 100 languages. This large and diverse dataset ensures that the model has broad language coverage and can perform well on cross-lingual tasks.

- **Evaluation Benchmarks**: The model has been evaluated on several benchmarks, including cross-lingual natural language inference, named entity recognition (NER), question answering, and the GLUE benchmark for English performance.

- **Performance**: The model has achieved state-of-the-art results on many cross-lingual understanding (XLU) benchmarks, demonstrating its effectiveness in processing and understanding multiple languages.

- **Training Details**: The model benefits from training MLMs for longer periods and with larger batch sizes. Performance on downstream tasks continues to improve even after validation perplexity has plateaued, which suggests that the model is robust to overfitting on pretraining tasks.

- **Checkpoint Size**: The download link provided for the `xlmr.base` model checkpoint is [xlm.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz). However, the exact size of the checkpoint file is not specified in the provided references.

- **Throughput, Start or End Time**: Specific details regarding the throughput, start or end time of the training process for the `xlm-roberta-base` model are not provided in the references. [More Information Needed]

- **Checkpoint Sizes**: The size of the model checkpoints for the `xlm-roberta-base` is not explicitly mentioned in the provided references. [More Information Needed]

In summary, the `xlm-roberta-base` model is a powerful multilingual model with a large number of parameters and a vast vocabulary size, trained on a substantial multilingual dataset, and has demonstrated excellent performance across various language tasks. Specific details about the training throughput, start/end times, and checkpoint sizes are not available in the provided references.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model `xlm-roberta-base` evaluates on the following benchmarks and datasets:

1. Cross-lingual Natural Language Inference (XNLI): The XNLI dataset includes groundtruth development and test sets in 15 languages, with an English training set that has been machine-translated into the other 14 languages.

2. Named Entity Recognition (NER): The model is evaluated using the CoNLL-2002 and CoNLL-2003 datasets for English, Dutch, Spanish, and German.

3. Cross-lingual Question Answering: The MLQA benchmark is used, which extends the English SQuAD benchmark to Spanish, German, Arabic, Hindi, Vietnamese, and Chinese.

4. GLUE Benchmark: The model's English performance is assessed on the GLUE benchmark, which includes multiple classification tasks such as MNLI, SST-2, and QNLI.

#### Factors

The model xlm-roberta-base is a multilingual model that has been trained with a focus on improving performance across a variety of languages, with particular attention to low-resource languages. Based on the references provided, the following characteristics are likely to influence the model's behavior:

1. **Language Resources**: The model's performance is influenced by the amount of training data available for each language. It has been pretrained on Common Crawl (CC), which has improved its performance, especially for low-resource languages like Swahili and Urdu, as indicated by a 7% and 4.8% improvement respectively (Reference 1). This suggests that the model may perform better for languages that have more extensive training data available.

2. **Cross-lingual Transfer**: The model relies on cross-lingual transfer to model languages, which can be beneficial for low-resource languages. However, the degree of transfer and the model's effectiveness can vary depending on the linguistic similarity between languages and the amount of shared vocabulary (Reference 1).

3. **Domain and Context**: The model has been evaluated on various benchmarks, including cross-lingual natural language inference, named entity recognition (NER), and question answering (Reference 4). Its performance in specific domains will depend on the representation of those domains in the training data and the model's ability to generalize from its training to new contexts.

4. **Evaluation Benchmarks**: The model has been fine-tuned and evaluated on datasets like CoNLL-2002 and CoNLL-2003 for NER in English, Dutch, Spanish, and German (Reference 2). Its performance on these tasks provides a measure of its capabilities in these specific languages and domains.

5. **Sampling Strategy**: The model's performance is affected by the sampling strategy used during training, which controls the rate at which examples from each language are sampled. This impacts the representation of high-resource and low-resource languages in the training data (Reference 6).

6. **Training Objective and Data**: The model uses a Transformer architecture trained with a multilingual masked language model (MLM) objective, using monolingual data from each language. The training objective and the nature of the data used can influence the model's behavior in terms of its ability to predict masked tokens and generalize across languages (Reference 7).

7. **Training Scale and Duration**: The scale of training data and the duration of training are important factors. Large-scale training and longer training times have been observed to improve model performance, suggesting that the model may continue to improve with additional training, even after standard stopping criteria are met (Reference 8).

In terms of evaluation and uncovering disparities in performance, it is important to disaggregate evaluation across languages, domains, and tasks. This would help identify any biases or weaknesses the model may have towards certain languages or contexts. For instance, performance metrics should be reported separately for high-resource and low-resource languages, and for different types of tasks (e.g., NER, question answering) to provide a comprehensive understanding of the model's capabilities and limitations. Without further disaggregated evaluation data, [More Information Needed] to make definitive statements about disparities in the model's performance across different subgroups.

#### Metrics

The evaluation metrics used for the model xlm-roberta-base will include the F1 score and the exact match (EM) score for cross-lingual transfer tasks, as well as average accuracy for multilingual masked language models. Specifically:

1. Named Entity Recognition (NER): We will use the F1 score to evaluate the model's performance on the CoNLL-2002 and CoNLL-2003 datasets in English, Dutch, Spanish, and German. The model will be fine-tuned and evaluated in three different settings: on the English set for cross-lingual transfer, on each language set for per-language performance, and on all sets combined for multilingual learning.

2. Cross-lingual Question Answering: For the MLQA benchmark, which includes languages such as Spanish, German, Arabic, Hindi, Vietnamese, and Chinese, we will report both the F1 score and the exact match (EM) score to assess the model's ability to transfer knowledge from English to other languages.

3. GLUE Benchmark: To evaluate the English performance of xlm-roberta-base, we will use the GLUE benchmark, which consists of multiple classification tasks. The specific metrics for GLUE vary by task but generally include accuracy, F1 score, and correlation coefficients, depending on the nature of the task (e.g., MNLI, SST-2, QNLI).

4. XNLI: For the cross-lingual natural language inference task, we will report average accuracy as a key metric.

The tradeoffs between different errors are not explicitly mentioned in the provided references, so we cannot provide a detailed analysis of how the model balances these tradeoffs without additional information. However, the choice of metrics like F1 score, which balances precision and recall, suggests an attempt to balance type I and type II errors in tasks like NER and question answering. For the GLUE and XNLI benchmarks, accuracy and average accuracy are used, which do not directly address the tradeoff between different types of errors but provide a general measure of model performance.

### Results

The evaluation results of the model `xlm-roberta-base` based on the factors and metrics mentioned in the references are as follows:

1. **Cross-lingual Natural Language Inference (XNLI)**: The model was evaluated on the XNLI benchmark, which is used to assess cross-lingual understanding. The performance of `xlm-roberta-base` was compared to monolingual BERT models on 7 languages, and it was found that multilingual models like `xlm-roberta-base` can outperform their monolingual counterparts, especially when the vocabulary size is increased for a fair comparison.

2. **Named Entity Recognition (NER)**: For NER, the model was evaluated using the CoNLL-2002 and CoNLL-2003 datasets in English, Dutch, Spanish, and German. The model was fine-tuned in three different settings: on the English set for cross-lingual transfer, on each language set for per-language performance, and on all sets for multilingual learning. The F1 score was reported and compared to baselines from previous studies.

3. **Cross-lingual Question Answering**: The model was tested on the MLQA benchmark, which extends the English SQuAD benchmark to other languages including Spanish, German, Arabic, Hindi, Vietnamese, and Chinese. The evaluation metrics used were the F1 score and the exact match (EM) score for cross-lingual transfer from English.

4. **GLUE Benchmark**: The English performance of `xlm-roberta-base` was evaluated on the GLUE benchmark, which includes multiple classification tasks such as MNLI, SST-2, and QNLI. The model's performance was compared to RoBERTa, with the goal of achieving strong performance on both cross-lingual understanding tasks and natural language understanding tasks for each language.

Unfortunately, specific numerical results and detailed comparisons for each of these benchmarks are not provided in the references, so [More Information Needed] to give exact figures or a more detailed analysis of the model's performance.

#### Summary

The evaluation results for the `xlm-roberta-base` model can be summarized as follows:

1. Cross-lingual Understanding Benchmarks: The model was evaluated on tasks such as cross-lingual natural language inference, named entity recognition (NER), and question answering. For NER, it was tested on the CoNLL-2002 and CoNLL-2003 datasets in English, Dutch, Spanish, and German, with the model fine-tuned in three different settings: on English data for cross-lingual transfer, on each language's data for per-language performance, and on all datasets for multilingual learning. The model's performance was measured using the F1 score and compared favorably against baselines from previous studies.

2. GLUE Benchmark: `xlm-roberta-base` was also evaluated on the GLUE benchmark to assess its performance on English natural language understanding tasks. The model aimed to achieve strong performance not only in cross-lingual tasks but also in monolingual (English) tasks.

3. XNLI Dataset: The model was tested on the XNLI dataset, which includes ground-truth development and test sets in 15 languages and a ground-truth English training set that has been machine-translated into the other 14 languages. The model was evaluated on its ability to transfer learning from English to other languages.

4. Comparison with Monolingual Models: The study compared the performance of `xlm-roberta-base` with monolingual BERT models on the XNLI benchmark across 7 languages. The results indicated that multilingual models like `xlm-roberta-base` can outperform their monolingual counterparts, challenging the criticism that multilingual models generally perform worse.

5. Training Data and Languages: `xlm-roberta-base` was trained on 2.5 terabytes of filtered CommonCrawl data across 100 languages, making it a highly diverse and robust model for cross-lingual understanding tasks.

In summary, `xlm-roberta-base` has demonstrated state-of-the-art results on various cross-lingual understanding benchmarks, showing strong performance in both cross-lingual transfer and English-specific tasks, and has even outperformed monolingual models in certain scenarios.

## Model Examination

Model Card: xlm-roberta-base

### Explainability/Interpretability

The xlm-roberta-base model is a multilingual model designed to understand and generate text across multiple languages. It is based on the RoBERTa architecture and has been trained on a large corpus of text from various languages. The model's performance has been evaluated on several benchmarks, including XNLI for cross-lingual understanding and the GLUE benchmark for natural language understanding tasks.

#### Model Capacity and Language Representation

The model has been pretrained on Wikipedia data in multiple languages, and its capacity has been scaled to accommodate the addition of languages. As the number of languages increased, the hidden size of the Transformer was also increased to mitigate the "curse of multilinguality," which refers to the dilution of language-specific features when a model is trained on many languages ([Reference 4](#)). However, the added capacity for xlm-roberta-base was carefully balanced to ensure that it does not lag behind due to higher vocabulary dilution.

#### Vocabulary and Sampling

The fixed vocabulary size of 150K subwords was used for all models, including xlm-roberta-base, to ensure a consistent representation across languages ([Reference 2](#)). The model employs an exponential smoothing of the language sampling rate, with a sampling rate proportional to the number of sentences in each corpus. The α parameter was optimized to balance the performance on high-resource and low-resource languages, with an optimal value of 0.3 used for xlm-roberta-base ([Reference 5](#)).

#### Cross-Lingual and Monolingual Performance

The xlm-roberta-base model has been compared to monolingual models to address concerns that multilingual models may underperform in comparison. The model has shown strong performance on cross-lingual tasks and is competitive with monolingual models on language-specific benchmarks ([Reference 6](#)).

#### Data and Training Improvements

The training data for xlm-roberta-base includes cleaned CommonCrawls, which significantly increase the amount of data available for low-resource languages. This approach has been shown to be effective for learning high-quality word embeddings in multiple languages and has contributed to the model's improved performance ([Reference 8](#)).

#### Limitations and Trade-offs

The development of xlm-roberta-base involved studying the trade-offs and limitations of scaling multilingual models to a large number of languages. The model's design reflects a balance between the need for language-specific capacity and the benefits of shared multilingual representation ([Reference 3](#)).

In summary, the xlm-roberta-base model is a robust multilingual model that has been carefully tuned to provide strong performance across a wide range of languages and tasks. While the model has been optimized for both high-resource and low-resource languages, further research into explainability and interpretability is necessary to fully understand the model's decision-making processes and to improve its transparency.

[More Information Needed] on specific methods or approaches used for explainability/interpretability of the xlm-roberta-base model, as the provided references do not contain detailed information on this aspect.

## Environmental Impact

- **Hardware Type:** The model xlm-roberta-base is trained on five-hundred 32GB Nvidia V100 GPUs.
- **Software Type:** The model `xlm-roberta-base` is trained on the software type of a Transformer model, as mentioned in the first reference. Specifically, it follows the approach of using a multilingual Masked Language Model (MLM) objective with a Transformer architecture.
- **Hours used:** [More Information Needed]
- **Cloud Provider:** The cloud provider that the model xlm-roberta-base is trained on is not explicitly mentioned in the provided references. Therefore, the answer is "[More Information Needed]".
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain information regarding the carbon emissions associated with the training of the xlm-roberta-base model. To calculate the carbon emissions, we would need data on the energy consumption of the hardware used during training, the duration of the training, and the energy mix (carbon intensity) of the electricity used.
## Technical Specification

### Model Architecture and Objective

The model architecture of `xlm-roberta-base` is based on the Transformer model (Vaswani et al., 2017) with the following specifications: it has 12 layers (L = 12), 768 hidden states (H = 768), and uses attention with 12 attention heads (A = 12). The model employs a large vocabulary size of 250,000 tokens and is trained without language embeddings to better handle code-switching scenarios.

The training objective for `xlm-roberta-base` is a multilingual Masked Language Model (MLM), following the approach of Devlin et al. (2018) and Lample and Conneau (2019). This involves predicting masked tokens in the input text, using monolingual data from various languages. The model is trained on streams of text sampled from each language, with a sampling distribution that uses a temperature of α = 0.3.

[More Information Needed] for any additional specifics not covered by the provided references.

### Compute Infrastructure

The compute infrastructure used for training the `xlm-roberta-base` model involved five-hundred 32GB Nvidia V100 GPUs with a batch size of 8192. The model was trained for 1.5 million updates.

## Citation

```
@misc{alexis-unsupervised,
    author = {Alexis Conneau and
              Kartikay Khandelwal and
              Naman Goyal and
              Vishrav Chaudhary and
              Guillaume Wenzek and
              Francisco Guzmán and
              Edouard Grave and
              Myle Ott and
              Luke Zettlemoyer and
              Veselin Stoyanov and
              Facebook Ai},
    title  = {Unsupervised Cross-lingual Representation Learning at Scale},
    url    = {https://arxiv.org/pdf/1911.02116.pdf}
}
```

