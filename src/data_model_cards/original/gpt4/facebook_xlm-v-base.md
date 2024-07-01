# Model Card for facebook/xlm-v-base

The model `facebook/xlm-v-base` is a multilingual language model with a one million token vocabulary, trained on 2.5TB of data from Common Crawl, designed to overcome the vocabulary bottleneck in multilingual models and to provide more semantically meaningful tokenizations, particularly benefiting low-resource languages. It has demonstrated superior performance over its predecessor XLM-R on a variety of natural language processing tasks.

## Model Details

### Model Description

Model Name: `facebook/xlm-v-base`

### Model Architecture:
The `facebook/xlm-v-base` model is a transformer-based encoder with 12 layers, equivalent in size to the XLM-R base model. It is a multilingual language model designed to handle over 100 languages.

### Training Procedures:
- The model was pretrained on the CC100 dataset using a sampling temperature of 0.3 to increase exposure to low- and medium-resource languages during training.
- The Adam optimizer was used with β1 = 0.9, β2 = 0.98, and ϵ = 1e-6.
- The learning rate was set to 6e-4 with a warmup of 15,000 steps.
- Training was distributed across 256 A100 GPUs with a batch size of 8,192.
- The model was trained for a total of 1.5 million iterations.
- Each batch consisted of examples concatenated up to a maximum sequence length of 512 tokens.
- The Masked Language Model (MLM) task was used for pretraining with a standard masking rate of 15%.

### Parameters:
- Vocabulary Size: 1 million tokens.
- Number of Layers: 12.
- Training Iterations: 1.5 million.
- Batch Size: 8,192.
- Sequence Length: Up to 512 tokens.

### Important Disclaimers:
- The model aims to reduce over-tokenization for low-resource languages by de-emphasizing token sharing between languages with little lexical overlap and assigning vocabulary capacity to ensure sufficient coverage for each language.
- Tokenizations using the XLM-V vocabulary are typically more semantically meaningful and shorter compared to XLM-R.
- XLM-V outperforms XLM-R on a variety of multilingual language understanding tasks, including XNLI, WikiAnn, MLQA, XQuAD, and TyDiQA, by an average of 3.5 points absolute.
- The model performs particularly well on low-resource evaluation datasets like AmericasNLI and MasakhaNER, outperforming XLM-R by 5.8% absolute accuracy and 11.2% absolute F1, respectively.
- The model's performance improvements are attributed to the improved vocabulary and training approach, which may not generalize to all possible tasks or datasets.

For further details and updates, the model documentation and updates can be found on the Huggingface Model Hub at [`facebook/xlm-v-base`](https://huggingface.co/facebook/xlm-v-base) and in the Transformers documentation [here](https://huggingface.co/docs/transformers/main/en/model_doc/xlm-v).

- **Developed by:** Davis Liang; Hila Gonen; Yuning Mao; Rui Hou; Naman Goyal; Marjan Ghazvininejad; Luke Zettlemoyer; Madian Khabsa; Meta Ai
- **Funded by:** The project for the model `facebook/xlm-v-base` is funded by Meta AI, as indicated by the authors' affiliations in the paper and the organization under which the model is published on Huggingface. The authors listed in the reference, such as Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, and Madian Khabsa, are associated with Meta AI.
- **Shared by:** The contributors that made the model `facebook/xlm-v-base` available online as a GitHub repo include Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, Madian Khabsa, and the Meta AI team.
- **Model type:** The model `facebook/xlm-v-base` is a multilingual masked language model trained using a large-scale unsupervised learning method on text data, representing a single modality.
- **Language(s):** The model facebook/xlm-v-base processes text in 116 languages from the CC100 corpus, with a focus on achieving semantically meaningful tokenizations and improved performance on a variety of natural language processing tasks.
- **License:** [More Information Needed]
- **Finetuned from model:** The model facebook/xlm-v-base is not fine-tuned from another model but is trained from scratch. It is a multilingual language model with a one million token vocabulary trained on 2.5TB of data from Common Crawl, as mentioned in the provided references. Therefore, there is no base model to provide a name or link for.
### Model Sources

- **Repository:** https://github.com/stefan-it/xlm-v-experiments
- **Paper:** https://arxiv.org/pdf/2301.10472.pdf
- **Demo:** The demo for the model `facebook/xlm-v-base` can be found on the Hugging Face Model Hub at the following link:

[facebook/xlm-v-base demo](https://huggingface.co/facebook/xlm-v-base)
## Uses

### Direct Use

The `facebook/xlm-v-base` model can be used without fine-tuning or post-processing by leveraging the pre-trained weights for tasks like masked language modeling (MLM). Since the model is trained on a diverse set of languages, it can be used to predict the masked token in a sentence for any of the languages it was trained on. Here's how you can use the model directly with the `transformers` library from Hugging Face:

```python
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Tokenize input
text = "Paris is the <mask> of France."
input_ids = tokenizer.encode(text, return_tensors='pt')

# Load pre-trained model (weights)
model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base')

# Predict all tokens
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs[0]

# Find the predicted token (we mask only one token, so we are interested in the first mask)
masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

# Replace <mask> with the predicted token
result = text.replace(tokenizer.mask_token, predicted_token)
print(result)
```

Please note that the above code snippet assumes that the `facebook/xlm-v-base` model is compatible with the `XLMRobertaTokenizer` and `XLMRobertaForMaskedLM` classes from the `transformers` library. If the model has specific compatibility requirements or if it's not officially integrated into the `transformers` library, you would need to follow the appropriate steps to load the model and tokenizer, which might be different from the provided code snippet. In such a case, you would refer to the official documentation or the model card for the correct usage instructions. If the model is not yet integrated and no specific instructions are provided, you would need to say "[More Information Needed]".

### Downstream Use

The `facebook/xlm-v-base` model is a multilingual language model that can be fine-tuned for various natural language processing (NLP) tasks such as question answering (QA), named entity recognition (NER), and natural language inference (XNLI). When fine-tuned, the model can be adapted to specific datasets and tasks, leveraging its pre-trained knowledge of multiple languages to improve performance.

For example, to fine-tune the model for a NER task, you would typically start with the pre-trained `facebook/xlm-v-base` model and continue training it on a labeled NER dataset. The fine-tuning process involves adjusting the model's weights based on the task-specific data, which allows the model to specialize in identifying named entities within the input text.

Once fine-tuned, the model can be integrated into a larger ecosystem or application, such as a multilingual chatbot, a content analysis tool, or a system for cross-lingual information retrieval. The model's ability to understand and generate text in multiple languages makes it particularly useful for applications serving diverse user bases.

Here is an example of how you might use the model in a Python application for the fill-mask task, based on the provided reference:

```python
from transformers import pipeline

# Load the fine-tuned model
unmasker = pipeline('fill-mask', model='facebook/xlm-v-base')

# Use the model to predict the masked word
result = unmasker("Paris is the <mask> of France.")

# Output the result
print(result)
```

Please note that the above code snippet assumes that the model has already been fine-tuned for the fill-mask task and is available on the Hugging Face Model Hub. If the model needs to be fine-tuned for a specific task, additional code and a task-specific dataset would be required.

### Out-of-Scope Use

As a sociotechnic on the deep learning model development team, it is my responsibility to consider the potential misuses of the facebook/xlm-v-base model and provide guidance to users on what they should not do with the model. Here are some foreseeable misuses and associated guidance:

1. **Biased or Discriminatory Applications**: Given that XLM-V has been trained on data from Common Crawl, it may have inherited biases present in the training data. Users should not use the model in ways that could amplify or perpetuate these biases, especially in sensitive applications such as hiring, law enforcement, or loan approvals.

2. **Misrepresentation of Language Capabilities**: While XLM-V has a large vocabulary and performs well on multiple languages, users should not overstate its capabilities, especially for low-resource languages. It is important to acknowledge the limitations of the model and not use it as a definitive authority on language understanding.

3. **Privacy Violations**: Users should not use XLM-V to analyze private or sensitive text data without proper consent and consideration of privacy laws and regulations. The model's ability to understand and generate text could be misused to infer private information about individuals.

4. **Generation of Harmful Content**: The model should not be used to generate harmful, abusive, or misleading content. This includes but is not limited to fake news, phishing emails, or propaganda that could be used to deceive or harm individuals or groups.

5. **Intellectual Property Infringement**: Users should not use the model to generate content that infringes on the intellectual property rights of others, such as automatically generating articles, books, or other creative works that are derivative of copyrighted material.

6. **Security Risks**: Given the model's capabilities, there is a risk of it being used in automated cyber-attacks, such as generating phishing emails or creating more convincing social engineering tactics. Users should not use the model for any form of cybercrime.

7. **Unfair Competition**: The model should not be used to create an unfair competitive advantage by generating large volumes of content to drown out competitors or manipulate search engine rankings.

In conclusion, while XLM-V is a powerful multilingual language model, it is crucial that users employ it responsibly and ethically. Misuse of the model can lead to serious societal and individual harm, and it is the responsibility of all users to ensure that their applications of XLM-V are in line with ethical guidelines and do not infringe upon the rights and well-being of others.

### Bias, Risks, and Limitations

The known and foreseeable issues stemming from the model facebook/xlm-v-base, as per the references provided, can be categorized into technical and sociotechnical limitations:

Technical Limitations:
1. Scalability Issues: The model's approach to scaling the vocabulary as a form of conditional compute can lead to increased pre-training times due to the computational complexity of the softmax over the entire vocabulary. This could be mitigated by adopting approximation techniques like adaptive softmax and adaptive inputs, but these solutions have not yet been implemented.
2. Memory Footprint: The large vocabulary size significantly increases the memory footprint of the model. While this is less of a concern for larger models where the number of non-embedding parameters outweighs the size of the vocabulary embedding matrix, it remains a challenge for deploying the model in resource-constrained environments.
3. Pre-training Resource Requirements: The model is trained on a single A100 GPU with float16 precision, which indicates that substantial computational resources are required for pre-training. This could limit the ability of researchers with less access to computational power to replicate or build upon the model.

Sociotechnical Limitations:
1. Language Representation: While the model aims to provide sufficient coverage for each individual language by de-emphasizing token sharing between languages with little lexical overlap, there may still be biases in representation, particularly for low-resource languages. This could result in uneven performance across different languages and potential marginalization of certain linguistic communities.
2. Misunderstandings and Misuse: Given the complexity of the model and its multilingual capabilities, there is a risk of misuse or misunderstanding of its outputs by users who may not fully grasp the nuances of language-specific contexts. This could lead to the propagation of misinformation or misinterpretation of the model's results.
3. Ethical Considerations: The use of a large dataset from Common Crawl raises questions about the ethical implications of training on data that may contain sensitive or private information. There is also the potential for the model to inadvertently perpetuate or amplify societal biases present in the training data.

In conclusion, while facebook/xlm-v-base presents significant advancements in multilingual language modeling, it also brings forth challenges related to scalability, resource requirements, language representation, potential misuse, and ethical considerations. Addressing these issues will require ongoing research and careful consideration of the sociotechnical impact of deploying such models.

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model facebook/xlm-v-base:

1. **Scalability of Vocabulary**: The model currently faces scalability issues due to the computational complexity of the softmax over the entire vocabulary during pre-training. To address this, it is recommended to adopt approximation techniques such as adaptive softmax (Joulin et al., 2017) and adaptive inputs (Baevski and Auli, 2018). These techniques can help reduce the computational burden and potentially decrease pre-training times.

2. **Memory Footprint**: The large vocabulary size significantly increases the memory footprint of the model. However, it is believed that as models become larger, the relative size of the vocabulary embedding matrix will be less of an issue compared to the number of non-embedding parameters. Therefore, the recommendation is to continue working with larger models where the memory impact of the vocabulary is proportionally smaller.

3. **Integration with Fairseq**: As of the last update, XLM-V is not officially integrated into the `fairseq` library, although it can be loaded with it. There is an open merge request to add the model and a usage readme into `fairseq`. It is recommended to monitor the progress of this merge request and contribute if necessary to ensure smooth integration.

4. **Further Research on Vocabulary Expansion**: The paper suggests that future work could involve increasing the vocabulary beyond 2M tokens while also using more data to investigate the Zipf ceiling discussed in Section 6. This could potentially improve the model's performance even further, especially for low-resource languages.

5. **Monitoring and Sharing Progress**: The first author of the XLM-V paper has shared the model weights via a tweet, indicating the importance of community engagement and transparency. It is recommended to continue sharing updates, progress, and resources with the community to facilitate collaboration and feedback.

6. **Performance on Low-Resource Languages**: XLM-V has shown outsized gains on tasks in low-resource languages. It is recommended to continue focusing on these languages to further enhance the model's capabilities and address the needs of underrepresented language communities.

7. **Semantic Tokenizations**: The model's vocabulary results in more semantically meaningful tokenizations and reduces average sequence length. It is recommended to leverage this strength in applications where semantic precision and efficiency are particularly important.

In summary, the recommendations include adopting approximation techniques for scalability, focusing on larger models to mitigate memory issues, contributing to the integration with `fairseq`, expanding the vocabulary in future research, engaging with the community, focusing on low-resource languages, and leveraging the model's strengths in semantic tokenizations.

## Training Details

### Training Data

The training data for the model facebook/xlm-v-base consists of the CC100 dataset, a multilingual corpus containing 2.5 TB of data split between 116 languages, which was used exclusively for constructing vocabularies and pretraining the model. This dataset was created from Common Crawl dumps and is designed to provide a diverse range of language examples, with a sampling temperature of 0.3 used during training to increase exposure to low- and medium-resource languages.

### Training Procedure

#### Preprocessing

The preprocessing for the data of the model `facebook/xlm-v-base` involves several steps to ensure that the tokenization is consistent and optimized for multilingual support. Here's a detailed description of the tokenization and preprocessing steps:

1. **Tokenizer Consistency**: The tokenizer used with `facebook/xlm-v-base` is designed to output the same ids/subtokens as the `fairseq` tokenizer. To ensure this consistency, the `xlm_v_tokenizer_comparison.py` script is utilized. This script loads sentences from all 176 languages in the WikiANN dataset, tokenizes each sentence, and compares the output to ensure that the tokenizer behaves as expected.

2. **Monolingual SentencePiece Models**: Individual monolingual SentencePiece models (SPM) are trained for each language using the Unigram Language Model (ULM) algorithm. This involves starting with a large initial vocabulary and iteratively pruning it to maximize the likelihood of the training corpus until the vocabulary size falls below a predetermined threshold.

3. **Lexical Representation Vectors**: Once the monolingual vocabularies are created, lexical representation vectors for each language are constructed. These vectors are used in the subsequent clustering step.

4. **Clustering and Vocabulary Capacity Allocation**: Languages are clustered using K-Means clustering with k=8, based on experiments showing that this number of clusters results in the best performance on downstream tasks. The vocabulary capacity for each cluster is determined using the ALP (Allocation of Language Probability) formula, which is correlated with downstream task performance. For low-resource languages not covered in previous work, a 2,000 token vocabulary budget is allocated.

5. **Final Multilingual Vocabulary**: Per-cluster SentencePiece models are trained, and the vocabularies from each cluster are combined into a single multilingual vocabulary. The final vocabulary consists of 901,629 unique tokens, with an overlap of 98,371 tokens between the clusters.

6. **Tokenization Quality**: The tokenizations using the constructed vocabulary are designed to be more semantically meaningful and shorter compared to those from XLM-R. This is achieved by de-emphasizing token sharing between languages with little lexical overlap and assigning vocabulary capacity to ensure sufficient coverage for each individual language.

7. **Training the Model**: Leveraging the improved vocabulary, `facebook/xlm-v-base` is trained as a multilingual language model with a one million token vocabulary. The model outperforms XLM-R on a variety of natural language processing tasks.

8. **Reuse of XLM-R Resources**: For the experiments, the publicly available XLM-R sentencepiece model and pretrained model checkpoint from `fairseq` are reused. This implies that the XLM-R vocabulary, which was created using the ULM algorithm on a corpus of 1 billion lines of text from CC100, is also utilized as a starting point or reference for the `facebook/xlm-v-base` model.

In summary, the preprocessing for `facebook/xlm-v-base` involves careful construction of a multilingual vocabulary that balances the need for cross-lingual subword sharing with robust representation of individual languages, resulting in a tokenizer that produces efficient and meaningful tokenizations for a wide range of languages.

#### Training Hyperparameters

The training hyperparameters used for the facebook/xlm-v-base model are as follows:

- **Optimizer**: Adam optimizer with β1 = 0.9, β2 = 0.98, and ϵ = 1e-6.
- **Learning Rate**: 6e-4.
- **Warmup Steps**: 15,000 steps.
- **Batch Size**: 8,192, distributed across 256 A100 GPUs.
- **Training Iterations**: 1.5 million iterations.
- **Sequence Length**: Maximum sequence length of 512.
- **Masked Language Model (MLM) Task**: Standard masking rate of 15%.

The model was pretrained without using any approximation tricks for the softmax layer, despite the increased vocabulary size and computational demands.

#### Speeds, Sizes, Times

The model `facebook/xlm-v-base` has been pretrained with a significant amount of computational resources and data. Here are the details regarding the throughput, start or end time, checkpoint sizes, and other relevant information:

- **Pretraining Timeframe**: The initial version of the repository was created on February 5, 2023. The model was made available under the Meta AI organization on Huggingface on May 8, 2023. This suggests that the pretraining likely occurred within this timeframe. However, the exact start or end time of the pretraining process is not specified in the provided references, so [More Information Needed] for precise dates.

- **Training Iterations**: The model was trained for a total of 1.5 million iterations.

- **Batch Size and GPUs**: During pretraining, the model used a batch size of 8,192, distributed across 256 A100 GPUs.

- **Learning Rate and Warmup**: A learning rate of 6e-4 was used with a warmup of 15,000 steps.

- **Optimizer**: The Adam optimizer was used with β1 = 0.9, β2 = 0.98, and ϵ = 1e-6.

- **Sequence Length**: Each batch consisted of examples concatenated up to the maximum sequence length of 512 tokens.

- **Checkpoint Size**: The checkpoint size is not explicitly mentioned in the provided references. Therefore, [More Information Needed] for the exact checkpoint sizes.

- **Throughput**: The throughput, which refers to the number of examples processed per second, is not directly mentioned in the provided references. Therefore, [More Information Needed] for specific throughput metrics.

- **Model Availability**: The model weights can be downloaded via the provided wget command:
  ```bash
  $ wget https://dl.fbaipublicfiles.com/fairseq/xlmv/xlmv.base.tar.gz
  ```
  The script `convert_xlm_v_original_pytorch_checkpoint_to_pytorch.py` is required to convert these weights into a Huggingface Transformers PyTorch model.

- **Integration with Fairseq**: As of the latest information provided, XLM-V is not officially integrated into the `fairseq` library, but there is an open merge request that adds the model and a usage readme into `fairseq`.

For more detailed information regarding the throughput, checkpoint sizes, and exact pretraining start or end times, additional data would be required that is not present in the provided references.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model `facebook/xlm-v-base` evaluates on the following benchmarks or datasets:

1. XNLI (Cross-lingual Natural Language Inference) for natural language inference tasks.
2. MLQA (Multilingual Question Answering) for question answering tasks.
3. XQuAD (Cross-lingual Question Answering Dataset) for question answering tasks in multiple languages.
4. TyDiQA (Typologically Diverse Question Answering) for question answering tasks across typologically diverse languages.
5. WikiAnn for named entity recognition tasks.
6. AmericasNLI for natural language inference tasks in low-resource languages of the Americas.
7. MasakhaNER for named entity recognition tasks in African languages.

These datasets are used to evaluate the performance of the model across various natural language understanding tasks, including question answering, natural language inference, and named entity recognition in multiple languages.

#### Factors

The model facebook/xlm-v-base is a multilingual language model with a large vocabulary size of 1 million tokens, designed to handle a variety of natural language processing tasks across multiple languages. Here are some characteristics that will influence its behavior:

1. **Domain and Context**: The model has been trained on datasets like XQuAD, MLQA, and XNLI, which are derived from Wikipedia and crowd-sourced annotations. This suggests that the model is likely to perform best on tasks related to general knowledge and encyclopedic content. Its performance may vary when applied to domain-specific contexts, such as legal, medical, or technical texts, where specialized vocabulary and knowledge are required.

2. **Population Subgroups**: Given that XLM-V has been shown to outperform its predecessor XLM-R, especially in low-resource languages, it is expected to be more effective for population subgroups speaking such languages. However, the model's performance may still be influenced by the amount and quality of data available for each language. High-resource languages with more training data are likely to yield better results than low-resource languages.

3. **Disaggregated Evaluation**: The model's evaluation on tasks like natural language inference, question answering, and named entity recognition across different languages suggests that its performance can vary by language. Disaggregated evaluation across languages is crucial to uncover disparities. For instance, the zero-shot cross-lingual transfer results on MasakhaNER indicate that the model's ability to generalize from English to unseen languages can be assessed, revealing potential performance gaps.

4. **Scalability Issues**: The model faces scalability challenges due to the computational complexity of the softmax over the entire vocabulary and the increased memory footprint. These issues may affect the model's deployment in resource-constrained environments or when scaling to even larger vocabularies and datasets.

5. **Tokenization Quality**: The improved tokenization using the model's vocabulary is more semantically meaningful and shorter compared to XLM-R, which could lead to better performance on tasks by reducing sequence length and improving the semantic representation of input texts.

In summary, the facebook/xlm-v-base model is expected to perform well across a range of languages and NLP tasks, with particular strengths in handling low-resource languages. However, its behavior will be influenced by the domain and context of the tasks, the availability and quality of data for each language, and the computational resources available for deploying the model. Disaggregated evaluation across these factors is essential to fully understand and address any disparities in the model's performance.

#### Metrics

The evaluation metrics used for the model facebook/xlm-v-base include accuracy and F1 score. These metrics are mentioned in the context of various downstream tasks such as natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), and named entity recognition (WikiAnn, MasakhaNER). Specifically, accuracy is used for tasks like XNLI and AmericasNLI, while F1 score is utilized for tasks that involve entity recognition, such as MasakhaNER. The model has been shown to outperform baselines and previous models like XLM-R on these metrics across different tasks and languages, with particularly notable improvements on low-resource evaluation datasets.

### Results

Evaluation results of the model facebook/xlm-v-base are as follows:

1. **Performance on Multilingual Language Understanding Tasks**: XLM-V demonstrates superior performance over XLM-R across various multilingual language understanding tasks. On average, XLM-V outperforms XLM-R by 3.5 points absolute, using metrics such as accuracy or F1 score depending on the task (Reference 1, 7).

2. **Cross-Lingual Transfer**: In cross-lingual transfer scenarios, where the model is trained on English and evaluated on other languages, XLM-V consistently outperforms XLM-R. This is evident from the results on all languages tested, with similar improvements noted on translate-train-all setups (Reference 1).

3. **Performance on Low-Resource Languages**: XLM-V shows significant improvements in low-resource languages. For instance, on the XNLI dataset, there is a 4.7% accuracy improvement for Swahili and a 2.9% improvement for Urdu. On the MasakhaNER dataset, which is a low-resource African language NER dataset, there is an average gain of 11.2% F1 (Reference 3).

4. **Americas NLI Results**: In zero-shot cross-lingual transfer on the Americas NLI dataset, XLM-V achieves an 18.2% absolute F1 improvement on Quechua and a 17.2% absolute improvement on Guaraní. These languages also saw the largest relative drop in average token count per sentence, indicating a more efficient tokenization by XLM-V compared to XLM-R (Reference 4).

5. **Tokenization Efficiency**: XLM-V produces tokenizations that are more semantically meaningful and shorter in length compared to XLM-R. This improved tokenization contributes to the model's overall performance and efficiency (Reference 6).

6. **Training Details**: The model was trained for 12 epochs on 8 A100 GPUs using float16 precision. The AdamW optimizer was used for training, with hyperparameters selected based on the best English performance on the development set (Reference 2, 5).

7. **Future Directions**: There is an interest in exploring the effects of increasing the vocabulary size beyond 2 million tokens and using more data to potentially further improve the model's performance (Reference 8).

In summary, facebook/xlm-v-base exhibits strong performance across a range of multilingual tasks, particularly in low-resource language contexts, and shows promise for further improvements with increased vocabulary size and data.

#### Summary

The evaluation results for the model facebook/xlm-v-base indicate that it consistently outperforms the previous model, XLM-R, across a variety of multilingual language understanding tasks. The improvements are particularly notable in low-resource languages, with significant gains in both accuracy and F1 scores. For instance, on the XNLI dataset, XLM-V shows a 4.7% and 2.9% accuracy improvement for Swahili and Urdu, respectively. On the MasakhaNER dataset, which focuses on low-resource African languages, there is an average F1 score increase of 11.2%.

In zero-shot cross-lingual transfer tasks, such as on the Americas NLI dataset, XLM-V demonstrates substantial improvements over XLM-R, with an 18.2% absolute F1 improvement for Quechua and a 17.2% absolute improvement for Guaraní. These languages also saw the largest relative drop in token count per sentence, indicating that XLM-V's tokenization is more efficient and semantically meaningful.

The model has been trained on the CC100 corpus for 1.5 million iterations with a large batch size and has been evaluated on tasks including natural language inference, question answering, and named entity recognition. Across all tasks tested, XLM-V outperforms XLM-R by an average of 3.5 points absolute.

Additionally, the tokenizer used with XLM-V in the 🤗 Transformers library has been carefully compared to the `fairseq` tokenizer to ensure consistency in tokenization across all 176 languages from the WikiANN dataset. This ensures that the tokenizations are semantically meaningful and contribute to the model's overall performance.

## Model Examination

Explainability/Interpretability Section for Model Card: facebook/xlm-v-base

The XLM-V model represents a significant advancement in multilingual language modeling, boasting a 1M token vocabulary that enables it to outperform its predecessor, XLM-R, across a variety of natural language processing tasks. The model's enhanced performance is particularly notable in low-resource languages, where it demonstrates outsized gains.

One of the key features of XLM-V is its ability to produce semantically meaningful tokenizations that are typically shorter than those generated by XLM-R. This not only contributes to the model's efficiency but also aids in interpretability, as shorter, more meaningful token sequences can be easier for humans to understand and analyze.

Despite these strengths, the model does present challenges in terms of scalability, particularly with respect to the computational complexity of the softmax operation over the extensive vocabulary. While this complexity can lead to increased pre-training times, we believe that approximation techniques such as adaptive softmax and adaptive inputs could be potential solutions to mitigate these issues.

In terms of memory footprint, the large vocabulary size of XLM-V could be a concern. However, as we move towards training larger models, the relative size of the vocabulary embedding matrix becomes less significant compared to the number of non-embedding parameters, suggesting that memory-related issues may diminish in importance with larger model architectures.

For future work on explainability and interpretability, we aim to explore the "Zipf ceiling" by potentially increasing the vocabulary size beyond 2M tokens and utilizing more data. This exploration may provide insights into how the model's performance and interpretability scale with vocabulary size and data availability.

In summary, while XLM-V has made strides in model interpretability through improved tokenization, there is ongoing work to address the scalability challenges that may impact explainability. Further research in this area will be crucial to enhance our understanding of the model's decision-making processes and to make its inner workings more transparent to users.

[More Information Needed] regarding specific methods or tools used for explainability/interpretability of the facebook/xlm-v-base model, as they are not detailed in the provided references.

## Environmental Impact

- **Hardware Type:** The model facebook/xlm-v-base is trained on A100 GPUs.
- **Software Type:** The model `facebook/xlm-v-base` is trained on the `fairseq` software.
- **Hours used:** The model facebook/xlm-v-base was trained for 12 epochs as mentioned in reference 1. However, the total amount of time used to train the model is not explicitly stated in the provided references. Therefore, I must say "[More Information Needed]" regarding the exact amount of time taken to train the model.
- **Cloud Provider:** The cloud provider that the model facebook/xlm-v-base is trained on is not explicitly mentioned in the provided references. Therefore, the answer is "[More Information Needed]".
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture for `facebook/xlm-v-base` is based on the transformer encoder design, which is equivalent to the XLM-R base model. This architecture consists of 12 transformer layers, which are designed to process and understand multilingual text.

The objective of the `facebook/xlm-v-base` model is to provide improved performance on a variety of natural language processing tasks by addressing the vocabulary bottleneck present in previous multilingual models like XLM-R. The model achieves this by de-emphasizing token sharing between languages with little lexical overlap and assigning vocabulary capacity to ensure sufficient coverage for each individual language. This results in tokenizations that are typically more semantically meaningful and shorter compared to XLM-R.

The model is pretrained using the Masked Language Model (MLM) task, which involves predicting randomly masked tokens in a sentence, encouraging the model to learn a deep understanding of language context and structure. The improved vocabulary and training approach allow XLM-V to outperform XLM-R on every task it was tested on, including natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), and named entity recognition (WikiAnn). It is particularly effective in reducing over-tokenization for low-resource languages and performs exceptionally well on low-resource evaluation datasets.

### Compute Infrastructure

The compute infrastructure used for training the facebook/xlm-v-base model involved the following specifications:

1. For pretraining, the model was trained on 256 A100 GPUs with a batch size of 8,192. [Reference 1]
2. The model was fine-tuned for the XQuAD benchmark on a single A100 GPU using float16 precision. [Reference 2]
3. For Named Entity Recognition (NER) tasks, the model was also trained on a single A100 GPU with float16 precision. [Reference 3]
4. The model was trained without the use of approximation techniques like adaptive softmax or adaptive inputs, which are known to reduce memory usage but can lead to slower convergence and increased training instability. [Reference 5]

The references do not provide information on the exact duration of the pretraining or the number of epochs for the pretraining phase, so for those details, [More Information Needed].

## Citation

```
@misc{davis-xlmv,
    author = {Davis Liang and
              Hila Gonen and
              Yuning Mao and
              Rui Hou and
              Naman Goyal and
              Marjan Ghazvininejad and
              Luke Zettlemoyer and
              Madian Khabsa and
              Meta Ai},
    title  = {XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models},
    url    = {https://arxiv.org/pdf/2301.10472.pdf}
}
```

