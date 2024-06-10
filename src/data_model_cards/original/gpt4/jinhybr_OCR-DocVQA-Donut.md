# Model Card for jinhybr/OCR-DocVQA-Donut

The model jinhybr/OCR-DocVQA-Donut is an OCR-free, end-to-end Transformer-based model designed for visual document understanding tasks, such as classification and information extraction, which achieves state-of-the-art performance without relying on external OCR engines.

## Model Details

### Model Description

Model Name: jinhybr/OCR-DocVQA-Donut

### Model Architecture:
The architecture of jinhybr/OCR-DocVQA-Donut, referred to as Donut, is an end-to-end Transformer-based model designed for document understanding without relying on traditional OCR systems. It consists of a visual encoder and a textual decoder. The visual encoder is based on the Swin Transformer, which has shown the best performance in preliminary studies for document parsing. This encoder converts input document images into a set of embeddings. The textual decoder is trained to generate a sequence of subword tokens that can be converted into a structured format, such as JSON, representing the desired output information.

### Training Procedures:
The model is trained on a task that involves reading all texts in an image in reading order, aiming to minimize the cross-entropy loss of next token prediction by jointly conditioning on the image and previous contexts. This is interpreted as a pseudo-OCR task. The model is pre-trained over visual corpora, specifically document images, and then fine-tuned for specific tasks by teaching it to understand the document image as a JSON prediction problem.

### Parameters:
The specific number of parameters in the model is not provided in the references. However, it is mentioned that Donut uses fewer parameters compared to models like LayoutLMv2 while achieving state-of-the-art performance and operating at a 2x faster speed.

### Visual Corpora:
For training, the IIT-CDIP dataset consisting of 11 million scanned English document images was used, with pseudo text labels obtained via a commercial CLOVA OCR API. To reduce language dependency and expand the dataset, a Synthetic Document Generator (SynthDoG) was used to create additional samples in Chinese, Japanese, Korean, and English, generating 0.5 million samples per language.

### Important Disclaimers:
- The model is designed to be OCR-free and does not rely on any off-the-shelf OCR engines or APIs.
- It is trained to perform well in English and other languages using synthetic data, but the performance may vary based on the language and the specific types of documents.
- The model aims to address issues such as high dependency on OCR, inflexibility of OCR models on different languages or document types, and OCR error propagation.
- [More Information Needed] for any additional disclaimers specific to the use or limitations of the model that are not covered in the provided references.

For further details or updates on the model, please feel free to reach out to the project organizer responsible for the model card.

- **Developed by:** Geewook Kim;  Hong; Moonbin Yim; Jeongyeon Nam; Jinyoung Park; Jinyeong Yim; Wonseok Hwang; Sangdoo Yun; Dongyoon Han; Seunghyun Park; Naver Clova; Naver Search; Naver Ai Lab;  Tmax;  Google;  Lbox
- **Funded by:** The organizations that fund the project of the model jinhybr/OCR-DocVQA-Donut include Naver Clova, Naver Search, Naver Ai Lab, Tmax, Google, and Lbox.
- **Shared by:** The contributors that made the model jinhybr/OCR-DocVQA-Donut available online as a GitHub repo include Geewook Kim, Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Additionally, the organizations involved are Naver Clova, Naver Search, Naver Ai Lab, Tmax, Google, and Lbox. The GitHub user [@NielsRogge](https://github.com/NielsRogge) also contributed by making Donut available at [huggingface/transformers 🤗](https://huggingface.co/docs/transformers/main/en/model_doc/donut).
- **Model type:** The model jinhybr/OCR-DocVQA-Donut is an end-to-end trained Transformer-based visual document understanding (VDU) model that uses a self-contained training method without relying on external OCR engines, employing supervised learning on both real and synthetic multilingual document images for text recognition and document analysis tasks.
- **Language(s):** The model jinhybr/OCR-DocVQA-Donut processes natural human language in a multi-lingual setting without relying on OCR, as it is trained end-to-end on synthetic data that can be extended to various languages.
- **License:** The model jinhybr/OCR-DocVQA-Donut is licensed under the MIT license. Here is the link to the license: [MIT License](https://opensource.org/licenses/MIT).
- **Finetuned from model:** The model jinhybr/OCR-DocVQA-Donut is fine-tuned from the base model `donut-base`. Here is the link to the base model: [donut-base](https://huggingface.co/naver-clova-ix/donut-base/tree/official).
### Model Sources

- **Repository:** https://github.com/clovaai/donut
- **Paper:** https://arxiv.org/pdf/2111.15664.pdf
- **Demo:** The demo of the model jinhybr/OCR-DocVQA-Donut can be found at the following link: [gradio space web demo](https://huggingface.co/spaces/nielsr/donut-docvqa).
## Uses

### Direct Use

The model `jinhybr/OCR-DocVQA-Donut` is designed to be an OCR-free Visual Document Understanding (VDU) model that can understand and extract text from document images without the need for traditional OCR systems. This model, named Donut, leverages a Transformer architecture and is pre-trained on a large corpus of document images with pseudo text labels obtained from a commercial OCR API and further enhanced with synthetic data generated for multiple languages.

To use `jinhybr/OCR-DocVQA-Donut` without fine-tuning, post-processing, or plugging it into a pipeline, you would typically need to load the pre-trained model and run inference directly on your document images. The model is trained to predict the text in reading order, and it outputs the recognized text along with its confidence scores.

Here is a hypothetical code snippet for using the model directly for inference, assuming that the model and necessary libraries are available on Huggingface's model hub and that the API supports direct inference:

```python
from transformers import DonutModel, DonutTokenizer

# Load pre-trained model and tokenizer
model_name = "jinhybr/OCR-DocVQA-Donut"
model = DonutModel.from_pretrained(model_name)
tokenizer = DonutTokenizer.from_pretrained(model_name)

# Load and preprocess the image
image = load_and_preprocess_image("path_to_your_document_image.jpg")

# Perform inference
inputs = tokenizer(image, return_tensors="pt")
outputs = model(**inputs)

# Extract the text from the output
predicted_text = tokenizer.decode(outputs.logits.argmax(-1))

print(predicted_text)
```

Please note that the above code is a simplified example and assumes that the necessary functions like `load_and_preprocess_image` are defined elsewhere. The actual usage may require additional steps or modifications depending on the specific implementation details of the `jinhybr/OCR-DocVQA-Donut` model and the available APIs.

Since the provided references do not include a direct code block for using the model without fine-tuning, post-processing, or plugging into a pipeline, the above code is a general representation of how one might use a pre-trained model from Huggingface for inference. If the actual model requires a different approach or additional steps, then [More Information Needed] to provide an accurate code snippet.

### Downstream Use

The jinhybr/OCR-DocVQA-Donut model is designed to read and understand document images by converting them into structured JSON format. When fine-tuned for a specific task, this model can be used to extract and interpret information from various types of documents, such as invoices, tickets, business cards, and menus, by recognizing and grouping related information.

For example, in a business application, the model could be integrated into a document management system to automate the extraction of key information from scanned documents. This could significantly reduce manual data entry and improve data accuracy. In a customer service application, the model could be used to quickly retrieve information from customer-submitted documents, enhancing response times and service quality.

When plugged into a larger ecosystem or app, the model can serve as a component that provides document understanding capabilities. It can be used in conjunction with other services, such as databases, to store extracted information, or with user interfaces to display the structured data to end-users.

Here's a hypothetical code snippet showing how the model could be used after fine-tuning for a specific task, assuming the necessary libraries and the fine-tuned model are already installed and imported:

```python
from donut_python import DonutModel, DonutTokenizer
from PIL import Image
import requests

# Load the fine-tuned model and tokenizer
model = DonutModel.from_pretrained('jinhybr/OCR-DocVQA-Donut')
tokenizer = DonutTokenizer.from_pretrained('jinhybr/OCR-DocVQA-Donut')

# Load an image of a document
image_url = "http://example.com/document.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Preprocess the image and convert to the format expected by the model
inputs = tokenizer(image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Convert the token sequence to JSON
decoded_output = tokenizer.decode(outputs.logits.argmax(-1), skip_special_tokens=True)
structured_output = tokenizer.convert_to_json(decoded_output)

# The structured_output is now a JSON object containing the extracted information
print(structured_output)
```

Please note that the above code is a simplified example and assumes that the model has been fine-tuned for a task where the output is structured as JSON. The actual implementation may require additional steps, such as error handling, integration with other systems, and user interface design.

### Out-of-Scope Use

As the sociotechnic on the deep learning model development team for jinhybr/OCR-DocVQA-Donut, it is my responsibility to consider the potential for misuse of our technology and to advise on ethical guidelines for its use. The model we have developed is an OCR-free visual document understanding (VDU) framework that can directly map document images to structured outputs. While it has shown state-of-the-art performance and robustness, especially with handwritten documents, there are several ways in which it could be misused.

1. **Privacy Violations**: Users should not use jinhybr/OCR-DocVQA-Donut to process documents containing sensitive or private information without the explicit consent of the individuals to whom the information pertains. This includes, but is not limited to, personal identification documents, financial records, and private correspondence.

2. **Copyright Infringement**: The model should not be used to digitize copyrighted material in a manner that violates the copyright holder's rights, especially given its ability to bypass traditional OCR and potentially evade detection methods that rely on OCR-generated text.

3. **Forgery and Fraud**: Users must not employ the model for the creation or dissemination of forged documents or for any fraudulent purposes. The model's proficiency in understanding and structuring document content could be exploited to create convincing forgeries.

4. **Unethical Surveillance**: The model should not be used as a tool for unauthorized surveillance, such as analyzing personal documents or communications without consent.

5. **Bias Propagation**: While not explicitly mentioned in the provided references, any machine learning model can inadvertently learn and propagate biases present in its training data. Users should be cautious not to use the model in ways that could reinforce or exacerbate societal biases, particularly when making decisions based on the model's outputs.

6. **Security Risks**: Given that the model can process documents without relying on OCR, it may be used to extract information from images that are designed to be secure against text extraction methods. Users should not use the model to circumvent security measures that are in place to protect document integrity and confidentiality.

In line with the MIT license under which the model is provided, users should adhere to the conditions of the license, which include not holding the authors or copyright holders liable for any misuse. Furthermore, users should respect the "AS IS" nature of the software, acknowledging that there is no warranty for the model and that they use it at their own risk.

In conclusion, while jinhybr/OCR-DocVQA-Donut has the potential to significantly benefit various industries and domains in document understanding tasks, it is imperative that users employ the model responsibly and ethically, avoiding any activities that could harm individuals or groups or that could be considered illegal or unethical.

### Bias, Risks, and Limitations

The model jinhybr/OCR-DocVQA-Donut presents several known or foreseeable issues that can be categorized into technical and sociotechnical limitations:

1. **Dependency on Input Resolution**: As indicated in the references, Donut's performance improves with larger input sizes, particularly for documents with many small texts (Reference 2). However, this also leads to increased computational costs. Users with limited computational resources may not be able to leverage the model's full potential, which could lead to inequitable access to the technology.

2. **Computational Efficiency**: The model uses the original Transformer architecture without an efficient attention mechanism (Reference 2). While this was a deliberate choice to maintain simplicity, it may not be as computationally efficient as possible, which could be a limitation for users with resource constraints.

3. **Robustness in Low-Resource Situations**: Although Donut has shown robust performance with limited training data (Reference 1), there may still be challenges in extremely low-resourced situations. The model's performance in such scenarios may not meet the needs of all users, particularly those in underrepresented regions or languages.

4. **Generalization to Other Tasks**: The model shows little gains in tasks that require general knowledge of images and texts, such as image captioning (Reference 3). This suggests that the model may not be suitable for a broader range of applications beyond its intended use case.

5. **Text Localization**: While the model can attend to desired locations in a given image (Reference 4), there may be limitations in its ability to accurately localize text in more complex document layouts or in the presence of noise and distortions.

6. **Sociotechnical Considerations**: The model's reliance on pre-training with synthetic document image generators (Reference 7) could lead to biases if the synthetic data does not adequately represent the diversity of real-world documents. This could result in lower performance on documents from underrepresented groups or regions.

7. **Language and Document Type Flexibility**: The model aims to address the inflexibility of OCR models on languages or types of documents (Reference 11). However, without further information, it is unclear how well Donut performs across a wide range of languages and document types, which could be a limitation for global applicability.

8. **Error Propagation**: By not depending on OCR, Donut aims to avoid error propagation from OCR to subsequent processes (Reference 11). However, any errors inherent to Donut's own processing could similarly propagate, affecting downstream tasks and potentially leading to misinformation or misinterpretation.

9. **Accessibility and Usability**: The model's performance and usability have not been explicitly discussed in terms of accessibility for users with disabilities. It is important to consider how the model's interface and outputs can be made accessible to all potential users.

10. **Ethical and Legal Considerations**: The use of the model in industries (Reference 6) raises questions about data privacy, consent, and the ethical use of automated document understanding. There may be legal implications if the model is used to process sensitive or personal information without proper safeguards.

In summary, while jinhybr/OCR-DocVQA-Donut presents a novel approach to visual document understanding, there are several technical and sociotechnical issues that need to be considered. These include computational efficiency, robustness in low-resource situations, generalization to other tasks, sociotechnical considerations such as bias and accessibility, and ethical and legal implications of its use.

### Recommendations

Based on the provided references, here are some recommendations with respect to the foreseeable issues about the model jinhybr/OCR-DocVQA-Donut:

1. **Performance on Low-Resolution Inputs**: As indicated in reference 3, the performance of Donut improves with larger input sizes. This suggests that for documents with small text or lower resolution, the model may not perform optimally. It is recommended to preprocess images to ensure they are of sufficient resolution and size to maintain accuracy. Additionally, exploring efficient attention mechanisms could be beneficial for maintaining performance without incurring high computational costs.

2. **Computational Efficiency**: Increasing the input size for precise results leads to bigger computational costs (reference 3). It is recommended to balance the input resolution with the available computational resources. For applications where computational efficiency is critical, it may be necessary to optimize the model further or use hardware acceleration.

3. **Robustness to Handwritten Text**: Donut has shown robustness to handwritten documents (reference 7), which are typically challenging for OCR systems. However, the degree of variability in handwriting styles could still pose challenges. Continuous evaluation and fine-tuning on diverse handwritten datasets may be necessary to maintain high performance across different handwriting styles.

4. **Dependence on Synthetic Data**: The model has been verified with synthetic images, which were sufficient for document understanding tasks (reference 2). However, reliance on synthetic data may not fully capture the complexity of real-world documents. It is recommended to periodically update the model with real-world data to ensure it remains effective in practical applications.

5. **Model Interpretability**: The visualization of cross-attention maps (reference 4) can serve as an auxiliary indicator of the model's behavior. It is recommended to provide users with tools to interpret these attention maps to build trust and understanding of the model's decision-making process.

6. **Generalization to Various Tasks**: Donut has been shown to be effective for specific tasks like document classification (reference 6), but other tasks requiring general knowledge of images and texts may see little gains (reference 2). It is recommended to evaluate the model on a broader range of tasks and consider additional fine-tuning for tasks where performance is not satisfactory.

7. **Ethical and Societal Considerations**: As a sociotechnic, it is important to consider the ethical implications of deploying an OCR system. Issues such as privacy, data security, and the potential for misuse of sensitive information extracted from documents should be addressed. Clear guidelines and safeguards should be established to ensure the responsible use of the technology.

8. **Accessibility and Inclusivity**: Ensure that the model is accessible to users with varying levels of technical expertise and that it supports documents in multiple languages and formats to promote inclusivity.

9. **Legal Compliance**: The model should comply with data protection laws such as GDPR, especially since it processes potentially sensitive information. Users should be informed about the data handling practices and consent should be obtained where necessary.

10. **Continuous Monitoring and Updating**: The model should be continuously monitored for performance degradation and updated as necessary to adapt to changes in document formats, styles, and the emergence of new types of documents.

By addressing these recommendations, the team can help ensure that the model remains effective, efficient, and ethically sound in its deployment and use.

## Training Details

### Training Data

The training data for the model jinhybr/OCR-DocVQA-Donut consists of 11 million scanned English document images from the IIT-CDIP dataset, with pseudo text labels obtained using a commercial CLOVA OCR API. Additionally, to support multiple languages and reduce dependency on English datasets, 0.5 million synthetic document samples per language (Chinese, Japanese, Korean, and English) were generated using the Synthetic Document Generator (SynthDoG), leveraging Wikipedia for text and ImageNet for background textures.

### Training Procedure

#### Preprocessing

The preprocessing steps for the data used in the model `jinhybr/OCR-DocVQA-Donut` include several key components to prepare the document images for training:

1. **Tokenization**: The textual data is tokenized into a sequence of subword tokens. This is done to construct the desired structured output from the document images. The tokenization process is likely based on the BART model's tokenizer, as the decoder uses the BART architecture. The tokenizer converts text into a sequence of tokens that can be processed by the model.

2. **Resizing**: The input document images are resized to fit the model's input dimensions. During the pre-training phase, the input resolution of Donut is set to 2560×1920. For downstream tasks, the input resolutions are controlled and can be different. For example, a smaller size of input resolution, such as 1280×960, is tested for certain tasks, which reduces the model training cost.

3. **Rewriting**: There is no direct mention of rewriting in the context of preprocessing the data. The model aims to understand the document images as they are, without rewriting the content. [More Information Needed]

4. **Image Patching and Embedding**: The visual encoder, specifically a Swin Transformer, first splits the input document image into patches. These patches are then converted into a set of embeddings, which are the latent vectors that the encoder uses to represent the visual features of the document.

5. **Synthetic Data Generation**: To supplement the training data and alleviate dependencies on datasets for languages other than English, a Synthetic Document Generator (SynthDoG) is used. This generator creates synthetic document images using heuristic random patterns to mimic real documents. The components of these synthetic images include background (sampled from ImageNet), document texture (sampled from collected paper photos), and text and layout (sampled from Wikipedia in different languages).

6. **OCR-Free Approach**: It is important to note that Donut does not rely on traditional OCR modules for feature extraction. Instead, it uses a visual encoder to extract features directly from the document images, which are then mapped into a sequence of subword tokens by the textual decoder.

7. **Visual Corpora**: The model is trained on a visual corpus that includes the IIT-CDIP dataset of 11 million scanned English document images. Pseudo text labels for these images are obtained using a commercial CLOVA OCR API. Additionally, synthetic samples generated by SynthDoG are used for multiple languages.

In summary, the preprocessing for the `jinhybr/OCR-DocVQA-Donut` model involves tokenization using a BART-based tokenizer, resizing images to the required input dimensions, converting images into patches and embeddings using a Swin Transformer, and generating synthetic data for training. The model adopts an OCR-free approach, directly extracting features from the images without relying on traditional OCR technology.

#### Training Hyperparameters

The training hyperparameters for the model jinhybr/OCR-DocVQA-Donut are as follows:

- **Precision**: The model is trained using half-precision (fp16) to optimize the training process and reduce memory consumption.
- **Optimizer**: Adam optimizer is used for training the model.
- **Learning Rate**: The initial learning rate for pre-training is set to 1e-4. For fine-tuning, the learning rate is selected from a range between 1e-5 to 1e-4, depending on the specific requirements of the task.
- **Training Steps**: The model is pre-trained for 200,000 steps.
- **Hardware**: Training is conducted on 64 NVIDIA A100 GPUs.
- **Mini-batch Size**: A mini-batch size of 196 is used during training.
- **Training Duration**: The pre-training phase takes about 2-3 GPU days.
- **Gradient Clipping**: A gradient clipping technique is applied with a maximum gradient norm selected from a range between 0.05 to 1.0 to prevent exploding gradients.

Please note that specific details such as the number of epochs for fine-tuning or the exact learning rate used for fine-tuning are not provided in the references and would require [More Information Needed].

#### Speeds, Sizes, Times

The model `jinhybr/OCR-DocVQA-Donut` is an OCR-free visual document understanding (VDU) model that has demonstrated state-of-the-art performance on various VDU tasks. Below are the details regarding the model's throughput, timing, and checkpoint sizes:

- **Throughput and Timing**: The model shows competitive performance in terms of speed. For instance, on the CORD dataset with an input resolution of 1280×960, the model processes at a speed of 0.7 seconds per image (reference 5). This indicates that the model is capable of fast inference, which is 2x faster than the speed of LayoutLMv2 while using fewer parameters (reference 1). However, it is noted that larger input resolutions, while providing robust accuracies, can slow down the model (reference 5).

- **Checkpoint Sizes**: The references do not provide explicit information about the checkpoint sizes of the `jinhybr/OCR-DocVQA-Donut` model. Therefore, [More Information Needed] regarding the exact checkpoint sizes.

- **Start or End Time**: Specific start or end times for the model training or inference are not mentioned in the provided references. Therefore, [More Information Needed] regarding the start or end times.

- **Additional Information**: The model is available on Huggingface Transformers (reference 6), and it has been shown to be robust even to handwritten documents (reference 2). It maintains stable performance regardless of the dataset size and task complexity (reference 3). The model is also robust in low-resource situations (reference 9).

For further details on the model's performance and characteristics, users are encouraged to refer to the official documentation and release notes linked in the references (reference 6) or to the model's repository for the most up-to-date and comprehensive information.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model jinhybr/OCR-DocVQA-Donut has been evaluated on the following benchmarks or datasets:

1. Document Classification tasks, as mentioned in the references, but the specific datasets used for classification are not listed in the provided text. [More Information Needed]
2. Document Visual Question Answering (DocVQA) task, where the dataset is explicitly named as "DocVQA" from Document Visual Question Answering. [Reference: 9]
3. Six different datasets including both public benchmarks and private industrial service datasets for various Visual Document Understanding (VDU) applications. However, the specific names of these datasets are not provided in the text. [More Information Needed]

Overall, the model has been tested on a variety of tasks and datasets, demonstrating its robustness and state-of-the-art performance in OCR-free VDU tasks.

#### Factors

The model jinhybr/OCR-DocVQA-Donut exhibits several characteristics that will influence its behavior in various domains and contexts, as well as across different population subgroups. Based on the provided references, the following factors are likely to affect the model's performance:

1. **OCR Engine Independence**: Donut does not rely on off-the-shelf OCR engines, which means its performance is not contingent on the capabilities or limitations of these external systems. This characteristic is beneficial for consistent behavior across different domains where the use of external OCR engines might introduce variability.

2. **Low Resourced Situations**: Donut has demonstrated robust performance even when the size of the training set is limited, as shown in the evaluation with a reduced CORD dataset. This suggests that the model could perform well in scenarios where only limited data is available, although the exact impact on various subgroups that might be represented in such data is not specified. [More Information Needed] to determine if there are disparities in performance across different population subgroups in low-resource settings.

3. **Document Classification**: Donut's ability to classify different types of documents and generate a JSON with class information indicates that it can be applied to tasks requiring document categorization. However, the performance across different document types or in contexts with novel or less common document formats is not detailed. [More Information Needed] to assess if there are any disparities in classification accuracy for certain document types or in specific domains.

4. **Input Resolution**: The model's performance improves with larger input sizes, which is particularly evident in tasks involving larger images with small text, such as DocVQA. However, this comes at the cost of increased computational resources. The impact of input resolution on performance may vary across different contexts where document image quality and size can differ significantly.

5. **Text Localization**: Visualization of cross-attention maps indicates that Donut can effectively attend to relevant locations in unseen document images. This suggests that the model could be useful in applications requiring text localization. However, the performance in localizing text in documents from diverse domains or with complex layouts is not explicitly discussed.

6. **Robustness to Handwritten Documents**: Donut is noted to be robust to handwritten documents, which are typically challenging for OCR systems. This implies that the model could be particularly useful in contexts where handwritten text is prevalent. However, the level of robustness across different handwriting styles or languages is not mentioned.

7. **Performance and Efficiency**: Donut is reported to have state-of-the-art performance and efficiency compared to other general-purpose VDU models, with fewer parameters and faster speed. This suggests that the model is suitable for applications requiring high throughput and low latency. However, the performance may still vary depending on the specific task and the computational resources available.

8. **General Visual Document Understanding (VDU)**: While Donut has shown advances in VDU, the references do not provide detailed information on its performance across all possible VDU tasks. [More Information Needed] to fully understand its capabilities and limitations in this broad area.

In summary, while Donut shows promising characteristics such as independence from external OCR engines, robustness in low-resource situations, and efficiency, there is a need for more information to fully evaluate its performance across different domains, contexts, and population subgroups. Disaggregated evaluation across these factors is essential to uncover any disparities in performance and to ensure the model's applicability and fairness in diverse real-world scenarios.

#### Metrics

For evaluating the model jinhybr/OCR-DocVQA-Donut, we will use two metrics: field-level F1 score and Tree Edit Distance (TED) based accuracy. The field-level F1 score will check whether the extracted field information matches the ground truth, with the caveat that if even a single character is missed, the extraction is considered failed. This metric is straightforward but does not account for partial overlaps or measure the predicted structure, such as groups and nested hierarchies.

On the other hand, TED based accuracy will provide a more nuanced evaluation by considering the structural accuracy of the extracted information, which can be particularly important for documents with complex layouts or hierarchical data.

These metrics will help us understand the tradeoffs between different types of errors, such as complete misses versus partial extractions, and structural versus content accuracy.

### Results

The evaluation results of the model jinhybr/OCR-DocVQA-Donut based on the provided references are as follows:

1. **Performance on Document Visual Question Answering (DocVQA)**: Donut achieves competitive scores compared to baselines that rely on external OCR engines. It demonstrates robustness to handwritten documents, which are typically challenging for OCR systems. Specific scores or comparative figures are not provided in the references, so [More Information Needed] for exact numbers.

2. **Document Classification**: Donut shows state-of-the-art performance among general-purpose Visual Document Understanding (VDU) models, surpassing the accuracy of LayoutLMv2 while using fewer parameters and offering a 2x faster speed. However, exact accuracy figures and speed metrics are not provided in the references, so [More Information Needed] for detailed results.

3. **Document Information Extraction**: Donut is evaluated on four different document information extraction tasks and shows promising results. The references mention the use of a conventional BIO-tagging-based IE approach, but specific performance metrics such as precision, recall, or F1 scores are not provided, so [More Information Needed] for detailed results.

4. **Performance in Low-Resource Situations**: Donut exhibits stable performance regardless of the dataset size and task complexity. It is particularly robust in low-resource situations, as demonstrated by performance curves in the references. However, the exact performance metrics in these situations are not detailed, so [More Information Needed] for specific figures.

5. **OCR-Free VDU Model**: Donut is highlighted as a simple OCR-free VDU model that can achieve state-of-the-art performance in terms of both speed and accuracy. This suggests that it does not rely on external OCR engines, which is a significant advantage in terms of system complexity and maintenance cost. However, specific comparisons to OCR-based models in terms of speed and accuracy are not provided, so [More Information Needed] for quantitative results.

6. **Dependence on OCR Engines**: The references indicate that the performance of conventional OCR-based methods heavily relies on the off-the-shelf OCR engine used. Donut, by contrast, does not depend on such engines, which implies a more consistent and potentially more efficient performance. However, no direct comparison figures are provided, so [More Information Needed] for a detailed analysis.

In summary, the model jinhybr/OCR-DocVQA-Donut is presented as a robust, efficient, and OCR-independent solution for various VDU tasks, showing state-of-the-art performance and speed. However, for precise evaluation results such as accuracy, speed metrics, and performance on specific tasks or datasets, [More Information Needed] as the references do not provide explicit numbers or comparative tables.

#### Summary

The model jinhybr/OCR-DocVQA-Donut demonstrates competitive performance in visual document understanding (VDU) tasks without relying on external OCR engines. It outperforms general-purpose VDU models such as LayoutLM and LayoutLMv2 in terms of accuracy, while also being more efficient, using fewer parameters and offering a 2x faster speed. Notably, Donut is robust to handwritten documents, which are typically challenging for OCR systems.

Donut's performance remains stable across different dataset sizes and task complexities, indicating its suitability for industrial applications. It also shows strong performance in low-resource situations, maintaining robustness even with limited training data.

The model's independence from off-the-shelf OCR engines is a significant advantage, as it avoids the need for additional model parameters and processing time associated with conventional OCR-based methods. This approach also reduces the overall system size and maintenance costs.

In summary, Donut provides a simple yet effective OCR-free solution for VDU tasks, achieving state-of-the-art performance in both speed and accuracy.

## Model Examination

### Model Card for jinhybr/OCR-DocVQA-Donut

#### Explainability/Interpretability

Our model, jinhybr/OCR-DocVQA-Donut, has been designed with a focus on understanding and visualizing the internal mechanisms that contribute to its performance on document understanding tasks. Below are some insights into the explainability and interpretability of our model:

1. **Text Localization Visualization**: We have visualized the cross-attention maps of the decoder when processing unseen document images. As illustrated in Figure 8 of our references, the model demonstrates the ability to attend to relevant locations on the image, providing an auxiliary indicator of its internal reasoning process. This visualization can be used to interpret how the model is reading and understanding the document layout and content.

2. **Performance on Various Input Resolutions**: Our experiments, as shown in Figure 7(c), indicate that the model's performance improves with larger input sizes, especially in the DocVQA dataset, which contains larger images with smaller text. This suggests that the model can effectively scale its understanding with the resolution, although at a higher computational cost. This trade-off is an important consideration for users who need to balance accuracy with efficiency.

3. **Robustness to Data Scarcity**: In low-resourced situations, such as when the size of the training set is limited (as with the CORD dataset), Donut has shown robust performance. The performance curves in the right Figure 9 demonstrate this resilience, which is crucial for understanding the model's behavior under different data availability scenarios.

4. **OCR-Free Document Understanding**: Unlike conventional OCR-based methods, which rely heavily on the performance of external OCR engines, our model operates without such dependencies. This independence from OCR systems is a significant aspect of Donut's design, as it simplifies the architecture and reduces the need for additional model parameters and processing time associated with OCR frameworks.

5. **Information Extraction Capabilities**: Donut goes beyond simple text reading and understanding by also predicting the relations and structures among the extracted information. This capability is essential for tasks that require a comprehensive understanding of the entire document, as it involves not just recognizing text but also interpreting its meaning and context within the document.

6. **Comparative Performance**: Our model has achieved state-of-the-art performance in document classification tasks without relying on external OCR engines, surpassing other general-purpose VDU models like LayoutLM and LayoutLMv2 in accuracy and speed. This demonstrates Donut's strong understanding ability and efficiency, which are key factors in its interpretability.

In summary, the jinhybr/OCR-DocVQA-Donut model exhibits a strong understanding of document content and structure, with visualizations and performance metrics that provide insights into its interpretability. Its robustness to various conditions and independence from external OCR engines make it a powerful tool for visual document understanding tasks.

## Environmental Impact

- **Hardware Type:** The model jinhybr/OCR-DocVQA-Donut is trained on NVIDIA A100 GPUs.
- **Software Type:** The model jinhybr/OCR-DocVQA-Donut is trained on a deep learning software framework that utilizes the Transformer architecture. While the specific software framework (like TensorFlow, PyTorch, etc.) is not explicitly mentioned in the provided references, the model employs common deep learning tools and techniques such as half-precision (fp16) training, the Adam optimizer, and gradient clipping. The training was conducted using NVIDIA A100 GPUs, indicating the use of a software framework compatible with CUDA-enabled GPU acceleration. However, for the exact software type, such as TensorFlow or PyTorch, [More Information Needed].
- **Hours used:** The model jinhybr/OCR-DocVQA-Donut was pre-trained for 200K steps with 64 NVIDIA A100 GPUs, which took about 2-3 GPU days. Fine-tuning on specific datasets like CORD or Ticket took approximately 0.5 hours with one A100 GPU.
- **Cloud Provider:** The cloud provider that the model jinhybr/OCR-DocVQA-Donut is trained on is not explicitly mentioned in the provided references. Therefore, the answer is "[More Information Needed]".
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The architecture of the model jinhybr/OCR-DocVQA-Donut is based on a Transformer, which includes a visual encoder and a textual decoder. The visual encoder is responsible for converting the input document image into a set of embeddings, and in this model, we use the Swin Transformer due to its superior performance in our preliminary studies on document parsing. The textual decoder then maps these embeddings into a sequence of subword tokens to construct the desired structured output.

The objective of the model is to read all texts in the image in reading order, from top-left to bottom-right, and to minimize the cross-entropy loss of next token prediction by jointly conditioning on the image and previous contexts. This task is akin to a pseudo-OCR task, where the model is trained as a visual language model over visual corpora, i.e., document images, without relying on traditional OCR modules. Instead, it directly maps from a raw input image to the desired output, which can be in various formats such as JSON.

Donut stands for Document understanding transformer and is an OCR-free VDU (Visual Document Understanding) model. It is designed to address the limitations of OCR, such as the dependency on OCR engines, inflexibility on languages or types of documents, and error propagation from OCR to subsequent processes. The model achieves state-of-the-art performance on various VDU tasks, offering both speed and accuracy advantages over traditional OCR-dependent methods.

### Compute Infrastructure

The compute infrastructure used for the model jinhybr/OCR-DocVQA-Donut includes the following:

1. The model was pre-trained on 64 NVIDIA A100 GPUs, which are high-performance GPUs designed for deep learning workloads.
2. The pre-training was conducted with a mini-batch size of 196.
3. The training process took about 2-3 GPU days, indicating the total amount of time the GPUs were actively used for training.
4. For fine-tuning tasks such as Train Tickets and Business Card parsing, the input resolution was set to 960×1280, and the model could be fine-tuned in approximately 0.5 hours with one A100 GPU.
5. The speed of the model was measured on a P40 GPU, which is noted to be slower than the A100 GPU.

For more specific details about the compute infrastructure, such as the exact configurations used for the Adam optimizer or the gradient clipping technique, [More Information Needed] as the references provided do not contain that level of detail.

## Citation

```
@misc{geewook-ocrfree,
    author = {Geewook Kim and
              Hong and
              Moonbin Yim and
              Jeongyeon Nam and
              Jinyoung Park and
              Jinyeong Yim and
              Wonseok Hwang and
              Sangdoo Yun and
              Dongyoon Han and
              Seunghyun Park and
              Naver Clova and
              Naver Search and
              Naver Ai Lab and
              Tmax and
              Google and
              Lbox},
    title  = {OCR-free Document Understanding Transformer},
    url    = {https://arxiv.org/pdf/2111.15664.pdf}
}
```

