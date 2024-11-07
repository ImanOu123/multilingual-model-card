# Model Card for jinhybr/OCR-DocVQA-Donut

Donut is an OCR-free end-to-end Transformer model for document understanding that achieves state-of-the-art performance on various visual document understanding tasks such as document classification and information extraction. It is a self-contained model with a simple architecture consisting of a visual encoder and textual decoder, and does not rely on any OCR-related modules.

## Model Details

### Model Description

Model: jinhybr/OCR-DocVQA-Donut

Model Architecture:
- Donut is an end-to-end VDU (Visual Document Understanding) model for general understanding of document images.
- It consists of a Transformer-based visual encoder and textual decoder.
- The visual encoder converts the input document image into a set of embeddings. Swin Transformer is used as the encoder network.
- The textual decoder maps the derived features into a sequence of subword tokens to construct the desired structured output.

Training Procedures:
- The model is trained as a visual language model over the visual corpora (document images).
- The objective is to minimize cross-entropy loss of next token prediction by jointly conditioning on the image and previous contexts.
- In the application stage (fine-tuning), the model is taught to understand the document image by interpreting all downstream tasks as a JSON prediction problem.
- The decoder is trained to generate a token sequence that can be converted into a JSON representing the desired output information.

Parameters:
[More Information Needed]

Important Disclaimers:
- Donut does not rely on any modules related to OCR functionality, but uses a visual encoder for extracting features from a given document image.
- The model goes beyond the traditional OCR-based framework by modeling a direct mapping from a raw input image to the desired output.
- Donut addresses issues induced by OCR-dependency, such as the cost of building an OCR system, inflexibility of OCR models on languages or types of documents, and OCR error propagation.

For more information or any questions, please contact the project organizer at [More Information Needed].

- **Developed by:** Geewook Kim;  Hong; Moonbin Yim; Jeongyeon Nam; Jinyoung Park; Jinyeong Yim; Wonseok Hwang; Sangdoo Yun; Dongyoon Han; Seunghyun Park; Naver Clova; Naver Search; Naver Ai Lab;  Tmax;  Google;  Lbox
- **Funded by:** Based on the provided references, there is no clear information about the specific people or organizations that funded the jinhybr/OCR-DocVQA-Donut model project. The references mention the model and provide some technical details, but do not specify funding sources.

[More Information Needed]

To properly answer this question for the model card, more details would need to be provided about any funding, grants, or sponsorships that supported the development of the jinhybr/OCR-DocVQA-Donut model. The project organizer should seek out and include that information if available.
- **Shared by:** Based on the provided references, the main contributors that made the model jinhybr/OCR-DocVQA-Donut available online as a GitHub repo are:

Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park from NAVER Corp (including Naver Clova, Naver Search, and Naver AI Lab teams).

The model code, weights, synthetic data and generator were first released on 2022-07-20.

Additional contributions:
- @estaudere published a hands-on tutorial on the Donut model on Towards Data Science on 2022-08-05.
- @NielsRogge made the Donut model available on Hugging Face Transformers on 2022-08-12.

[More Information Needed] on the specific roles and contributions of the individual authors from NAVER Corp.
- **Model type:** The jinhybr/OCR-DocVQA-Donut model is an end-to-end document understanding transformer model that is trained on visual corpora using self-supervised learning to generate structured text outputs from document images without relying on external OCR engines.
- **Language(s):** Based on the provided references, the Donut model is an OCR-free end-to-end transformer model for document understanding, but there is no specific information about the natural human language(s) it uses or processes. [More Information Needed]
- **License:** The model jinhybr/OCR-DocVQA-Donut is using the MIT License, as stated in the references:

"MIT license

Copyright (c) 2022-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:"

The full text of the MIT License is provided in the references.
- **Finetuned from model:** Based on the information provided in the references, the model jinhybr/OCR-DocVQA-Donut is likely fine-tuned from the donut-base model, which is linked here:

[`donut-base`](https://huggingface.co/naver-clova-ix/donut-base/tree/official)

The references mention that the DocVQA task was fine-tuned using the donut-base model:

"| [DocVQA Task1](https://rrc.cvc.uab.es/?ch=17) (Document VQA) |  0.78       | 67.5 | [donut-base-finetuned-docvqa](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa/tree/official) |"

So the jinhybr/OCR-DocVQA-Donut model is probably a further fine-tuned version of the donut-base-finetuned-docvqa model, which itself was fine-tuned from the original donut-base model.
### Model Sources

- **Repository:** https://github.com/clovaai/donut
- **Paper:** https://arxiv.org/pdf/2111.15664.pdf
- **Demo:** [More Information Needed]

The provided references do not contain a direct link to a demo for the specific model "jinhybr/OCR-DocVQA-Donut". The references mention demos for other Donut models like "donut-base-finetuned-docvqa" and "donut-base-finetuned-cord-v2", but not for the requested model. More information would be needed to determine if a demo exists for "jinhybr/OCR-DocVQA-Donut".
## Uses

### Direct Use

The model jinhybr/OCR-DocVQA-Donut can be used without fine-tuning, post-processing, or plugging into a pipeline in the following way:

Donut is an end-to-end model that jointly learns visual and textual information from document images. It takes a document image as input and directly generates the desired output in a structured format, such as JSON. 

For example, for document classification, Donut can generate a JSON containing the predicted class:

{ "class":"receipt" }

For document visual question answering (DocVQA), Donut can take the question as a prompt and generate the answer:

{ "question": "what is the price of choco mochi?", "answer": "14,000" }

This allows Donut to be used directly for inference on downstream tasks without the need for additional fine-tuning, post-processing steps like OCR, or integration into a pipeline. The uniformity of the task-solving method (generating structured output) makes Donut versatile and easy to use.

[More Information Needed] for a specific code snippet demonstrating the usage.

In summary, the end-to-end nature and structured output generation capability of Donut enables it to be used directly for various document understanding tasks without relying on external components or task-specific fine-tuning.

### Downstream Use

The Donut model (jinhybr/OCR-DocVQA-Donut) can be fine-tuned for various visual document understanding tasks such as document classification, information extraction, and visual question answering. When fine-tuned, the model learns to generate a JSON output that represents the desired task-specific information.

For example, in the document classification task, the model is trained to generate a JSON like {"class": "memo"} to classify the document type. This allows the model to maintain a uniform task-solving approach across different downstream applications.

When plugged into a larger ecosystem or app, Donut can be used as an OCR-free document understanding component. It can process document images directly, without relying on external OCR engines, and provide the extracted information in a structured JSON format. This makes it easier to integrate Donut into existing workflows and pipelines.

[More Information Needed] for a specific code snippet demonstrating the fine-tuning or integration process.

Overall, Donut's ability to handle various document understanding tasks without OCR dependencies, combined with its state-of-the-art performance and efficient inference, makes it a valuable asset for applications dealing with visual documents.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the Donut model (jinhybr/OCR-DocVQA-Donut) could foreseeably be misused and what users ought not do:

- The model should not be used on sensitive, private or confidential documents without proper authorization, as it could extract information that is not intended to be shared. The references do not provide details on what safeguards are in place to protect private data.

- Users should be cautious about relying on the model outputs for high-stakes decisions or applications where errors could have serious consequences. While the model achieves competitive performance, the references note it is not perfect and can make mistakes, especially on challenging inputs like handwritten text. 

- The model should not be represented as being an authoritative OCR tool or replacement for dedicated OCR engines in all scenarios. The references position it as an OCR-free alternative for certain visual document understanding tasks, but traditional OCR may still be needed for some use cases.

- [More Information Needed] on what specific languages, document types, and domains the model was trained on and evaluated for. Users should not assume it will work for all languages and document types without further testing.

- [More Information Needed] on how the model handles potential bias and fairness issues that could arise from the training data. Users should be cautious about applying it to sensitive use cases without deeper analysis.

In summary, while a promising approach, users should be thoughtful about appropriate use cases, be transparent about limitations, and not treat the model as an OCR replacement without further diligence. More details on the training data, evaluation, and intended use would help address potential misuse.

### Bias, Risks, and Limitations

Based on the provided references about the OCR-DocVQA-Donut model, here are some known or foreseeable issues:

Technical limitations:
- The model's performance heavily relies on the input resolution. Using larger input sizes leads to better accuracy, especially on tasks like DocVQA with many tiny texts, but incurs higher computational costs. (Reference 2, 4)
- [More Information Needed] on how well the model handles extremely low-resource situations with very limited training data.

Sociotechnical limitations: 
- The model was trained on synthetic document images from SynthDoG to reduce reliance on large real document image datasets. However, [More Information Needed] on how representative SynthDoG is of real-world documents and any potential biases or gaps.
- [More Information Needed] on the model's performance and fairness when applied to documents from diverse languages, cultures, and communities beyond the training data.

Potential misunderstandings:
- While the model achieves competitive performance without relying on external OCR engines, users should not assume it outperforms all OCR-based methods in all scenarios. Conventional OCR may still be advantageous for some use cases. (Reference 1, 8)
- The model shows robustness on challenging handwritten documents, but [More Information Needed] on its limitations and failure modes on diverse handwritten inputs.

Foreseeable harms:
- [More Information Needed] on risks of the model being misused for unauthorized processing of sensitive personal documents.
- [More Information Needed] on potential negative impacts if the model underperforms on documents from certain demographics or communities.

To responsibly deploy this model, more information and analysis is needed in the areas marked [More Information Needed] to fully characterize limitations and proactively mitigate risks. Ongoing monitoring and maintenance is also recommended to identify and address any issues that may emerge post-deployment.

### Recommendations

Based on the provided references about the Donut model for document understanding, here are some recommendations regarding foreseeable issues:

1. OCR dependency: While Donut aims to be an OCR-free visual document understanding model, the references mention comparing performance to OCR-based methods. It's important to clearly communicate the extent to which Donut relies on or is independent from OCR to set appropriate expectations. [More Information Needed] on how Donut's OCR-free approach impacts accuracy, speed, and flexibility compared to OCR-dependent models.

2. Language and document type flexibility: The references indicate Donut can be extended to a multi-lingual setting, but [More Information Needed] on the specific languages and scripts supported, and how performance varies across them. Similarly, more details should be provided about what document types and formats Donut handles well vs. struggles with.

3. Computational cost vs. resolution: The references note that increasing input resolution improves Donut's performance, especially on tasks like DocVQA with large images and small text. However, higher resolutions incur greater computational costs. Guidance should be provided to users on recommended resolutions to balance accuracy and efficiency for different use cases.

4. Limited training data: While the SynthDoG synthetic document generator is proposed to alleviate the need for large real document datasets, [More Information Needed] on the quantity and diversity of data required to train Donut effectively for different tasks. Limitations related to smaller training sets should be clearly documented.

5. Responsible usage: As with any AI system, it's important to provide guidance on responsible usage of Donut, such as not relying on it for high-stakes decisions without human oversight. Potential failure modes and biases should be transparently discussed.

In summary, key issues to address are the relationship to OCR, flexibility across languages and document types, computational trade-offs, training data requirements, and responsible usage guidelines. More specific information is needed in several of these areas to comprehensively assess limitations and societal implications.

## Training Details

### Training Data

The model jinhybr/OCR-DocVQA-Donut was trained on the IIT-CDIP dataset, which contains 11M scanned English document images, as well as 0.5M synthetic document images per language (Chinese, Japanese, Korean and English) generated using SynthDoG and text from Wikipedia.

[More Information Needed] for links to documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Here are the details about preprocessing for the data of the model jinhybr/OCR-DocVQA-Donut, based on the provided references:

Tokenization:
[More Information Needed]

Image Resizing:
The input resolution of Donut is set to 2560×1920 at the pre-training phase. In some downstream document IE experiments, such as CORD, Ticket and Business Card, smaller resolutions like 960×1280 are used.

Other Preprocessing:
The decoder is trained to generate a token sequence that can be converted into a JSON that represents the desired output information. Special tokens (e.g., [memo]) are introduced to represent certain classes, if such replacement is available in the target task.

[More Information Needed] for other specific preprocessing steps.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters used for the jinhybr/OCR-DocVQA-Donut model:

- Visual encoder: Swin-B with layer numbers and window size set as {2, 2, 14, 2} and 10 respectively
- Decoder: First four layers of BART
- Optimizer: Adam 
- Learning rate: Selected from 1e-5 to 1e-4, scheduled during training. Initial rate of 1e-4 for pre-training and 1e-5 to 1e-4 for fine-tuning.
- Input resolution: 2560×1920 for pre-training, controlled for downstream tasks (e.g. 1280×960 for CORD, Ticket and Business Card datasets)
- Max decoder length: 1536
- Precision: Half-precision (fp16) training
- Gradient clipping: Maximum gradient norm selected from 0.05 to 1.0
- Pre-training: 200K steps with 64 A100 GPUs and mini-batch size of 196
- Fine-tuning: Monitored edit distance over token sequences. Took ~0.5 hours with one A100 GPU on CORD or Ticket datasets with 1280×960 input resolution.

[More Information Needed] on the exact learning rate schedule, gradient clipping value, and fine-tuning steps/epochs used.

#### Speeds, Sizes, Times

Here are the details about the model jinhybr/OCR-DocVQA-Donut based on the provided references:

Training throughput: [More Information Needed]

Training start time: [More Information Needed] 

Training end time: [More Information Needed]

Checkpoint sizes: [More Information Needed]

Other relevant details:
- Donut was trained with 8 V100 GPUs for approximately 5 days
- The model architecture consists of:
Encoder layers: {2,2,18,2}  
Decoder layers: 4
- Input image size used was 2048x1536
- Swin window size of 8 was used
- Pre-training data included SynthDoG datasets in English, Japanese, Korean, with 0.4M images each

The references do not provide specific information about the training throughput, start/end times, or checkpoint sizes for the jinhybr/OCR-DocVQA-Donut model. More details would be needed from the model developers to include that information.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model jinhybr/OCR-DocVQA-Donut was evaluated on the following datasets:

1. DocVQA dataset (Reference 3 and 7): The model was tested on the DocVQA (Document Visual Question Answering) task, where given a document image and a question, the model predicts the answer by capturing both visual and textual information within the image.

2. [More Information Needed] for other specific datasets or benchmarks, as they are not explicitly mentioned in the provided references. The references mention evaluating the model on "various VDU tasks and datasets" (Reference 8) and "six different datasets including both public benchmarks and private industrial service datasets" (Reference 6), but do not provide the names of these datasets.

#### Factors

Based on the provided references, here are some foreseeable characteristics that may influence how the OCR-DocVQA-Donut model behaves:

1. Document types and complexity: The model's performance may vary across different types of documents, as suggested by the document classification task used to evaluate if the model can distinguish between document types (Reference 1).

2. OCR system used: The model's performance and speed can be influenced by the choice of OCR engine, as conventional OCR-based methods heavily rely on the off-the-shelf OCR system (Reference 2).

3. Size of training data: The model's performance may be affected by the size of the training dataset. However, Donut shows robust performance even in low-resource situations (Reference 2).

4. Input resolution: The model's performance improves with larger input sizes, especially for tasks like DocVQA where images are larger with many tiny texts. However, increasing the input size also leads to higher computational costs (Reference 4).

5. Handwritten documents: Donut shows robustness in processing handwritten documents, which are known to be challenging for conventional approaches (Reference 9).

[More Information Needed] Regarding specific population subgroups or factors that may lead to disparities in performance, as the provided references do not contain sufficient information to address this aspect.

In summary, the OCR-DocVQA-Donut model's performance can be influenced by factors such as document types, OCR systems, training data size, input resolution, and the presence of handwritten content. However, more information is needed to assess potential disparities across population subgroups or other specific factors.

#### Metrics

Based on the provided references, the evaluation metric used for the DocVQA task and the jinhybr/OCR-DocVQA-Donut model is ANLS (Average Normalized Levenshtein Similarity). Specifically, reference 10 states:

"The evaluation metric is ANLS (Average Normalized Levenshtein Similarity) which is an edit-distancebased metric. The score on the test set is measured via the evaluation site."

ANLS measures the similarity between the predicted answer and the ground truth answer based on edit distance. 

The references also mention evaluating trade-offs between accuracy and inference speed at different input resolutions. Higher resolutions tend to improve accuracy but slow down the model (reference 7). However, no specific information is provided on how different error types are traded off in the evaluation.

[More Information Needed] on the details of how ANLS handles different error types and any additional metrics used to evaluate specific accuracy-speed tradeoffs for this model.

### Results

Based on the provided references, here are the evaluation results for the model jinhybr/OCR-DocVQA-Donut:

Performance:
- Donut achieves competitive scores with baselines that depend on external OCR engines on the DocVQA dataset. It is robust on handwritten documents which are challenging to process. (Reference 1)
- Donut shows state-of-the-art performance on Document Classification among general-purpose VDU models like LayoutLM and LayoutLMv2, surpassing LayoutLMv2 accuracy while using fewer parameters and 2x faster speed. (Reference 2) 
- On Document Visual Question Answering (DocVQA dataset), Donut shows stable performance regardless of dataset size and task complexity compared to other baselines. (Reference 3)

Efficiency: 
- Donut is 2x faster than LayoutLMv2 while using fewer parameters. OCR-based models must consider additional parameters and speed for the entire OCR framework. (Reference 2)
- [More Information Needed] on exact inference speed of Donut.

Low-Resource Performance:
- Donut shows robust performance when limiting the training set size of CORD dataset compared to other models. (Reference 4)

[More Information Needed] for additional factors and metrics like model size, memory usage, training time, etc. The provided references focus mainly on accuracy and some efficiency comparisons to other models.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model jinhybr/OCR-DocVQA-Donut:

- Donut achieves state-of-the-art performance on document classification tasks compared to general-purpose VDU models like LayoutLM and LayoutLMv2, while using fewer parameters and being 2x faster. (Reference 1)

- Donut shows stable performance regardless of dataset size and task complexity, which is significant for industry applications. (Reference 2) 

- On the DocVQA dataset, Donut achieves competitive scores with baselines that rely on external OCR engines. Donut is robust on challenging handwritten documents. (Reference 3)

- Donut's performance on DocVQA improves rapidly with larger input image resolutions, at the cost of increased computation. (Reference 4)

- Across public and private datasets, Donut achieves the best scores in terms of both F1 and TED-based accuracy for extracting key information and predicting complex structures. (Reference 5)

- [More Information Needed] on specific visualization and analysis of Donut's strengths and challenges. (Reference 7)

- Donut keeps architecture uniform by generating answers with the question as a starting prompt. (Reference 8)

- Donut demonstrates strong document understanding ability across various VDU tasks, achieving state-of-the-art speed and accuracy without relying on OCR. (Reference 10)

## Model Examination

Based on the provided references about the Donut model (jinhybr/OCR-DocVQA-Donut), here are some key points related to the model's explainability and interpretability:

- Donut shows striking characteristics through experiments and visualization, indicating its effectiveness as a simple yet powerful approach for tasks like OCR and DocVQA. (Reference 1)

- The performance of Donut grows rapidly with larger input image sizes, especially for tasks like DocVQA where images contain many tiny texts. However, increasing input size also leads to higher computational costs. (Reference 3)

- Donut demonstrates strong understanding ability through extensive evaluation on various visual document understanding (VDU) tasks and datasets. (Reference 7)

- [More Information Needed] on specific techniques or methods used for explaining or interpreting Donut's predictions and internal workings.

- [More Information Needed] on any visualizations, attribution methods, or other explainability tools applied to Donut.

In summary, while the provided references highlight Donut's effectiveness and some general characteristics related to its performance and understanding capabilities, more specific information is needed on the explainability and interpretability techniques applied to the model to provide a comprehensive answer to the question.

## Environmental Impact

- **Hardware Type:** According to the references provided, the Donut model was pre-trained using 64 NVIDIA A100 GPUs for 200K steps with a mini-batch size of 196. The fine-tuning on specific datasets like CORD or Ticket took approximately 0.5 hours with one A100 GPU.

For the specific model jinhybr/OCR-DocVQA-Donut, [More Information Needed] on the exact hardware used for training, as it is not explicitly mentioned in the given references.
- **Software Type:** Based on the provided references, the model jinhybr/OCR-DocVQA-Donut is trained using the following software:

- NVIDIA A100 GPUs (References 4 and 5)
- Adam optimizer (References 1 and 4)
- Half-precision (fp16) training (Reference 4)

The references do not explicitly mention the deep learning framework or other specific software used for training the model. [More Information Needed] on the exact deep learning framework (e.g., PyTorch, TensorFlow) and any other software dependencies for training and deploying the model.
- **Hours used:** Based on the references provided, the model jinhybr/OCR-DocVQA-Donut was pre-trained for 200K steps with 64 NVIDIA A100 GPUs and a mini-batch size of 196, which took about 2-3 GPU days.

For the fine-tuning phase on downstream tasks like CORD or Ticket datasets, it took approximately 0.5 hours with one A100 GPU when using an input resolution smaller than the pre-training resolution of 2560×1920.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the jinhybr/OCR-DocVQA-Donut model. More information would be needed from the model development process to determine the cloud provider, if any, that was used.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the model jinhybr/OCR-DocVQA-Donut. To accurately report the carbon emissions, more details would be needed such as the hardware used for training, total training time, energy consumption, and carbon intensity of the electricity used.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The architecture of Donut consists of a Transformer-based visual encoder and textual decoder. The visual encoder converts the input document image into a set of embeddings using Swin Transformer. The textual decoder then maps the derived features into a sequence of subword tokens to construct the desired structured output.

Objective:
Donut is an end-to-end OCR-free visual document understanding model. It directly maps an input document image into a desired structured output format (e.g., JSON) without relying on any OCR engines or APIs. The model is trained using cross-entropy loss as the pre-training objective.

[More Information Needed] on specific implementation details and hyperparameters used in the jinhybr/OCR-DocVQA-Donut model.

### Compute Infrastructure

Based on the provided references, here is the compute infrastructure information for the model jinhybr/OCR-DocVQA-Donut:

- Pre-training:
  - 64 NVIDIA A100 GPUs
  - Mini-batch size of 196
  - 200K training steps, which takes about 2-3 GPU days
  - Half-precision (fp16) training
  - Input resolution set to 2560×1920

- Fine-tuning:
  - [More Information Needed] on the number and type of GPUs used
  - Approximately 0.5 hours with one A100 GPU for fine-tuning on CORD or Ticket datasets
  - Input resolutions are controlled, e.g., 960×1280 used for Train Tickets and Business Card parsing tasks
  - Half-precision (fp16) training

- Inference:
  - Speed measured on a P40 GPU, which is slower than A100

Additional notes:
- Adam optimizer used for both pre-training and fine-tuning
- Learning rate decreased as training progresses
  - Pre-training initial learning rate: 1e-4
  - Fine-tuning initial learning rate: selected from 1e-5 to 1e-4
- Gradient clipping applied with maximum gradient norm selected from 0.05 to 1.0

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

