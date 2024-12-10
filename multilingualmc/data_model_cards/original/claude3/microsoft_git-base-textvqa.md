# Model Card for microsoft/git-base-textvqa

The microsoft/git-base-textvqa model is a generative image-to-text transformer (GIT) that unifies vision-language tasks such as visual question answering (VQA). It achieves state-of-the-art performance on various VQA benchmarks by interpreting the question as a caption prefix and generating the answer directly, without pre-defined candidate answers.

## Model Details

### Model Description

Model Architecture:
The model consists of an image encoder and a text decoder in a transformer architecture. The image encoder is based on a contrastive pre-trained model that takes a raw image as input and outputs a compact 2D feature map, which is flattened and projected to D dimensions. The text is tokenized, embedded into D dimensions, and concatenated with the image features as input to the transformer module. The text is decoded auto-regressively until the [EOS] token or maximum steps are reached.

Training Procedures:
The model is pre-trained on large-scale image-text paired data with a language modeling objective to generate the associated text description given an input image. For fine-tuning on visual question answering tasks like TextVQA, the question and ground-truth answer are concatenated as a special caption, with the language modeling loss only applied to the answer and [EOS] tokens. The model learns to predict the answer in a generative manner without pre-defined candidate answers.

Parameters:
[More Information Needed]

Important Disclaimers:
- The model is a generative approach to visual question answering, which imposes more challenges compared to discriminative methods that use pre-defined answer candidates. Due to the difficulty of the generative task, the model may have slightly worse performance on some VQA benchmarks compared to discriminative approaches.
- [More Information Needed]

For model card updates or further information, please contact the project organizer of the microsoft/git-base-textvqa development team.

- **Developed by:** Jianfeng Wang; Zhengyuan Yang; Xiaowei Hu; Linjie Li; Kevin Lin; Zicheng Liu; Ce Liu; Lijuan Wang
- **Funded by:** Based on the provided references, the model microsoft/git-base-textvqa appears to be funded by Microsoft. This can be inferred from a few key points:

1. The model name itself includes "microsoft", indicating it is a Microsoft project.

2. Reference 4 mentions Microsoft trademarks and logos, further suggesting this is a Microsoft project. 

3. Reference 5 mentions this project has adopted the Microsoft Open Source Code of Conduct.

4. Many of the authors listed in Reference 3 seem to be Microsoft affiliated based on their email contact in Reference 5 (opencode@microsoft.com).

However, to say definitively that Microsoft is the sole funder, [More Information Needed]. The references do not explicitly state all funding sources. It's possible there are other funding sources not mentioned here. But Microsoft's involvement and likely funding seems clear based on the available information.
- **Shared by:** Based on the references provided, the contributors who made the model microsoft/git-base-textvqa available online as a GitHub repo are:

Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zicheng Liu, Ce Liu, and Lijuan Wang.

The references do not provide complete information about the specific roles and contributions of each individual in making the model available on GitHub. [More Information Needed] on those details.
- **Model type:** The microsoft/git-base-textvqa model is a generative image-to-text model that uses a language modeling loss for fine-tuning on the visual question answering task, with an image encoder and text decoder architecture trained on image-text pairs.
- **Language(s):** The model microsoft/git-base-textvqa processes natural language text descriptions associated with input images, based on being pre-trained on large-scale image-text pairs.
- **License:** [More Information Needed]

The provided references do not contain any information about the specific license being used for the microsoft/git-base-textvqa model. More details would be needed from the model developers or documentation to determine the license under which this model is being released.
- **Finetuned from model:** The model microsoft/git-base-textvqa is fine-tuned from a base model initialized from contrastive pretraining, as mentioned in Reference 2:

"In our design, the image encoder is initialized from the contrastive pretraining."

However, the specific name and link to the base contrastive pretraining model is not provided in the given references. [More Information Needed] on the exact base model used for initialization.
### Model Sources

- **Repository:** https://github.com/microsoft/GenerativeImage2Text
- **Paper:** https://arxiv.org/pdf/2205.14100.pdf
- **Demo:** Based on the provided references, there is no direct mention of a demo link for the model microsoft/git-base-textvqa. The references mainly discuss the model architecture, training details, and evaluation metrics, but do not provide a specific demo URL.

[More Information Needed] A demo link showcasing the capabilities and usage of the microsoft/git-base-textvqa model would be helpful to include in the model card description. If available, please provide the relevant URL.
## Uses

### Direct Use

The model microsoft/git-base-textvqa can be used for visual question answering without fine-tuning, post-processing, or plugging into a pipeline. Here's a code snippet demonstrating how to use the model for inference on a single image:

```shell
AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', \
      'image_path': 'aux_data/images/1.jpg', \
      'model_name': 'GIT_BASE', \
      'prefix': 'what is it?', \
}"
```

In this example:
- The `image_path` specifies the path to the input image.
- The `model_name` is set to 'GIT_BASE', indicating the use of the base model.
- The `prefix` contains the question to be answered based on the image.

The model generates the answer in an auto-regressive manner without predefined candidate answers. It predicts at least two correct tokens: one for the answer and another for the [EOS] token.

[More Information Needed] regarding the specific performance details and comparisons with existing approaches for visual question answering tasks.

### Downstream Use

The microsoft/git-base-textvqa model can be fine-tuned for various tasks such as image captioning, visual question answering (VQA), and image classification. Here are a few examples of how it can be used:

For image captioning, the model can be fine-tuned using the same LM task as in pre-training, since the training data format is the same (Reference 5).

For VQA tasks like VQAv2, the model takes a question as input (concatenated with the image) and generates the answer in an auto-regressive manner. This is more challenging than classification-based approaches as the model needs to predict the correct answer tokens and the [EOS] token (References 1, 3).

```python
AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', \
        'image_path': ['aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg', 'aux_data/images/1.jpg'], \
        'model_name': 'GIT_BASE_MSRVTT_QA', \
        'prefix': 'what is it?', \
  }"
```

For image classification, the model predicts the class name in an auto-regressive way, which is beneficial when adding new data and categories to an existing dataset (Reference 2).

The model can be plugged into larger ecosystems or apps for tasks like generating captions for images, answering questions about images/videos, or classifying images. The specific integration would depend on the application.

[More Information Needed] on detailed code snippets for fine-tuning or integration into larger systems.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the microsoft/git-base-textvqa model may be misused and what users should avoid doing:

The model shows some gender and skin tone biases ranging from 0.7% to 5.3% across evaluation metrics when tested on COCO images (reference 4-5). Users should be aware of these biases and not rely on the model's outputs in sensitive contexts where such biases could lead to harmful or discriminatory impacts.

The model is designed for specific vision-language tasks like image captioning and visual question answering. Applying it to significantly different tasks or data domains it wasn't trained for may lead to unreliable, nonsensical, or misleading outputs. Users should only use the model as intended and documented.

[More Information Needed] on other specific technical limitations, failure modes, or potential vulnerabilities that malicious actors may try to exploit to misuse the model.

In general, users should not treat the model's outputs as perfectly accurate, unbiased, or authoritative, especially for high-stakes decisions. Outputs should be carefully reviewed by humans. The model should be used to assist and augment human intelligence, not fully automate important judgments.

[More Information Needed] on intended uses that may raise ethical concerns around privacy, surveillance, manipulation, etc. that users should avoid, even if the model is capable of those applications from a technical perspective.

### Bias, Risks, and Limitations

Based on the provided references about the GIT model, here are some known or foreseeable issues:

Potential Biases and Harms:
- The model was pre-trained on large-scale data that may contain toxic language, which could poison the output (ref 7, 9). More research is needed to control the output and mitigate potential harms before deploying the model in practice.
- Some gender and skin tone biases were measured in the model's performance on COCO images, ranging from 0.7% to 5.3% normalized performance difference between groups (ref 4, 5). 

Technical Limitations:
- The base model's performance dropped with the largest 0.8B pre-training dataset on COCO, possibly because the model has limited capacity to effectively benefit from very large-scale noisy data (ref 1). Scaling the model size helps more.
- Scaling up the text decoder size did not improve performance, possibly due to limited text data or the decoder's simpler language modeling task (ref 2, 6). The image encoder seems more important to scale.

[More Information Needed] on other specific technical limitations, potential misunderstandings or negative sociotechnical impacts. The references focus more on the model's capabilities and performance. A deeper analysis of limitations and risks from multiple perspectives would be valuable to include in the model card to promote responsible use.

### Recommendations

Based on the provided references about the GIT (General Image-to-Text Transformer) model, here are some recommendations regarding foreseeable issues:

1. Bias and fairness: The model shows relatively low bias (0.7% to 5.3% NPD) across gender groups in the evaluation (Ref 3). However, more analysis may be needed to assess bias and fairness issues across other demographic groups and image types.

2. Robustness to noisy data: The model's performance on COCO drops with 0.8B pre-training data compared to 14M, possibly because the 0.8B data is noisier and less similar to COCO (Ref 2). The model's robustness to noisy and out-of-domain data should be further evaluated and improved if needed.

3. Limitations in text generation: The model shows difficulty in effectively scaling up the text decoder (Ref 1, 4). More research may be needed to enhance the model's language generation capabilities, especially for complex or less common text patterns.

4. Responsible use and potential misuse: While the model shows impressive results in recognizing and describing a wide range of visual content (Ref 7), its capabilities could potentially be misused, e.g. for generating deceptive or harmful content. Responsible use guidelines and safeguards should be provided.

5. Transparency on training data: To help users better understand the model's behavior and limitations, more details could be provided on the pre-training data sources, sizes, and characteristics (Ref 5).

[More Information Needed] on other aspects like privacy, security, environmental impact, etc. Further interdisciplinary analysis from legal, ethical and sociological perspectives is recommended to identify and address potential issues proactively.

## Training Details

### Training Data

The training data consists of image-text pairs, where the images are preprocessed to have the shorter side no larger than 384 pixels and the longer side no larger than 640 pixels while maintaining the aspect ratio, and are saved in JPEG format with quality 90. [More Information Needed] on the specific datasets used for training the microsoft/git-base-textvqa model.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the data of the model microsoft/git-base-textvqa:

For pre-training data:
- Images are preprocessed to make sure the shorter length is no larger than 384 and the longer side no larger than 640 while maintaining the aspect ratio. 
- All images are re-saved with quality being 90 in the JPEG format.
- This preprocessing results in 39 terabytes of data.

For fine-tuning data (including TextVQA):
- No such preprocessing as described above is applied on the fine-tuning dataset.

[More Information Needed] on specifics of tokenization and resizing/rewriting for the TextVQA data.

The references mention extending to video by extracting features from multiple sampled frames and concatenating them as the video representation, but [More Information Needed] on if this was done for the microsoft/git-base-textvqa model specifically.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the model microsoft/git-base-textvqa:

- The learning rate is warmed up in the first 500 iterations, and then follows cosine decay to 0. 
- The learning rate is 1e-5 for the image encoder and is multiplied by 5 for the randomly initialized text decoder.

The references do not provide complete information about other hyperparameters such as batch size, optimizer, number of training epochs, etc. [More Information Needed] for those details.

#### Speeds, Sizes, Times

Based on the provided references, there is not enough specific information to provide details about the throughput, start or end time, checkpoint sizes, etc. for the model microsoft/git-base-textvqa. The references mention some high-level training details like using A100 GPUs on Azure Machine Learning, but do not give the requested specifics for this particular model.

[More Information Needed] on the detailed throughput, training times, and checkpoint sizes for microsoft/git-base-textvqa.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model microsoft/git-base-textvqa evaluates on the following VQA benchmarks and datasets:

VQAv2 (Goyal et al., 2017)
TextVQA
VizWiz-VQA (Gurari et al., 2018)
ST-VQA (Biten et al., 2019)
OCR-VQA (Mishra et al., 2019)

The references also mention evaluating on video QA datasets like MSVD-QA and MSRVTT-QA, but it's not clear if the specific microsoft/git-base-textvqa model was evaluated on those. More information would be needed to confirm.

#### Factors

Based on the provided references about the model microsoft/git-base-textvqa, here are some foreseeable characteristics that may influence the model's behavior:

Domain and Context:
- The model is designed for visual question answering tasks, specifically on datasets like VQAv2, TextVQA, VizWiz-VQA, ST-VQA, and OCR-VQA (Reference 8). Performance may vary across these different VQA datasets.
- The model uses a generative approach without pre-defined answer candidates, which poses more challenges compared to discriminative models (Reference 9). This generative nature could impact performance.
- Pre-training data containing scene text descriptions plays a critical role in the model's capability to handle scene text and generate informative captions (Reference 7).

Population Subgroups:
- The model shows some bias when evaluated on COCO images annotated with gender (male/female) and skin type (light/dark). The normalized performance difference (NPD) ranges from 0.7% to 5.3% across metrics (Reference 6). This indicates potential disparities in performance across these subgroups.

[More Information Needed] on further disaggregated evaluation across other relevant factors and subgroups to uncover any additional performance disparities.

Overall, the model's behavior is influenced by the specific VQA datasets it is applied to, its generative nature, reliance on pre-training data with scene text, and exhibits some bias based on gender and skin type annotations. More disaggregated evaluation would help provide a clearer picture of the model's performance characteristics across different subgroups and contexts.

#### Metrics

Based on the provided references, the following metrics will likely be used to evaluate the microsoft/git-base-textvqa model:

On the VQA (Visual Question Answering) task, common evaluation benchmarks include:
- VQAv2
- TextVQA 
- VizWiz-VQA
- ST-VQA
- OCR-VQA

The references mention that generative VQA models require predicting both the correct answer and the [EOS] token, making them more difficult to evaluate compared to discriminative models.

For video QA tasks, evaluation datasets mentioned include:
- MSVD-QA
- MSRVTT-QA

[More Information Needed] on the specific metrics used for VQA evaluation on these datasets, such as accuracy, BLEU, METEOR, CIDEr, SPICE etc. The tradeoffs between different error types are also not explicitly discussed for the microsoft/git-base-textvqa model.

In summary, while several relevant VQA evaluation datasets are listed, more details are needed on the exact metrics and error tradeoffs considered for this specific model's evaluation.

### Results

Based on the provided references, here is the evaluation information I could find for the microsoft/git-base-textvqa model:

Factors:
- The model was evaluated on the TextVQA dataset, which is one of the VQA evaluation benchmarks mentioned in reference 1.

Metrics:
[More Information Needed] - The specific metrics and results on TextVQA were not provided in the given references.

The references mention evaluating GIT models on various VQA datasets, but do not give the specific results for the microsoft/git-base-textvqa model on the TextVQA dataset. More information would be needed from the model developers to report the quantitative performance metrics this model achieved on the TextVQA evaluation.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model microsoft/git-base-textvqa:

The model was evaluated on various VQA benchmarks including VQAv2, TextVQA, VizWiz-VQA, ST-VQA, and OCR-VQA. The generative approach of the model requires predicting more tokens correctly compared to a discriminative approach, which may explain slightly lower performance compared to some other models like Florence that use a discriminative approach with RoBERTa text encoder.

In zero-shot/few-shot settings on TextVQA without vocabulary knowledge, the model achieves only 1.93% exact match accuracy. However, if relaxing the requirement to count a prediction as correct if it contains the ground truth, accuracy is 40.88%, showing the model can identify image content well.

The model exhibits strong performance in recognizing and inferring scene text, even when occluded, by leveraging both visual and textual context. It can read handwritten text and long pieces of scene text, organizing the words into fluent sentences.

[More Information Needed] on the specific performance metrics achieved by the microsoft/git-base-textvqa model on each benchmark. The provided references focus more on general trends and qualitative analysis rather than stating the quantitative results for this specific model.

## Model Examination

Based on the provided references about the GIT (Generative Image-to-Text) model, here is the experimental section on explainability/interpretability for the microsoft/git-base-textvqa model:

The GIT model achieves impressive performance on image captioning and visual question answering tasks by scaling up pre-training data and model size (Reference 7). To better understand how the model works:

- [More Information Needed] on analyzing the impact of pre-training data size and similarity to downstream tasks. Reference 4 mentions performance trends with 4M, 14M and 0.8B pre-training examples on COCO, TextCaps and VizWiz-QA, but more details would be needed to include an analysis here.

- [More Information Needed] on the role of the image encoder vs text decoder in the model. Reference 3 hypothesizes the image encoder handles object recognition while the decoder organizes terms into natural language, but experiments to validate this would improve interpretability.

- The model's strong zero-shot capabilities to recognize and describe a wide variety of visual concepts like text, charts, food, logos, landmarks, etc. (Reference 7) could be further probed to understand the extent and limitations of the visual knowledge captured during pre-training.

- [More Information Needed] on comparing the generation-based classification approach to predicting a category likelihood directly (Reference 6), to assess pros/cons for model explainability.

In summary, while GIT demonstrates remarkable performance, additional experiments on the pre-training data, model components, visual knowledge, and classification approach could improve understanding of its capabilities and limitations. We leave these to future work on explainable AI for GIT models.

## Environmental Impact

- **Hardware Type:** The model microsoft/git-base-textvqa is trained on A100 GPUs provisioned by Azure Machine Learning, according to the information provided in the references:

"The data are stored in Azure Blob Storage, and the training is conducted on A100 provisioned by Azure Machine Learning."
- **Software Type:** Based on the references provided, the model microsoft/git-base-textvqa is trained using the following software:

- Python
- PyTorch
- DeepSpeed
- Transformers
- maskrcnn-benchmark
- CLIP
- OSCAR
- VirTex

The training is conducted on A100 GPUs provisioned by Azure Machine Learning. The data used for training is stored in Azure Blob Storage.
- **Hours used:** [More Information Needed]

The provided references do not contain specific information about the training time for the microsoft/git-base-textvqa model. More details would be needed from the model developers or documentation to determine the amount of time used for training this particular model.
- **Cloud Provider:** The model microsoft/git-base-textvqa is trained on Azure. Specifically, the data are stored in Azure Blob Storage, and the training is conducted on A100 GPUs provisioned by Azure Machine Learning.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the microsoft/git-base-textvqa model. While the references mention some details about the training setup, such as using A100 GPUs provisioned by Azure Machine Learning and the number of training images, they do not provide the carbon emission data. To accurately report the carbon footprint, more details would be needed, such as the total training time, energy consumption of the hardware, and the carbon intensity of the electricity used.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The model architecture of microsoft/git-base-textvqa consists of one image encoder and one text decoder. The image encoder is based on a contrastive pre-trained model (e.g., CLIP/ViT-B/16) that takes a raw image as input and outputs a compact 2D feature map, which is flattened into a list of features. These features are then projected into D dimensions using an extra linear layer and a layernorm layer, and serve as input to the text decoder. The text decoder is a transformer module that takes the concatenated image features and text embeddings as input. The text begins with a [BOS] token and is decoded auto-regressively until the [EOS] token or reaching the maximum steps.

Model Objective:
The objective of microsoft/git-base-textvqa is to map the input image to the associated text description using a single language modeling task. The model is pre-trained on large-scale image-text pairs to unify vision-language tasks such as image/video captioning and question answering. By simplifying the architecture to one image encoder and one text decoder, the model aims to provide a consistent network architecture between pre-training and fine-tuning without relying on external modules like object detectors or optical character recognition (OCR).

[More Information Needed] on specific pre-training hyperparameters, dataset size, and model scaling details.

### Compute Infrastructure

Based on the provided references, here is the information about the compute infrastructure used for the microsoft/git-base-textvqa model:

The data are stored in Azure Blob Storage, and the training is conducted on A100 GPUs provisioned by Azure Machine Learning. The code is in Python with packages including PyTorch, DeepSpeed, Transformers, maskrcnn-benchmark, CLIP, OSCAR, and VirTex.

For inference, the model can be run using Docker with the following command:

```shell
nvidia-docker run --ipc=host amsword/setup:py38pt19u20cu11 \
    bash -c "mkdir /tmp/code \
            && cd /tmp/code \
            && pip install git+https://github.com/microsoft/azfuse.git \
            && git clone https://github.com/amsword/generativeimage2text.git \
            && cd generativeimage2text \
```

[More Information Needed] on the exact compute resources like number of GPUs, memory, etc. used for training the base model.

## Citation

```
@misc{jianfeng-git,
    author = {Jianfeng Wang and
              Zhengyuan Yang and
              Xiaowei Hu and
              Linjie Li and
              Kevin Lin and
              Zicheng Liu and
              Ce Liu and
              Lijuan Wang},
    title  = {GIT: A Generative Image-to-text Transformer for Vision and Language},
    url    = {https://arxiv.org/pdf/2205.14100.pdf}
}
```

