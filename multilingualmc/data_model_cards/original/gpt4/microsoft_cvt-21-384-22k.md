# Model Card for microsoft/cvt-21-384-22k

The model microsoft/cvt-21-384-22k is a Convolutional vision Transformer (CvT) that introduces convolutions to the Vision Transformer architecture, enhancing performance and efficiency. It is pretrained on the ImageNet-22k dataset and achieves state-of-the-art results on ImageNet-1k validation set without the need for positional encoding, making it suitable for high-resolution vision tasks.

## Model Details

### Model Description

Model Name: microsoft/cvt-21-384-22k

### Model Architecture:
The microsoft/cvt-21-384-22k is a Convolutional vision Transformer (CvT) that introduces convolution-based operations into the Vision Transformer architecture. It employs a multi-stage hierarchy design with three stages, each consisting of a Convolutional Token Embedding layer followed by a stack of Convolutional Transformer Blocks. The Convolutional Token Embedding layer is implemented as a convolution with overlapping patches, which allows for spatial downsampling and increased feature richness. The Convolutional Transformer Block includes a depth-wise separable convolution operation, referred to as Convolutional Projection, for query, key, and value embeddings. This model aims to combine the desirable properties of CNNs, such as shift, scale, and distortion invariance, with the merits of Transformers, like dynamic attention and global context.

### Training Procedures:
The model is pretrained on the ImageNet-22k dataset and can be fine-tuned for downstream tasks. Training configurations can be adjusted using a YAML file, and additional command-line parameters allow for further customization, such as changing the learning rate. The default training command is provided in the reference, and checkpoint, model, and log files are saved in a specified output directory.

### Parameters:
The CvT-21 model has 31.54 million parameters. It is one of the basic models defined in the CvT series, with 21 Transformer Blocks in total. The model also removes the positional encoding, which is typically crucial in Vision Transformers, simplifying the design for higher resolution vision tasks.

### Important Disclaimers:
[More Information Needed] - The references provided do not include specific disclaimers about the model. However, common disclaimers for deep learning models may include limitations regarding the data the model was trained on, potential biases in the model, and the intended use cases for the model. It is also important to note that while the model achieves state-of-the-art performance, results may vary depending on the specific task and dataset.

### Contact Information:
For updates or inquiries regarding the model card, please contact the project organizer (the role you are assuming in this scenario).

(Note: The above model card description is based on the provided references and may require additional details or updates based on the actual model implementation and results.)

- **Developed by:** Haiping Wu; Bin Xiao; Noel Codella; Mengchen Liu; Xiyang Dai; Lu Yuan; Lei Zhang
- **Funded by:** [More Information Needed]
- **Shared by:** The contributors who made the model microsoft/cvt-21-384-22k available online as a GitHub repo are Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang.
- **Model type:** The model microsoft/cvt-21-384-22k is a Convolutional vision Transformer (CvT) that is pre-trained on the ImageNet-22k dataset, representing a supervised learning approach in the computer vision modality.
- **Language(s):** The model microsoft/cvt-21-384-22k does not use or process any natural human language as it is a computer vision model pre-trained on ImageNet-22k for image classification tasks.
- **License:** The license information for the model `microsoft/cvt-21-384-22k` is not explicitly mentioned in the provided references. Therefore, the answer is:

[More Information Needed]
- **Finetuned from model:** The model `microsoft/cvt-21-384-22k` is fine-tuned from a base model that was pre-trained on ImageNet-22k. However, the specific name of the base model is not directly provided in the references above. The references do mention that the CvT-W24 model was pre-trained on ImageNet-22k and obtained a top-1 accuracy of 87.7% on the ImageNet-1k val set, but it does not explicitly state that this is the base model for `microsoft/cvt-21-384-22k`.

For the exact base model name and link, [More Information Needed].
### Model Sources

- **Repository:** https://github.com/microsoft/CvT
- **Paper:** https://arxiv.org/pdf/2103.15808.pdf
- **Demo:** [More Information Needed]
## Uses

### Direct Use

The model `microsoft/cvt-21-384-22k` is a Convolutional Vision Transformer (CvT) that has been pre-trained on the ImageNet-22k dataset. It is designed to handle image classification tasks by capturing both local and global dependencies within an image. The model can be used without fine-tuning, post-processing, or plugging into a pipeline for image classification tasks where the classes are a subset of the ImageNet-22k dataset, as it has already learned a rich representation of image features during pre-training.

To use the model directly for inference, you would typically load the pre-trained model and pass an image through it to obtain the predicted class probabilities. However, since no direct code snippet is provided in the references for using the model without fine-tuning or further processing, I cannot provide a specific code example. If the Huggingface library supports this model, you would typically use their API to load the model and perform inference, but without a direct reference to a code block, I must say [More Information Needed] regarding the exact code snippet.

### Downstream Use

The `microsoft/cvt-21-384-22k` model is a Convolutional Vision Transformer (CvT) that has been pre-trained on the ImageNet-22k dataset. This model is designed to be fine-tuned on a variety of downstream tasks, leveraging its pre-trained weights to achieve high performance even with a smaller number of parameters compared to some larger models.

When fine-tuning `microsoft/cvt-21-384-22k` for a specific task, users can benefit from the model's ability to transfer learning from the large and diverse ImageNet-22k dataset to their task of interest. The model has been shown to perform well on tasks such as image classification on ImageNet-1k, CIFAR-10, CIFAR-100, and Oxford-IIIT Pets and Flowers-102 datasets.

To fine-tune the model, users would typically follow these steps:

1. Prepare the dataset for the specific task, ensuring it is in a format compatible with the model's expected input.
2. Modify the configuration file to specify the new dataset, hyperparameters, and any other necessary changes for the task.
3. Use the provided `run.sh` script to start the fine-tuning process with the desired configuration.

For example, if a user wants to fine-tune the model on a new dataset `DATASET_X` with a specific learning rate, they would:

- Create a new configuration file under `experiments/DATASET_X/cvt/cvt-21-384-22k.yaml` with the appropriate settings.
- Run the training script with the modified learning rate:

```sh
bash run.sh -g 8 -t train --cfg experiments/DATASET_X/cvt/cvt-21-384-22k.yaml TRAIN.LR 0.1
```

After fine-tuning, the model can be integrated into a larger ecosystem or app by loading the fine-tuned weights and using the model for inference. The model can be used for various applications such as image recognition systems, content moderation tools, or any other visual understanding tasks within an application.

For inference, the user would typically load the fine-tuned model and pass the input images to it to obtain predictions. Here's a simplified example of how the model might be used for inference, assuming the necessary libraries and the fine-tuned model file are available:

```sh
bash run.sh -t test --cfg experiments/DATASET_X/cvt/cvt-21-384-22k.yaml TEST.MODEL_FILE ${PRETRAINED_MODEL_FILE}
```

Please note that the actual code for loading the model and running inference would depend on the specific libraries and frameworks used in the application. The above command is a placeholder for the process of testing the model with a given configuration and a pre-trained model file.

### Out-of-Scope Use

As the sociotechnic on the deep learning model development team, it is my responsibility to consider the potential societal impacts and misuse of the model microsoft/cvt-21-384-22k. This model, which incorporates convolutional projections into the Vision Transformer architecture, demonstrates state-of-the-art performance in image recognition tasks (Reference 3, 7, 8). While the model has been designed with computational efficiency and performance in mind, it is important to address potential areas of misuse:

1. **Unauthorized Surveillance**: The model's capabilities in image recognition could be misused for unauthorized surveillance purposes. Users should not employ this model to develop applications that monitor individuals without their consent, which would violate privacy rights and could lead to a breach of trust and legal consequences.

2. **Biased Decision-Making**: If the pre-training data (ImageNet-22k) contains biases, the model may inadvertently perpetuate or amplify these biases, leading to unfair or discriminatory outcomes when used in decision-making systems. Users should not use the model in contexts where biased outputs could lead to discrimination, such as hiring, law enforcement, or access to services, without rigorous bias mitigation and ethical oversight.

3. **Deepfakes and Misinformation**: The model's image recognition capabilities could potentially be adapted to create or propagate deepfakes or other forms of visual misinformation. Users should not use this model to create, distribute, or promote deceptive media content that could undermine trust in digital media and have serious social and political ramifications.

4. **Intellectual Property Violations**: Users should not use the model to analyze or generate images that are subject to copyright without the appropriate permissions, as this could lead to intellectual property rights violations.

5. **Security and Safety**: Given the model's ability to process images without positional embeddings and handle variable input resolutions (Reference 1), there is a risk of misuse in security-sensitive applications. Users should not use the model in ways that could compromise public safety or national security.

6. **Compliance with Open Source Code of Conduct**: As per the Microsoft Open Source Code of Conduct (Reference 5), users should ensure that their use of the model aligns with the principles of respect, fairness, and openness, and should refrain from engaging in disruptive or harassing behavior.

In conclusion, while the microsoft/cvt-21-384-22k model represents a significant advancement in image recognition technology, it is crucial that users consider the ethical and societal implications of its deployment. Users should adhere to legal and ethical standards, respect privacy and intellectual property rights, and be vigilant against the model's potential to exacerbate biases or be used in harmful ways. It is the responsibility of all stakeholders to ensure that the model is used for the benefit of society and not to its detriment.

### Bias, Risks, and Limitations

The model microsoft/cvt-21-384-22k represents an advancement in the field of computer vision by integrating convolutional neural network (CNN) properties with Vision Transformer (ViT) architecture. However, there are several known and foreseeable issues that may arise from the deployment of this model:

1. **Complexity and Resource Requirements**: Reference 2 highlights the high computational and memory costs associated with common input image sizes. While the CvT model aims to address these issues by incorporating pyramid structures and convolutional projections, the complexity of the model may still pose challenges in terms of resource requirements, particularly when deployed on devices with limited computational power.

2. **Generalization to Various Tasks**: Although the model has been shown to perform well on ImageNet-1k and ImageNet-22k datasets (Reference 11), there may be limitations in its ability to generalize to other tasks or datasets that have different characteristics. The performance on tasks not covered in the pre-training or fine-tuning phases may not be as robust.

3. **Positional Embeddings**: The model's design allows for the removal of positional embeddings due to the introduction of Convolutional Projections (Reference 5). While this simplifies the design and may benefit tasks with variable input resolution, it could also lead to unforeseen issues in tasks where positional information is crucial.

4. **Transferability**: Reference 9 discusses the model's transferability to various tasks after fine-tuning. While the results are promising, there may be unforeseen issues when transferring to tasks with significantly different data distributions or when fine-tuning is not feasible due to data limitations.

5. **Sociotechnical Considerations**: As a sociotechnic, it is important to consider the broader implications of deploying this model. For instance, there may be ethical concerns regarding the use of the model in surveillance systems, potential biases in the datasets used for training, and the impact on privacy. The model's interpretability and the ability to explain its decisions are also crucial, especially in sensitive applications.

6. **Code of Conduct and Open Source Considerations**: Reference 7 mentions the adoption of the Microsoft Open Source Code of Conduct. While this promotes ethical use and collaboration, there may be issues related to the misuse of the model, adherence to the code of conduct by all users, and potential legal implications of the model's deployment in various jurisdictions.

7. **Concurrent Work and Innovation Pace**: Reference 8 points out the existence of concurrent work that also aims to improve ViT by incorporating elements of CNNs. The rapid pace of innovation in the field may lead to the model quickly becoming outdated or superseded by new approaches.

In conclusion, while the microsoft/cvt-21-384-22k model shows state-of-the-art performance and introduces innovative design elements, there are technical and sociotechnical limitations and foreseeable issues that need to be considered. These include resource requirements, generalization capabilities, the importance of positional information, transferability to diverse tasks, ethical considerations, adherence to open-source guidelines, and the pace of concurrent innovation.

### Recommendations

Based on the provided references, here are some recommendations with respect to the foreseeable issues about the model microsoft/cvt-21-384-22k:

1. **Model Generalization and Transferability**: The model has demonstrated strong transferability when fine-tuned on various tasks after being pre-trained on ImageNet-22k. However, it is important to continuously evaluate the model on diverse datasets to ensure that it generalizes well across different domains and does not overfit to specific characteristics of the ImageNet dataset.

2. **Positional Encoding**: The model's architecture allows for the removal of positional encoding without performance degradation. This simplification is beneficial for higher resolution vision tasks. However, it is recommended to assess whether this holds true across a wide range of vision tasks, especially those that may rely heavily on positional information.

3. **Computational Efficiency**: The model benefits from a pyramid structure and convolutional projections, which help reduce computational complexity. It is recommended to monitor the computational efficiency when scaling to larger datasets or more complex tasks to ensure that the model remains practical for deployment.

4. **Design Choices**: The model's design choices, such as the use of Convolutional Projections and Convolutional Token Embedding, have shown to improve performance. It is recommended to document these design choices clearly in the model card to aid in reproducibility and further research.

5. **Ethical Considerations**: As a sociotechnic, it is crucial to consider the ethical implications of deploying this model. This includes assessing the potential for bias in the model's predictions, especially given the diversity of real-world scenarios and the data it was trained on. It is recommended to conduct bias and fairness evaluations and to be transparent about the limitations of the model in this regard.

6. **Legal and Rights Advocacy**: Ensure compliance with data protection laws and intellectual property rights when using the model for various applications. It is recommended to provide clear guidelines on the legal use of the model and any restrictions that apply based on the training data or intended use cases.

7. **Societal Impact**: Consider the broader societal impact of the model's deployment, including privacy concerns and the potential for misuse. It is recommended to engage with stakeholders, including ethicists and rights advocates, to develop guidelines for responsible use.

8. **Accessibility**: Make sure that the model and its associated code are accessible to a wide range of users, including those with disabilities. This includes providing clear documentation and support for users to understand and utilize the model effectively.

In summary, while the model shows promising results, it is important to consider generalization, computational efficiency, ethical implications, legal compliance, societal impact, and accessibility when deploying the model in real-world scenarios.

## Training Details

### Training Data

The training data for the model microsoft/cvt-21-384-22k consists of the ImageNet-22k dataset, which includes approximately 14 million images spanning 22,000 classes. For data preparation and pre-processing instructions, please refer to the provided script structure under reference 4. Additional details on data handling and filtering are not provided in the references and may require [More Information Needed].

### Training Procedure

#### Preprocessing

The preprocessing for the data of the model `microsoft/cvt-21-384-22k` involves the following steps:

1. **Image Patch Tokenization**: As per reference [7], images are first split into discrete non-overlapping patches (e.g., 16x16 pixels). This process is analogous to tokenization in natural language processing (NLP), where words are treated as tokens. Here, each image patch is treated as a token.

2. **Positional Encoding**: After the image patches are tokenized, they are summed with a special positional encoding. This step is crucial as it helps to represent coarse spatial information within the image, allowing the model to maintain an understanding of where each patch is located relative to others.

3. **Convolutional Token Embedding**: Reference [1] describes a Convolutional Token Embedding layer that adjusts the token feature dimension and the number of tokens at each stage by varying parameters of the convolution operation. This layer progressively decreases the token sequence length while increasing the token feature dimension, allowing the tokens to represent increasingly complex visual patterns over larger spatial footprints.

4. **Convolutional Projection**: As detailed in reference [2], the Convolutional Projection involves a depth-wise separable convolution that includes a Depth-wise Conv2d followed by BatchNorm2d and a Point-wise Conv2d. This operation is applied to the token input for Q/K/V matrices at each layer.

5. **Resizing and Rewriting Tokens**: Reference [3] explains that the input to each stage is a 2D image or a 2D-reshaped output token map from a previous stage, which is then mapped into new tokens with a channel size C_i using a 2D convolution operation. The new token map has its height and width calculated based on the formula provided, taking into account the kernel size, stride, padding, and other convolution parameters.

6. **Normalization and Complexity Considerations**: Reference [6] highlights that operations like Tokens-to-Token (T2T) differ fundamentally from convolutions, especially in normalization details, and that concatenation of multiple tokens can greatly increase complexity in computation and memory. The CvT model avoids such complexity by employing convolution operations instead.

In summary, the preprocessing for the `microsoft/cvt-21-384-22k` model involves tokenizing images into patches, adding positional encodings, and then applying convolutional operations to embed and project these tokens through a multi-stage hierarchy, which is designed to capture both local and global spatial contexts within the image.

#### Training Hyperparameters

The training hyperparameters for the model `microsoft/cvt-21-384-22k` are as follows:

- Optimizer: AdamW with a weight decay of 0.1, as specified for CvT-21 models.
- Initial Learning Rate: 0.02, with a cosine learning rate decay scheduler.
- Batch Size: A total batch size of 2048.
- Training Epochs: The model is trained for 300 epochs.
- Data Augmentation and Regularization: The same methods as used in ViT are adopted.
- Input Size: Although the default input size for ImageNet models is 224 × 224, since the model name includes '384', it suggests that the input size might be 384 × 384. However, this specific detail is not provided in the references, so [More Information Needed] for confirmation.
- Pretraining Dataset: The model is pretrained on ImageNet-22k.
- Positional Encoding: The positional encoding is removed in this model, simplifying the design for higher resolution vision tasks.

Please note that the exact command to run the training with these hyperparameters is not provided in the references, and the input size is assumed based on the model name. For exact commands or any additional hyperparameters not listed, [More Information Needed].

#### Speeds, Sizes, Times

The model `microsoft/cvt-21-384-22k` is a part of the CvT (Convolutional vision Transformer) series, which introduces a novel architecture that combines the strengths of convolutional neural networks (CNNs) with the Transformer architecture for image classification tasks. This model has been pre-trained on the ImageNet-22k dataset and can be fine-tuned on various downstream tasks.

Regarding the specific details requested:

- Throughput: [More Information Needed]
- Start or end time of training: [More Information Needed]
- Checkpoint sizes: [More Information Needed]

The references provided do not include explicit information about the throughput, start or end time of the training process, or the checkpoint sizes for the `microsoft/cvt-21-384-22k` model. These details are typically logged during the training process and saved in the output directory as mentioned in the notes, but the exact figures are not provided in the references.

For further details on the model's performance and efficiency, reference 3 mentions that the computational cost is reduced by 4 times for the later Multi-Head Self-Attention (MHSA) operation with a minimal performance penalty. Reference 10 indicates that the model maintains performance gains when pre-trained on larger datasets like ImageNet-22k and fine-tuned on downstream tasks, achieving a top-1 accuracy of 87.7% on the ImageNet-1k validation set.

To obtain the specific details requested, one would typically need to access the training logs or the output directory where the checkpoint and log files are saved by default, as mentioned in reference 6. However, without access to these logs or additional information, we cannot provide the exact throughput, start/end times, or checkpoint sizes.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model microsoft/cvt-21-384-22k evaluates on the following benchmarks or datasets:

1. ImageNet (1k classes, 1.3M images) [Reference 6]
2. ImageNet-22k (22k classes, 14M images) [Reference 6]
3. ImageNet Real [Reference 3]
4. ImageNet V2 [Reference 3]
5. CIFAR-10 [Reference 6]
6. CIFAR-100 [Reference 6]
7. Oxford-IIIT-Pet [Reference 6]
8. Oxford-IIIT-Flower [Reference 6]

#### Factors

The model microsoft/cvt-21-384-22k is a Convolutional vision Transformer (CvT) designed for image classification tasks. Based on the provided references, the following characteristics can influence how the model behaves:

1. **Domain and Context**: The model has been evaluated on large-scale image classification datasets and has been fine-tuned on various downstream tasks with all models being pre-trained on ImageNet-22k. This suggests that the model is likely to perform well on tasks similar to those found in ImageNet-22k, which includes a wide variety of natural images across different categories. However, its performance on domains significantly different from natural images, such as medical imaging or satellite imagery, may not be as strong without further domain-specific fine-tuning.

2. **Input Image Size**: The model incorporates pyramid structures to handle different input image sizes, which suggests that it should be robust to variations in image resolution. However, the performance may still vary with image size, and this is an area where disaggregated evaluation would be beneficial to understand the model's performance across different resolutions.

3. **Population Subgroups**: The references do not provide specific information on the performance of the model across different population subgroups. Since the model is trained on ImageNet-22k, it may inherit any biases present in that dataset. For instance, if certain demographic groups are underrepresented in the training data, the model may perform less well on images of individuals from those groups. Disaggregated evaluation across demographic factors is necessary to uncover any disparities in performance.

4. **Modeling Local Spatial Relationships**: The introduction of Convolutional Projections and Convolutional Token Embedding allows the model to capture local spatial relationships without the need for position embeddings. This suggests that the model may be less sensitive to the absolute position of objects within an image, which could be beneficial for tasks where object location varies.

5. **Transferability**: The model has demonstrated the ability to transfer to various tasks after being pre-trained on ImageNet-22k. However, the degree to which it can transfer effectively to tasks that are significantly different from natural image classification is not detailed in the references provided.

6. **Architectural Variants**: The model is one variant within a family that includes models with different numbers of parameters and computational complexity (e.g., CvT-13, CvT-21, CvT-W24). The specific variant, CvT-21-384-22k, may behave differently in terms of performance and efficiency compared to other variants, and these differences could influence its suitability for certain applications or deployment contexts.

In summary, while the CvT-21-384-22k model shows promise in terms of handling variable input image sizes and transferring to various tasks, further evaluation is needed to fully understand its performance across different domains, contexts, and population subgroups. Disaggregated evaluation across these factors is crucial to uncover any potential disparities in performance.

#### Metrics

The evaluation of the model microsoft/cvt-21-384-22k will primarily use the Top-1 Accuracy metric on the ImageNet dataset, as indicated by the references to ImageNet Top-1 accuracy improvements and performance measurements. Additionally, the model's computational efficiency is considered through the measurement of FLOPs (floating-point operations per second), with references to reductions in computational cost and memory usage. The tradeoffs between different errors are considered by evaluating the impact of design choices such as Convolutional Projection with different strides, the introduction of Convolutional Token Embedding, and the removal of positional encoding on both accuracy and computational efficiency.

To summarize, the metrics used for evaluation will include:
- Top-1 Accuracy on ImageNet (as a measure of model performance)
- FLOPs (as a measure of computational efficiency)

These metrics will help in understanding the tradeoffs between model accuracy and computational resources required.

### Results

The evaluation results of the model `microsoft/cvt-21-384-22k` based on the factors and metrics are as follows:

- **ImageNet Top-1 Accuracy**: The CvT-21 model achieves a Top-1 accuracy of 82.5% on the ImageNet dataset.
- **Parameters and FLOPs**: Compared to DeiT-B, CvT-21 has 63% fewer parameters and 60% fewer FLOPs, indicating a more efficient model in terms of computational resources.
- **Comparison with Other Models**: CvT-21 outperforms several concurrent Transformer-based models such as PVT-Small, T2T-ViT t-14, and TNT-S with respective Top-1 accuracy improvements of 1.7%, 0.8%, and 0.2%.
- **Pre-training and Fine-tuning**: The subscript '22k' indicates that the model was pre-trained on the ImageNet22k dataset and fine-tuned on ImageNet1k with an input size of 384 × 384 pixels.
- **Transfer Learning**: The model demonstrates strong transfer learning capabilities, having been pre-trained on ImageNet-22k and fine-tuned on various downstream tasks, showing competitive performance even against models with significantly more parameters.

For more detailed evaluation results or specific metrics on downstream tasks, [More Information Needed] as the provided references do not include those specific details.

#### Summary

The model `microsoft/cvt-21-384-22k` is a Convolutional vision Transformer (CvT) with 21 Transformer blocks, pre-trained on the ImageNet-22k dataset and fine-tuned on ImageNet1k with an input resolution of 384×384 pixels. The model achieves a top-1 accuracy of 82.5% on the ImageNet-1k validation set, which is 0.5% higher than the DeiT-B model while having 63% fewer parameters and 60% fewer FLOPs. This demonstrates the model's efficiency and effectiveness in large-scale image classification tasks.

The model outperforms several concurrent Transformer-based models and CNN-based models, showcasing its superior advantages in terms of accuracy with fewer parameters. The design of the CvT architecture allows for the removal of positional encoding, which is typically a crucial component in Vision Transformers, thereby simplifying the design for higher resolution vision tasks.

The model's architecture can potentially be further improved in terms of parameters and FLOPs through neural architecture search (NAS), which could optimize the stride for each convolution projection and the expansion ratio for each MLP layer.

Overall, the `microsoft/cvt-21-384-22k` model presents a significant advancement in the field of vision Transformers, offering high accuracy with a more efficient use of parameters and computational resources.

## Model Examination

Model Card for microsoft/cvt-21-384-22k

## Model Description

The microsoft/cvt-21-384-22k is a Convolutional vision Transformer (CvT) model designed for large-scale image classification tasks and transfer learning to various downstream datasets. This model is a part of the CvT series, which introduces a novel architecture that combines the strengths of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) to address the computational and memory costs associated with ViTs, especially for common input image sizes.

## Technical Details

- **Architecture**: The CvT-21 model incorporates a pyramid structure similar to CNNs into the Transformer structure, employing convolutions with stride to spatially subsample the feature map or key/value matrices in projection. This design allows for the fusion of local neighboring information, which is crucial for performance.
- **Convolutional Projections**: Unlike traditional ViTs that use position-wise linear projections, CvT replaces these with convolutional projections, which enables the model to capture local spatial relationships effectively.
- **Positional Embeddings**: The CvT architecture's reliance on convolutional projections and token embedding allows for the removal of positional embeddings without impacting performance. This simplification is beneficial for vision tasks with variable input resolutions.
- **Model Variants**: The CvT-21 model is one of the basic models with 31.54 million parameters. It is part of a family that includes other variants like CvT-13 and CvT-W24, which is a wider model with a larger token dimension and significantly more parameters (298.3 million).
- **Pretraining and Fine-tuning**: The model is pretrained on the ImageNet-22k dataset and has shown impressive transfer learning capabilities when fine-tuned on various downstream tasks.

## Experimental Results

- **Image Classification**: Our experiments demonstrate that the CvT-21 model achieves high performance on large-scale image classification datasets.
- **Ablation Studies**: Various ablation experiments were conducted to validate the design choices of the CvT architecture. For instance, replacing the Convolutional Token Embedding with non-overlapping Patch Embedding resulted in a performance drop, highlighting the effectiveness of our approach.
- **Transfer Learning**: The CvT models, including CvT-21, have shown excellent ability to transfer to different tasks, outperforming models with significantly more parameters.

## Explainability/Interpretability

[More Information Needed]

## Additional Information

- **Code Availability**: The code for the CvT models will be released at https://github.com/leoxiaobin/CvT.
- **Pretraining Dataset**: The model was pretrained on the ImageNet-22k dataset.
- **Top-1 Accuracy**: While not specific to CvT-21, the wider CvT-W24 model achieved a top-1 accuracy of 87.7% on the ImageNet-1k validation set after pretraining on ImageNet-22k.

For more detailed information and results, please refer to our comprehensive experiments and ablation studies outlined in the references provided.

*Note: This model card is a summary based on the provided references and does not include additional experimental details or results that may be available in the full model documentation or research papers.*

## Environmental Impact

- **Hardware Type:** [More Information Needed]
- **Software Type:** The model microsoft/cvt-21-384-22k is trained on PyTorch. This information is derived from reference 3, which mentions the requirement of having PyTorch and TorchVision installed and provides the official instruction link for their installation. The code development and testing are specifically mentioned for PyTorch version 1.7.1.
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The architecture of the model `microsoft/cvt-21-384-22k` is based on the Convolutional vision Transformer (CvT), which introduces convolution-based operations into the Vision Transformer architecture to improve performance and efficiency. The model employs a multi-stage hierarchy design with three stages, each consisting of Convolutional Token Embedding and a stack of Convolutional Transformer Blocks.

The Convolutional Token Embedding is implemented as a convolution with overlapping patches, where tokens are reshaped to a 2D spatial grid. This allows for spatial downsampling and increased feature dimension richness through the stages. An additional layer normalization is applied to the tokens.

The Convolutional Transformer Block, which forms the remainder of each stage, includes a depth-wise separable convolution operation referred to as Convolutional Projection. This operation is applied to query, key, and value embeddings to model local spatial contexts, from low-level edges to higher-order semantic primitives, over the multi-stage hierarchy.

The model's objective is to achieve state-of-the-art performance in image classification tasks by combining the desirable properties of CNNs (such as shift, scale, and distortion invariance) with the merits of Transformers (such as dynamic attention, global context, and better generalization). The CvT architecture does this while maintaining fewer parameters and lower FLOPs compared to other Vision Transformers and ResNets.

Pretrained on ImageNet-22k, the `microsoft/cvt-21-384-22k` model achieves a top-1 accuracy of 87.7% on the ImageNet-1k validation set. Notably, the model removes the need for positional encoding, which is a common component in existing Vision Transformers, thus simplifying the design for higher resolution vision tasks.

### Compute Infrastructure

The compute infrastructure used for training the model microsoft/cvt-21-384-22k is not explicitly detailed in the provided references. However, we can infer some aspects of the training setup based on the information given:

1. The training command suggests the use of a distributed training setup with multiple GPUs, as indicated by the `-g 8` flag, which implies that the training was conducted on a system with at least 8 GPUs.

2. The model utilizes efficient convolutions, specifically depth-wise separable convolutions, to reduce the number of parameters and floating-point operations per second (FLOPs), suggesting that the model is designed to be computationally efficient.

3. The model was pre-trained on the ImageNet-22k dataset and fine-tuned on various downstream tasks, indicating that a significant amount of computational resources would be required for pre-training, given the size of the ImageNet-22k dataset.

4. The AdamW optimizer was used with specific weight decay settings for different model variants, and a cosine learning rate decay scheduler was employed. This indicates that the training setup was capable of utilizing these optimization techniques.

5. The training was conducted with an initial learning rate of 0.02 and a total batch size of 2048 for 300 epochs, which suggests that the infrastructure was able to handle a large batch size and a substantial number of training epochs.

6. The models were trained with an input size of 224 × 224, which is a common resolution for image models and implies that the infrastructure could support the processing of images at this resolution.

Without more specific details on the exact hardware specifications, such as the type of GPUs, CPU, memory, storage, and networking used, we cannot provide a complete description of the compute infrastructure for the model microsoft/cvt-21-384-22k. Therefore, for any information not inferred from the provided references, we must say "[More Information Needed]".

## Citation

```
@misc{haiping-cvt,
    author = {Haiping Wu and
              Bin Xiao and
              Noel Codella and
              Mengchen Liu and
              Xiyang Dai and
              Lu Yuan and
              Lei Zhang},
    title  = {CvT: Introducing Convolutions to Vision Transformers},
    url    = {https://arxiv.org/pdf/2103.15808.pdf}
}
```

