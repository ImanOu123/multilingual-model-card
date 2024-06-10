# Model Card for google/mobilenet_v1_0.75_192

The model google/mobilenet_v1_0.75_192 is a MobileNet model, which is a class of efficient convolutional neural network models designed for mobile and embedded vision applications. It uses depthwise separable convolutions and two global hyperparameters (width multiplier of 0.75 and resolution of 192x192) to build a lightweight deep neural network that efficiently trades off between latency and accuracy.

## Model Details

### Model Description

Model Architecture:
- MobileNet is based on depthwise separable convolutions, which factorize a standard convolution into a depthwise convolution and a 1x1 pointwise convolution. This significantly reduces computation and model size.
- The model uses 3x3 depthwise separable convolutions, which use 8-9 times less computation than standard convolutions.
- The architecture is defined in Table 1 [More Information Needed]. It uses depthwise separable convolutions in all layers except the first, which is a full convolution. Layers are followed by batchnorm and ReLU, except the final fully connected layer.

Training Procedures:
- Trained in TensorFlow using RMSprop with asynchronous gradient descent
- Used less regularization and data augmentation compared to training large models, as small models are less prone to overfitting
- No side heads or label smoothing used
- Reduced image distortions by limiting small crop sizes
- Put very little or no weight decay (L2 regularization) on depthwise filters due to having few parameters

Parameters:
- Width Multiplier: Thins the network uniformly at each layer to reduce computation by roughly [More Information Needed]
- Resolution Multiplier: Reduces the input image resolution to further reduce computation

Important Disclaimers:
[More Information Needed]

Please reach out if you need any additional details or have suggestions for improving the model card. We aim to provide transparent and responsible documentation.

- **Developed by:** Andrew G Howard; Menglong Zhu; Bo Chen; Dmitry Kalenichenko; Weijun Wang; Tobias Weyand; Marco Andreetto; Hartwig Adam
- **Funded by:** [More Information Needed]

Based on the provided references, there is no information about the funding sources for the google/mobilenet_v1_0.75_192 model. The references only contain performance metrics and download links for various MobileNet v1 model configurations. To answer this question, more details would be needed about the organizations or individuals that financially supported the development of this specific model.
- **Shared by:** Based on the provided references, there is no information about the specific contributors who made the model google/mobilenet_v1_0.75_192 available online as a GitHub repo. The references discuss the MobileNet architecture and training details, but do not mention the individuals involved in publishing this specific pre-trained model. To answer the question of who the contributors were, [More Information Needed] beyond what is provided in the given references.
- **Model type:** The google/mobilenet_v1_0.75_192 model is a convolutional neural network trained on ImageNet using asynchronous gradient descent with RMSprop optimizer, utilizing depthwise separable convolutions for efficient mobile and embedded vision applications, with width multiplier α=0.75 and resolution multiplier ρ=192/224 for reduced computation.
- **Language(s):** The model google/mobilenet_v1_0.75_192 processes images and does not use or process natural human language.
- **License:** [More Information Needed]

The provided references do not contain any information about the specific license being used for the google/mobilenet_v1_0.75_192 model. More details would be needed from the model creators or documentation to determine the license under which this model is being released.
- **Finetuned from model:** Based on the provided references, the model google/mobilenet_v1_0.75_192 does not appear to be fine-tuned from another model. It seems to be one of the base MobileNet v1 models with a width multiplier of 0.75 and input resolution of 192x192.

The references mention several MobileNet v1 models with different width multipliers and input resolutions, but do not indicate that any of them are fine-tuned from others. They appear to be independently trained base models.

[More Information Needed] on whether google/mobilenet_v1_0.75_192 was fine-tuned from a different model. The given references do not provide that information.
### Model Sources

- **Repository:** https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
- **Paper:** https://arxiv.org/pdf/1704.04861.pdf
- **Demo:** Based on the information provided in the References, the link to the demo of the model google/mobilenet_v1_0.75_192 is:

[MobileNet_v1_0.75_192](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_192.tgz)
## Uses

### Direct Use

The MobileNet model google/mobilenet_v1_0.75_192 can be used for image classification tasks without requiring fine-tuning, post-processing, or plugging into a pipeline. It has been pre-trained on the ILSVRC-2012-CLS dataset.

To use the model for evaluation:

[More Information Needed]

The model has a width multiplier of 0.75 and an input resolution of 192x192 pixels. It offers a trade-off between accuracy, latency and model size compared to other MobileNet v1 configurations.

[More Information Needed] on specific accuracy metrics for this model variant.

The model can also potentially be used for other applications like fine-grained recognition or as a base for compression via knowledge distillation, but [More Information Needed] on using this specific variant for those use cases.

In summary, the google/mobilenet_v1_0.75_192 model provides a good balance of accuracy and efficiency for image classification, and can be easily deployed as-is for that purpose. More details would be needed to provide code snippets or determine its suitability for additional scenarios.

### Downstream Use

The google/mobilenet_v1_0.75_192 model can be fine-tuned for various tasks such as:

1. Fine-grained recognition: The model can be pre-trained on a large but noisy dataset collected from the web and then fine-tuned on a smaller, clean dataset like the Stanford Dogs dataset to achieve near state-of-the-art performance on fine-grained dog breed classification.

2. Face attribute classification: The model can be used to compress and speed up large face attribute classifiers. By distilling knowledge from a larger model into MobileNet, the resulting classifier can maintain similar mean average precision while requiring only 1% of the computational cost.

3. Image classification on mobile devices: MobileNets are designed to be efficient and run well on resource-constrained devices. They can be used with TensorFlow Lite for classification tasks on mobile devices.

To fine-tune or use the model, it can be built and trained using TensorFlow. Here's an example code snippet:

```
$ bazel build -c opt --config=cuda mobilenet_v1_{eval,train}
```

[More Information Needed] on detailed fine-tuning code and plugging the model into a larger ecosystem or app.

In summary, google/mobilenet_v1_0.75_192 is a versatile model that can be adapted for various computer vision tasks, especially those requiring efficiency on mobile and embedded devices. Its small size and low latency make it a good choice for integrating into applications with limited computational resources.

For more details or to discuss integrating this model into a specific project, feel free to reach out to me as the project organizer and model card maintainer.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the MobileNet V1 model (google/mobilenet_v1_0.75_192) could be misused and what users should avoid doing:

- The model should not be used for critical applications like medical diagnosis or autonomous driving without extensive further testing and validation, as the references do not provide sufficient information about performance and limitations in such high-stakes domains. [More Information Needed]

- Users should not assume the model is free of bias or will perform equally well across all demographics. The references do not discuss fairness evaluations or debiasing techniques. Thorough testing for bias is advised before deploying the model. [More Information Needed]

- The model appears to be trained primarily for classification tasks like ImageNet. Users should be cautious about off-label uses like facial recognition, as the reference to a FaceNet model does not provide enough details to assess suitability or performance of this specific MobileNet model for those applications. Facial recognition use cases also raise significant privacy concerns that would require careful consideration.

- As a compressed, mobile-optimized model, MobileNet may be more vulnerable to adversarial attacks compared to larger models. Users should not rely on it for sensitive security applications without thorough robustness testing. [More Information Needed]

In general, the provided references focus mainly on MobileNet's architecture and benchmarks, without substantive discussion of responsible usage limitations. More information would be needed from the model developers to fully assess foreseeable misuse potential and establish clear usage guidelines. Users should be cautious about high-stakes applications of the model without further validation.

### Bias, Risks, and Limitations

Based on the provided references about the MobileNet model architecture, here are some potential issues and limitations:

Technical limitations:
- The model uses a width multiplier α to reduce computation and model size, which trades off accuracy for efficiency. Smaller α values result in lower accuracy.
- Similarly, the resolution multiplier ρ reduces input resolution to decrease computational cost, again impacting accuracy. 
- To achieve the desired size/latency, the reduced MobileNet models need to be retrained from scratch, which requires additional time and resources.

Potential misunderstandings:
- Users may assume the compressed MobileNet models maintain the same accuracy as the original, when in fact the multipliers trade off accuracy for improved efficiency.

[More Information Needed] on other specific technical limitations or failure modes of the model.

Foreseeable harms and sociotechnical issues:
- [More Information Needed] The references do not provide enough context on the intended use cases and deployment environments to assess potential harms or sociotechnical issues that could arise from using the model. More information is needed on factors like what the model will be used for, who will be using it, and how it will impact end users and society.

To summarize, the key known limitations are the accuracy trade-offs from using the width and resolution multipliers to compress the model. However, more context is needed to fully understand other potential technical issues and sociotechnical implications. I would recommend gathering more information on the intended use and deployment of the model to round out this section of the model card.

### Recommendations

Based on the provided references about the MobileNet architecture and the specific google/mobilenet_v1_0.75_192 model, here are some recommendations regarding foreseeable issues:

1. Accuracy-efficiency trade-off: The model uses a width multiplier α of 0.75, which reduces the model size and computational cost but also leads to some accuracy drop compared to the baseline MobileNet (α=1). Users should be aware of this trade-off and evaluate if the reduced accuracy is acceptable for their specific use case.

2. Limited application scope: The references mainly discuss MobileNet's performance on ImageNet classification and a few other tasks like object detection and face attributes. However, there is limited information on the model's generalizability to a wider range of applications. More extensive testing may be needed to assess the model's effectiveness in diverse domains.

3. Potential biases: [More Information Needed] The references do not provide insights into any biases the model might have inherited from its training data. A thorough analysis of the training dataset and resulting biases is recommended to ensure responsible deployment.

4. Robustness and security: [More Information Needed] The provided information does not cover the model's robustness against adversarial attacks or its security vulnerabilities. Further investigation into these aspects is crucial, especially if the model is applied in sensitive or high-stakes scenarios.

5. Environmental impact: [More Information Needed] While the model's efficiency is discussed in terms of computational cost and latency, there is no mention of its environmental footprint. An assessment of the model's energy consumption and carbon emissions during training and inference would be valuable for users concerned about sustainability.

In summary, while google/mobilenet_v1_0.75_192 offers a good balance between efficiency and accuracy, more information is needed to comprehensively address potential issues related to biases, robustness, security, and environmental impact. Users should carefully consider these aspects and gather additional insights before deploying the model in their specific contexts.

## Training Details

### Training Data

The model google/mobilenet_v1_0.75_192 was trained on the ImageNet dataset, which consists of over 14 million images across 1000 classes. The training data was pre-processed with less regularization and data augmentation techniques compared to larger models, as mentioned in Reference 1. [More Information Needed] for details on any additional data filtering or pre-processing steps.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the google/mobilenet_v1_0.75_192 model:

Resizing:
The model uses a resolution multiplier ρ to reduce the input image resolution and subsequently the internal representation of every layer by the same multiplier. For the google/mobilenet_v1_0.75_192 model, the input resolution is implicitly set to 192x192 pixels by setting ρ = 0.75 (reference 6, 7).

Tokenization:
[More Information Needed]

Other preprocessing details:
The model uses less regularization and data augmentation techniques compared to training large models, as small models have less trouble with overfitting. Specifically, side heads, label smoothing, and the amount of image distortions are reduced by limiting the size of small crops used in training (reference 1).

[More Information Needed] for any additional preprocessing steps.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters used for the google/mobilenet_v1_0.75_192 model:

Optimizer: RMSprop with asynchronous gradient descent (Reference 1)

Regularization and data augmentation:
- Less regularization and data augmentation compared to training large models, as small models have less trouble with overfitting (Reference 1)
- No side heads or label smoothing used (Reference 1)
- Reduced image distortions by limiting the size of small crops (Reference 1)
- Very little or no weight decay (l2 regularization) on the depthwise filters due to their small number of parameters (Reference 4)

Model architecture:
- Width multiplier α = 0.75 (Reference 2, Figure 4, Figure 5)
- Input resolution of 192x192 (Reference 2, Figure 4, Figure 5)
- Depthwise separable convolutions used in all layers except the first layer, which is a full convolution (Reference 3)
- Batchnorm and ReLU nonlinearity applied after each layer, except for the final fully connected layer (Reference 3)

[More Information Needed] for specific values of hyperparameters such as learning rate, batch size, number of epochs, and any additional regularization techniques used.

#### Speeds, Sizes, Times

Based on the provided references, here is the information I could find about the model google/mobilenet_v1_0.75_192:

Throughput: [More Information Needed]

Start or end time: [More Information Needed] 

Checkpoint size: [More Information Needed]

The references mention various MobileNet models with different width multipliers (α) and resolutions, but do not provide specific details about throughput, start/end times, or checkpoint sizes for the mobilenet_v1_0.75_192 model.

The closest relevant information is in reference 7, which lists the mobilenet_v1_0.75_224 model as having 2.59 million parameters and 68.4% top-1 accuracy on ImageNet. The 192 resolution model would likely have slightly fewer parameters and slightly lower accuracy, but exact figures are not provided.

To summarize, more specific information would be needed to report the requested metrics for the mobilenet_v1_0.75_192 model based on the given references.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the MobileNet model google/mobilenet_v1_0.75_192 was evaluated on the following benchmarks and datasets:

1. Stanford Dogs dataset for fine-grained dog recognition. MobileNet achieved near state-of-the-art results compared to a larger model.

2. [More Information Needed] on the exact dataset used for the PlaNet re-training with MobileNet. 

3. ImageNet classification for evaluating resource and accuracy tradeoffs. MobileNets showed strong performance compared to other popular models.

4. COCO dataset for object detection, comparing MobileNet to VGG and Inception V2 under the Faster-RCNN and SSD frameworks.

[More Information Needed] on the specific results and metrics for the model on these datasets. The references discuss MobileNets in general but do not provide the exact evaluation details for the specific google/mobilenet_v1_0.75_192 model.

#### Factors

Based on the provided references, here are some foreseeable characteristics that may influence how the MobileNet V1 0.75 192 model behaves:

Domain and Context:
- The model was trained on millions of geo-tagged photos for the task of determining where on Earth a photo was taken (PlaNet) [4]
- It was also used for fine-grained dog breed classification on the Stanford Dogs dataset [2]
- The model was applied to object detection, fine-grain classification, face attributes, and large-scale geo-localization [9]

Population Subgroups:
[More Information Needed] - The references do not provide details on performance across different population subgroups.

Other Factors:
- MobileNet uses depthwise convolutions and reduces network width rather than depth to achieve a smaller, more efficient architecture [3] 
- The model's performance is resilient to aggressive shrinking, maintaining good accuracy while significantly reducing computation and size [6, 10]
- Less regularization and data augmentation was used during training since small models are less prone to overfitting [11]

To fully evaluate potential performance disparities, more disaggregated evaluation across relevant factors and subgroups would be needed. The current references lack sufficient information to draw conclusions there.

#### Metrics

Based on the provided references, the key metrics used to evaluate the tradeoffs for the google/mobilenet_v1_0.75_192 model are:

1. Accuracy: The model's performance on tasks like image classification and fine-grained recognition. The references mention accuracy dropping off smoothly as the model is made smaller using the width multiplier α and resolution multiplier ρ.

2. Computation cost: Measured by the number of Multiply-Accumulates (MACs), which counts the fused Multiplication and Addition operations. The computation cost scales with ρ^2 when using the resolution multiplier.

3. Model size: Proportional to the number of parameters in the network. The model size scales roughly with α^2 when using the width multiplier.

4. Latency: The references mention that latency scales with the number of MACs, so reducing computation cost also reduces latency.

The key tradeoff is between model accuracy and resource efficiency (computation cost, size, latency). The references show that the MobileNet architecture can be effectively shrunk using width and resolution multipliers to fit different budgets while maintaining reasonable accuracy. Specific results for the google/mobilenet_v1_0.75_192 variant are [More Information Needed].

### Results

Based on the provided references, here is the evaluation information I could find for the google/mobilenet_v1_0.75_192 model:

Model Performance:
The MobileNet model with a width multiplier of 0.75 and input resolution of 192x192 provides a good tradeoff between accuracy, computation, and model size. Accuracy drops off smoothly as the model is made smaller through the width multiplier (Reference 8).

Computational Cost: 
[More Information Needed] The exact computation cost in terms of mult-adds is not provided for this specific model configuration.

Model Size:
[More Information Needed] The exact model size in terms of number of parameters is not provided for this specific model configuration.

Evaluation Datasets:
The model was evaluated on the following datasets:
- ImageNet classification (Reference 7) 
- Stanford Dogs fine-grained classification (Reference 2)
- PlaNet geolocation (Reference 1, 4)

Evaluation Metrics:
The following metrics were used to evaluate the model:
- Accuracy for ImageNet and Stanford Dogs classification (Reference 2, 7, 8)
- Mean average precision (mAP) for face attribute classification (Reference 6)

To summarize, the google/mobilenet_v1_0.75_192 model provides a good balance of accuracy and efficiency, performing well on tasks like ImageNet classification, fine-grained classification, and face attribute detection while having a reduced size and computational cost compared to larger models. However, more specific details on the computation and size for this particular model configuration would be needed to include in a complete model card.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model google/mobilenet_v1_0.75_192:

The MobileNet_v1_0.75_192 model is a compact version of the MobileNet architecture with a width multiplier of 0.75 and an input resolution of 192x192 pixels. It has fewer parameters and mult-adds compared to the full MobileNet model.

[More Information Needed] on the specific performance metrics like accuracy, number of parameters, and mult-adds for the MobileNet_v1_0.75_192 model, as the provided references do not contain this information directly.

The references mention that MobileNet models in general can achieve performance close to state-of-the-art results on tasks like fine-grained recognition (Stanford Dogs dataset) and geo-localization (PlaNet), while being much more compact than larger models like Inception V3.

To provide a complete evaluation summary, more specific information on the performance, size, and computational cost of the MobileNet_v1_0.75_192 model would be needed.

## Model Examination

Based on the provided references about the MobileNet architecture and the specific model google/mobilenet_v1_0.75_192, here is the information I can provide regarding explainability and interpretability:

The MobileNet architecture is designed to be small, low-latency, and low-power to meet resource constraints for various use cases, especially on mobile devices. The key architectural choices that contribute to its efficiency are:

1. Depthwise separable convolutions: This breaks the interaction between the number of output channels and the kernel size, reducing computational cost.

2. Width multiplier (α): This parameter allows uniformly thinning the network at each layer, reducing the number of input and output channels. For the model google/mobilenet_v1_0.75_192, α is set to 0.75, meaning the number of channels is reduced to 75% of the original MobileNet architecture.

3. Resolution multiplier: MobileNets can be trained with reduced input resolutions, trading off accuracy for computational cost. The model google/mobilenet_v1_0.75_192 has an input resolution of 192x192 pixels.

These architectural choices make the model more interpretable by reducing the complexity of the network while maintaining acceptable accuracy. However, the provided references do not go into detail about specific explainability techniques or interpretability analysis for this particular model.

[More Information Needed] on how the depthwise separable convolutions, width multiplier, and resolution multiplier specifically impact the interpretability of the learned features and decision-making process of the google/mobilenet_v1_0.75_192 model.

[More Information Needed] on any visualization techniques, such as saliency maps or activation maximization, that could be applied to this model to gain insights into its learned representations.

In summary, while the MobileNet architecture has inherent properties that make it more interpretable than larger, more complex models, further analysis and experiments would be needed to provide a comprehensive understanding of the explainability and interpretability of the google/mobilenet_v1_0.75_192 model specifically.

## Environmental Impact

- **Hardware Type:** Based on the provided references, there is no specific information about the hardware type that the model google/mobilenet_v1_0.75_192 was trained on. The references mention that MobileNet models in general were trained using TensorFlow with asynchronous gradient descent, but do not specify the hardware used for this particular model variant.

[More Information Needed]
- **Software Type:** Based on the references provided, the MobileNet models were trained using TensorFlow, as mentioned in reference 1:

"MobileNet models were trained in TensorFlow [1] using RMSprop [33] with asynchronous gradient descent similar to Inception V3 [31]."

Therefore, the model google/mobilenet_v1_0.75_192 was likely trained using TensorFlow.
- **Hours used:** [More Information Needed]

The provided references do not contain specific information about the training time for the google/mobilenet_v1_0.75_192 model. The references discuss various aspects of the MobileNet architecture and training approach, but do not give details on how long this particular model variant took to train.
- **Cloud Provider:** [More Information Needed]

The provided references do not specify the cloud provider used for training the MobileNet model. More information would be needed from the model development process to determine the specific cloud provider, if any, that was utilized.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any specific information about the amount of carbon emitted during the training of the MobileNet V1 0.75 192 model. The references discuss the model architecture, training techniques, and performance on various datasets, but do not mention the carbon footprint of the training process.

To accurately report the carbon emissions, more details would be needed, such as:
- The hardware used for training (GPUs, CPUs, etc.)
- The energy consumption of the training hardware
- The duration of the training process 
- The energy sources powering the training hardware (renewable vs. fossil fuels)
- Any carbon offset or renewable energy credits purchased

Without this additional information, it is not possible to provide a reliable estimate of the carbon emissions from training this specific MobileNet model variant based solely on the given references.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The MobileNet architecture is based on depthwise separable convolutions, which factorize a standard convolution into a depthwise convolution and a 1x1 pointwise convolution. The model is built using 3x3 depthwise separable convolutions, except for the first layer which uses a full convolution. Batchnorm and ReLU activations are applied after each layer, except the final fully connected layer which feeds into a softmax layer.

The model google/mobilenet_v1_0.75_192 uses a width multiplier α of 0.75 to uniformly thin the network at each layer, reducing the number of input and output channels. This results in a smaller and faster model compared to the base MobileNet architecture.

Objective:
The objective of the MobileNet model, specifically google/mobilenet_v1_0.75_192, is to provide an efficient and lightweight deep neural network for mobile and embedded vision applications. By using depthwise separable convolutions and the width multiplier, the model aims to achieve a good balance between latency and accuracy, allowing it to be easily matched to the design requirements of resource-constrained devices.

[More Information Needed] on the specific performance metrics, training dataset, and downstream tasks for the google/mobilenet_v1_0.75_192 model.

### Compute Infrastructure

Based on the provided references, here is the information I can provide about the compute infrastructure used for the google/mobilenet_v1_0.75_192 model:

The MobileNet models were trained using TensorFlow with asynchronous gradient descent, similar to the training of the Inception V3 model. RMSprop optimizer was used for training.

Some key points about the training infrastructure and methodology:

- Less regularization and data augmentation techniques were used compared to training larger models, since smaller models like MobileNet have less trouble with overfitting. 
- Side heads, label smoothing were not used. The amount of image distortions was reduced by limiting the size of small crops used in training.
- Most MobileNet layers are followed by batchnorm and ReLU, except the final fully connected layer which feeds into a softmax.
- Very little or no weight decay (L2 regularization) was used on the depthwise filters since they have very few parameters.

[More Information Needed] on the specific compute hardware, distributed training setup, training times, hyperparameters, etc. used for this particular MobileNet_v1_0.75_192 variant, as those details are not provided in the given references.

## Citation

```
@misc{andrew-mobilenets,
    author = {Andrew G Howard and
              Menglong Zhu and
              Bo Chen and
              Dmitry Kalenichenko and
              Weijun Wang and
              Tobias Weyand and
              Marco Andreetto and
              Hartwig Adam},
    title  = {MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications},
    url    = {https://arxiv.org/pdf/1704.04861.pdf}
}
```

