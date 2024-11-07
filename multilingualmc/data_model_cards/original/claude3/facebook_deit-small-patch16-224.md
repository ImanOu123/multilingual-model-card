# Model Card for facebook/deit-small-patch16-224

The facebook/deit-small-patch16-224 model is a vision transformer architecture for image classification that is trained on ImageNet-1k only, without using external data. It achieves competitive performance compared to convolutional networks while having a lower memory footprint, and can be trained efficiently on a single 8-GPU node in 2-3 days.

## Model Details

### Model Description

Model Architecture:
- Vision transformer (ViT) architecture that processes input images as a sequence of patches
- Fixed-size input RGB image is decomposed into N patches of 16x16 pixels each (N = 14x14 for 224x224 input)
- Each patch is linearly projected to dimension 768
- Transformer block consists of Multi-head Self-Attention (MSA) and Feed-Forward Network (FFN) layers
- Class token is appended to patch tokens and used to predict the output class

Training Procedures:
- Trained on ImageNet-1k dataset
- Employs strong data augmentation techniques like RandAugment and Random Erasing
- Distillation from a teacher model (like RegNetY) to improve performance
- [More Information Needed] on exact training hyperparameters and schedule

Parameters:
- [More Information Needed] on exact parameter count
- Based on ViT-S (small) architecture

Important Disclaimers:
- Requires larger training data compared to CNNs due to lack of inductive biases
- [More Information Needed] on potential biases and limitations

For more details and code, please refer to:
https://github.com/facebookresearch/deit

- **Developed by:** Hugo Touvron; Matthieu Cord; Matthijs Douze; Francisco Massa; Alexandre Sablayrolles; Hervé Jégou; Facebook Ai
- **Funded by:** Based on the provided references, the model facebook/deit-small-patch16-224 appears to have been developed by researchers at Facebook AI, including Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. The project was likely funded by Facebook (now Meta).
- **Shared by:** Based on the references provided, the main contributors who made the model facebook/deit-small-patch16-224 available online as a GitHub repo are:

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou from Facebook AI.

The model and code are available in the GitHub repository: https://github.com/facebookresearch/deit

The references also thank Ross Wightman for sharing his ViT code and bootstrapping training method with the community, which likely helped in the development of the DeiT model. However, more information is needed to determine his exact role in making the specific facebook/deit-small-patch16-224 model available.
- **Model type:** The facebook/deit-small-patch16-224 model is a vision transformer trained on ImageNet using a data-efficient training strategy called DeiT (Data-efficient image Transformers), which employs hard distillation from a teacher model and introduces a distillation token to enable learning from the teacher's output.
- **Language(s):** The model facebook/deit-small-patch16-224 processes input images, not natural human language. It decomposes fixed-size RGB images into patches which are then projected and fed into a transformer architecture adapted from Natural Language Processing for image classification tasks.
- **License:** The model facebook/deit-small-patch16-224 is released under the Apache 2.0 license, as stated in the references:

"This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file."

The link to the license file is not directly provided in the given references, so [More Information Needed] for the specific link.
- **Finetuned from model:** Based on the provided references, there is no explicit mention of the model facebook/deit-small-patch16-224 being fine-tuned from another model. The references discuss the DeiT (Data-efficient image Transformers) architecture in general, and mention variants like DeiT-S and DeiT-Ti, but do not specify the exact model name or any fine-tuning details for facebook/deit-small-patch16-224.

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/datasets.py
- **Paper:** https://arxiv.org/pdf/2012.12877.pdf
- **Demo:** [More Information Needed]

The provided references do not contain a direct link to a demo of the facebook/deit-small-patch16-224 model. The references mention an open-source implementation available on GitHub at https://github.com/facebookresearch/deit, but this is not a demo link specifically for the deit-small-patch16-224 variant. More information would be needed to provide a demo link for this specific model in the model card.
## Uses

### Direct Use

The model facebook/deit-small-patch16-224 can be used for image classification tasks without requiring fine-tuning, post-processing or plugging into a pipeline. It has been pre-trained on the ImageNet dataset and can make predictions on new images at a resolution of 224x224 pixels.

To use the model, you would first need to load the pre-trained weights and set up the model architecture. Then you can pass an appropriately pre-processed image through the model to get the predicted class probabilities.

[More Information Needed] for providing a specific code snippet, as the references do not contain a directly applicable code block. The high-level steps would be:
1. Load the pre-trained model weights 
2. Preprocess the input image to 224x224 resolution and normalize
3. Pass the image through the model
4. Get the predicted class probabilities

The model achieves competitive performance on ImageNet and other transfer learning datasets compared to convolutional architectures, while having a lower memory footprint for a given accuracy. This makes it an attractive choice for image classification tasks, even without dataset-specific fine-tuning.

### Downstream Use

The DeiT (Data-efficient Image Transformers) model facebook/deit-small-patch16-224 can be fine-tuned for various vision tasks or integrated into larger applications. Here are a few ways it can be used:

1. Fine-tuning for transfer learning: DeiT performs well on transfer learning tasks by fine-tuning on different datasets, as shown in the reference (point 2). It achieves results on par with state-of-the-art convolutional architectures.

2. Training from scratch on small datasets: DeiT can be trained from scratch on small datasets like CIFAR-10 without ImageNet pre-training (points 3 and 4). This demonstrates its ability to learn reasonable representations even with limited data.

3. Plugging into larger ecosystems: DeiT can be integrated into larger ecosystems or applications that require vision capabilities. Its lower memory footprint compared to convolutional networks (point 9) makes it an attractive choice for resource-constrained environments.

[More Information Needed] for a specific code snippet demonstrating fine-tuning or integration into a larger ecosystem.

Overall, DeiT's strong performance on transfer learning tasks, ability to learn from small datasets, and lower memory footprint make it a versatile model for various vision applications. It can be fine-tuned for specific tasks or integrated into larger systems to provide vision capabilities.

For more information and an open-source implementation, refer to the GitHub repository: https://github.com/facebookresearch/deit (point 9).

### Out-of-Scope Use

Based on the provided references, there is not enough information to comprehensively address how the model facebook/deit-small-patch16-224 may be misused or what users ought not do with it. The references focus mainly on technical details of the model architecture, training data, and performance comparisons to other models. They do not appear to discuss potential misuse cases or provide guidance on responsible usage.

To properly answer the question, more information would be needed, such as:
[More Information Needed]
- Details on the intended use cases and scope of the model 
- Analysis of potential harmful applications or unintended consequences
- Ethical considerations around the training data and potential biases
- Guidelines from the developers on what constitutes misuse
- Terms of use or other policies that users must agree to

Without additional context from the model developers on these aspects, it is difficult to specify what would be considered misuse or improper application of this model. More details and guidance from the team that created facebook/deit-small-patch16-224 would be required to fully address the question from the perspective of a sociotechnic role.

### Bias, Risks, and Limitations

Based on the provided references about the DeiT (Data-efficient Image Transformers) model facebook/deit-small-patch16-224, here are some potential issues and limitations:

Technical limitations:
- The model's performance is still slightly below state-of-the-art convolutional networks like EfficientNet when trained only on ImageNet data. Further improvements may be needed to fully close the gap.
- [More Information Needed] on the model's robustness to adversarial attacks, distribution shifts, or out-of-distribution data.

Sociotechnical issues:
- As an image classification model trained on ImageNet, it may perpetuate biases present in the ImageNet dataset related to gender, race, culture, etc. [More Information Needed] on analysis of biases.
- Misuse or over-reliance on the model for high-stakes decisions without human oversight could be problematic, especially if the model's failure modes are not well understood.
- [More Information Needed] on the potential for misuse in surveillance, profiling, or invasion of privacy if the model is applied beyond its intended image classification use case.

To responsibly publish this model, more information and analysis is needed in several key areas to fully characterize foreseeable harms and limitations. The provided references focus primarily on technical details and do not substantially address broader sociotechnical implications. I would recommend the team conduct and document further analysis before publication.

### Recommendations

Based on the provided references about the DeiT (Data-efficient Image Transformers) model facebook/deit-small-patch16-224, here are some recommendations regarding foreseeable issues:

1. Data efficiency and generalization: While DeiT models have shown impressive performance on ImageNet classification, closing the gap with convolutional networks, it's important to further evaluate their generalization capabilities on other datasets through transfer learning tasks. The references mention evaluating transfer learning, but more extensive testing across diverse datasets would help assess the model's robustness and identify potential limitations.

2. Fairness and bias: [More Information Needed] The references do not provide information about analyzing the model for potential biases or fairness concerns. It's crucial to evaluate the model's performance across different demographics, assess for any disparate impacts, and take steps to mitigate any identified biases.

3. Transparency and explainability: [More Information Needed] The references focus on the model's architecture and performance but do not discuss interpretability or explainability aspects. Providing insights into how the model makes predictions, visualizing attention maps, or employing explainable AI techniques can enhance transparency and build trust with users.

4. Ethical considerations: [More Information Needed] The references do not explicitly address ethical considerations surrounding the model's development and deployment. It's important to consider potential misuse cases, ensure responsible use of the technology, and establish guidelines for appropriate application domains.

5. Environmental impact: The references mention that DeiT models have a lower memory footprint compared to convolutional networks for a given accuracy. This is a positive aspect in terms of computational efficiency and environmental sustainability. However, it's still important to assess the overall energy consumption and carbon footprint associated with training and deploying these models at scale.

6. Continuous monitoring and updates: As the model is deployed and used in real-world scenarios, it's crucial to establish mechanisms for continuous monitoring, gathering user feedback, and addressing any emerging issues or concerns. Regular updates and refinements to the model based on real-world performance and user feedback can help ensure its long-term effectiveness and alignment with societal needs.

These recommendations highlight areas where more information and analysis may be needed to comprehensively assess the societal implications of the DeiT model. Collaboration among the development team, including sociotechnical experts, can help address these considerations and ensure the model is developed and deployed responsibly.

## Training Details

### Training Data

The model facebook/deit-small-patch16-224 was trained on the ImageNet-1k dataset, which consists of 1.2 million training images with 1000 object categories. The training data was preprocessed using strong data augmentation techniques such as RandAugment and random erasing to improve the model's performance and data efficiency.

[More Information Needed] for links to documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Here are the details about preprocessing for the data of the model facebook/deit-small-patch16-224, based on the provided references:

Tokenization:
The fixed-size input RGB image is decomposed into a batch of N patches of a fixed size of 16 × 16 pixels (N = 14 × 14). Each patch is projected with a linear layer that conserves its overall dimension 3 × 16 × 16 = 768.

Resizing:
The model is initially trained at resolution 224×224, and after fine-tuning at resolution 384×384. When increasing the resolution of an input image, the patch size is kept the same, therefore the number N of input patches changes.

Data Augmentation:
The model relies on extensive data augmentation to train with datasets of the same size as models that integrate more priors. The following data augmentation techniques are used:
- Rand-Augment 
- Random erasing
[More Information Needed] on the specific parameters used for these augmentations.

Other Preprocessing Details:
- [More Information Needed]

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters used for the model facebook/deit-small-patch16-224:

- Initialization of weights: Truncated normal distribution (Reference 2)
- Default training resolution: 224x224 (Reference 3, 10)
- Fine-tuning resolution: 384x384 (Reference 3, 10)
- Data augmentation methods used: Rand-Augment, random erasing (Reference 4, 6)
- Distillation parameters: τ = 3.0, λ = 0.1 (Reference 2)
- Longer training with distillation improves performance, e.g. 300 epochs (Reference 7)

[More Information Needed] for the following hyperparameters:
- Exact number of training epochs
- Optimizer used and its settings (learning rate, weight decay, etc.)
- Batch size
- Exact Rand-Augment and random erasing settings

The references do not provide code blocks with the exact hyperparameter settings. More details would be needed to fully specify the training setup for reproducibility.

#### Speeds, Sizes, Times

Here are the details I could find about the model facebook/deit-small-patch16-224 based on the provided references:

Throughput: 
The throughput of DeiT models is measured as the number of images processed per second on a V100 GPU, as shown in Figure 1. The exact throughput for the DeiT-S (small) model is not explicitly stated.
[More Information Needed]

Training Time:
DeiT-S and DeiT-Ti are trained in less than 3 days on 4 GPUs.

Checkpoint Sizes:
[More Information Needed]

Additional Details:
- DeiT is an image transformer model that is on par with convolutional networks in accuracy while having a lower memory footprint.
- DeiT-S is trained on ImageNet-1k only, without relying on external data. 
- DeiT models use a transformer-specific distillation approach referred to with the ⚗ symbol.
- The model processes input images as a sequence of patches of size 16x16 pixels, projected to an embedding dimension of 768.

To summarize, the DeiT-S model achieves strong accuracy on ImageNet while being efficient to train in less than 3 days. However, more specific information would be needed to report the exact throughput and checkpoint size.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the DeiT (Data-efficient Image Transformers) model facebook/deit-small-patch16-224 was evaluated on the following benchmarks and datasets:

1. ImageNet-1k (ILSVRC2012) dataset for image classification. The model achieves competitive performance compared to state-of-the-art convolutional architectures.

2. Transfer learning tasks by fine-tuning on datasets mentioned in Table 6 of the referenced paper. The specific datasets are not provided in the given references, so [More Information Needed] for the exact datasets used for transfer learning evaluation.

3. ImageNet V2 and ImageNet Real datasets, which have test sets distinct from the ImageNet validation set, to reduce overfitting. The model outperforms state-of-the-art models on the trade-off between accuracy and inference time on GPU.

The references also mention comparisons with EfficientNet and ViT models, highlighting the competitive performance of DeiT in terms of accuracy and throughput trade-off when trained on ImageNet-1k only.

#### Factors

Based on the provided references about the DeiT (Data-efficient Image Transformers) model facebook/deit-small-patch16-224, here are some key characteristics that may influence its behavior:

Domain and Context:
- The model was primarily trained and evaluated on the ImageNet dataset for image classification tasks. Its performance on other domains or tasks beyond image classification would require further evaluation through transfer learning (Reference 1).

Population Subgroups:
[More Information Needed] 
The provided references do not contain information on how the model's performance varies across different population subgroups or image categories. Disaggregated evaluation would be needed to uncover any performance disparities.

Other Factors:
- The model's performance is influenced by training resolution. It was trained at 224x224 resolution and fine-tuned at 384x384 (Reference 10). Behavior at other resolutions would need testing.

- Distillation from a teacher model (e.g. RegNetY) can boost DeiT's performance (Reference 6). The choice of teacher model and distillation parameters (τ, λ) affect the student DeiT model (Reference 5).

- Weight initialization using truncated normal distribution was important for model convergence (Reference 5).

- Hyper-parameters like learning rate, optimizer, etc. listed in Table 9 (not provided) impact model training (Reference 3). Their values matter for replicating performance.

In summary, key factors like the training dataset, resolution, distillation setup, initialization and hyper-parameters influence the model's behavior. More disaggregated evaluation across different data subgroups and domains would help further characterize model performance. The provided references lack some specifics to fully answer the question.

#### Metrics

Based on the references provided, the key metrics used to evaluate the DeiT (Data-efficient Image Transformers) models, including facebook/deit-small-patch16-224, are:

1. Top-1 classification accuracy on ImageNet dataset
2. Throughput (images processed per second) on a single 16GB V100 GPU
3. Transfer learning performance on downstream tasks like CIFAR-10, CIFAR-100, Oxford-102 flowers, Stanford Cars, and iNaturalist-18/19

The evaluation focuses on the trade-off between accuracy and throughput, comparing DeiT models with state-of-the-art convolutional networks (ConvNets) like EfficientNet.

[More Information Needed] on the specific tradeoffs between different errors for the facebook/deit-small-patch16-224 model.

### Results

Based on the provided references, here are the evaluation results I could find for the model facebook/deit-small-patch16-224:

Model Architecture:
- The DeiT-S (small) model has a fixed number of heads, keeping d (the head dimension) fixed at 64. The exact architecture details are: [More Information Needed]

Performance on ImageNet:
- DeiT models perform well on ImageNet and almost close the gap between vision transformers and convolutional networks when trained on ImageNet only. 
- DeiT models outperform previous ViT models trained on ImageNet1k only by a large margin (+6.3% top-1 accuracy in a comparable setting).
- Exact ImageNet accuracy numbers for DeiT-S: [More Information Needed]

Transfer Learning Performance: 
- DeiT models evaluated on transfer learning tasks by fine-tuning are on par with competitive convolutional models.
- Exact transfer learning dataset results for DeiT-S: [More Information Needed]

Inference Speed:
- DeiT models aim to optimize the trade-off between accuracy and inference time on GPU.
- Exact throughput numbers (images/sec) for DeiT-S: [More Information Needed]

Impact of Resolution:
- DeiT models are trained at 224x224 resolution by default and fine-tuned at 384x384.
- Influence of other resolutions on DeiT-S performance: [More Information Needed]

In summary, while the references highlight strong overall results for the DeiT architecture on ImageNet accuracy, transfer learning, and inference speed, I would need more specific information to report the exact metrics for the DeiT-S model variant.

#### Summary

Based on the provided references about the DeiT (Data-efficient Image Transformers) model, here is a summary of the evaluation results for the facebook/deit-small-patch16-224 model:

- DeiT models trained only on ImageNet-1k achieve competitive performance compared to state-of-the-art convolutional networks like EfficientNet, closing the gap between vision transformers and ConvNets. 

- DeiT models outperform previous ViT models trained only on ImageNet-1k by a significant margin (+6.3% top-1 accuracy).

- When benefiting from knowledge distillation from a RegNetY teacher model, DeiT (referred to as DeiT⚗) outperforms EfficientNet.

- On transfer learning tasks, DeiT models fine-tuned on other datasets perform on par with competitive ConvNet models, demonstrating good generalization.

- DeiT models provide a favorable trade-off between inference throughput (images processed per second) and top-1 accuracy on ImageNet compared to other models.

- The DeiT models are trained at 224x224 resolution by default and can be fine-tuned at higher resolutions like 384x384 to further boost performance.

[More Information Needed] on the exact performance metrics of the facebook/deit-small-patch16-224 model variant, as the provided references discuss DeiT models in general without specifying results for this particular model size.

## Model Examination

Explainability/Interpretability:

The DeiT (Data-efficient image Transformers) model, including the facebook/deit-small-patch16-224 variant, is based on the Vision Transformer (ViT) architecture introduced by Dosovitskiy et al. It processes images as a sequence of fixed-size patches, projecting each patch with a linear layer while maintaining its overall dimension. The transformer block used in DeiT is invariant to the order of the patch embeddings.

[More Information Needed] on specific explainability or interpretability techniques applied to the DeiT model. Further research and experiments would be required to provide insights into how the model makes its predictions and to visualize the learned representations.

The DeiT model has shown competitive performance compared to convolutional networks (ConvNets) like EfficientNet when trained on ImageNet1k alone. With distillation from a weaker teacher model (RegNetY), DeiT⚗ outperforms EfficientNet and improves upon the original ViT-B model by 1% top-1 accuracy when pre-trained on [More Information Needed].

As the development of explainable and interpretable AI progresses, we expect more techniques to be applied to transformer-based vision models like DeiT, providing a better understanding of their decision-making process and learned representations.

## Environmental Impact

- **Hardware Type:** Based on the provided references, the model facebook/deit-small-patch16-224 (DeiT-S) is trained on GPUs. Specifically, it is mentioned that "DeiT-S and DeiT-Ti are trained in less than 3 days on 4 GPU." (Reference 8)

However, the exact GPU model is not specified in the given references. To provide a more complete answer, additional information about the specific GPU model used for training DeiT-S would be needed.
- **Software Type:** The model facebook/deit-small-patch16-224 is trained using PyTorch and the timm library, as mentioned in Reference 2:

"We build upon PyTorch [39] and the timm library [55]."
- **Hours used:** According to the references provided, DeiT-S (which likely refers to the facebook/deit-small-patch16-224 model) is trained in less than 3 days on 4 GPUs (Reference 1). No more specific information about the exact training time for this particular model variant is given.

[More Information Needed] on the precise number of hours or days needed to train the facebook/deit-small-patch16-224 model.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the model facebook/deit-small-patch16-224. More information would be needed to determine the cloud provider, such as explicit mention of AWS, Google Cloud, Azure, or other providers in the context of model training.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted when training the model facebook/deit-small-patch16-224. To accurately report the carbon emissions, more details would be needed such as the computing resources used for training, training duration, energy consumption, and carbon intensity of the electricity grid powering the training hardware.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The model architecture is based on the Vision Transformer (ViT) introduced by Dosovitskiy et al. It processes input images as a sequence of patches. The fixed-size input RGB image is decomposed into a batch of N patches of a fixed size of 16 × 16 pixels (N = 14 × 14). Each patch is projected with a linear layer to a dimension of 768. The model uses the standard transformer block architecture with Multi-Head Self-Attention (MSA) followed by a Feed-Forward Network (FFN). A learnable class token is appended to the sequence of patch embeddings before being processed by the transformer layers. The class token is then used to predict the output class through a linear layer.

Model Objective:
The objective of the DeiT (Data-efficient image Transformers) model is to train vision transformers effectively on smaller datasets like ImageNet-1k and achieve performance competitive with convolutional networks, while maintaining the computational efficiency of transformers. The model introduces training strategies to improve generalization and a novel distillation technique using a distillation token to learn from a teacher model. The goal is to demonstrate the potential of transformer-based models for image classification tasks, even in data-constrained scenarios.

[More Information Needed] on specific hyperparameters, training details, and performance metrics for the facebook/deit-small-patch16-224 model variant.

### Compute Infrastructure

Based on the provided references, here is the compute infrastructure information I could find about the model facebook/deit-small-patch16-224:

DeiT-S (which likely refers to the deit-small model) can be trained in less than 3 days on 4 GPUs.

[More Information Needed] on the exact GPU model, CPU, memory, batch size, and other training infrastructure details for the deit-small-patch16-224 model specifically. The references mention some compute details for DeiT models in general, but do not provide the full infrastructure setup for this particular variant.

## Citation

```
@misc{hugo-training,
    author = {Hugo Touvron and
              Matthieu Cord and
              Matthijs Douze and
              Francisco Massa and
              Alexandre Sablayrolles and
              Hervé Jégou and
              Facebook Ai},
    title  = {Training data-efficient image transformers & distillation through attention},
    url    = {https://arxiv.org/pdf/2012.12877.pdf}
}
```

