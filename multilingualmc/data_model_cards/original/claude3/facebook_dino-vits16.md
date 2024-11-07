# Model Card for facebook/dino-vits16

The facebook/dino-vits16 model is a self-supervised Vision Transformer (ViT) trained using the DINO (self-distillation with no labels) method. It achieves strong performance on various computer vision tasks, such as image classification and segmentation, without requiring labeled data during pretraining.

## Model Details

### Model Description

Model Architecture:
- Backbone: Vision Transformer (ViT-S/16)
- Projection head: 3-layer MLP with hidden dimension 2048, followed by l2 normalization, and a weight normalized fully connected layer with K dimensions
- Number of parameters: 21M
- The model is BN-free

Training Procedure:
- Pretrained on ImageNet dataset without labels
- Optimizer: AdamW 
- Batch size: 1024, distributed over 16 GPUs
- Learning rate: Linearly ramped up to 0.0005 * batchsize/256 during first 10 epochs, then decayed with cosine schedule
- Weight decay: Cosine schedule from 0.04 to 0.4
- Temperature τ s: 0.1, τ t: Linear warm-up from 0.04 to 0.07 during first 30 epochs
- Data augmentations: BYOL augmentations (color jittering, Gaussian blur, solarization) and multi-crop with bicubic interpolation
- Trained using two 8-GPU servers for 3 days

Important Parameters:
- Patch size: 16x16 works best for DINO
- Momentum encoder and cross-entropy loss are important for good performance
- No predictor in student network

Disclaimers:
[More Information Needed]

- **Developed by:** Mathilde Caron; Hugo Touvron; Ishan Misra; Hervé Jegou; Julien Mairal; Piotr Bojanowski; Armand Joulin; Facebook Ai Research
- **Funded by:** Based on the provided references, there is no explicit mention of the people or organizations that fund the DINO project with the ViT-S/16 model. The references discuss technical details about applying DINO to the Vision Transformer (ViT) architecture, but do not specify funding sources.

[More Information Needed]

To properly answer this question for the model card, more information would need to be provided about the funding and support for this specific DINO ViT-S/16 model development project.
- **Shared by:** Based on the provided references, the main contributors to the DINO self-supervised learning method and the ViT (Vision Transformer) model are:

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jegou, Julien Mairal, Piotr Bojanowski, and Armand Joulin from Facebook AI Research.

The specific model facebook/dino-vits16 is likely a ViT-Small model pretrained with DINO by the authors at Facebook AI Research. However, more specific information would be needed to confirm the exact contributors who made this particular model variant available online as a GitHub repo.
- **Model type:** The facebook/dino-vits16 model is a self-supervised Vision Transformer (ViT) model trained using the DINO (self-DIstillation with NO labels) method, which can be interpreted as a form of knowledge distillation without using labeled data, and it operates on the image modality.
- **Language(s):** The facebook/dino-vits16 model processes visual features from images and does not use or process natural human language. [More Information Needed]
- **License:** [More Information Needed]

The provided references do not contain any information about the specific license being used for the facebook/dino-vits16 model. More details would be needed from the model creators or documentation to determine the license under which this model is being released.
- **Finetuned from model:** Based on the provided references, there is no direct mention of the model facebook/dino-vits16 being fine-tuned from another model. The references discuss training Vision Transformer (ViT) models from scratch using the DINO self-supervised learning framework, but do not specify a particular base model that facebook/dino-vits16 is fine-tuned from.

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/facebookresearch/dino
- **Paper:** https://arxiv.org/pdf/2104.14294.pdf
- **Demo:** [More Information Needed]

Based on the provided references, there is no information about a demo link for the model facebook/dino-vits16. The references discuss the DINO framework, its similarities to knowledge distillation, and the synergy between DINO and Vision Transformers (ViTs). However, they do not mention a specific demo for the dino-vits16 model. More information would be needed from the model developers or documentation to determine if a demo link is available.
## Uses

### Direct Use

The model facebook/dino-vits16 can be used without fine-tuning, post-processing, or plugging into a pipeline in the following ways:

1. The model can be loaded directly using PyTorch hub:

```python
import torch
vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
```

2. The model outputs a 1536d descriptor for an input image. This descriptor can be used directly for tasks like copy detection, image retrieval, and k-NN classification, without any further training or fine-tuning.

3. For copy detection, the descriptors can be compared using cosine similarity to recognize distorted images.

4. For image retrieval tasks like the revisited Oxford and Paris datasets, the descriptors can be used with a simple k-NN classifier to achieve competitive performance.

5. [More Information Needed] on how to use the model for weakly supervised image segmentation without fine-tuning.

Overall, the model provides general-purpose image descriptors that can be used directly for various downstream tasks without the need for task-specific fine-tuning or complex post-processing steps.

### Downstream Use

The facebook/dino-vits16 model, which is a Vision Transformer (ViT) architecture pretrained with the self-supervised learning method DINO, can be fine-tuned on various downstream tasks to achieve strong performance. As mentioned in the references, the self-supervised pretraining with DINO transfers better to downstream tasks compared to supervised pretraining, especially for ViT architectures.

To use the pretrained facebook/dino-vits16 model for a specific task, you can follow these steps:
1. Load the pretrained model weights.
2. Replace the final classification layer with a new layer suitable for your specific task.
3. Fine-tune the model on your downstream task using task-specific labeled data.

The references suggest that the facebook/dino-vits16 model can be fine-tuned for tasks such as image classification on datasets like ImageNet, Places205, PASCAL VOC, and Oxford-102 flowers. Fine-tuning the model on these tasks has been shown to achieve strong performance.

[More Information Needed] regarding the specific code snippet for fine-tuning the model, as no direct code block is provided in the given references.

When plugged into a larger ecosystem or app, the facebook/dino-vits16 model can serve as a feature extractor for various computer vision tasks. The self-supervised pretraining with DINO allows the model to learn meaningful visual representations that can be leveraged in downstream applications.

Overall, the facebook/dino-vits16 model offers a powerful and efficient solution for various computer vision tasks, thanks to its self-supervised pretraining with DINO and the Vision Transformer architecture. The model can be fine-tuned for specific tasks or used as a feature extractor in larger ecosystems or apps to achieve strong performance while reducing computational requirements compared to convolutional networks of comparable sizes.

### Out-of-Scope Use

Based on the provided references about the DINO self-supervised learning method and ViT (Vision Transformer) models, here are some potential ways the facebook/dino-vits16 model could be misused and what users ought not do:

The model should not be used for surveillance, facial recognition, or tracking individuals without consent, as the visual features it learns could enable identifying and following people in concerning ways. More analysis is needed on privacy risks.

Users should not assume the model's outputs are fully accurate for sensitive applications like medical diagnosis or autonomous vehicle control, as it was trained in a self-supervised way on web images, not specialized datasets. Thorough testing is required before high-stakes deployment.

The model may reflect biases or offensive content present in its web training data. Users should be cautious applying it to diverse populations and contexts without checking for fairness. [More Information Needed] on the specific data used.

In general, the model is a powerful visual representation learner but not a full solution for complex real-world vision tasks. Users should validate it is fit-for-purpose and augment it responsibly, not misuse it as an off-the-shelf system without careful adaptation and human oversight. Transparency on limitations is key.

[More Information Needed] on further technical ways the model architecture and training process could be exploited or misused. Ongoing monitoring and testing by the developers is advised to proactively surface and mitigate risks as the model is adopted.

### Bias, Risks, and Limitations

Based on the provided references about the DINO ViT model, here are some potential issues and limitations:

Technical limitations:
- Performance is highly dependent on patch size. Smaller patches improve performance but reduce throughput significantly (e.g. 5x5 patches have 44 im/s throughput vs 180 im/s for 8x8 patches). Careful tuning is needed to balance accuracy and efficiency.
- The model does not work well without a momentum encoder. More advanced techniques like Sinkhorn-Knopp are needed to prevent collapse without momentum.
- [More Information Needed] on how well the model generalizes to other datasets and tasks beyond ImageNet classification.

Potential misuse and harms:
- [More Information Needed] The references do not discuss what types of images the model was trained on. If the training data contains biases or underrepresentation, the model may exhibit biased or unfair behavior.
- [More Information Needed] No analysis is provided on whether the model encodes personal information from training data that could be extracted.
- As a general visual feature extractor, the model could potentially be used for unintended purposes like surveillance. Careful usage restrictions may be needed.

Sociotechnical considerations:
- Releasing a powerful visual representation model could accelerate progress in computer vision, but may advantage organizations with more computing resources.
- [More Information Needed] The environmental impact of training large vision models is not discussed.
- The model is a step toward developing "BERT-like models for vision" which could have broad societal impact, but more analysis is needed to understand potential consequences.

In summary, key limitations seem to be technical trade-offs, while more information is needed to fully assess fairness, privacy, and broader sociotechnical implications. I would recommend conducting more analysis before a full release, focusing on training data composition, potential for misuse, and environmental and societal impact.

### Recommendations

Based on the provided references about the DINO self-supervised learning framework applied to Vision Transformers (ViT), here are some recommendations regarding foreseeable issues with the facebook/dino-vits16 model:

1. Patch size sensitivity: The performance of the model is highly dependent on the patch size used during training. Smaller patch sizes lead to better performance but reduced throughput. Careful consideration should be given to the trade-off between accuracy and efficiency when deploying the model. [Reference 1, 5]

2. Collapse avoidance: The model relies on centering and target sharpening techniques to avoid collapse during training. Monitoring and tuning these components may be necessary to ensure stable training and prevent uniform or dominated output. [Reference 8]

3. Lack of batch normalization: Unlike standard convolutional networks, the ViT architecture used in DINO does not include batch normalization layers. This design choice should be taken into account when integrating the model into existing pipelines or comparing it with other models. [Reference 11]

4. [More Information Needed] on the specific pretraining dataset and any potential biases or limitations it may introduce.

5. [More Information Needed] on the computational resources and environmental impact of training and deploying the model at scale.

6. Potential misuse: The strong performance of the model in tasks like copy detection and image retrieval [Reference 10] raises concerns about potential misuse, such as unauthorized tracking or surveillance. Clear guidelines and safeguards should be put in place to prevent unethical applications.

7. Interpretability and transparency: [More Information Needed] on the interpretability of the learned features and the decision-making process of the model. Efforts should be made to provide explanations and visualizations to build trust and accountability.

## Training Details

### Training Data

The model facebook/dino-vits16 is pretrained on the ImageNet dataset without labels. [More Information Needed] for details about data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Here are the details about preprocessing for the data of the model facebook/dino-vits16, based on the provided references:

Tokenization/Patch Extraction:
The ViT architecture takes as input a grid of non-overlapping contiguous image patches of resolution 16x16. (Reference 1)

Resizing:
We follow the data augmentations of BYOL (color jittering, Gaussian blur and solarization) and multi-crop with a bicubic interpolation to adapt the position embeddings to the scales. (Reference 4)

Other Preprocessing:
[More Information Needed]

Code:
[More Information Needed]

#### Training Hyperparameters

Here are the detailed training hyperparameters for the model facebook/dino-vits16 based on the provided references:

Model architecture:
- Vision Transformer (ViT) small variant with patch size 16x16

Training dataset:
- ImageNet dataset without labels

Optimizer:
- AdamW optimizer
- Batch size: 1024, distributed over 16 GPUs
- Learning rate schedule:
  - Linearly ramped up during first 10 epochs to base value of lr = 0.0005 * batchsize/256
  - Then decayed with cosine schedule
- Weight decay schedule: 
  - Follows cosine schedule from 0.04 to 0.4

Training duration:
- 100 epochs
- Training time is 1.75 days on 8 GPUs

Momentum encoder:
- EMA (exponential moving average) of the student encoder
- Momentum parameter m = 0.996

Temperatures:
- Student temperature τ_s = 0.1
- Teacher temperature τ_t linearly warmed up from 0.04 to 0.07 during first 30 epochs

Data augmentation:
- BYOL augmentations (color jittering, Gaussian blur, solarization) 
- Multi-crop with bicubic interpolation to adapt position embeddings to scales

[More Information Needed] on exact crop scales used for multi-crop augmentation.

#### Speeds, Sizes, Times

Based on the provided references, here are the details about the model facebook/dino-vits16:

Training throughput and time:
- Training DINO with ViT-S/16 on ImageNet without labels using two 8-GPU servers takes 3 days to achieve 76.1% top-1 accuracy. (Reference 1, 9)
- Using multi-crop improves the accuracy/running-time tradeoff. For example, using 2×224²+10×96² crops reaches 74.6% accuracy in 24 hours, compared to 72.5% after 46 hours without multi-crop (2×224²). (Reference 4)

GPU memory requirements:
- Training ViT-S/16 with 2×224² crops uses 9.3G GPU memory.
- Training ViT-S/16 with 2×224²+10×96² crops uses 15.4G GPU memory. (Reference 4)

Checkpoint sizes:
[More Information Needed]

Start or end time: 
[More Information Needed]

Implementation details:
- Pretrained on ImageNet dataset without labels
- Used adamw optimizer with batch size 1024, distributed over 16 GPUs for ViT-S/16
- Learning rate linearly ramped up to 0.0005 * batchsize/256 over first 10 epochs, then decayed with cosine schedule
- Weight decay follows cosine schedule from 0.04 to 0.4
- Temperature τ_s=0.1, τ_t warmed up from 0.04 to 0.07 over first 30 epochs
- Augmentations: color jittering, Gaussian blur, solarization, multi-crop
(Reference 5)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the DINO model with ViT-S/16 architecture (facebook/dino-vits16) evaluates on the following benchmarks and datasets:

1. Standard self-supervised benchmark on ImageNet for feature validation
2. ImageNet linear classification benchmark, achieving 80.1% top-1 accuracy with a ViT-Base with small patches
3. Image retrieval performance comparison on ImageNet and Google Landmarks datasets

The references do not provide specific details on the exact benchmarks or metrics used for evaluating the image retrieval performance. [More Information Needed]

#### Factors

Based on the provided references about the DINO self-supervised learning framework applied to Vision Transformers (ViT), here are some foreseeable characteristics that may influence the behavior of the facebook/dino-vits16 model:

Domain and Context:
- The model was pretrained on ImageNet, so it is expected to perform well on natural images similar to that dataset. Performance on other visual domains like medical imaging, satellite imagery, etc. is unclear. [More Information Needed]

- The model uses self-supervised pretraining, which has been shown to transfer better than supervised pretraining for ViT architectures on downstream tasks (Reference 1). So it may generalize well to new visual tasks.

- Evaluation focused on tasks like copy detection, nearest neighbor search, and transfer learning (References 2, 6, 9). But real-world robustness in diverse contexts needs further study. [More Information Needed]

Population Subgroups:
- No information is provided about performance disparities across demographic subgroups (age, gender, skin tone, etc). Disaggregated evaluation is needed to uncover potential biases or unfairness. [More Information Needed]

- Geographic diversity of pretraining data is not discussed. The model may underperform for cultural contexts and visual concepts underrepresented in the training set. [More Information Needed]

Other Factors:
- Increasing ViT model size, pretraining duration, and number of self-attention heads tends to improve accuracy, but at the cost of computational efficiency (References 5, 7). Practical tradeoffs should be considered.

- The momentum teacher was important for preventing collapse in the self-supervised framework (References 8, 10). Careful hyperparameter tuning is likely needed for optimal performance.

In summary, while the provided studies demonstrate promising results for the DINO framework on academic benchmarks, more information is needed to fully characterize the model's real-world behavior across diverse domains and populations. Disaggregated evaluation focused on robustness and fairness should be a priority.

#### Metrics

Based on the references provided, the following metrics will be used to evaluate the DINO ViT-S/16 model:

1. Top-1 accuracy on ImageNet using a linear classifier trained on frozen features (linear evaluation protocol). This measures the quality of the learned features without finetuning.

2. Top-1 accuracy on ImageNet by finetuning the pretrained weights on the downstream task. This measures the performance when adapting the model to the specific task.

3. Accuracy using a k-Nearest Neighbor (k-NN) classifier with the learned features. Specifically, a 20-NN classifier is used as it consistently performs the best. This evaluation protocol does not require learning additional parameters.

4. Image retrieval performance using the pretrained features on datasets like ImageNet and Google Landmarks v2.

[More Information Needed] on the specific tradeoffs between different errors that will be considered during evaluation.

### Results

Evaluation results of the model facebook/dino-vits16 based on the Factors and Metrics:

Factors:
- Architecture: ViT-small (ViT-S), which follows the design of DeiT-S and is similar to ResNet-50 in terms of number of parameters (21M vs 23M)
- Pre-training framework: DINO (self-supervised learning)

Metrics:
- ImageNet k-NN accuracy: [More Information Needed]
- ImageNet linear probing accuracy: [More Information Needed] 
- Image retrieval performance on ImageNet: [More Information Needed]
- Image retrieval performance on Google Landmarks: [More Information Needed]
- Object discovery: [More Information Needed]
- Transfer learning: [More Information Needed]

Code:
[More Information Needed]

Note: The provided references do not contain specific evaluation results for the facebook/dino-vits16 model. More information would be needed from the model developers to fill in the metric values and provide any relevant code.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the facebook/dino-vits16 model:

- The model achieves 73.3% accuracy on k-NN evaluation and 76.0% accuracy on linear evaluation on ImageNet. (Reference 10)

- The model's performance is compared with other self-supervised methods using the same ViT-small architecture. (Reference 1)

- The model's features are evaluated for image retrieval on the revisited Oxford and Paris datasets. DINO features outperform supervised features, especially when pretrained on a landmark dataset. (References 3, 8)

- The model's self-attention maps are probed, showing that different heads attend to different semantic regions of an image, even when occluded or small. (Reference 6)

- [More Information Needed] on the model's performance on dense recognition tasks like segmentation. The references mention that variants with small patches ("/8") perform much better, but no specific results are provided for the dino-vits16 model. (Reference 7)

In summary, the dino-vits16 model demonstrates strong performance on ImageNet classification, image retrieval, and produces interpretable self-attention maps, showcasing the effectiveness of the DINO self-supervised learning framework.

## Model Examination

Based on the provided references, here is the experimental section on explainability/interpretability for the model facebook/dino-vits16:

The self-attention visualizations from the last layer of DINO ViT-S/8 (a model similar to facebook/dino-vits16) provide insights into what the model focuses on for several reference points in images randomly selected from the COCO validation set, which were not used during training. These visualizations offer a glimpse into the internal representations learned by the self-supervised Vision Transformer model.

[More Information Needed] on specific code examples related to generating these visualizations for the facebook/dino-vits16 model.

Exploring the interpretability of features learned by large Vision Transformer models pre-trained with DINO on random, uncurated images could potentially push the limits of visual features and improve our understanding of how these models capture and represent visual information.

[More Information Needed] on additional explainability techniques, such as attribution methods or concept activation vectors, that could be applied to the facebook/dino-vits16 model to further investigate its learned representations and decision-making process.

## Environmental Impact

- **Hardware Type:** The model facebook/dino-vits16 is trained on two 8-GPU servers over 3 days, as mentioned in Reference 1:

"Overall, training DINO with Vision Transformers achieves 76.1 top-1 accuracy using two 8-GPU servers for 3 days."

Reference 2 also confirms the hardware setup:

"In Tab. 8, we detail the time and GPU memory requirements when running ViT-S/16 DINO models on two 8-GPU machines."
- **Software Type:** The model facebook/dino-vits16 is trained using PyTorch, as evidenced by this excerpt from the references:

"Please install [PyTorch](https://pytorch.org/) and download the [ImageNet](https://imagenet.stanford.edu/) dataset. This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2."
- **Hours used:** Based on the provided references, the training time for the facebook/dino-vits16 model is as follows:

Training the DINO model with ViT-S/16 on the ImageNet dataset without labels takes 2.6 days using 16 GPUs (Reference 6).

Some additional details:
- Using multi-crop improves the accuracy / running-time tradeoff for DINO runs. For example, DINO with 2×224^2 +10×96^2 crop setting reaches 74.6% accuracy in 24 hours, compared to 72.5% after 46 hours of training without multi-crop (2×224^2) (Reference 3).
- The resulting pretrained model reaches 73.3% on k-NN evaluation and 76.0% on linear evaluation (Reference 6).

[More Information Needed] on the exact hyperparameters and hardware setup used for the specific facebook/dino-vits16 model.
- **Cloud Provider:** [More Information Needed]

The provided references do not specify the cloud provider used for training the facebook/dino-vits16 model. The references mention using Slurm and submitit for distributed training across multiple nodes and GPUs, but do not indicate a specific cloud provider.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the facebook/dino-vits16 model. To accurately report the carbon emissions, details such as the hardware used for training, total training time, energy consumption, and the carbon intensity of the electricity grid would be required.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The model uses a Vision Transformer (ViT) backbone with the following configuration:
- ViT-Small (ViT-S) architecture
- 16x16 input patch size ("/16")
- 6 Transformer blocks
- 384 channel dimension
- 6 attention heads
- 197 input tokens for 224x224 resolution
- 21.7M total parameters (excluding projection head)

The projection head consists of:
- 3-layer MLP with hidden dimension 2048
- 2 normalization layers
- Weight normalized fully connected output layer with K dimensions

The model does not use any batch normalization layers, making it entirely BN-free.

Training Objective:
DINO (self-DIstillation with NO labels) is a self-supervised learning approach that can be interpreted as a form of knowledge distillation with no labels. The training involves a student network and a teacher network, both sharing the same ViT architecture but with different parameters.

The training objective is to minimize the cross-entropy loss between the output of the student network and the sharpened and centered output of the teacher network. This encourages the student to predict the output of the teacher network without using any labeled data.

[More Information Needed] on the specific loss function and optimization details.

### Compute Infrastructure

Based on the provided references, the compute infrastructure used for training the facebook/dino-vits16 model is as follows:

- The model was trained on 2 nodes with 8 GPUs each, for a total of 16 GPUs (Reference 1).
- Slurm and submitit were used for distributed training (Reference 1).
- The exact command used for training on 2 nodes with 16 GPUs total is (Reference 1):
```
python run_with_submitit.py --nodes 2 --ngpus 8 --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```
- The model was trained with a batch size of 1024, distributed over 16 GPUs when using ViT-S/16 (Reference 4).
- [More Information Needed] on the exact GPU model used.
- [More Information Needed] on the CPU and memory specifications of the nodes.
- The model achieves 76.1% top-1 accuracy using two 8-GPU servers for 3 days (Reference 3).

## Citation

```
@misc{mathilde-emerging,
    author = {Mathilde Caron and
              Hugo Touvron and
              Ishan Misra and
              Hervé Jegou and
              Julien Mairal and
              Piotr Bojanowski and
              Armand Joulin and
              Facebook Ai Research},
    title  = {Emerging Properties in Self-Supervised Vision Transformers},
    url    = {https://arxiv.org/pdf/2104.14294.pdf}
}
```

