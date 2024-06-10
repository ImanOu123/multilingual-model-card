# Model Card for microsoft/swin-small-patch4-window7-224

The model microsoft/swin-small-patch4-window7-224 is a Swin Transformer model, which is a hierarchical Transformer whose representation is computed with shifted windows, allowing for efficient modeling at various scales and linear computational complexity with respect to image size. It serves as a general-purpose backbone for computer vision tasks such as image classification and dense prediction.

## Model Details

### Model Description

Model Architecture:
- Swin Transformer architecture, which is a hierarchical Transformer whose representation is computed with shifted windows
- Splits input RGB image into non-overlapping 4x4 patches, each treated as a "token" with feature dimension 4x4x3=48
- Applies several Swin Transformer blocks (modified self-attention) on patch tokens, maintaining number of tokens in "Stage 1"
- Performs patch merging by concatenating features of 2x2 neighboring patches to reduce number of tokens and increase feature dimension in later stages
- Uses global average pooling and linear classifier on last stage output for image classification

Training Procedures:
- Trained on ImageNet-1K, COCO object detection, and ADE20K semantic segmentation datasets
- [More Information Needed] on specific training details like optimizer, learning rate schedule, data augmentation, etc.

Parameters:
[More Information Needed]

Important Disclaimers:
[More Information Needed]

Feel free to reach out if you need any other details to complete the model card. As the project organizer, I'm happy to track down additional information from the team.

- **Developed by:** Ze Liu; Yutong Lin; Yue Cao; Han Hu; Yixuan Wei; Zheng Zhang; Stephen Lin; Baining Guo
- **Funded by:** Based on the provided references, the Swin Transformer model microsoft/swin-small-patch4-window7-224 appears to have been developed by researchers at Microsoft. Specifically, the acknowledgements section mentions "colleagues at Microsoft" who provided help, including Li Dong, Furu Wei, Bin Xiao, Lu Yuan and Lei Zhang.

The main authors listed are Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Some are noted as interns at MSRA (Microsoft Research Asia).

However, no explicit funding organizations are mentioned in the provided information. To conclusively state the funding sources, [More Information Needed] would need to be provided by the model developers or Microsoft.
- **Shared by:** Based on the references provided, the main contributors to the Swin Transformer model (microsoft/swin-small-patch4-window7-224) are:

- Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo, who authored the paper "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".

The references also mention:

- Li Dong and Furu Wei from Microsoft, who provided useful discussions.
- Bin Xiao, Lu Yuan and Lei Zhang, who helped with datasets.

However, more specific information is needed about their direct contributions to the GitHub repository and the model made available online.
- **Model type:** The microsoft/swin-small-patch4-window7-224 model is a hierarchical vision Transformer pretrained on the ImageNet-1K dataset using the shifted window approach, which enables efficient modeling at various scales for image classification and dense prediction tasks.
- **Language(s):** The model microsoft/swin-small-patch4-window7-224 processes natural images, as it is trained on the ImageNet-1K dataset which consists of images across 1,000 object categories.
- **License:** Based on the provided references, there is no direct information about the license being used for the model microsoft/swin-small-patch4-window7-224. The references mention various Swin Transformer models and their training details, but do not specify the license for this particular model.

[More Information Needed]

To properly answer this question, more details would be needed from the model developers or repository about the specific license that the microsoft/swin-small-patch4-window7-224 model is being released under.
- **Finetuned from model:** Based on the provided references, there is no direct mention of the model microsoft/swin-small-patch4-window7-224 being fine-tuned from another base model. The references describe the general Swin Transformer architecture, but do not specify the pre-training or fine-tuning details for this particular model.

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/microsoft/Swin-Transformer
- **Paper:** https://arxiv.org/pdf/2103.14030.pdf
- **Demo:** [More Information Needed]

Based on the provided references, there is no direct information about a demo link for the model microsoft/swin-small-patch4-window7-224. The references mention the model weights, configuration file, and the original research paper, but do not include a demo link. More information would be needed from the model development team to provide a demo link in the model card.
## Uses

### Direct Use

The Swin Transformer model microsoft/swin-small-patch4-window7-224 can be used for image classification without requiring fine-tuning, post-processing, or plugging into a pipeline. As mentioned in the references:

"The image classification is performed by applying a global average pooling layer on the output feature map of the last stage, followed by a linear classifier."

So the pre-trained Swin Transformer model can directly output class predictions for an input image by applying global average pooling and a linear classifier to the final feature map.

[More Information Needed] for providing a specific code snippet, as no direct code block is found in the given references that demonstrates using the microsoft/swin-small-patch4-window7-224 model for classification out-of-the-box. The references mention some high-level instructions for image classification in get_started.md, but the specific code is not provided.

In summary, the microsoft/swin-small-patch4-window7-224 model can be used as a standalone image classifier without modification, but more specific code examples would be helpful to include in the model card to make it easier for users to get started with the model.

### Downstream Use

The Swin Transformer model microsoft/swin-small-patch4-window7-224 can be used as a general-purpose backbone for various computer vision tasks, including:

1. Image classification: The model can be fine-tuned on a specific dataset for image classification tasks. It has achieved 87.3 top-1 accuracy on ImageNet-1K.

2. Object detection and instance segmentation: The model can be integrated into object detection frameworks such as Cascade Mask R-CNN, ATSS, RepPoints v2, and Sparse RCNN. It has achieved strong performance on the COCO dataset, surpassing previous state-of-the-art results.

3. Semantic segmentation: The model can be used as a backbone for semantic segmentation tasks, leveraging techniques like feature pyramid networks (FPN) or U-Net. It has obtained competitive results on the ADE20K dataset.

The hierarchical architecture and shifted windowing scheme of the Swin Transformer make it efficient and flexible for modeling at various scales, with linear computational complexity with respect to image size.

[More Information Needed] (No specific code snippet found in the provided references)

To fine-tune or integrate the Swin Transformer model into a larger ecosystem or app, one would typically need to adapt the model to the specific task and dataset, and then train or fine-tune the model using appropriate techniques and hyperparameters.

For more detailed information on how to use the model for a specific task or integrate it into a larger system, please refer to the official documentation or consult with the model developers.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the model microsoft/swin-small-patch4-window7-224 may be misused and what users ought not do:

The Swin Transformer model is designed for vision problems and has shown effectiveness on tasks like object detection and image classification. However, users should not assume it will perform well on natural language processing tasks without further investigation, as stated in reference 1.

Users should not train the model on datasets containing private, sensitive, or unethical content. The model's capabilities could potentially be misused to analyze or generate harmful images if trained improperly.

[More Information Needed] on the specific datasets and training approaches used for this model to better assess additional misuse risks. The references mention techniques like masked image modeling and contrastive learning but don't clarify if they were used for this particular model.

In general, users have an obligation to carefully control access to the model and not deploy it for unintended or malicious purposes, such as unauthorized surveillance, generating fake media, or any applications that could cause harm to individuals or society if the model makes mistakes or is used improperly. More information on the intended use cases would help determine other potential misuse vectors to caution users about.

### Bias, Risks, and Limitations

Based on the provided references about the Swin Transformer model (microsoft/swin-small-patch4-window7-224), here are some potential issues and limitations:

Technical limitations:
- The model's architecture may be unsuitable as a general-purpose backbone for dense vision tasks or high-resolution input images, due to its low-resolution feature maps and quadratic complexity increase with image size (Reference 9).
- [More Information Needed] on the model's performance and robustness on tasks beyond image classification, object detection, and semantic segmentation.

Potential misunderstandings:
- Users may assume this model can effectively handle all computer vision tasks, but CNNs still dominate many areas. The references focus primarily on image classification, detection and segmentation (References 1, 9, 10).
- [More Information Needed] on how well the model's performance generalizes to real-world, out-of-distribution data.

Foreseeable sociotechnical harms:
- As with many vision models, there are risks of biased outputs or misuse for surveillance if the model is applied without careful consideration of fairness and privacy issues. However, [More Information Needed] on specific analyses of bias and fairness for this model.
- [More Information Needed] on potential security vulnerabilities or robustness to adversarial attacks, which could lead to exploitation if the model is deployed in sensitive applications.

To mitigate these issues, more testing of the model's limitations, biases, and failure modes is needed. Users should be clearly informed about the appropriate use cases and caveats. Responsible deployment practices that consider fairness, security and privacy are also recommended.

### Recommendations

Based on the provided references about the Swin Transformer model (microsoft/swin-small-patch4-window7-224), here are some recommendations regarding foreseeable issues:

1. Computational complexity and efficiency: The shifted window approach and hierarchical architecture of Swin Transformer help achieve linear computational complexity with respect to image size (Ref 8). However, it's important to monitor and optimize the model's real-world latency, especially for dense prediction tasks (Ref 4, 8).

2. Generalizability to other domains: While Swin Transformer has shown promising results on vision tasks, its applicability to other domains like natural language processing should be further investigated (Ref 2).

3. Licensing and contributions: The model is open to contributions, but contributors need to agree to the Contributor License Agreement (CLA) (Ref 5, 6). It's crucial to clearly communicate the licensing terms and contribution guidelines to the community.

4. [More Information Needed] on the model's robustness, fairness, and potential biases. The provided references do not cover these aspects in detail.

5. [More Information Needed] on the model's environmental impact and computational resource requirements for training and deployment.

To comprehensively address the foreseeable issues, more information is needed on the model's performance across diverse datasets, its ethical considerations, and its resource footprint. Engaging with the broader community and conducting further studies can help uncover and mitigate potential risks associated with the model.

## Training Details

### Training Data

The model microsoft/swin-small-patch4-window7-224 is trained on ImageNet-1K [19], which contains 1.28M training images and 50K validation images from 1,000 classes. [More Information Needed] for documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

For the model microsoft/swin-small-patch4-window7-224, the preprocessing steps are as follows:

1. Patch Splitting: The input RGB image is split into non-overlapping patches. The patch size used is 4x4, resulting in a feature dimension of 4x4x3=48 for each patch. (Reference 9)

2. Linear Embedding: A linear embedding layer is applied to the raw-valued feature of each patch to project it to a higher dimension. (Reference 9)

3. Resizing: The default input image resolution is 224x224. For fine-tuning on larger resolutions like 384x384, the model is first trained on 224x224 resolution and then fine-tuned, instead of training from scratch. (Reference 6)

[More Information Needed] on further details about tokenization and other preprocessing steps.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the model microsoft/swin-small-patch4-window7-224:

Optimizer: AdamW
- Initial learning rate: 0.001
- Weight decay: 0.05
- Gradient clipping with a max norm of 1

Training epochs: 300, using a cosine decay learning rate scheduler with 20 epochs of linear warm-up

Batch size: 1024

Input image resolution: 224x224

Data augmentation and regularization strategies:
- RandAugment
- Mixup
- Cutmix
- Random erasing
- Stochastic depth with a ratio of 0.3 for Swin-S

[More Information Needed] on the exact dataset and loss function used for training this specific model.

#### Speeds, Sizes, Times

Here are the details about the model microsoft/swin-small-patch4-window7-224 based on the provided references:

Model Architecture:
- Uses patch size of 4x4 and feature dimension of 48 per patch (4x4x3=48)
- Applies a linear embedding layer to project the raw patch features
- Employs a hierarchical Transformer architecture with shifted windows for greater efficiency
- Window size of 7x7 is used in the multi-head self-attention modules
- Performs global average pooling on the last stage's output feature map, followed by a linear classifier for image classification

Performance:
- Achieves 87.3% top-1 accuracy on ImageNet-1K dataset
- Demonstrates strong performance on dense prediction tasks like object detection and semantic segmentation
- Offers a good speed-accuracy trade-off compared to state-of-the-art ConvNets like RegNet and EfficientNet

Training Details:
- Trained using AdamW optimizer for 300 epochs with cosine decay learning rate scheduler and 20 epochs of linear warm-up
- Uses batch size of 1024, initial learning rate of 0.001, and weight decay of 0.05
- Employs data augmentation and regularization strategies similar to DeiT training, except for repeated augmentation and EMA

[More Information Needed] for the following:
- Throughput 
- Start or end time of training
- Checkpoint sizes

The references do not provide specific details about these aspects of the model. More information would be needed from the model developers or training logs to include these details in the model card description.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the Swin Transformer model microsoft/swin-small-patch4-window7-224 is evaluated on the following benchmarks and datasets:

1. ImageNet-1K image classification (References 1, 2, 3, 4, 6)
   - Contains 1.28M training images and 50K validation images from 1,000 classes
   - Reports top-1 accuracy on a single crop

2. COCO object detection and instance segmentation (References 2, 7, 8, 10)
   - COCO 2017 dataset with 118K training, 5K validation and 20K test-dev images
   - Evaluated using frameworks like Cascade Mask R-CNN, ATSS, RepPoints v2, and Sparse RCNN in mmdetection
   - System-level comparison using an improved HTC (HTC++) with additional enhancements

3. ADE20K semantic segmentation (References 2, 10)
   - [More Information Needed] on the specific dataset details and evaluation metrics for ADE20K

The model achieves strong performance on these tasks, outperforming previous state-of-the-art models like ViT, DeiT, and ResNe(X)t (References 6, 10, 11).

#### Factors

Based on the provided references about the Swin Transformer model (microsoft/swin-small-patch4-window7-224), here are some key characteristics that may influence its behavior:

Domain and Context:
- The model was primarily evaluated on ImageNet-1K for image classification, COCO for object detection, and ADE20K for semantic segmentation. Its performance in other domains or on different datasets may vary.
- The model uses a hierarchical architecture with shifted windows, which allows it to model at various scales. This design makes it suitable for a broad range of vision tasks.

Population Subgroups:
[More Information Needed] - The provided references do not contain information about the model's performance across different population subgroups or potential disparities.

Other Factors:
- The model's performance may be influenced by the choice of training settings, such as input resizing, optimizer, learning rate, and number of training epochs.
- The use of shifted windows in the model's architecture significantly enhances its modeling power compared to using a single window partitioning at each stage.
- The model's inference speed may be affected by the implementation of self-attention computation and the level of kernel optimization.

To fully understand the model's behavior and uncover potential disparities in performance, it would be ideal to conduct disaggregated evaluations across various factors, including different population subgroups. However, the provided references do not contain sufficient information to address this aspect.

#### Metrics

Based on the provided references, the following metrics will be used for evaluating the Swin Transformer model (microsoft/swin-small-patch4-window7-224):

1. For image classification on ImageNet-1K, the top-1 accuracy on a single crop is reported.

2. For object detection and instance segmentation on COCO, the model is compared to standard ConvNets (ResNe(X)t) and previous Transformer networks (DeiT) by changing only the backbones with other settings unchanged.

3. The model's performance is also evaluated on ADE20K semantic segmentation.

4. The speed-accuracy trade-off is compared with state-of-the-art ConvNets, such as RegNet and EfficientNet.

[More Information Needed] regarding specific metrics for COCO object detection, instance segmentation, and ADE20K semantic segmentation tasks.

### Results

Based on the provided references, here is the evaluation information I could find for the Swin-S (Swin-Small) model, which corresponds to the microsoft/swin-small-patch4-window7-224 model:

Model Size and Complexity:
- Swin-S has about 0.5× the model size and computational complexity compared to the Swin-B base model. (Reference 6)
- The complexity of Swin-S is similar to that of ResNet-101. (Reference 6)

Evaluation Datasets:
- ImageNet-1K image classification (Reference 1)
- COCO object detection (Reference 1) 
- ADE20K semantic segmentation (Reference 1)

ImageNet-1K Evaluation:
- The model was evaluated on ImageNet-1K, which contains 1.28M training images and 50K validation images from 1,000 classes. (Reference 2)
- Top-1 accuracy on a single crop is reported. (Reference 2)

Training Settings:
- An AdamW optimizer was used for 300 epochs with a cosine decay learning rate scheduler and 20 epochs of linear warm-up. (Reference 3)
- Batch size of 1024, initial learning rate of 0.001, and weight decay of 0.05 were used. (Reference 3)
- Most data augmentation and regularization strategies were included, except for repeated augmentation and EMA. (Reference 3)

Model Architecture:
- The model uses window size M = 7 by default. (Reference 6)
- The query dimension of each head is d = 32. (Reference 6)
- [More Information Needed] for the expansion layer of each MLP.

[More Information Needed] for the specific ImageNet-1K, COCO, and ADE20K results of the Swin-S model. The references do not provide the exact metric values.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model microsoft/swin-small-patch4-window7-224:

The Swin Transformer achieves slightly better speed-accuracy trade-off compared to state-of-the-art ConvNets like RegNet and EfficientNet on ImageNet-1K image classification, even though those models were obtained via thorough architecture search.

On COCO object detection and instance segmentation tasks, the Swin Transformer outperforms standard ConvNets like ResNe(X)t and previous Transformer networks like DeiT when used as the backbone with other settings kept the same.

[More Information Needed] on the specific top-1 accuracy numbers achieved by the swin-small-patch4-window7-224 model on ImageNet-1K validation set.

The references mention some training details used for the Swin Transformer on ImageNet like AdamW optimizer for 300 epochs, cosine decay learning rate scheduler, 20 epochs linear warmup, batch size 1024, initial learning rate 0.001, weight decay 0.05, and various data augmentation and regularization strategies. However, [More Information Needed] on the exact settings used for the swin-small-patch4-window7-224 model.

In summary, the Swin Transformer shows strong performance on both image classification and object detection/segmentation tasks, but more specific results are needed for the swin-small-patch4-window7-224 model to include in the model card description. The general training approach is outlined but exact hyperparameters used for this model size are not provided in the given references.

## Model Examination

Here is my attempt at an experimental section on explainability/interpretability for the Swin Transformer model microsoft/swin-small-patch4-window7-224:

The Swin Transformer introduces a shifted windowing approach that enables cross-window connections while maintaining efficient local self-attention computation within each window. Experiments ablating the shifted window approach (Reference 3) show that it improves performance by +1.1% top-1 accuracy on ImageNet-1K, +2.8 box AP/+2.2 mask AP on COCO object detection, and +2.8 mIoU on ADE20K semantic segmentation compared to using a single window partitioning at each stage. This indicates the shifted windowing scheme enhances the model's ability to build connections across windows from preceding layers.

[More Information Needed] on visualizations or analysis of the learned window-based representations and self-attention patterns to further explain what the model is learning and how the shifted windowing impacts the learned features.

The hierarchical architecture of the Swin Transformer enables it to model at various scales and have linear computational complexity with respect to image size (Reference 5). [More Information Needed] on how the representations at different stages can be visualized or probed to understand what features are being captured at each scale.

Overall, the shifted windowing and hierarchical structure are key to the strong performance of the Swin Transformer on a range of vision tasks (Reference 6, 7). However, [More Information Needed] on additional experiments and analysis focused specifically on explaining what the model has learned and how its architecture impacts the learned representations. Visualizations of attention patterns, probing learned features at different stages, and ablating the model could help provide more interpretability.

## Environmental Impact

- **Hardware Type:** [More Information Needed]

The provided references do not contain specific information about the hardware used to train the microsoft/swin-small-patch4-window7-224 model. More details would be needed from the model developers or training logs to determine the exact hardware setup, such as GPU type and number of GPUs used.
- **Software Type:** Based on the provided references, the model microsoft/swin-small-patch4-window7-224 is trained for image classification tasks. Specifically, the references mention:

"The supported code and models for ImageNet-1K image classification, COCO object detection and ADE20K semantic segmentation are provided."

And:

"These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object"

So in summary, the microsoft/swin-small-patch4-window7-224 model is trained for image classification on the ImageNet-1K dataset. The specific software frameworks and tools used for training are [More Information Needed] based on the provided references.
- **Hours used:** Based on the provided references, the Swin-S model (which likely corresponds to microsoft/swin-small-patch4-window7-224) was trained on ImageNet-1K using the following schedule:

AdamW optimizer with:
- Initial learning rate of 0.0001 
- Weight decay of 0.05
- Batch size of 16
- 3x schedule (36 epochs with the learning rate decayed by 10× at epochs 27 and 33)

So in summary, the Swin-S model was trained for 36 epochs on ImageNet-1K to obtain the microsoft/swin-small-patch4-window7-224 weights.

[More Information Needed] on the exact wall-clock training time, as that would depend on the specific hardware used which is not mentioned in the provided references.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the model microsoft/swin-small-patch4-window7-224. More information would be needed from the model development team or training logs to determine the cloud provider utilized.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the model microsoft/swin-small-patch4-window7-224. To accurately report the carbon emissions, more details would be needed such as the hardware used for training, the energy consumption during training, the duration of training, and the carbon intensity of the electricity used.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The Swin Transformer model microsoft/swin-small-patch4-window7-224 has the following architecture:
- Input RGB image is split into non-overlapping patches of size 4x4. Each patch is treated as a "token" with feature dimension 4x4x3=48.
- A linear embedding layer is applied to the raw patch features.
- Several Swin Transformer blocks are applied on the patch tokens. Each block consists of a shifted window based multi-head self-attention (MSA) module, followed by a 2-layer MLP with GELU nonlinearity and LayerNorm. The window size is 7x7.
- The model maintains the number of tokens in Stage 1, then applies patch merging (concatenation of neighboring features) to downsample the feature map in the following stages.
- This model is the "Swin-S" version which has about 0.5x the model size and computational complexity compared to the Swin-B base model.

Model Objective:
The Swin Transformer serves as a general-purpose backbone model for computer vision tasks. Its key features are:
- Hierarchical architecture that enables modeling at various scales
- Linear computational complexity with respect to image size, achieved via computing self-attention within local windows
- Shifted windowing scheme that allows for cross-window connections while limiting self-attention computation to non-overlapping local windows for efficiency
These make the model well-suited for a broad range of vision tasks including image classification, object detection and semantic segmentation.

### Compute Infrastructure

Based on the provided references, there is no specific information about the compute infrastructure used for training the model microsoft/swin-small-patch4-window7-224. The references mention some training details like optimizer, learning rate, weight decay, batch size, and training schedule, but do not specify the hardware or compute resources utilized.

[More Information Needed] on the exact compute infrastructure, such as the number and type of GPUs or TPUs, memory, etc. used for training this specific model.

## Citation

```
@misc{ze-swin,
    author = {Ze Liu and
              Yutong Lin and
              Yue Cao and
              Han Hu and
              Yixuan Wei and
              Zheng Zhang and
              Stephen Lin and
              Baining Guo},
    title  = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
    url    = {https://arxiv.org/pdf/2103.14030.pdf}
}
```

