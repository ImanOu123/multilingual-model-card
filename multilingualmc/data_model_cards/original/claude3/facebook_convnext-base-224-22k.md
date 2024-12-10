# Model Card for facebook/convnext-base-224-22k

The facebook/convnext-base-224-22k model is a pure ConvNet model constructed entirely from standard ConvNet modules, which competes favorably with Transformers in terms of accuracy, scalability and robustness across various computer vision tasks such as image classification, object detection, and semantic segmentation.

## Model Details

### Model Description

Model Card for facebook/convnext-base-224-22k

Model Architecture:
- ConvNeXt is a pure ConvNet model that outperforms Swin Transformers on ImageNet-1K classification, COCO detection, and ADE20K segmentation tasks.
- It uses standard ConvNet modules and design choices adapted from vision Transformers.
- The model architecture is similar to ResNet, but with modifications such as separate downsampling layers.

Training Procedures:
- Trained using modern training techniques close to DeiT and Swin Transformer, including:
  - 300 epochs (extended from 90 epochs for ResNets)
  - AdamW optimizer
  - Data augmentation (Mixup, Cutmix, RandAugment, Random Erasing)
  - Regularization (Stochastic Depth, Label Smoothing)
- These training techniques significantly improved performance compared to traditional ConvNets.
- Layer Normalization is used instead of Batch Normalization, resulting in slightly better performance.

Parameters:
[More Information Needed]

Important Disclaimers:
- The design choices in ConvNeXt have been researched separately over the last decade, but not collectively.
- The model's scaling behavior and performance on downstream tasks are key factors that distinguish it from vision Transformers.

For more information or updates, please contact the project organizer.

- **Developed by:** Zhuang Liu; Hanzi Mao; Chao-Yuan Wu; Christoph Feichtenhofer; Trevor Darrell; Saining Xie; A I Facebook;  Research; U C Berkeley
- **Funded by:** Based on the provided references, it appears that the ConvNeXt model was developed by researchers from Facebook AI Research and UC Berkeley, including Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie.

However, there is no explicit mention of the specific funding sources for this project in the given information. To definitively answer the question of who funded the development of the facebook/convnext-base-224-22k model, [More Information Needed].
- **Shared by:** Based on the references provided, the contributors who made the model facebook/convnext-base-224-22k available online are:

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell and Saining Xie from Facebook AI Research and UC Berkeley.

The model is based on their paper "A ConvNet for the 2020s" published at CVPR 2022.
- **Model type:** The facebook/convnext-base-224-22k model is a ConvNeXt model, which is a pure convolutional neural network (CNN) architecture, pre-trained on the ImageNet-22K dataset with 21841 classes and ~14M images using modern training techniques such as AdamW optimizer, data augmentation, regularization, and fine-tuned on the ImageNet-1K dataset for image classification.
- **Language(s):** The model facebook/convnext-base-224-22k processes images, as it is a pure ConvNet model designed for image classification tasks, trained on the ImageNet-22K dataset at 224x224 resolution.
- **License:** [More Information Needed]

The provided references do not contain any information about the specific license being used for the facebook/convnext-base-224-22k model. To complete the model card, more information would need to be gathered from the model creators or associated documentation about the license under which the model is being released.
- **Finetuned from model:** The model facebook/convnext-base-224-22k is not fine-tuned from another model. It is pre-trained on the ImageNet-22k dataset from scratch, as indicated by the "22k" in the model name.
### Model Sources

- **Repository:** https://github.com/facebookresearch/ConvNeXt
- **Paper:** https://arxiv.org/pdf/2201.03545.pdf
- **Demo:** [More Information Needed]

The provided references do not contain a direct link to a demo of the facebook/convnext-base-224-22k model. The references discuss the ConvNeXt architecture and its robustness evaluation, but do not mention a specific demo link for this pre-trained model variant.

To provide the demo link in the model card, more information would need to be gathered from the model developers or official model repository.
## Uses

### Direct Use

The facebook/convnext-base-224-22k model can be used for image classification tasks without requiring fine-tuning, post-processing, or being plugged into a pipeline. As a pre-trained ConvNeXt model, it can directly make predictions on input images.

[More Information Needed] for providing a specific code snippet, as the references do not include a directly applicable code block for using the model out-of-the-box. However, the model should be able to take an input image and output the predicted class probabilities, leveraging the power of the ConvNeXt architecture that has been shown to perform well on image classification tasks.

### Downstream Use

The ConvNeXt model, such as facebook/convnext-base-224-22k, can be fine-tuned for various downstream vision tasks, including:

1. Object detection and instance segmentation on COCO dataset using Mask R-CNN or Cascade Mask R-CNN frameworks. The pre-trained ConvNeXt weights can be used as network initializations.

2. Semantic segmentation on ADE20K dataset. The pre-trained ConvNeXt weights can be used as network initializations.

To fine-tune the model, you can use popular toolboxes like MMDetection and MMSegmentation. Here's a code snippet for fine-tuning ConvNeXt on COCO using MMDetection:

[More Information Needed]

When plugged into a larger ecosystem or app, ConvNeXt can serve as a powerful backbone for various computer vision tasks. Its fully-convolutional nature makes it simple to deploy in different settings. However, for multi-modal learning tasks, a cross-attention module may be preferable for combining information from different modalities.

Overall, ConvNeXt offers a strong alternative to Transformers, achieving competitive performance on major benchmarks while maintaining the simplicity and efficiency of standard ConvNets.

### Out-of-Scope Use

Based on the provided references about the ConvNeXt model, here are some potential ways the model may be misused and what users ought not do:

- The model was trained on the large-scale ImageNet-22K dataset. If users wish to pre-train the model on their own data instead, they need to be very careful in data selection to avoid introducing biases. Using biased data for pre-training could lead to an unfair or biased model. (Reference 5)

- ConvNeXt achieves strong performance with its large model variants, but training and using these huge models requires enormous computing resources and can result in increased carbon emissions. Users should consider if such large models are truly necessary for their use case and strive for simplicity and efficiency when possible. (References 2, 4) 

- While ConvNeXt shows promising robustness on some benchmarks, its robustness and fairness still needs further investigation, especially compared to Transformers. Users should not assume ConvNeXt will be robust and fair in all scenarios without proper testing. (References 1, 6)

- ConvNeXt may be well-suited for certain computer vision tasks, but not necessarily all. In particular, it may be less flexible than Transformers for tasks requiring multi-modal learning, discretized/sparse outputs, or structured outputs. Users should carefully consider if ConvNeXt fits their specific task and data. (References 3, 4)

[More Information Needed] on other specific ways the model could be misused or abused, as the provided references focus more on the model's capabilities and limitations rather than directly discussing potential misuse.

In summary, users ought not to assume ConvNeXt is unbiased, robust, efficient, or suitable for all tasks without careful consideration of their data and requirements. Responsible use requires circumspect data selection, evaluating model size vs efficiency, testing for fairness and robustness, and aligning model choice to the task at hand.

### Bias, Risks, and Limitations

Based on the provided references about the ConvNeXt model, here are some known or foreseeable issues:

Robustness and Fairness Concerns:
- Large models and datasets like those used in ConvNeXt can present issues in terms of model robustness and fairness. Further investigation is needed on the robustness behavior of ConvNeXt vs. Transformers. (Reference 1)
- While ConvNeXt exhibits promising robustness on some benchmarks, more evaluation is needed, especially without using specialized modules or fine-tuning. (Reference 9, 10)

Computational and Environmental Costs: 
- The huge model variants of ConvNeXt that perform best inevitably result in increased carbon emissions. Striving for simplicity in model design is important to mitigate this. (Reference 2)

Data Selection and Bias:
- Using large public datasets like ImageNet-22K can introduce potential data bias concerns. A more responsible approach to data selection is needed if individuals acquire their own pre-training data. (Reference 3)

Task Flexibility Limitations:
- While ConvNeXt performs well on many vision tasks, Transformers may be more flexible for tasks requiring discretized/sparse/structured outputs or multi-modal learning. Architecture choice should meet the task needs. (Reference 4, 5)

[More Information Needed] on specific technical limitations, failure modes, potential misuse cases, or downstream effects on different demographics. Conducting further studies on these aspects would help create a more complete picture of ConvNeXt's foreseeable issues and limitations.

### Recommendations

Based on the provided references about the ConvNeXt model, here are some recommendations regarding foreseeable issues:

Robustness and Fairness:
The references mention that large models and datasets like those used in ConvNeXt can present challenges in terms of model robustness and fairness. Further research is needed to investigate the robustness behavior of ConvNeXt compared to Transformers. Careful data selection is also required to avoid potential data bias concerns.

Computational Resources: 
The references note that visual representation learning with large models like ConvNeXt places enormous demands on computing resources and can increase carbon emissions. Striving for simplicity in model design is recommended as one way to address this.

Applicability to Diverse Tasks:
While ConvNeXt performs well on many vision tasks, the references suggest it may be more or less suited for certain applications compared to Transformers. In particular, Transformers may be more flexible for multi-modal learning tasks that benefit from cross-attention modules.

Scaling Behavior:
[More Information Needed] on how ConvNeXt's performance scales with increased model and data size compared to vision Transformers.

Downstream Task Performance:
[More Information Needed] on ConvNeXt's ability to compete with Swin Transformers on downstream computer vision tasks like object detection and semantic segmentation, which is noted as a central concern for practitioners.

## Training Details

### Training Data

The model facebook/convnext-base-224-22k is pre-trained on the ImageNet-22K dataset, which consists of 21841 classes (a superset of the 1000 ImageNet-1K classes) with approximately 14 million images. [More Information Needed] for documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the data of the model facebook/convnext-base-224-22k:

Resizing/Cropping:
- The model is fine-tuned at 384x384 resolution.
- When fine-tuning at 384x384 resolution, a crop ratio of 1.0 (i.e., no cropping) is used during testing, following [2,74,80].

Data Augmentation:
- RandAugment [14] with parameters (9, 0.5) is used.
- Mixup [90] with alpha=0.8 is used.
- CutMix [89] with alpha=1.0 is used.
- Random Erasing [91] with probability 0.25 is used.

Other Preprocessing:
- Label smoothing [69] with a smoothing factor of 0.1 is used.

[More Information Needed] for details on tokenization, as it is not directly mentioned in the provided references.

#### Training Hyperparameters

Based on the provided references, the following training hyperparameters were used for the ConvNeXt-B model (which corresponds to facebook/convnext-base-224-22k):

- Warmup: 50 epochs
- Layer scale: Disabled
- Stochastic depth rate: 0.2
- Data augmentation: RandAugment with (9, 0.5)
- Mixup: 0.8
- Cutmix: 1.0
- Random erasing: 0.25
- Label smoothing: 0.1
- Gradient clipping: [More Information Needed]
- Exponential moving average (EMA): [More Information Needed]

The ConvNeXt-B model differs from other variants (T/S/L) only in the number of channels C and blocks B per stage. The exact values for C and B in ConvNeXt-B are [More Information Needed].

#### Speeds, Sizes, Times

Based on the provided references, here are the details I can provide about the model facebook/convnext-base-224-22k:

Throughput:
- ConvNeXt models achieve faster inference throughput on A100 GPUs compared to Swin Transformers, sometimes up to 49% faster, when using PyTorch 1.10 with the "Channel Last" memory layout. (Reference 2)
- [More Information Needed] for the specific throughput numbers of the facebook/convnext-base-224-22k model.

Start or end time: [More Information Needed]

Checkpoint sizes: [More Information Needed]

Other relevant information:
- ConvNeXt-B is the end product of the "modernizing" procedure on the ResNet-200 regime. (Reference 3)
- ConvNeXt models compete favorably with Swin Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation. (Reference 8)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the ConvNeXt model facebook/convnext-base-224-22k evaluates on the following benchmarks and datasets:

1. ImageNet-1K: The model is pre-trained on ImageNet-22K and then fine-tuned on ImageNet-1K for evaluation. The top-1 accuracy on the ImageNet-1K validation set is reported.

2. Robustness benchmarks:
   - ImageNet-A
   - ImageNet-R
   - ImageNet-Sketch
   - ImageNet-C: Mean corruption error (mCE) is reported.
   - ImageNet-C: Corruption error is reported.

3. Object detection and segmentation on COCO: The model is used as a backbone for Mask R-CNN and Cascade Mask R-CNN, and the performance is evaluated on the COCO dataset.

[More Information Needed] regarding the specific performance metrics and results obtained on these benchmarks and datasets for the facebook/convnext-base-224-22k model.

#### Factors

Based on the provided references about the ConvNeXt model, here are some foreseeable characteristics that may influence how facebook/convnext-base-224-22k behaves:

Domain and Context:
- The model is designed for computer vision tasks like image classification, object detection/segmentation, and semantic segmentation. It may perform differently across these various domains and contexts.
- ConvNeXt models benefit from pre-training on large-scale datasets like ImageNet-22K. Performance may vary depending on the pre-training data used.

Population Subgroups:
- [More Information Needed] The references do not provide specific information about performance across different population subgroups. Disaggregated evaluation across factors like demographics, geographies, etc. would be needed to uncover any performance disparities.

Robustness and Fairness:
- ConvNeXt shows promising robustness, outperforming some transformer models on benchmarks like ImageNet-A/R/Sketch, indicating potential for strong domain generalization. 
- However, the references note that large models and datasets used can present challenges in terms of robustness and fairness that require further investigation. A circumspect approach to data selection is needed to avoid potential data bias concerns.

Efficiency and Scalability:
- ConvNeXt maintains the efficiency of standard ConvNets and has a fully-convolutional architecture that allows for simple adaptation to different vision tasks.
- Performance scales well with larger model sizes, but this also introduces challenges around increased computational demands and carbon emissions. Striving for simplicity in the model design is an important consideration.

[More Information Needed] Code cannot be provided, as there are no direct code block references from the given text to answer this question. More specific details would be required to assess and compare model behaviors across different subgroups and contexts in order to identify potential disparities or unintended consequences.

#### Metrics

Based on the provided references, the following metrics will be used for evaluating the ConvNeXt models, including facebook/convnext-base-224-22k:

1. Top-1 Accuracy on ImageNet-1K at 224x224 resolution (Table 2)
2. Object detection and segmentation performance (AP) on COCO using Mask R-CNN and Cascade Mask R-CNN (specific AP numbers for ConvNeXt-B are [More Information Needed])
3. Frames per second (FPS) measured on an A100 GPU (specific FPS for ConvNeXt-B is [More Information Needed])
4. FLOPs calculated with image size (1280, 800) (specific FLOPs for ConvNeXt-B is [More Information Needed])
5. Robustness evaluation on benchmark datasets (Table 8):
   - Top-1 Accuracy on ImageNet-A, ImageNet-R, and ImageNet-Sketch
   - Mean corruption error (mCE) and corruption error on ImageNet-C/C

The tradeoffs between different errors are not explicitly discussed in the provided references for the specific ConvNeXt-B model. [More Information Needed] on how the team prioritizes and balances these metrics.

### Results

Based on the provided references, here is the evaluation information I could find for the facebook/convnext-base-224-22k model:

Performance:
- Achieves strong results on ImageNet-1K classification at 224x224 resolution. Exact accuracy numbers not provided. [3]

Efficiency: 
- Exhibits improved efficiency compared to vision Transformers due to ConvNet inductive bias. [2]
- FPS measured on A100 GPU, FLOPs calculated at (1280, 800) image size. Exact FPS and FLOPs numbers not provided. [5]

Robustness:
- Shows promising robustness, outperforming SOTA robust transformer models on several benchmarks. [6] 
- Achieves 69.3% / 68.2% / 55.0% accuracy on ImageNet-A/R/Sketch benchmarks respectively when trained with extra ImageNet-22K data, demonstrating strong domain generalization. [6]
- Reports mean corruption error (mCE) on ImageNet-C and corruption error on ImageNet-C. Exact numbers not provided. [4]

Model Architecture:
- Differs from other ConvNeXt variants (T/S/L/XL) only in number of channels C and blocks B per stage. [1] 
- Is the result of "modernizing" a ResNet-200 regime. [1]
- Number of channels doubles at each stage, following ResNets and Swin Transformers. [1]

[More Information Needed] for exact accuracy, FPS, FLOPs, and corruption error numbers, as well as full model configuration details. The references do not contain code blocks directly relevant for inclusion.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model facebook/convnext-base-224-22k:

The model achieves 85.820% top-1 accuracy and 97.868% top-5 accuracy on ImageNet-1K validation set, with a loss of 0.563.

Additional robustness evaluation was performed on several benchmark datasets:
- ImageNet-A
- ImageNet-R 
- ImageNet-Sketch
- ImageNet-C/C
Mean corruption error (mCE) is reported for ImageNet-C, corruption error for ImageNet-C, and top-1 accuracy for the other datasets. However, the specific results are [More Information Needed].

The model was also evaluated on COCO dataset for object detection and segmentation tasks by fine-tuning Mask R-CNN and Cascade Mask R-CNN. The ConvNeXt model demonstrated improved efficiency compared to vision Transformers, attributed to the ConvNet inductive bias rather than the self-attention mechanism. [More Information Needed] on the specific detection and segmentation metrics.

Overall, the ConvNeXt-B model, which facebook/convnext-base-224-22k is based on, benefited from pre-training on large-scale datasets. Further research on model robustness and fairness is suggested.

## Model Examination

Based on the provided references, here is the experimental section on explainability/interpretability for the facebook/convnext-base-224-22k model:

ConvNeXt demonstrates promising robustness behaviors, outperforming state-of-the-art robust transformer models on several benchmarks (Reference 7). With extra ImageNet-22K data, ConvNeXt-XL shows strong domain generalization capabilities, achieving high accuracy on ImageNet-A, ImageNet-R, and ImageNet-Sketch datasets (Reference 7).

Additional robustness evaluation results for ConvNeXt models are presented, directly testing ImageNet-1K trained/fine-tuned classification models on robustness benchmark datasets such as ImageNet-A, ImageNet-R, ImageNet-Sketch, ImageNet-C, and ImageNet-C (Reference 8). Mean corruption error (mCE) is reported for ImageNet-C, corruption error for ImageNet-C, and top-1 accuracy for all other datasets (Reference 8).

[More Information Needed] on specific explainability/interpretability techniques applied to the ConvNeXt model, such as visualizing feature maps, attention maps, or using methods like LIME or SHAP.

The robustness evaluation results were obtained without using any specialized modules or additional fine-tuning procedures (Reference 7), indicating the inherent robustness of the ConvNeXt architecture.

[More Information Needed] on further analysis of the model's behavior and decision-making process to provide deeper insights into its explainability and interpretability.

## Environmental Impact

- **Hardware Type:** Based on the provided references, there is no direct mention of the specific hardware used for training the facebook/convnext-base-224-22k model. The references discuss inference performance on V100 and A100 GPUs, but do not specify the hardware used for training.

[More Information Needed] on the exact hardware used to train the facebook/convnext-base-224-22k model.
- **Software Type:** Based on the references provided, the ConvNeXt model is built using the timm library, as mentioned in reference 7:

"This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [DeiT](https://github.com/facebookresearch/deit) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repositories."

The timm library is a PyTorch-based library for training state-of-the-art image models. Therefore, the software type that the model facebook/convnext-base-224-22k is trained on is PyTorch.
- **Hours used:** Based on the provided references, there is no specific information about the training time for the model facebook/convnext-base-224-22k. The references mention training settings such as number of epochs, learning rate, batch size, and data augmentation techniques, but do not provide the actual training time.

[More Information Needed] on the amount of time used to train the model facebook/convnext-base-224-22k.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the ConvNeXt-base model on ImageNet-22K. More information would be needed from the model development team to determine the cloud provider utilized for training this particular model.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emissions generated during the training of the facebook/convnext-base-224-22k model. While the references mention that investigating model designs like ConvNeXt can lead to increased carbon emissions, no concrete values are provided for this particular model.

To accurately report the carbon footprint of training the facebook/convnext-base-224-22k model in the model card description, more detailed information would be needed, such as the specific hardware used for training, the duration of the training process, and the energy consumption during that period.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
ConvNeXt is a pure ConvNet model that adopts several design choices from vision Transformers, including:
- Separate downsampling layers, like the "patchify" stem in ViT
- Inverted bottleneck block structure similar to Swin Transformer 
- Replacing BatchNorm with LayerNorm
- Depthwise convolutions for efficient computation

The detailed architecture specifications are:
[More Information Needed]

Objective:
The objective was to construct a pure ConvNet model using only standard ConvNet modules that can compete favorably with state-of-the-art vision Transformers like Swin in terms of accuracy, scalability and robustness across major vision benchmarks like ImageNet classification, COCO object detection/segmentation, and ADE20K semantic segmentation, while maintaining the simplicity and efficiency of ConvNets.

The ConvNeXt models achieved strong results, such as 87.8% top-1 accuracy on ImageNet-1K and outperformed Swin Transformers on COCO and ADE20K, demonstrating that the design choices inspired by Transformers can also benefit pure ConvNets. The results challenge some widely held views and highlight the continued importance of convolutions in computer vision.

### Compute Infrastructure

Based on the provided references, here is the information about the compute infrastructure used for the ConvNeXt-B model pre-trained on ImageNet-22K and fine-tuned on ImageNet-1K:

- The model was pre-trained on ImageNet-22K for 90 epochs with a warmup of 5 epochs, using a batch size of 4096. (Reference 1)

- The model was fine-tuned on ImageNet-1K for 30 epochs, using AdamW optimizer, a learning rate of 5e-5, cosine learning rate schedule, layer-wise learning rate decay, no warmup, a batch size of 512, and weight decay of 1e-8. (Reference 5)

- The inference throughput of the model was benchmarked using NVIDIA A100 GPUs, which support the TensorFloat32 (TF32) tensor cores. PyTorch version 1.10 was used to leverage the "Channel Last" memory layout for further speedup. (References 3 and 6)

[More Information Needed] about the specific number and configuration of GPUs used for training and the total training time.

## Citation

```
@misc{zhuang-a,
    author = {Zhuang Liu and
              Hanzi Mao and
              Chao-Yuan Wu and
              Christoph Feichtenhofer and
              Trevor Darrell and
              Saining Xie and
              A I Facebook and
              Research and
              U C Berkeley},
    title  = {A ConvNet for the 2020s},
    url    = {https://arxiv.org/pdf/2201.03545.pdf}
}
```

