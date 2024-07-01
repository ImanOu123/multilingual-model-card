# Model Card for hustvl/yolos-tiny

hustvl/yolos-tiny is a tiny-sized object detection model based on the Vision Transformer (ViT) architecture. It achieves impressive performance compared to well-established and highly-optimized tiny-sized CNN object detectors, while being competitive in FLOPs and FPS.

## Model Details

### Model Description

Model Description for hustvl/yolos-tiny:

Model Architecture:
- Closely follows the original Vision Transformer (ViT) architecture, optimized for object detection
- Uses a simple setup to reveal characteristics of the Transformer family in object detection
- Detector heads are implemented by one MLP with separate parameters containing two hidden layers with ReLU activation functions
- Uses [DET] tokens to bind objects for final predictions, avoiding additional 2D inductive biases and task-specific heuristics
- No additional convolutions or performance-oriented design inspired by modern CNN architectures

Training Procedures:
- Pre-trained on ImageNet-1k dataset using a data-efficient training strategy
- Fine-tuned on COCO object detection benchmark
- Initialized from ImageNet-1k pre-trained weights, except for MLP heads and [DET] tokens which are randomly initialized
- Trained on a single node with 8 × 12G GPUs
- Learning rate: 2.5 × 10^-5, batch size: 8, cosine learning rate decay, [More Information Needed] weight decay
- Input images resized with the shortest side between 480 and 800 pixels, longest side at most 1333 pixels
- Random crop augmentations applied during training
- Uses 100 [DET] tokens
- Loss function and weights kept the same as DETR
- No dropout or stochastic depth applied during fine-tuning

Parameters:
[More Information Needed]

Important Disclaimers:
- The model is designed to reveal the versatility and transferability of pre-trained Transformers from image recognition to object detection in a pure sequence-to-sequence manner, with minimal knowledge about the input spatial structure.
- The intentionally simple setup is not designed for better detection performance, but to uncover characteristics of the Transformer family in object detection.

For more information or updates, please contact the project organizer at [More Information Needed].

- **Developed by:** Yuxin Fang; Bencheng Liao; Xinggang Wang; Jiemin Fang; Jiyang Qi; Rui Wu; Jianwei Niu; Wenyu Liu
- **Funded by:** Based on the provided references, the following organizations fund the YOLOS project which includes the hustvl/yolos-tiny model:

NSFC (National Natural Science Foundation of China) under Grant No. 61876212, No. 61733007, and No. 61773176
Zhejiang Laboratory under Grant 2019NB0AB02

The references do not provide more specific information about funding for the hustvl/yolos-tiny model variant in particular. To fully answer this question, [More Information Needed] from the model developers about funding sources for hustvl/yolos-tiny.
- **Shared by:** Based on the provided references, the main contributors to the YOLOS model are:

Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, and Wenyu Liu

The references mention that the work was supported by NSFC grants and the Zhejiang Laboratory. The model builds upon prior works like DETR, DeiT, DINO and the timm library.

However, there is no explicit information provided about who specifically made the hustvl/yolos-tiny model variant available as a GitHub repo. More information would be needed to determine that.
- **Model type:** The hustvl/yolos-tiny model is a tiny-sized object detection model based on the Vision Transformer (ViT) architecture, pre-trained on the ImageNet-1k dataset using a data-efficient training strategy and fine-tuned on the COCO object detection benchmark.
- **Language(s):** The hustvl/yolos-tiny model processes natural images and does not use or process natural human language.
- **License:** [More Information Needed]

The provided references do not contain any information about the license being used for the hustvl/yolos-tiny model. More details would be needed from the model developers to determine the specific license that applies to this model.
- **Finetuned from model:** Based on the provided references, the hustvl/yolos-tiny model is likely fine-tuned from a pre-trained ViT (Vision Transformer) model, as evidenced by these key points:

1. Reference 2 mentions studying "the effects of different pre-training strategies (both label-supervised and self-supervised) when transferring ViT (DeiT-Ti and DeiT-S) from ImageNet-1k to the COCO object detection benchmark via YOLOS."

2. Reference 5 states: "Here we perform a preliminary transfer learning experiment on YOLOS-S using MoCo-v3 [13] and DINO [11] self-supervised pre-trained ViT weights in Tab. 3."

However, the exact name and link to the base model used for fine-tuning hustvl/yolos-tiny are not explicitly provided in the given references. To complete this information, [More Information Needed] from the model developers or additional documentation.
### Model Sources

- **Repository:** https://github.com/hustvl/YOLOS
- **Paper:** https://arxiv.org/pdf/2106.00666.pdf
- **Demo:** [More Information Needed]

The references provided do not contain a link to a demo of the hustvl/yolos-tiny model. The information is mainly about the model design, contributions, and where to find the code and pre-trained models (on GitHub at https://github.com/hustvl/YOLOS). However, there is no mention of a specific demo link for the tiny variant of the YOLOS model.
## Uses

### Direct Use

Based on the provided references, there is no direct information on how the hustvl/yolos-tiny model can be used without fine-tuning, post-processing, or plugging into a pipeline. The references mention that:

- The model is pre-trained on ImageNet-1k and then fine-tuned on COCO for object detection (Reference 2, 10). 
- The model closely follows the original ViT architecture and is optimized for object detection (Reference 6, 9).
- The model uses detector heads implemented as MLPs for classification and bounding box regression (Reference 7).

However, there are no specific details on using the model out-of-the-box without any additional steps. To provide a complete answer, [More Information Needed] on:

- The exact input format and preprocessing steps required by the model
- The output format of the model predictions 
- Whether the model can perform object detection on arbitrary images without any fine-tuning or additional processing steps

Without these details, providing example code or instructions for direct usage is not possible based solely on the given references. More specific documentation on the hustvl/yolos-tiny model API and intended usage would be needed to address this question fully.

### Downstream Use

The hustvl/yolos-tiny model can be fine-tuned on the COCO object detection benchmark to perform object detection tasks. As mentioned in the references, the model is initialized with ImageNet-1k pre-trained weights, except for the MLP heads for classification & bounding box regression and the [DET] tokens, which are randomly initialized.

To fine-tune YOLOS-Ti on COCO:
[More Information Needed]

After fine-tuning, the YOLOS-Ti model can be integrated into various applications that require object detection capabilities, such as:
- Autonomous driving systems
- Surveillance and security systems
- Robotics and industrial automation
- Medical imaging analysis

The tiny-sized YOLOS model achieves impressive performance compared to well-established CNN object detectors while being competitive in terms of FLOPs and FPS. This makes it a promising choice for applications that require real-time object detection on resource-constrained devices.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the YOLOS-Tiny model could be misused and what users should not do:

The model should not be used for safety-critical object detection applications like autonomous driving or medical diagnosis, as the references do not provide sufficient information on the model's reliability and failure modes in such high-stakes domains. More rigorous testing would be needed first. [More Information Needed]

Users should not assume the model is unbiased or fair in its detections across all object categories, regions, and sizes. The references suggest the [DET] tokens specialize in certain areas, so the model may have blind spots. A more thorough fairness evaluation across diverse datasets would be required to understand biases. [More Information Needed]

The model should not be deployed as-is for production object detection systems, as the references frame YOLOS more as an exploratory research platform to study Transformers rather than an industrial-grade detector. Users ought to properly test and validate the model's fitness for their specific use case.

In general, the model should only be used by those with sufficient ML expertise to understand its limitations. It should not be presented to lay users as a fully reliable off-the-shelf object detector without clear communication of its experimental nature based on the references provided. More information would be needed on intended uses and performance limitations to define misuse cases in greater detail.

### Bias, Risks, and Limitations

Based on the provided references about the YOLOS-Tiny model, here are some known or foreseeable issues, limitations, and potential harms:

Technical Limitations:
- The detection results of YOLOS are far from state-of-the-art performance (Ref 1). More work is needed to improve accuracy.
- YOLOS still requires a significant amount of transfer learning (150 epochs) to adapt the pre-trained ViT for object detection (Ref 1). Reducing the fine-tuning costs is an area for improvement.
- The model's performance on multi-scale inputs during training or inputs with varying sizes/aspect ratios during inference is not discussed. Handling these scenarios may require adjusting the positional encodings (Ref 3).

Sociotechnical Considerations:
- [More Information Needed] on potential negative societal impacts or misuse cases of the object detection capabilities.
- [More Information Needed] regarding fairness evaluations across different demographics and mitigation of any biases.
- The environmental and financial costs of the compute resources required to train and deploy the model are not provided. Efficiency improvements would help reduce these impacts (Ref 8).

Potential Misunderstandings:
- YOLOS is primarily designed to study the characteristics and capabilities of Transformer architectures for object detection, not to surpass CNN-based detectors in efficiency metrics like FLOPs and FPS (Ref 9).
- The model's strong performance compared to tiny CNN detectors (Ref 8, 10) could be misinterpreted as overall superiority, but its full potential and limitations at larger scales are unknown.

To mitigate these issues, more research is needed to improve YOLOS' accuracy, efficiency, and robustness. Societal impact evaluations and cost-benefit analyses should guide responsible development and deployment. Clear communication of the model's intended use cases and limitations is also important.

### Recommendations

Based on the provided references about the YOLOS-Tiny model, here are some recommendations regarding foreseeable issues:

1. Model size and efficiency: The model uses intermediate positional encodings (PEs) for object detection, which can be expensive in terms of model size (Ref 2). Consider optimizing the PE usage to reduce redundancy and improve efficiency (Ref 6).

2. Transfer learning and adaptation: The model requires a significant amount of transfer learning (150 epochs) to adapt the pre-trained ViT for object detection, and the results are not yet on par with state-of-the-art (Ref 1). Further research is needed to improve the transfer learning process and reduce the required labeled data (Ref 3, 7).

3. Inductive biases and priors: While the model closely follows the original ViT architecture, it's important to consider the right inductive biases and priors for the target task of object detection (Ref 7, 10). [More Information Needed] on specific architectural optimizations for improved performance.

4. Sensitivity to pre-training: The object detection results are sensitive to the pre-training scheme, and the performance is not saturated (Ref 11). The model can serve as a benchmark to evaluate different pre-training strategies for ViT in object detection.

5. Societal impact: [More Information Needed] on the potential societal implications of the model, such as fairness, bias, and privacy considerations, as well as any intended use cases and limitations.

Overall, while YOLOS-Tiny shows promising results compared to tiny-sized CNN detectors (Ref 9), further research and optimization are needed to address the above issues and improve its performance and efficiency for real-world applications.

## Training Details

### Training Data

The hustvl/yolos-tiny model is pre-trained on the ImageNet-1k dataset for 300 epochs, and then fine-tuned on the COCO object detection benchmark for 150 epochs. [More Information Needed] for details on data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the hustvl/yolos-tiny model:

Tokenization:
The input patch size for all YOLOS models, including YOLOS-Tiny, is 16 × 16.

Resizing:
For the tiny model, [More Information Needed] about the specific input image resizing strategy during fine-tuning on the COCO object detection benchmark. The references only provide resizing details for the small and base models.

Other preprocessing details:
During pre-training on ImageNet-1k, data augmentation techniques such as RandAugment and random erasing are used, as implemented in the timm library.

[More Information Needed] about other specific preprocessing steps for the hustvl/yolos-tiny model.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the hustvl/yolos-tiny model:

Pre-training on ImageNet-1k:
- Parameters initialized with truncated normal distribution
- Optimizer: AdamW
- Learning rate: 1 × 10^-3
- Batch size: 1024
- Learning rate decay: cosine 
- Weight decay: 0.05
- Data augmentation: Rand-Augment and random erasing (implemented by timm library)
- [More Information Needed] on number of training epochs

Fine-tuning on COCO object detection benchmark:
- Parameters initialized from ImageNet-1k pre-trained weights, except for:
  - MLP heads for classification & bounding box regression 
  - One hundred [DET] tokens (randomly initialized)
- Training setup: single node with 8 × 12G GPUs
- Learning rate: 2.5 × 10^-5
- Batch size: 8
- Learning rate decay: cosine
- [More Information Needed] on weight decay and number of training epochs
- Input image processing:
  - Resize shortest side to be between 480 and 800 pixels, longest side at most 1333 pixels
  - Apply random crop augmentations during training
- Number of [DET] tokens: 100
- Loss function and weights kept the same as DETR
- No dropout or stochastic depth applied during fine-tuning

#### Speeds, Sizes, Times

Based on the provided references about the YOLOS-Tiny model (hustvl/yolos-tiny), here is the available information relevant to the question:

Throughput:
- FLOPs and FPS data are measured over the first 100 images of COCO val split during inference. FPS is measured with batch size 1 on a single 1080Ti GPU. (Reference 5)
- Exact FLOPs and FPS values for YOLOS-Tiny: [More Information Needed]

Start or end time:
[More Information Needed]

Checkpoint sizes:
[More Information Needed]

Other details:
- YOLOS-Tiny is trained on a single node with 8 × 12G GPUs. The learning rate is 2.5 × 10^-5 and batch size is 8. (Reference 2)
- YOLOS-Tiny achieves impressive performance compared with well-established and highly-optimized CNN object detectors. It is strong in AP and competitive in FLOPs & FPS. (Reference 1)

To provide more complete information about throughput, start/end time, and checkpoint sizes, additional details would be needed beyond what is given in the references.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the hustvl/yolos-tiny model is evaluated on the following benchmark and dataset:

COCO object detection benchmark: The model is fine-tuned and evaluated on the COCO object detection benchmark (References 1, 2, 6, 10). COCO is a challenging dataset for object detection tasks.

The references do not provide specific information about other benchmarks or datasets used for evaluating the hustvl/yolos-tiny model. More information would be needed to determine if the model was evaluated on any additional datasets or benchmarks.

#### Factors

Based on the provided references about the YOLOS object detection model, here are some key characteristics that could influence the model's behavior and performance:

Domain and Context:
- YOLOS is designed for the object detection task on the COCO benchmark dataset. Its performance and behavior in other visual domains or on datasets with significantly different characteristics than COCO would need further evaluation. [More Information Needed]

Model Architecture and Design:
- YOLOS closely follows the original Vision Transformer (ViT) architecture with minimal modifications for object detection. This intentionally simple design is meant to reveal the characteristics of Transformers in object detection in an unbiased way, rather than being optimized for peak performance.
- The detection performance of YOLOS is quite sensitive to the pre-training scheme used for the underlying ViT. Different pre-training strategies (supervised, self-supervised) can significantly impact results.
- Compared to highly-optimized tiny CNN object detectors, YOLOS achieves competitive performance in terms of AP, FLOPs and FPS. It could serve as a promising starting point for Transformer-based model scaling in object detection.

Handling of Variable Input Sizes:
- Unlike image classification, object detection benchmarks usually have variable image resolutions and aspect ratios. YOLOS has to adapt the position embeddings (PE) in the underlying ViT to handle this.
- The references study two types of PE settings, but [More Information Needed] on the details and impact of those settings.
- About 1/3 of the model weights are used for providing positional information via the PEs. While this is expensive in terms of model size, the authors argue it allows the PEs to learn positional relations from scratch without injecting additional inductive biases.

Population Subgroups and Disaggregated Evaluation:
- [More Information Needed] The provided references do not contain information on YOLOS' performance across different population subgroups or factors that could reveal disparities. Disaggregated evaluation across relevant factors would be needed to uncover and understand any such issues.

#### Metrics

Based on the provided references, the key evaluation metrics for the hustvl/yolos-tiny model on the COCO object detection benchmark would likely be:

- Average Precision (AP) - the primary metric used to measure object detection performance. References 9 and 10 mention AP results.

- FLOPs - to measure computational complexity. References 3, 4 and 9 discuss FLOPs.

- FPS (Frames Per Second) - to measure inference speed. Reference 9 measures FPS.

The references don't provide full details on the specific AP metrics used (e.g. AP@[.5:.95], AP50, etc.) or the tradeoffs considered between them. More information would be needed on the exact AP metrics reported and the relative prioritization of AP, FLOPs and FPS for evaluating the model's speed-accuracy tradeoffs.

### Results

Based on the provided references, here are the evaluation results for the hustvl/yolos-tiny model:

Model: YOLOS-Ti (tiny-sized YOLOS model)

Performance:
- Achieves impressive performance compared to well-established and highly-optimized CNN object detectors (Reference 1)
- Can serve as a promising model scaling start point (Reference 1)
- AP (Average Precision): 28.7 (Reference 8)

Computational Efficiency:
- Competitive in FLOPs and FPS, even though Transformer is not intentionally designed to optimize these factors (Reference 1)
- Input resolution for inference: 512 (Reference 8)

Pre-training:
- Initialized with weights from DeiT (Reference 6)
- Pre-trained for 300 epochs on ImageNet-1k dataset (Reference 8)

[More Information Needed] for additional factors and metrics not directly mentioned in the provided references, such as robustness, fairness, and domain shift performance.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the hustvl/yolos-tiny model:

The tiny-sized YOLOS model achieves impressive performance compared to well-established and highly-optimized CNN object detectors. YOLOS-Ti is strong in Average Precision (AP) and competitive in FLOPs and FPS, even though Transformer is not intentionally designed to optimize these factors. 

YOLOS-Ti can serve as a promising starting point for model scaling. The performance numbers are meaningful, as YOLOS is designed to precisely reveal the transferability of Vision Transformers (ViT) in object detection, rather than being purposefully designed for better performance.

[More Information Needed] on the exact AP, FLOPs and FPS values for the hustvl/yolos-tiny model specifically. The references discuss YOLOS-Ti in general but do not provide the metrics for this particular variant.

Overall, the evaluation demonstrates that Transformer architectures like YOLOS can be successfully applied to challenging object detection tasks, with minimal modifications. This reveals the versatility and generality of Transformers.

## Model Examination

Based on the provided references, here is the experimental section on explainability and interpretability for the hustvl/yolos-tiny model:

We conducted a qualitative analysis of the self-attention maps of the [DET] tokens in the last layer of YOLOS-S, a model similar to YOLOS-Tiny. The visualizations showed that different self-attention heads focus on different patterns and locations, with some being more interpretable than others [1, 2].

A quantitative study revealed a strong negative linear correlation (ρ = -0.80) between the cosine similarity of [DET] token pairs and the Euclidean distances of their corresponding predicted bounding box centers [3, 4]. This suggests that [DET] tokens with high cosine similarity tend to make predictions for nearby objects.

However, there was no strong linear correlation (ρ = -0.07) between the cosine similarity of [DET] token pairs and the cosine similarity of their corresponding output features from the classifier [5].

[More Information Needed] on the specific architecture and design choices of YOLOS-Tiny that contribute to its interpretability and explainability.

YOLOS-Tiny achieves competitive performance compared to well-established and highly-optimized CNN object detectors, making it a promising starting point for model scaling [7].

While the performance of YOLOS-Tiny may not be state-of-the-art, it is designed to reveal the transferability of Vision Transformers (ViT) in object detection tasks with minimal modifications [8].

## Environmental Impact

- **Hardware Type:** Based on the provided references, the YOLOS models are trained on a single node with 8 × 12G GPUs, as stated in reference 1:

"We train YOLOS on a single node with 8 × 12G GPUs."

Therefore, the hardware type that the model hustvl/yolos-tiny is trained on is:

A single node with 8 GPUs, each having 12GB of memory.
- **Software Type:** Based on the references provided, the YOLOS models, including hustvl/yolos-tiny, are trained using:

Python version 3.6
PyTorch 1.5+
torchvision 0.6+

The codebase also requires installing additional libraries:
```setup
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
- **Hours used:** [More Information Needed]

The provided references do not contain specific information about the training time for the hustvl/yolos-tiny model. More details would be needed from the model developers or training logs to determine the exact amount of time used for training this particular model.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the hustvl/yolos-tiny model. More information would be needed from the model developers or training logs to determine the cloud provider.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the carbon emissions from training the hustvl/yolos-tiny model. To accurately estimate the carbon footprint, more details would be needed such as:

- The hardware used for training (GPU models, number of GPUs, etc.)
- Total training time 
- Energy consumption of the hardware
- Carbon intensity of the electricity used

Without these specifics, it is not possible to give a reliable estimate of the carbon emissions from the model training process based solely on the information provided in the references.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
YOLOS closely follows the original ViT architecture, and is optimized for object detection. The key architectural changes from ViT to YOLOS are:
1. YOLOS replaces one [CLS] token for image classification in ViT with one hundred [DET] tokens for object detection. 
2. The classification and bounding box regression heads are implemented by one MLP with separate parameters containing two hidden layers with intermediate ReLU non-linearity activation functions.
3. YOLOS adds minimal additional inductive biases upon ViT, with no non-degenerated (e.g., 3 × 3 or other non 1 × 1) convolutions.

Model Objective:
The objective of YOLOS is to unveil the versatility and transferability of pre-trained canonical Transformer from image recognition to the more challenging object detection task, with minimal modifications, region priors, and task-specific inductive biases injected. 

YOLOS aims to perform object detection from a pure sequence-to-sequence perspective, without re-interpreting the output sequence to 2D feature maps. It investigates whether Transformers can perform 2D object-and region-level recognition with minimal knowledge about the 2D spatial structure.

[More Information Needed] on specific details of the hustvl/yolos-tiny model variant, such as model size, dataset used for pre-training and fine-tuning, and performance metrics.

### Compute Infrastructure

Based on the provided references, here is the information about the compute infrastructure used for the hustvl/yolos-tiny model:

For pre-training on ImageNet-1k, the following settings were used:
- Batch size: 1024
- Learning rate: 1 × 10^-3 
- Learning rate decay: cosine
- Weight decay: 0.05
- Data augmentation: Rand-Augment and random erasing implemented by timm library

For fine-tuning on COCO object detection benchmark:
- Single node with 8 × 12G GPUs
- Batch size: 8
- Learning rate: 2.5 × 10^-5
- Learning rate decay: cosine
- Weight decay: [More Information Needed]

The exact training command for YOLOS-Tiny is not provided in the references. More information is needed to specify the full compute infrastructure and hyperparameters used for training this specific model variant.

## Citation

```
@misc{yuxin-you,
    author = {Yuxin Fang and
              Bencheng Liao and
              Xinggang Wang and
              Jiemin Fang and
              Jiyang Qi and
              Rui Wu and
              Jianwei Niu and
              Wenyu Liu},
    title  = {You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection},
    url    = {https://arxiv.org/pdf/2106.00666.pdf}
}
```

