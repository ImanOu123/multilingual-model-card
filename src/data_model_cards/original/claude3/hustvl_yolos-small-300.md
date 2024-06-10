# Model Card for hustvl/yolos-small-300

The hustvl/yolos-small-300 model is an object detection model based on the Vision Transformer (ViT) architecture, which performs 2D object detection in a pure sequence-to-sequence manner by taking a sequence of fixed-sized non-overlapping image patches as input, with minimal modifications and inductive biases.

## Model Details

### Model Description

Here are the basic details about the model hustvl/yolos-small-300:

Model Architecture:
- Closely follows the original ViT (Vision Transformer) architecture, optimized for object detection
- Uses [DET] tokens as proxies for object representations, replacing the [CLS] token in ViT
- Classification and bounding box regression heads are implemented by one MLP with separate parameters, containing two hidden layers with ReLU activation functions
- Minimal additional inductive biases injected compared to the original ViT

Training Procedures:
- Pre-trained on ImageNet-1k dataset using data-efficient training strategy
- Parameters initialized with truncated normal distribution, optimized using AdamW
- Learning rate: 1 × 10^-3, batch size: 1024, cosine learning rate decay, weight decay: 0.05
- Data augmentation: RandAugment and random erasing
- Fine-tuned on COCO object detection benchmark
- Initialized from ImageNet-1k pre-trained weights, except for MLP heads and [DET] tokens (randomly initialized)
- Fine-tuning hyperparameters:
  - Learning rate: 2.5 × 10^-5
  - Batch size: 8
  - Cosine learning rate decay
  - Weight decay: [More Information Needed]
  - Trained on a single node with 8 × 12G GPUs

Model Parameters:
- Input patch size: 16 × 16
- Specific configuration details for YOLOS-S (Small):
  [More Information Needed]

Important Disclaimers:
- The intentionally simple setup is not designed for better detection performance, but to reveal characteristics of the Transformer family in object detection as unbiased as possible
- [More Information Needed]

For more information or updates, please contact the project organizer at [More Information Needed].

- **Developed by:** Yuxin Fang; Bencheng Liao; Xinggang Wang; Jiemin Fang; Jiyang Qi; Rui Wu; Jianwei Niu; Wenyu Liu
- **Funded by:** Based on the provided references, the project that developed the model hustvl/yolos-small-300 is in part supported by:

- NSFC (National Natural Science Foundation of China) under Grant No. 61876212, No. 61733007, and No. 61773176
- The Zhejiang Laboratory under Grant 2019NB0AB02

The references also thank Zhuowen Tu for valuable suggestions, but it's unclear if he directly funded the project.
- **Shared by:** Based on the references provided, the main contributors who made the YOLOS model (including hustvl/yolos-small-300) available online as a GitHub repo are:

Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, and Wenyu Liu

The paper was authored by Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu and Wenyu Liu, as stated in reference 3. Yuxin Fang and Bencheng Liao contributed equally to this work, and Xinggang Wang is the corresponding author, as mentioned in reference 6.

The code and pre-trained models are available at https://github.com/hustvl/YOLOS, as stated in reference 6.

[More Information Needed] on the specific individual contributions of each author to the GitHub repository.
- **Model type:** The hustvl/yolos-small-300 model is a label-supervised pre-trained Vision Transformer fine-tuned for object detection on the COCO dataset, using a pure sequence-to-sequence perspective with minimal modifications and inductive biases.
- **Language(s):** The model hustvl/yolos-small-300 processes natural images and does not use or process natural human language.
- **License:** [More Information Needed]

The provided references do not contain any information about the specific license being used for the hustvl/yolos-small-300 model. More details would be needed from the model developers to determine the license under which this model is being released.
- **Finetuned from model:** Based on the provided references, the model hustvl/yolos-small-300 is likely fine-tuned from a pre-trained ViT (Vision Transformer) model on ImageNet-1k for 300 epochs, as mentioned in this excerpt:

"We pre-train all the models for 300 epochs on ImageNet-1k with input resolution determined by the corresponding scaling strategies, and then fine-tune these models on COCO for 150 epochs."

However, the specific base model name and link are not provided in the given references. More information would be needed to definitively state the exact base model used.

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/hustvl/YOLOS
- **Paper:** https://arxiv.org/pdf/2106.00666.pdf
- **Demo:** [More Information Needed]

The references provided do not contain a direct link to a demo of the hustvl/yolos-small-300 model. The references mention that code and pre-trained models are available on GitHub at https://github.com/hustvl/YOLOS, but there is no specific mention of a demo for the yolos-small-300 variant.

To provide a link to a demo, more information would be needed, such as confirmation that a demo exists for this specific model and the URL where it can be accessed.
## Uses

### Direct Use

Based on the provided references, there is no direct information on how the hustvl/yolos-small-300 model can be used without fine-tuning, post-processing, or plugging into a pipeline. The references mention that:

- The model is pre-trained on ImageNet-1k for 300 epochs and then fine-tuned on COCO for 150 epochs (Reference 2). 
- The model closely follows the original ViT architecture and is optimized for object detection (References 5, 6, 9).
- The model is fine-tuned on the COCO object detection benchmark in a similar way as Carion et al. (Reference 10).

However, there are no specific details on using the model as-is without any modifications or additional steps.

[More Information Needed] A code snippet cannot be provided as there are no direct code block references in the given information.

To summarize, more information would be needed from the model developers or documentation to explain if and how the hustvl/yolos-small-300 model can be used out-of-the-box without fine-tuning, post-processing steps, or integration into a larger pipeline. The current references focus more on the model architecture and training process rather than deployment and usage.

### Downstream Use

The hustvl/yolos-small-300 model can be fine-tuned on the COCO object detection benchmark to adapt it for object detection tasks. As mentioned in the references, the pre-trained ViT model can be converted to a YOLOS detector with minimal modifications:

1. Replace the single [CLS] token used for image classification with one hundred [DET] tokens for object detection. 
2. Replace the image classification head with object detection heads (MLP heads for classification & bounding box regression).

The model is fine-tuned on COCO for 150 epochs, initializing all parameters from the ImageNet-1k pre-trained weights except for the newly added [DET] tokens and detection heads, which are randomly initialized.

[More Information Needed] for the specific fine-tuning hyperparameters and code snippet.

Once fine-tuned, the YOLOS model can be plugged into larger systems or applications that require object detection capabilities. It takes an image as input and outputs the detected object classes and bounding boxes.

[More Information Needed] for details on integrating the fine-tuned model into a larger ecosystem or application.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the YOLOS-small-300 model could be misused and what users should not do:

The model should not be used for real-time object detection in critical applications like autonomous driving or surveillance, as the references indicate the detection speed (FPS) is not optimized compared to purpose-built CNN detectors. More testing would be needed to ensure adequate real-time performance.

Users should not assume the model will perform well on object detection tasks significantly different from its training data (COCO dataset). The references discuss the significant training still required to adapt vision transformers like YOLOS to new tasks. Applying it to very different detection problems without finetuning could lead to poor results.

[More Information Needed] on the specific object categories and data the model was trained on, to better assess potential misuse cases around biased or unethical applications (e.g. using it for unauthorized facial recognition).

The references do not provide enough information to comment on security vulnerabilities or risks of the model being used to generate harmful synthetic media. More details on the model architecture and training would be needed.

In general, the model should only be used for research purposes in object detection, not deployed to production systems without extensive further testing. Users must critically examine its suitability and performance for their specific use case.

### Bias, Risks, and Limitations

Based on the provided references about the YOLOS model, here are some potential issues and limitations:

Technical limitations:
- The detection results of YOLOS are far from state-of-the-art, suggesting room for improvement in adapting pre-trained ViT models to object detection tasks (Reference 1)
- Using intermediate positional encodings for object detection is expensive in terms of model size, with about 1/3 of model weights used only for positional information (Reference 3)
- Adjusting positional encodings is needed to handle multi-scale inputs during training or different input sizes/aspect ratios during inference (Reference 3)
- Pre-trained computer vision models like YOLOS still require a lot of labeled data to transfer to downstream tasks, unlike language models that can perform few-shot or zero-shot learning (Reference 4)

Sociotechnical limitations: 
- [More Information Needed] The references do not directly discuss sociotechnical limitations.

Potential misunderstandings:
- YOLOS is designed to reveal characteristics of Transformers in object detection in an unbiased way, not necessarily to achieve the best detection performance (Reference 9)

Foreseeable harms:
- [More Information Needed] The references do not directly discuss foreseeable harms of the model.

In summary, the key limitations seem to be the technical challenges in efficiently adapting pre-trained vision Transformers to downstream tasks like object detection while minimizing additional data and inductive biases. More information would be needed to assess potential sociotechnical limitations and harms.

### Recommendations

Based on the provided references about the YOLOS model, here are some recommendations regarding foreseeable issues with the hustvl/yolos-small-300 model:

1. The model's object detection performance is likely far from state-of-the-art, as the goal was to precisely reveal ViT transferability rather than optimize for best results. More work is needed to improve detection accuracy if the model is to be used in production settings.

2. Detection results may be quite sensitive to the pre-training scheme used. Different supervised or self-supervised pre-training strategies should be carefully evaluated to ensure robust performance. The model could serve as a benchmark to compare pre-training approaches.

3. [More Information Needed] on the specific dataset and metrics used to train and evaluate the yolos-small-300 variant. This information is important to properly characterize model performance and limitations.

4. Computational efficiency and model size may be a concern, as the references indicate using intermediate position embeddings is expensive in terms of model size. Practical deployment will require analyzing inference speed and resource usage.

5. [More Information Needed] on failure modes and error analysis of the model's predictions. Understanding where and why the model makes mistakes is critical to assessing risks and guiding further improvements.

6. As an object detection model, the societal impacts and potential for misuse should be carefully considered, such as privacy concerns from unauthorized surveillance or inappropriate applications in sensitive domains. Responsible disclosure and deployment practices are advised.

## Training Details

### Training Data

The model hustvl/yolos-small-300 is pre-trained for 300 epochs on the ImageNet-1k dataset, and then fine-tuned on the COCO object detection benchmark for 150 epochs. During inference on COCO, the input shorter size is 800 for this small model.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the YOLOS-S (Small) model:

Input Resizing:
- For small models, the input images are resized such that the shortest side is at least 480 and at most 800 pixels while the longest side is at most 1333 pixels. (Reference 3)

Tokenization:
- The input patch size for all YOLOS models is 16 × 16. (Reference 1)
- [More Information Needed] about specific tokenization details.

Data Augmentation:
- Random crop augmentations are applied during training, following Carion et al. (Reference 3)
- Rand-Augment and random erasing implemented by the timm library are used for data augmentation during pre-training. (Reference 2)
- [More Information Needed] about other specific data augmentation techniques used.

Other Preprocessing Details:
- [More Information Needed]

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the hustvl/yolos-small-300 model:

Pre-training on ImageNet-1k:
- Initialization: Truncated normal distribution 
- Optimizer: AdamW
- Learning rate: 1 × 10^-3
- Batch size: 1024
- Learning rate decay: Cosine 
- Weight decay: 0.05
- Data augmentation: RandAugment and random erasing

Fine-tuning on COCO object detection:
- Initialization: ImageNet-1k pre-trained weights, except for randomly initialized MLP heads for classification & bounding box regression and 100 [DET] tokens
- Hardware: Single node with 8 × 12G GPUs
- Learning rate: 2.5 × 10^-5 
- Batch size: 8
- Learning rate decay: Cosine
- Weight decay: [More Information Needed]
- Input resolution: Shortest side between 480 and 800 pixels, longest side at most 1333 pixels
- Data augmentation: Random crop
- Number of [DET] tokens: 100
- Loss function and weights: Same as DETR
- Regularization: No dropout or stochastic depth during fine-tuning

#### Speeds, Sizes, Times

Based on the provided references about the YOLOS model, here are the details I can provide about the hustvl/yolos-small-300 model:

Throughput:
- FLOPs and FPS data are measured over the first 100 images of COCO val split during inference. FPS is measured with batch size 1 on a single 1080Ti GPU. (Reference 5)
- [More Information Needed] for the specific FLOPs and FPS values of the yolos-small-300 model.

Training Time:
- All the models are pre-trained for 300 epochs on ImageNet-1k, and then fine-tuned on COCO for 150 epochs. (Reference 6) 
- [More Information Needed] for the specific start and end training times.

Checkpoint Sizes:
- [More Information Needed]

Other Details:
- YOLOS-Ti (tiny-sized model) can serve as a promising model scaling start point based on comparisons with other tiny-sized CNN detectors. (Reference 1)
- The model is fine-tuned on COCO with a learning rate of 2.5 × 10−5, batch size of 8, cosine learning rate decay, and weight decay. Training is done on 8 × 12G GPUs. (Reference 2)
- The model closely follows the original ViT architecture and is optimized for object detection similar to DETR. (Reference 8)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model hustvl/yolos-small-300 is evaluated on the COCO object detection benchmark. Specifically, reference 1 mentions that "We fine-tune all YOLOS models on COCO object detection benchmark [36] in a similar way as Carion et al. [10]." and reference 8 states "quite competitive performance on the challenging COCO object detection benchmark, e.g., YOLOS-Base directly adopted from BERT-Base architecture can obtain 42.0 box AP on COCO val."

No other benchmarks or datasets are explicitly mentioned for evaluating the hustvl/yolos-small-300 model in the given references.

#### Factors

Based on the provided references about the YOLOS object detection model, here are some key characteristics that may influence the model's behavior:

Domain and Context:
- The model is designed for the task of object detection in images, as evidenced by references to the COCO object detection benchmark and comparisons to CNN-based object detection models.
- It is unclear what specific image domains (e.g. natural images, medical images, satellite imagery) the model was trained and evaluated on. [More Information Needed]

Population Subgroups:
- The references mention that the [DET] tokens in YOLOS specialize in detecting objects of certain sizes (e.g. some are sensitive to small objects while others focus on large objects). This suggests the model's performance could vary for images containing objects of different scales.
- No other clear information is provided about performance on different population subgroups, such as performance across different object categories, lighting conditions, occlusion levels, etc. Ideally results should be disaggregated across such factors, but [More Information Needed].

Other Factors:
- The model's detection performance appears quite sensitive to the pre-training scheme used for the Vision Transformer backbone. Different pre-training datasets and strategies are likely to impact results.
- Changing input image resolutions and aspect ratios between pre-training and fine-tuning requires care in handling the position embeddings. The approach used here could affect model size and performance.
- As a pure Vision Transformer model without explicit region priors or inductive biases for object detection, the model may have different characteristics than CNN-based detectors. However, the references suggest it still learns to attend to relevant object regions and sizes.

In summary, key factors that may influence model behavior include the image domains used for training/evaluation, object sizes and scales, pre-training approach, and unique properties of the Vision Transformer architecture. More disaggregated evaluation results would help better characterize model performance. The provided information focuses mainly on model design and high-level behavior, so [More Information Needed] to draw strong conclusions about performance across different subgroups and contexts.

#### Metrics

Based on the provided references, the key evaluation metrics for the YOLOS model variants, including hustvl/yolos-small-300, are:

1. AP (Average Precision) on the COCO object detection benchmark. This is the primary metric used to compare YOLOS performance against other object detectors like DETR.

2. FLOPs (Floating Point Operations) to measure the computational complexity of the model. This is important to consider the tradeoff between model performance and efficiency.

3. FPS (Frames Per Second) to measure the inference speed of the model, which is crucial for real-time applications. FPS is measured with batch size 1 on a single 1080Ti GPU.

The references discuss comparisons between different YOLOS variants and DETR in terms of AP, FLOPs, and FPS. However, there is no specific information about the tradeoffs or error analysis for the hustvl/yolos-small-300 model. [More Information Needed] on the specific tradeoffs and error analysis for this particular model variant.

### Results

Based on the provided references, here is the evaluation information I can provide for the hustvl/yolos-small-300 model:

Model Architecture:
- The model uses the YOLOS-S (YOLOS-Small) architecture, which corresponds to the DeiT-S (DeiT-Small) architecture.
- It performs width scaling on the YOLOS-Ti (YOLOS-Tiny) model. 
- The input patch size is 16x16.

Pre-training:
- Pre-trained on ImageNet-1k for 300 epochs using weights from DeiT (Facebook Research).

Object Detection Performance on COCO:
- Achieves 36.1 AP (Average Precision) on COCO val split.

[More Information Needed] for:
- Detailed model configurations like number of parameters, FLOPs, layers, channels, etc.
- Comparison of performance, speed and model size against other YOLO and DETR models.
- Evaluation metrics like AP across different object sizes, AR, FPS on specific hardware.

To evaluate the model on COCO:
[More Information Needed] 
The exact evaluation command is not provided in the references. A template is given but path placeholders need to be filled based on the user's setup.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model hustvl/yolos-small-300:

The YOLOS-S (Small) model, which the hustvl/yolos-small-300 is likely based on, performs worse than a similar-sized DETR model by 0.8 AP, despite having more computations. The performance gain of YOLOS-S with dwr scaling over the DETR counterpart cannot be clearly explained.

[More Information Needed] on the specific evaluation metrics and results for the hustvl/yolos-small-300 model, as the references do not provide those details. The references compare YOLOS variants in general to DETR models, but lack specifics on the -small-300 version.

To visualize the model's predictions:
```
python cocoval_predjson_generation.py --coco_path /path/to/coco --batch_size 1 --backbone_name small --eval --eval_size 800 --init_pe_size 512 864 --mid_pe_size 512 864 --resume /path/to/yolos-s-model.pth --output_dir ./visualization
```
This will generate a JSON file with the model's predictions on the COCO validation set. [More Information Needed] on the exact performance of those predictions for the -small-300 variant specifically.

In summary, more details on the evaluation metrics and results are needed to fully characterize the performance of the hustvl/yolos-small-300 model. The provided references give useful context on YOLOS models in general, but lack specifics on this particular variant.

## Model Examination

Based on the provided references, here is the experimental section on explainability and interpretability for the hustvl/yolos-small-300 model:

We analyzed the self-attention patterns of the [DET] tokens in the last layer of the YOLOS-S model pre-trained for 300 epochs on ImageNet-1k. Visualizations show that different attention heads focus on different patterns and locations, with some being more interpretable than others. 

Comparing to a YOLOS-S model pre-trained for 200 epochs, which achieves the same AP of 36.1, the 300 epoch model exhibits different attention map patterns. This indicates that attention maps vary between models, even with similar performance.

Quantitatively, we found a strong negative Pearson correlation (ρ=-0.80) between the cosine similarity of [DET] token pairs and the L2 distance between their predicted bounding box centers, averaged over the COCO val set. This means [DET] tokens with high similarity tend to make predictions for nearby locations.

[More Information Needed] on any code or architecture details, as the references do not contain direct code blocks related to explainability. 

In summary, our YOLOS-S model shows interpretable self-attention patterns that differ from other checkpoints and correlate with the spatial structure of predictions. Further work is needed to use these insights to improve object detection performance and robustness.

## Environmental Impact

- **Hardware Type:** The model hustvl/yolos-small-300 is trained on a single node with 8 × 12G GPUs, according to the information provided in the references:

"We train YOLOS on a single node with 8 × 12G GPUs."
- **Software Type:** Based on the provided references, the YOLOS models are trained using:

- Python version 3.6
- PyTorch 1.5+
- torchvision 0.6+

The references also mention installing additional dependencies:

```setup
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

However, there is no specific information about the exact software versions used for training the hustvl/yolos-small-300 model. More details would be needed to definitively state the specific software setup for this particular model variant.
- **Hours used:** Based on the provided references, the YOLOS-S model with 300 epoch pretrained DeiT-S was trained for 150 epochs during the fine-tuning stage on the COCO object detection benchmark.

The exact amount of time needed to train the model is [More Information Needed], as it depends on factors such as the hardware used (e.g., GPU specifications) and the computational resources available, which are not specified in the given references.
- **Cloud Provider:** Based on the provided references, there is no direct information about the cloud provider used for training the hustvl/yolos-small-300 model. The references mention training commands and configurations, but do not specify the cloud platform.

[More Information Needed]
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted when training the hustvl/yolos-small-300 model. To accurately estimate the carbon emissions, additional details would be needed, such as:

- The hardware used for training (GPU models, number of GPUs, etc.)
- The total training time or number of GPU hours
- The energy consumption of the hardware during training
- The carbon intensity of the electricity grid where the training was performed

Without these specifics, it is not possible to provide a reliable estimate of the carbon emissions associated with training this particular model based solely on the given references.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
YOLOS closely follows the original ViT (Vision Transformer) architecture with minimal modifications for object detection. The key changes are:
1. YOLOS replaces the single [CLS] token used for image classification in ViT with 100 [DET] tokens for object detection.
2. The classification head of ViT is replaced with an MLP containing two hidden layers with ReLU activation for both object classification and bounding box regression.
3. The position embeddings are adapted to handle variable input resolutions and aspect ratios in object detection.
YOLOS avoids injecting inductive biases and prior knowledge about 2D spatial structure during label assignment.

Model Objective:
The objective of YOLOS is to investigate whether Transformers can perform 2D object-level recognition from a pure sequence-to-sequence perspective with minimal knowledge about the 2D spatial structure. YOLOS aims to reveal the characteristics and capabilities of the Transformer architecture for object detection in an unbiased manner, rather than being optimized for maximum performance.

[More Information Needed] on the specific details of the hustvl/yolos-small-300 model variant, such as the exact model size, pre-training dataset, and performance metrics.

### Compute Infrastructure

Based on the provided references, here is the information about the compute infrastructure used for the hustvl/yolos-small-300 model:

For fine-tuning YOLOS models on COCO object detection benchmark:
- Single node with 8 × 12G GPUs
- Batch size: 8
- Learning rate: 2.5 × 10^-5

For pre-training YOLOS / ViT models on ImageNet-1k dataset:
- [More Information Needed] (specific compute infrastructure not provided)
- Batch size: 1024 
- Learning rate: 1 × 10^-3

To train the YOLOS-S model with 300 epoch pretrained Deit-S, the following command can be used:

```
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --coco_path /path/to/coco
    --batch_size 1 \
    --lr 2.5e-5 \
    --epochs 150 \
    --backbone_name small \
    --pre_trained /path/to/deit-small-300epoch.pth\
    --eval_size 800 \
    --init_pe_size 512 864 \
    --mid_pe_size 512 864 \
```

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

