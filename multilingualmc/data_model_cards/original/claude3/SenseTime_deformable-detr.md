# Model Card for SenseTime/deformable-detr

Deformable DETR is an efficient and fast-converging end-to-end object detector that mitigates the high complexity and slow convergence issues of DETR via a novel sampling-based efficient attention mechanism called deformable attention modules.

## Model Details

### Model Description

Model Card for SenseTime/deformable-detr

Model Details:
- Deformable DETR is an end-to-end object detector that replaces the Transformer attention modules in DETR with deformable attention modules. This makes the model more efficient and faster to converge.
- The deformable attention module is designed to process convolutional feature maps and has a complexity of O(2N_qC^2 + min(HWC^2, N_qKC^2)) when the number of key elements M_K is relatively small. 
- In the encoder, multi-scale feature maps are extracted from the output of stages C3 through C5 in a ResNet backbone. The decoder uses object queries to attend to the encoder output via cross-attention and to each other via self-attention.

Training:
- The model is trained on the COCO dataset for object detection.
- An ImageNet pre-trained ResNet-50 is used as the backbone. 
- Training largely follows the setup of DETR, using the Adam optimizer with a base learning rate of 2 x 10^-4, betas of (0.9, 0.999), and weight decay of 10^-4.
- Focal loss is used for bounding box classification, and the number of object queries is increased to 300.
- Models are trained for 50 epochs by default, with the learning rate decayed by 0.1 at the 40th epoch.

Parameters:
- M = 8 and K = 4 are used for the deformable attention modules by default.
- Parameters of the deformable Transformer encoder are shared across feature levels.

[More Information Needed] on the total number of parameters in the model.

Important Disclaimers:
[More Information Needed]

For questions about this model, please contact [More Information Needed].

- **Developed by:** Xizhou Zhu; Weijie Su; Lewei Lu; Bin Li; Xiaogang Wang; Jifeng Dai; Sensetime Research
- **Funded by:** Based on the references provided, the Deformable DETR project is funded by the following organizations:

1. National Key R&D Program of China (2020AAA0105200)
2. Beijing Academy of Artificial Intelligence
3. National Natural Science Foundation of China under grant No.U19B2044 and No.61836011

The work is done by researchers from SenseTime Research.
- **Shared by:** According to the references provided, the contributors who made the model SenseTime/deformable-detr available online as a GitHub repo are:

Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai from SenseTime Research.

The work was supported by the National Key R&D Program of China, Beijing Academy of Artificial Intelligence, and the National Natural Science Foundation of China.
- **Model type:** SenseTime/deformable-detr is an end-to-end object detection model that utilizes a novel deformable attention mechanism to efficiently process image features, trained in a supervised manner.
- **Language(s):** Based on the provided references, the SenseTime/deformable-detr model does not appear to use or process natural human language. The model is described as an end-to-end object detector that processes image feature maps using an efficient attention mechanism. [More Information Needed] on any natural language capabilities of this computer vision model.
- **License:** The SenseTime/deformable-detr project is released under the Apache 2.0 license. The link to the license is provided in the references: https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE
- **Finetuned from model:** Based on the provided references, Deformable DETR is not explicitly mentioned to be fine-tuned from another model. The references describe Deformable DETR as an improvement over the original DETR model by replacing the Transformer attention modules with deformable attention modules. However, there is no clear indication that Deformable DETR is fine-tuned from a specific base model.

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/fundamentalvision/Deformable-DETR
- **Paper:** https://arxiv.org/pdf/2010.04159.pdf
- **Demo:** [More Information Needed]

The references provided do not contain a direct link to a demo of the SenseTime/deformable-detr model. The references discuss the model architecture, performance, and provide a link to the code repository on GitHub, but do not mention a publicly accessible demo. More information would be needed from the model developers to determine if a demo link is available to include in the model card.
## Uses

### Direct Use

The SenseTime/deformable-detr model can be used for object detection without requiring fine-tuning, post-processing, or plugging into a pipeline. To evaluate the pre-trained Deformable DETR model on the COCO 2017 validation set, you can run the following command:

```bash
<path to config file> --resume <path to pre-trained model> --eval
```

You can also run distributed evaluation using the provided scripts:

```bash
./tools/run_dist_launch.sh
```

or

```bash
./tools/run_dist_slurm.sh
```

[More Information Needed] on how to use the model for inference on custom images or videos.

### Downstream Use

Deformable DETR can be fine-tuned for object detection tasks on custom datasets. To use it in a larger ecosystem or app, you would first fine-tune the model on your dataset, then use the fine-tuned model to generate object detections on new images.

The key advantages of using Deformable DETR are:
- It is an efficient and fast-converging end-to-end object detector, achieving better performance (especially on small objects) with 10x less training epochs compared to the original DETR
- The deformable attention modules enable processing image feature maps efficiently 
- It eliminates the need for hand-crafted components like anchor generation, rule-based training target assignment, and non-maximum suppression post-processing
- Variants like iterative bounding box refinement and two-stage Deformable DETR can further improve accuracy

To fine-tune Deformable DETR on a custom dataset:
[More Information Needed]

After fine-tuning, the model can be plugged into an application to detect objects in images. For example:
[More Information Needed]

The fine-tuned model would output bounding box coordinates and class labels for each detected object, which can then be used in the downstream application logic.

In summary, Deformable DETR's efficient end-to-end architecture and fast convergence make it an attractive choice to use as an object detector module in larger systems when fine-tuned on relevant data. The deformable attention enables processing high resolution feature maps in a scalable way.

### Out-of-Scope Use

Based on the provided references about the Deformable DETR model, here are some potential misuses and what users ought not do with the model:

1. The model should not be used for surveillance or tracking of individuals without their consent, as this would be a violation of privacy rights. The object detection capabilities should only be applied in ethical and legally permissible ways.

2. Users should not rely solely on the model's outputs for high-stakes decision making, such as in medical diagnosis or autonomous vehicles, without extensive testing, validation, and human oversight. The model may have limitations or biases that could lead to harmful outcomes if misused.

3. The model should not be used to discriminate against protected groups or enable human rights violations. Users must ensure the model is applied fairly and without reinforcing societal biases.

4. [More Information Needed] on specific misuse cases related to the model architecture and training data. The references do not provide enough details to assess domain-specific risks.

5. [More Information Needed] on potential security vulnerabilities or ways the model could be exploited by malicious actors. More technical details would be required to analyze these risks.

In general, users have a responsibility to deploy the model ethically, consider unintended consequences, and put appropriate safeguards in place to mitigate misuse. They should not assume the model is infallible or use it for purposes that violate laws, regulations, or societal values. Ongoing monitoring and adjustment may be needed to address emerging risks.

### Bias, Risks, and Limitations

Based on the provided references about the Deformable DETR model, here are some potential issues and limitations:

Technical limitations:
- While Deformable DETR improves upon DETR in detecting small objects, the references don't quantify how well it performs on very small objects compared to other state-of-the-art detectors. More specific performance metrics would be needed to assess limitations in this area.
- The complexity and computational requirements of Deformable DETR compared to other object detection approaches are not clearly stated. More information is needed on inference speed, memory usage, etc.

Potential misuse or harms:
- Object detection models like Deformable DETR could potentially be misused for harmful surveillance applications if put in the wrong hands. Strict usage guidelines and restrictions may be needed.
- [More Information Needed] on how Deformable DETR handles different demographics and if there is any bias in performance across skin tones, ages, genders, etc. Fairness evaluations should be conducted and reported.

Societal implications:
- Powerful, general-purpose object detection could displace some human workers performing visual detection tasks. The economic impacts should be studied and mitigated.
- [More Information Needed] on if Deformable DETR was trained on any sensitive or personal data, which would raise privacy concerns. The training data sources and privacy implications need to be clearly documented.

To summarize, while Deformable DETR represents an important technical advance, more information is needed to fully characterize its technical limitations, potential for misuse, and broader societal implications. These factors need to be rigorously evaluated and transparently communicated to users.

### Recommendations

Based on the provided references about the Deformable DETR model, here are some recommendations regarding foreseeable issues:

Efficiency and Speed:
- The model achieves fast convergence and efficiency compared to DETR, with 10x less training epochs needed. However, it is still slightly slower than traditional convolution-based methods like Faster R-CNN + FPN due to unordered memory access. Further optimizations may be needed to improve runtime speed.

Complexity and Memory:
- The deformable attention module has a complexity of O(HWC^2) when applied in the DETR encoder, which scales linearly with spatial size. This could lead to high memory usage for large feature maps. Strategies to manage memory consumption should be considered.

Generalization and Robustness:
- While the model shows strong performance on the COCO benchmark, its generalization to other datasets and real-world scenarios needs further evaluation. Testing the model's robustness to variations in object scales, occlusions, and domain shifts is recommended.

[More Information Needed]:
- The provided references do not give sufficient information about potential ethical, societal, or legal implications of the model. A more comprehensive analysis of these aspects would be necessary to provide informed recommendations.

- Details about the training data, including its sources, diversity, and potential biases, are not provided. Understanding the data characteristics is crucial for assessing the model's limitations and potential issues.

- Information about the intended use cases and deployment scenarios of the model is lacking. Knowing the specific applications and contexts in which the model will be used is important for identifying potential risks and mitigation strategies.

In summary, while the Deformable DETR model shows promising performance and efficiency, further analysis is needed to fully assess its potential impact and recommend appropriate measures to address any foreseeable issues. Collaboration with domain experts and stakeholders is advised to gather the necessary information and develop comprehensive guidelines for responsible use and deployment of the model.

## Training Details

### Training Data

The SenseTime/deformable-detr model is trained on the COCO 2017 dataset. Specifically, it is trained on the COCO 2017 train set, and evaluated on the COCO 2017 val set and test-dev set.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the data of the SenseTime/deformable-detr model:

Backbone and Feature Extraction:
- ImageNet pre-trained ResNet-50 is utilized as the backbone
- Multi-scale feature maps are extracted from the output feature maps of stages C3 through C5 in ResNet (transformed by a 1x1 convolution)
- The lowest resolution feature map is obtained via a 3x3 stride 2 convolution on the final C5 stage
- FPN (Feature Pyramid Network) is not used

Input Feature Maps:
- The input multi-scale feature maps of the encoder are extracted from the output feature maps of stages C3 through C5 in ResNet
- The feature maps are of 4 different scales (L = 4)

[More Information Needed] on specific details about tokenization and resizing/rewriting of the input data, as the references do not provide explicit information about these preprocessing steps.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the SenseTime/deformable-detr model:

- Optimizer: Adam
  - Base learning rate: 2 × 10^-4
  - β1 = 0.9
  - β2 = 0.999
  - Weight decay: 10^-4
- Learning rate schedule: 
  - Decay at the 40-th epoch by a factor of 0.1
  - Learning rates of linear projections for predicting object query reference points and sampling offsets are multiplied by 0.1
- Number of training epochs: 50 (achieves better performance than DETR with 10x less training epochs)
- Batch size: 32
- Loss function: Focal Loss with loss weight of 2 for bounding box classification
- Number of object queries: 300
- Backbone: ImageNet pre-trained ResNet-50
- Multi-scale features: Extracted without FPN
- Deformable attention parameters:
  - M = 8 
  - K = 4
- Parameter sharing: Deformable Transformer encoder parameters are shared among different feature levels

[More Information Needed] on other hyperparameters like learning rate warmup, data augmentation, etc.

#### Speeds, Sizes, Times

Here are the details about the SenseTime/deformable-detr model based on the provided references:

Training throughput and time:
- All models were trained with a total batch size of 32
- Deformable DETR achieves better performance than DETR with 10x less training epochs
- By default, models are trained for 50 epochs with the learning rate decayed at the 40th epoch by 0.1

Inference speed:
- Measured on NVIDIA Tesla V100 GPU
- Deformable DETR is 1.6x faster than DETR-DC5
- "Batch Infer Speed" refers to inference with batch size = 4 to maximize GPU utilization
- Deformable DETR is about 25% slower than Faster R-CNN + FPN

Checkpoint sizes:
[More Information Needed]

Other notes:
- The original implementation was based on an internal codebase, so there may be slight differences in final accuracy and running time due to platform differences
- Code is publicly released at https://github.com/fundamentalvision/Deformable-DETR

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model SenseTime/deformable-detr is evaluated on the COCO 2017 dataset. Specifically, it is trained on the COCO 2017 train set, and evaluated on the COCO 2017 val set and test-dev set.

#### Factors

Based on the provided references about the Deformable DETR model, here are some foreseeable characteristics that may influence the model's behavior:

Domain and Context:
- The model is designed for object detection tasks in images. Performance on other visual domains like video is not discussed. [More Information Needed]
- The references focus on model architecture and training details. Specific application contexts are not covered. [More Information Needed]

Population Subgroups:
- The model's performance on detecting small objects is highlighted as an improvement area. This suggests potential disparities in performance across object sizes.
- No information is provided on how the model performs across different object categories, image backgrounds, lighting conditions, etc. Disaggregated evaluation on these factors would help uncover any performance gaps. [More Information Needed]

Other Factors:
- Inference speed is reported, showing the model can achieve good efficiency. However, computational requirements may still limit accessibility.
- Training the model requires significant compute resources (32 batch size, 50 epochs on Tesla V100 GPUs). This could create barriers for some users.

In summary, the references provide useful technical details on Deformable DETR, but more information is needed to fully characterize foreseeable factors that may lead to disparities in real-world behavior. Disaggregated evaluation across relevant subgroups is recommended to better understand the model's strengths and limitations.

#### Metrics

Based on the provided references, the main evaluation metrics used for the SenseTime/deformable-detr model are:

1. AP (Average Precision) at different IoU thresholds, specifically:
   - AP: overall Average Precision
   - AP_S: AP for small objects
   - AP_M: AP for medium objects 
   - AP_L: AP for large objects

2. Training epochs needed to converge, compared to other models like Faster R-CNN + FPN and DETR.

3. Inference speed (FPS) and FLOPs, to assess the computational efficiency and runtime performance.

The model is evaluated on the COCO benchmark dataset. The tradeoffs considered are between detection accuracy (especially for small objects), training convergence speed, and inference efficiency.

[More Information Needed] on specific details about how different error types are weighted or prioritized in the evaluation metrics.

### Results

Here are the evaluation results for the SenseTime/deformable-detr model based on the provided references:

Model Performance:
- Deformable DETR achieves 44.5 AP, 59.6 AP_L, 47.6 AP_M, and 27.1 AP_S on COCO val set, outperforming DETR especially on small objects
- Two-stage Deformable DETR further improves performance to 46.9 AP, 61.6 AP_L, 50.1 AP_M, and 29.6 AP_S

Training Efficiency: 
- Deformable DETR converges in 50 epochs, 10x faster than DETR which requires 500 epochs
- Models are trained with Adam optimizer, base learning rate 2e-4, batch size 32 on 8 GPUs

Inference Efficiency:
- Deformable DETR has similar FLOPs as Faster R-CNN+FPN and DETR-DC5
- Runtime is 1.6x faster than DETR-DC5, only 25% slower than Faster R-CNN+FPN
- Inference speed measured on NVIDIA Tesla V100 GPU

[More Information Needed] for additional factors and metrics not directly mentioned in the provided references.

#### Summary

Here is a summary of the evaluation results for the SenseTime/deformable-detr model based on the provided references:

- Compared to Faster R-CNN + FPN and DETR, Deformable DETR achieves better performance, especially on small objects, with 10x fewer training epochs needed to converge. 

- Deformable DETR has similar FLOPs as Faster R-CNN + FPN and DETR-DC5, but runs 1.6x faster than DETR-DC5, though still 25% slower than Faster R-CNN + FPN due to unordered memory access from the deformable attention mechanism.

- All Deformable DETR models were trained with a total batch size of 32. Training and inference speeds were measured on an NVIDIA Tesla V100 GPU.

- Deformable DETR's multi-scale deformable attention module can adapt its sampling points and attention weights based on the different scales and shapes of foreground objects. 

- Visualization shows that Deformable DETR attends to the extreme boundary points of objects to determine the bounding box coordinates and size, while also looking at interior object pixels to predict the category, differing from the original DETR.

- [More Information Needed] on the exact AP evaluation metrics achieved by Deformable DETR on the COCO benchmark.

- Code is publicly released at https://github.com/fundamentalvision/Deformable-DETR

## Model Examination

Based on the provided references, here is an experimental section on the explainability and interpretability of the Deformable DETR model:

Deformable DETR provides insights into how the model detects objects through visualizations of the learned attention mechanisms and gradient norms:

1. The instances are separated in the encoder, while the decoder focuses on the whole foreground instance rather than just extreme points (Reference 1). This suggests that Deformable DETR uses both extreme points and interior points to determine object category (Reference 2).

2. Visualizing the gradient norm of each item in the final prediction (x/y coordinates, width/height, category score) with respect to each pixel shows which pixels the model relies on for detection (Reference 3). Deformable DETR attends to the left/right boundary for x coordinate and width, and top/bottom boundary for y coordinate and height, while also looking at interior pixels for predicting category (Reference 2).

3. Visualizing the sampling points and attention weights of the last layer in the encoder and decoder demonstrates how the multi-scale deformable attention module adapts to different scales and shapes of foreground objects (References 4, 5).

[More Information Needed] on specific code examples or implementations related to these visualizations and interpretability techniques.

Overall, the deformable attention modules enable Deformable DETR to efficiently process image feature maps and adapt to various object scales and shapes, leading to improved performance and convergence compared to the original DETR (References 6, 7, 8).

## Environmental Impact

- **Hardware Type:** The model SenseTime/deformable-detr is trained on NVIDIA Tesla V100 GPU, according to the references:

"Training and inference speed are measured on NVIDIA Tesla V100 GPU."

"Run time is evaluated on NVIDIA Tesla V100 GPU."
- **Software Type:** Based on the provided references, the software used to train the SenseTime/deformable-detr model includes:

- Operating System: Linux
- CUDA: version >= 9.2
- GCC: version >= 5.4 
- Python: version >= 3.7
- PyTorch: version >= 1.5.1
- torchvision: version >= 0.6.1

The model was trained using PyTorch. An example is provided for installing PyTorch 1.5.1 and torchvision 0.6.1 with CUDA 9.2:

```bash
conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
```

Additional Python package requirements are specified in a requirements.txt file and can be installed using:

```bash
pip install -r requirements.txt
```
- **Hours used:** Based on the provided references, Deformable DETR can achieve better performance than DETR with 10× less training epochs. DETR needs 500 epochs to converge on the COCO benchmark, which is around 10 to 20 times slower than Faster R-CNN.

Therefore, the amount of time used to train the Deformable DETR model is approximately 50 epochs, as mentioned in the references:

"By default, models are trained for 50 epochs and the learning rate is decayed at the 40-th epoch by a factor of 0.1."
- **Cloud Provider:** Based on the provided references, the model SenseTime/deformable-detr is trained on NVIDIA Tesla V100 GPU, as stated in reference 2:

"Training and inference speed are measured on NVIDIA Tesla V100 GPU."

However, the specific cloud provider is not mentioned in the given references. Therefore, for the cloud provider information:

[More Information Needed]
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the SenseTime/deformable-detr model. To accurately report the carbon emissions, details such as the total training time, hardware used, and energy consumption would be needed.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
Deformable DETR is an end-to-end object detector that uses a novel deformable attention mechanism in the transformer architecture. The key components are:

1. Deformable Transformer Encoder: Replaces the standard transformer attention modules with multi-scale deformable attention modules. The encoder processes multi-scale feature maps extracted from a CNN backbone (e.g., ResNet).

2. Deformable Transformer Decoder: Uses deformable attention modules for both cross-attention (between object queries and encoder output) and self-attention (among object queries). The decoder refines the object queries to generate the final detections.

3. Detection Head: A 3-layer feed-forward neural network (FFN) and a linear projection are added on top of the object query features produced by the decoder. The FFN acts as the regression branch to predict bounding box coordinates.

Objective:
The main objectives of Deformable DETR are:

1. Improve the efficiency and convergence speed of DETR by introducing a novel deformable attention mechanism. The deformable attention modules only attend to a small set of key sampling points around a reference, reducing complexity and enabling faster convergence.

2. Achieve better performance than the original DETR by mitigating the issues of slow convergence and limited feature spatial resolution.

3. Explore more interesting and practical variants of end-to-end object detectors, such as the two-stage Deformable DETR, which generates region proposals in the first stage and refines them in the second stage.

[More Information Needed] for specific implementation details and code.

### Compute Infrastructure

Based on the provided references, the compute infrastructure details for the SenseTime/deformable-detr model are:

- OS: Linux
- CUDA: >=9.2
- GCC: >=5.4 
- Python: >=3.7 (recommend using Anaconda to create a conda environment)
- PyTorch: >=1.5.1
- torchvision: >=0.6.1
- GPU: NVIDIA Tesla V100 (used for measuring training and inference speed)

[More Information Needed] on the specific CPU, memory, and disk requirements.

## Citation

```
@misc{xizhou-title,
    author = {Xizhou Zhu and
              Weijie Su and
              Lewei Lu and
              Bin Li and
              Xiaogang Wang and
              Jifeng Dai and
              Sensetime Research},
    title  = {None},
    url    = {https://arxiv.org/pdf/2010.04159.pdf}
}
```

