# Model Card for hustvl/yolos-tiny

The model hustvl/yolos-tiny is a minimalistic object detection model based on the Vision Transformer architecture, designed to perform 2D object detection in a sequence-to-sequence manner with impressive performance compared to tiny-sized CNN detectors, serving as a benchmark for evaluating Transformer's versatility in object detection tasks.

## Model Details

### Model Description

Model Name: hustvl/yolos-tiny

### Model Architecture:
- The hustvl/yolos-tiny model is a part of the YOLOS (You Only Look at One Sequence) series, which is based on the vanilla Vision Transformer (ViT) architecture with minimal modifications for object detection.
- The detector head of YOLOS is simple, resembling the image classification layer of ViT, and uses a Multilayer Perceptron (MLP) with two hidden layers and ReLU activation functions for both classification and bounding box regression.
- YOLOS employs [DET] tokens, inspired by DETR, to represent objects without relying on 2D structure inductive biases or task-specific heuristics.
- The model avoids additional inductive biases by not using non-degenerated convolutions (e.g., 3x3) on top of ViT and maintains a pure sequence-to-sequence approach for object detection.

### Training Procedures:
- Pre-training: YOLOS models, including the tiny version, are pre-trained on the ImageNet-1k dataset using a data-efficient training strategy. The parameters are initialized with a truncated normal distribution and optimized using AdamW. The learning rate is set to 1x10^-3, and the batch size is 1024, with cosine learning rate decay and a weight decay of 0.05. Data augmentation techniques such as Rand-Augment and random erasing are employed.
- Fine-tuning: The model is fine-tuned on the COCO object detection benchmark. All parameters are initialized from ImageNet-1k pre-trained weights except for the MLP heads and [DET] tokens, which are randomly initialized. Training is performed on a single node with 8x12G GPUs, with a learning rate of 2.5x10^-5, a batch size of 8, cosine learning rate decay, and weight decay.

### Parameters:
- The model uses ImageNet-1k pre-trained weights for initialization, except for the newly introduced components for object detection, which are randomly initialized.
- The input resolution for tiny models during inference is 512 for the shorter side.

### Important Disclaimers:
- The YOLOS design is not aimed at achieving the best detection performance but rather to explore the capabilities of Transformers in object detection with minimal biases.
- The tiny-sized YOLOS model demonstrates impressive performance compared to tiny-sized CNN detectors, showing promise as a starting point for model scaling.
- The model's versatility and transferability from image recognition to object detection are highlighted, with minimal knowledge about the input spatial structure.
- [More Information Needed] for any additional disclaimers not covered by the provided references.

### Contact Information:
- For updates or inquiries regarding the model card, please contact the project organizer (the role you are assuming in this scenario).

(Note: The above model card description is based on the provided references and may require additional details or updates based on the latest information available on the Huggingface model repository or from the model development team.)

- **Developed by:** Yuxin Fang; Bencheng Liao; Xinggang Wang; Jiemin Fang; Jiyang Qi; Rui Wu; Jianwei Niu; Wenyu Liu
- **Funded by:** The project for the model hustvl/yolos-tiny is in part supported by NSFC (National Natural Science Foundation of China) under the grant numbers No. 61876212, No. 61733007, and No. 61773176. Additionally, the Zhejiang Laboratory has provided support under Grant 2019NB0AB02.
- **Shared by:** The contributors who made the model hustvl/yolos-tiny available online as a GitHub repo include Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, and Wenyu Liu, as mentioned in reference 3. These individuals are credited with the work on the paper titled "You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection," which the YOLOS model is based on.
- **Model type:** The hustvl/yolos-tiny model is a fine-tuned object detection model based on the Vision Transformer (ViT) architecture, utilizing a bipartite matching loss for set prediction and trained on the COCO benchmark with ImageNet-1k pre-trained weights, representing a single-modality approach in computer vision.
- **Language(s):** The model hustvl/yolos-tiny does not process natural human language; instead, it is an object detection model based on the Vision Transformer architecture designed to transfer pre-trained image recognition capabilities to object detection tasks.
- **License:** [More Information Needed]
- **Finetuned from model:** The model `hustvl/yolos-tiny` is fine-tuned from a Vision Transformer (ViT) base model. Specifically, it uses the DeiT (Data-efficient Image Transformer) and DINO (self-supervised learning method) pre-training strategies. However, the exact name of the base model used for fine-tuning `hustvl/yolos-tiny` is not explicitly mentioned in the provided references. Therefore, to provide the name and link to the base model, [More Information Needed] is required.
### Model Sources

- **Repository:** https://github.com/hustvl/YOLOS
- **Paper:** https://arxiv.org/pdf/2106.00666.pdf
- **Demo:** [More Information Needed]
## Uses

### Direct Use

The model `hustvl/yolos-tiny` is a pre-trained model based on the YOLOS (You Only Look One-level Series) architecture, which is a variant of the Vision Transformer (ViT) adapted for object detection tasks. The references provided do not include explicit instructions for using the `hustvl/yolos-tiny` model without fine-tuning, post-processing, or plugging it into a pipeline. However, they do provide general information about how YOLOS models are trained and evaluated.

To use `hustvl/yolos-tiny` without fine-tuning, you would typically load the pre-trained model and run inference directly on your images. Since there is no direct code snippet provided for `hustvl/yolos-tiny` in the references, and the references focus on fine-tuning and evaluation, I cannot provide a specific code snippet for using `hustvl/yolos-tiny` without fine-tuning.

However, if the model is available on HuggingFace Transformers as mentioned in the TL;DR section, you would generally use the Transformers library to load the model and perform inference. The code would look something like this:

```python
from transformers import AutoModelForObjectDetection, AutoFeatureExtractor
import torch
from PIL import Image
import requests

# Load the feature extractor and model from HuggingFace
feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-tiny")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

# Load an image from the web
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess the image and make predictions
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# The model outputs raw logits, which you would typically post-process to obtain bounding boxes and labels
# However, since the question specifies no post-processing, we stop here
# [More Information Needed] for details on how to interpret the raw logits without post-processing
```

Please note that the above code is a general example of how you might use a model from the HuggingFace Transformers library for inference. The actual usage of `hustvl/yolos-tiny` may require specific details that are not provided in the references. Additionally, without post-processing, the raw logits output by the model are not directly interpretable as object detections (bounding boxes and class labels), so [More Information Needed] on how to handle the outputs without post-processing.

### Downstream Use

The `hustvl/yolos-tiny` model is a small variant of the YOLOS (You Only Look One-level Series) models, which are based on the Vision Transformer (ViT) architecture adapted for object detection tasks. When fine-tuned on the COCO object detection benchmark, this model can be used for detecting and localizing objects within images across a wide range of categories.

In a larger ecosystem or application, `hustvl/yolos-tiny` can serve as a component for various tasks that require object detection capabilities. For instance, it can be integrated into surveillance systems for real-time object tracking, used in retail to analyze customer interactions with products, or implemented in autonomous vehicles for obstacle detection.

The model's small size makes it suitable for applications where computational resources are limited, such as on mobile devices or edge computing platforms. Despite its size, it has been pre-trained on the ImageNet-1k dataset and fine-tuned on COCO, which suggests that it can still perform robustly for object detection tasks.

To use `hustvl/yolos-tiny` for fine-tuning on a specific task, you would follow a procedure similar to the one described in the references for training on the COCO dataset. You would need to adjust the dataset path, batch size, learning rate, and other parameters as necessary for your specific use case.

Here is a code snippet for fine-tuning the `YOLOS-Ti` model, which is the tiny variant of YOLOS, as per the provided references:

```python
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --coco_path /path/to/coco \
    --batch_size 2 \
    --lr 5e-5 \
    --epochs 300 \
    --backbone_name tiny \
    --pre_trained /path/to/deit-tiny.pth \
    --eval_size 512
```

Please note that the above command is for training on the COCO dataset, and you would need to replace `/path/to/coco` with the path to your specific dataset and `/path/to/deit-tiny.pth` with the path to the pre-trained weights for the tiny model. Adjustments to the batch size, learning rate, and number of epochs may also be necessary depending on your dataset and computational resources.

For integration into an application, you would load the fine-tuned model and use it to make predictions on new images. However, since a direct code block for inference is not provided in the references, I would say [More Information Needed] for the exact code snippet for this part.

### Out-of-Scope Use

As a sociotechnic involved in the development of the `hustvl/yolos-tiny` model, it is my responsibility to consider the potential for misuse of our technology and to advise users on practices to avoid. Here are some foreseeable misuses and guidance on what users should not do with the model:

1. **Surveillance and Privacy Violations**: The `hustvl/yolos-tiny` model, being an object detection system, could be used for surveillance purposes without the consent of the individuals being monitored. Users should not employ this model in any form of surveillance that violates privacy rights or expectations, such as in private spaces or for tracking individuals without their consent.

2. **Unethical Monitoring**: The model should not be used for monitoring purposes that discriminate against any group of people based on race, gender, sexuality, or other protected characteristics. The use of object detection in such a manner could lead to biased enforcement and exacerbate social inequalities.

3. **Militarization and Lethal Autonomous Weapons**: The technology should not be repurposed for military use, especially in the development of lethal autonomous weapons systems that could make life-and-death decisions without human intervention.

4. **Deepfakes and Misinformation**: While `hustvl/yolos-tiny` is primarily an object detection model, the underlying technology could potentially be adapted to generate or refine deepfake videos or images, contributing to the spread of misinformation. Users should not use this model to create or propagate false or misleading content.

5. **Illegal Activities**: The model should not be used to aid in any illegal activities, such as theft, by identifying or tracking potential targets for criminal acts.

6. **Bias and Fairness**: As indicated in the references, the model is designed to be as unbiased as possible in its detection capabilities. However, users should be aware of and take steps to mitigate any potential biases that may arise from the training data or the context in which the model is deployed.

7. **Resource Intensive Applications**: Given that the model is described as expensive in terms of model size and positional embeddings (reference 5), users should avoid deploying it in resource-constrained environments where such resource intensity could lead to inefficiencies or exclusion of those with less computational power.

In conclusion, while `hustvl/yolos-tiny` is a powerful tool for object detection, it is crucial that users employ it responsibly, respecting privacy, legal, and ethical boundaries. Any application that could harm individuals or groups, violate rights, or be used for malicious purposes is strictly discouraged. Users are urged to consider the broader social implications of their use of this technology and to engage in practices that promote fairness, transparency, and accountability.

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the model hustvl/yolos-tiny can be categorized into technical limitations and sociotechnical considerations:

**Technical Limitations:**

1. **Transfer Learning Efficiency**: As mentioned in reference 1, YOLOS requires 150 epochs of transfer learning to adapt a pre-trained ViT for object detection, which indicates a significant computational cost and time investment for fine-tuning the model to specific tasks.

2. **Model Size and Positional Encodings (PEs)**: Reference 2 highlights that about one-third of the model weights are dedicated to providing positional information, which may be seen as inefficient. Although the randomly initialized intermediate PEs do not introduce additional inductive biases, they do require learning positional relations from scratch, which could be computationally expensive.

3. **Lack of Few-shot or Zero-shot Learning**: Unlike state-of-the-art language models that can adapt to new scenarios with little to no labeled data, as stated in reference 3, YOLOS still requires substantial supervision to transfer to downstream tasks, limiting its flexibility and ease of deployment in new domains.

4. **PEs in Tiny-sized Models**: Reference 4 suggests that using a large enough PE in the first layer might make intermediate PEs redundant. This is an area that requires further research, and the current instantiation may not be optimal.

5. **Model Scaling and Parameter Efficiency**: Reference 5 discusses the trade-offs in controlling model size and parameter efficiency, indicating that there may be a balance to strike between model complexity and computational resources.

**Sociotechnical Considerations:**

1. **Potential for Misunderstanding**: Reference 11 indicates that the performance of YOLOS may seem discouraging, but it is designed to reveal the transferability of ViT in object detection. Users may misunderstand the purpose of the model, expecting state-of-the-art performance rather than a demonstration of ViT's adaptability.

2. **Bias and Fairness**: [More Information Needed] - The references do not provide explicit information on biases in the dataset or model fairness, but it is a common issue in deep learning models that must be considered, especially in object detection tasks.

3. **Ethical Use and Misuse**: [More Information Needed] - The references do not discuss the ethical implications of using the model. However, as with any object detection technology, there is potential for misuse, such as surveillance without consent or in violation of privacy rights.

4. **Accessibility and Inclusivity**: [More Information Needed] - There is no information provided on how accessible the model is for users with varying levels of expertise or how inclusive it is in terms of recognizing diverse objects and scenarios.

5. **Environmental Impact**: The computational cost mentioned in reference 1 implies a significant energy expenditure for training and fine-tuning the model, which has environmental implications.

In summary, while the hustvl/yolos-tiny model demonstrates the potential of Transformer-based models in object detection, it comes with technical limitations that affect its efficiency and adaptability. Additionally, there are broader sociotechnical issues that need to be addressed, such as potential misuse, bias, and environmental impact, for which more information is needed to fully assess.

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model hustvl/yolos-tiny:

1. **Transfer Learning Efficiency**: The YOLOS model requires 150 epochs of transfer learning to adapt a pre-trained Vision Transformer (ViT) to perform object detection (Ref. 1). This indicates a significant computational cost for adaptation. It is recommended to explore methods to reduce the number of epochs required for transfer learning, possibly by investigating more efficient fine-tuning techniques or by leveraging more targeted pre-training.

2. **Supervision Requirement**: Unlike state-of-the-art language models that can perform few-shot or zero-shot learning, YOLOS and other pre-trained computer vision models still require substantial supervision to transfer to downstream tasks (Ref. 2). It is recommended to investigate strategies to reduce the supervision needed, such as semi-supervised or unsupervised learning approaches that could leverage unlabeled data more effectively.

3. **Positional Encoding (PE) Optimization**: The model uses a large PE in the first layer, which is a promising setting that will be studied more in the future (Ref. 3). It is recommended to continue research on optimizing PEs, especially considering the trade-off between model size and the ability to handle multi-scale inputs or inputs with varying sizes and aspect ratios (Ref. 5).

4. **Model Size and Efficiency**: The YOLOS-Ti model is competitive in terms of average precision (AP), floating-point operations per second (FLOPs), and frames per second (FPS) when compared to tiny-sized CNN detectors (Ref. 6). However, it is noted that about one-third of the model weights are dedicated to providing positional information (Ref. 5). It is recommended to explore ways to reduce the model size without compromising performance, possibly by investigating more parameter-efficient positional encoding methods or model pruning techniques.

5. **Interpretation of Results**: The YOLOS model is not designed for peak performance but rather to reveal the characteristics of the Transformer family in object detection (Ref. 7, 8). It is important to communicate this clearly to potential users, emphasizing that the model serves as a research tool to understand transferability and that there may be better-performing models for practical applications.

6. **Ethical and Societal Considerations**: As a sociotechnic, it is crucial to consider the broader implications of deploying object detection models like YOLOS-Ti. This includes addressing potential biases in the training data, ensuring privacy and ethical use of the technology, and being transparent about the model's limitations. It is recommended to conduct thorough bias and fairness assessments and to provide clear guidelines for ethical usage.

In summary, while the YOLOS-Ti model shows promise, there are several areas for improvement, particularly in terms of transfer learning efficiency, supervision requirements, model size optimization, and ethical considerations. Further research and development are needed to address these issues.

## Training Details

### Training Data

The training data for the model hustvl/yolos-tiny consists of the COCO object detection benchmark, with all parameters initialized from ImageNet-1k pre-trained weights except for the MLP heads and [DET] tokens. The model was fine-tuned on this dataset following the procedure outlined by Carion et al. [More Information Needed] on data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

The preprocessing steps for the data used in the `hustvl/yolos-tiny` model involve several key stages, as can be inferred from the provided references:

1. **Image Resizing and Patch Extraction**: As per reference 5 and 7, input images are first resized such that the shortest side is at least 480 pixels and at most 800 pixels, while the longest side is at most 1333 pixels. The images are then reshaped into a sequence of flattened 2D image patches (`x_PATCH`). The resolution of each image patch is `(P, P)`, and the number of patches `N` is determined by the formula `N = HW / P^2`, where `H` and `W` are the height and width of the resized image, and `C` is the number of input channels.

2. **Tokenization**: Reference 1 describes the tokenization process where the image patches are projected to `D` dimensions with a trainable linear projection (`E`). These projected patches are referred to as `[PATCH]` tokens. Additionally, one hundred randomly initialized learnable `[DET]` tokens are appended to the `[PATCH]` tokens to serve as proxies for object representations.

3. **Position Embeddings**: Position embeddings are added to all input tokens to retain positional information, following the standard learnable 1D position embeddings as mentioned in reference 1.

4. **Data Augmentation**: During training, random crop augmentations are applied as stated in reference 7. Additionally, reference 8 mentions the use of Rand-Augment and random erasing for data augmentation during the pre-training phase on the ImageNet-1k dataset.

5. **Normalization and Other Preprocessing Steps**: Although not explicitly mentioned in the provided references, it is common practice in deep learning models to normalize the image data before feeding it into the network. This typically involves scaling pixel values to a range of [0, 1] or normalizing them based on the dataset's mean and standard deviation. [More Information Needed] for specific normalization details used in `hustvl/yolos-tiny`.

6. **Fine-tuning**: Reference 6 indicates that the model is fine-tuned on the COCO object detection benchmark, with all parameters initialized from ImageNet-1k pre-trained weights except for the MLP heads and the `[DET]` tokens, which are randomly initialized.

The above steps outline the preprocessing pipeline for the `hustvl/yolos-tiny` model based on the provided references. It is important to note that specific implementation details such as the exact patch size `(P, P)`, the dimension `D` to which patches are projected, and the normalization parameters would be necessary to replicate the preprocessing exactly. However, these details are not provided in the references, so [More Information Needed] for those specifics.

#### Training Hyperparameters

The training hyperparameters for the model `hustvl/yolos-tiny` are as follows:

- **Initialization**: The parameters are initialized from ImageNet-1k pre-trained weights, except for the MLP heads for classification & bounding box regression and one hundred [DET] tokens, which are randomly initialized.
- **Training Environment**: The model is trained on a single node with 8 × 12G GPUs.
- **Learning Rate**: The learning rate is set to 2.5 × 10^-5.
- **Batch Size**: The batch size used is 8.
- **Learning Rate Decay**: A cosine learning rate decay schedule is used.
- **Weight Decay**: The specific weight decay value is not mentioned in the provided references, so [More Information Needed] for this detail.
- **Data Augmentation**: Random crop augmentations are applied during training, following Carion et al. [10].
- **Regularization**: Dropout or stochastic depth are not applied during fine-tuning as these methods were found to hurt performance.

Please note that some details such as the specific weight decay value during fine-tuning are not provided in the references, and thus more information would be needed to complete those details.

#### Speeds, Sizes, Times

The model `hustvl/yolos-tiny`, also referred to as YOLOS-Ti, is a tiny-sized variant of the YOLOS model family, which is designed to investigate the transferability of the Vision Transformer (ViT) architecture to the task of object detection. Here are the details based on the provided references:

- **Throughput**: The model achieves competitive throughput in terms of FLOPs (Floating Point Operations per second) and FPS (Frames Per Second). However, specific numbers for throughput are not provided in the references, so [More Information Needed] for exact values.

- **Start or End Time**: The references do not provide explicit start or end times for the training or inference processes. However, it is mentioned that the tiny-sized models are trained to be fully converged, and the FPS data are measured over the first 100 images of the COCO validation split during inference. [More Information Needed] for precise start or end times.

- **Checkpoint Sizes**: The references do not explicitly state the checkpoint sizes for the YOLOS-Ti model. Checkpoint size typically refers to the storage space required to save the model's weights and architecture configuration. [More Information Needed] for the exact checkpoint size.

Additional details that can be inferred from the references include:

- The YOLOS-Ti model is pre-trained on the ImageNet-1k dataset for 300 epochs and fine-tuned on the COCO object detection benchmark for 150 epochs.
- The input resolution for inference is selected from the range [480, 800], with the smallest resolution being chosen for each model during inference.
- The model is trained on a single node with 8 × 12G GPUs, with a learning rate of 2.5 × 10^-5 and a batch size of 8. The learning rate follows a cosine decay schedule, and the weight decay is not specified in the provided references.
- The input patch size for all YOLOS models is 16 × 16.
- The YOLOS-Ti model corresponds to the DeiT-Ti model in terms of scaling, and it is suggested that it can serve as a promising model scaling start point.

For a complete and accurate model card description, it would be necessary to have access to the full details of the model's training and evaluation, which are not fully covered in the provided references.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model hustvl/yolos-tiny evaluates on the COCO object detection benchmark.

#### Factors

The foreseeable characteristics that will influence how the model hustvl/yolos-tiny behaves are as follows:

1. **Model Design and Architecture**: YOLOS-Ti closely follows the original Vision Transformer (ViT) architecture and is optimized for object detection. The design is intentionally simple and not aimed at achieving the best detection performance but to reveal the characteristics of the Transformer family in object detection (Reference 6). This means that the model's behavior is influenced by the inherent properties of Transformers, such as their ability to handle sequential data and their requirement for large amounts of data to generalize well.

2. **Performance Sensitivity**: The model's object detection results are sensitive to the pre-training scheme, indicating that the choice of pre-training strategy can significantly influence the model's performance (Reference 7). This suggests that the domain and context in which the model is pre-trained will affect its subsequent behavior in object detection tasks.

3. **Detection Token Specialization**: The [DET] tokens used by YOLOS-Ti to represent detected objects are sensitive to object locations and sizes but are insensitive to object categories (Reference 2). This characteristic implies that the model may behave differently across various population subgroups that are defined by the size and location of objects within images.

4. **Transferability of ViT**: The model is designed to reveal the transferability of ViT in object detection, which was originally developed for NLP tasks (Reference 3). The transferability aspect may influence the model's behavior when detecting objects that do not align well with the patterns learned from NLP-focused architectures.

5. **Fine-Tuning on COCO Benchmark**: The model is fine-tuned on the COCO object detection benchmark, which means its behavior will be influenced by the distribution of objects and scenarios present in this dataset (Reference 8). If the COCO dataset has biases or lacks representation for certain subgroups or scenarios, this will likely be reflected in the model's performance.

6. **Comparison with Other Models**: YOLOS-Ti is competitive in terms of average precision (AP), FLOPs, and frames per second (FPS) when compared with other tiny-sized CNN detectors, suggesting that it can serve as a promising model scaling start point (Reference 1). However, disparities in performance may still exist when compared to more complex or differently optimized models.

7. **Hardware and Inference Speed**: The FPS data is measured on a single 1080Ti GPU, which indicates that the model's behavior in terms of inference speed is influenced by the hardware used (Reference 4). This could lead to disparities in performance across different deployment environments with varying hardware capabilities.

In summary, the behavior of hustvl/yolos-tiny is influenced by its Transformer-based design, pre-training schemes, specialization of detection tokens, the COCO benchmark dataset, and the hardware used for inference. To uncover disparities in performance, evaluation should be disaggregated across factors such as object size and location, pre-training data domains, and hardware environments.

#### Metrics

The evaluation of the model hustvl/yolos-tiny will primarily use the Average Precision (AP) metric, as it is a standard measure for object detection performance. AP provides a balance between precision and recall, and it is particularly relevant for the COCO benchmark which is mentioned in the references.

Additionally, the model will be evaluated based on its computational efficiency, which includes FLOPs (Floating Point Operations Per Second) and FPS (Frames Per Second). These metrics are important for understanding the trade-offs between the model's accuracy and its speed or resource consumption. FLOPs give an indication of the computational complexity of the model, while FPS measures how fast the model can process images, which is crucial for real-time applications.

The references also suggest that the model's design is not optimized for performance but rather to reveal the transferability of the Vision Transformer (ViT) to object detection tasks. Therefore, while AP, FLOPs, and FPS are the primary metrics for evaluation, the broader goal is to understand the characteristics and capabilities of Transformer architectures in the context of object detection.

In summary, the evaluation metrics for hustvl/yolos-tiny will include:
- Average Precision (AP) for accuracy assessment.
- FLOPs for computational complexity.
- FPS for inference speed.

These metrics will help in understanding the balance between detection accuracy and computational efficiency, as well as the model's potential as a starting point for further scaling and optimization of Transformer-based object detectors.

### Results

The evaluation results of the model `hustvl/yolos-tiny` based on the provided references are as follows:

- **Performance Comparison**: The tiny-sized YOLOS model, specifically `hustvl/yolos-tiny`, demonstrates impressive performance when compared with other tiny-sized CNN object detectors. It is particularly strong in Average Precision (AP) and is competitive in terms of FLOPs (Floating Point Operations Per Second) and FPS (Frames Per Second), despite Transformers not being intentionally designed to optimize these factors.

- **Model Scaling**: YOLOS-Ti can be considered a promising starting point for model scaling, indicating potential for further improvements and adaptations of the architecture.

- **Transferability of ViT**: The YOLOS model is designed to reveal the transferability of the Vision Transformer (ViT) to object detection tasks. It shows that a Transformer architecture, with minimal modifications, can be successfully applied to the COCO object detection benchmark, achieving an AP of 42.0.

- **Comparison with DETR**: Quantitative comparisons with DETR (DEtection TRansformer) models show that when fully converged, tiny-sized models like `hustvl/yolos-tiny` are competitive. The model's performance metrics, such as FLOPs and FPS, are measured on the COCO validation split and are indicative of its efficiency.

- **Detection Tokens Analysis**: The [DET] tokens used by YOLOS for object detection are sensitive to object locations and sizes but are less sensitive to object categories. This characteristic is visualized with color-coded points representing different object sizes.

- **Model Design**: `hustvl/yolos-tiny` closely follows the original ViT architecture and is optimized for object detection. The design is intentionally simple, aiming to reveal the characteristics of the Transformer family in object detection tasks.

- **COCO Benchmark Performance**: The YOLOS-Base model, which is closely related to `hustvl/yolos-tiny`, achieves a box AP of 42.0 on the COCO validation dataset, indicating that the tiny version also has competitive performance.

- **Fine-tuning Details**: The fine-tuning process for `hustvl/yolos-tiny` on the COCO benchmark involves initializing parameters from ImageNet-1k pre-trained weights, except for the MLP heads and [DET] tokens. The model is trained with a learning rate of 2.5 × 10^-5 and a batch size of 8, using cosine learning rate decay and weight decay.

For specific numerical results such as the exact AP, FLOPs, and FPS achieved by `hustvl/yolos-tiny`, [More Information Needed] as these details are not provided in the references.

#### Summary

The evaluation results for the model `hustvl/yolos-tiny` can be summarized as follows:

1. Pre-training and Fine-tuning: The model was pre-trained for 300 epochs on ImageNet-1k and fine-tuned on COCO for 150 epochs. During inference, the input shorter size is set to 512 for tiny models like `hustvl/yolos-tiny`.

2. Transfer Learning: The model demonstrates that self-supervised pre-training strategies, such as DINO for 800 epochs, can achieve comparable performance to label-supervised pre-training like DeiT for 300 epochs on COCO object detection tasks. This suggests the potential of self-supervised learning for vision transformers in complex recognition tasks.

3. Model Scaling: The `hustvl/yolos-tiny` model shows that traditional CNN scaling methods may not directly apply to vision transformers due to the different computational dynamics of spatial attention. This indicates a need for novel model scaling strategies that consider the complexity of spatial attention in transformers.

4. Performance Comparison: When compared to tiny-sized CNN detectors, `hustvl/yolos-tiny` achieves impressive performance in terms of average precision (AP) and is competitive in FLOPs (floating-point operations per second) and FPS (frames per second), despite transformers not being specifically optimized for these metrics.

5. Model Design: `hustvl/yolos-tiny` closely follows the original ViT architecture and is optimized for object detection. The design is intentionally simple to reveal the characteristics of the Transformer family in object detection tasks.

6. Comparison with DETR: Quantitative comparisons with DETR models show that tiny-sized models like `hustvl/yolos-tiny` are trained to full convergence. The FLOPs and FPS metrics are measured on the COCO validation split, with FPS measured on a single 1080Ti GPU with a batch size of 1.

In summary, `hustvl/yolos-tiny` is a promising tiny-sized model that balances performance and efficiency, showing the effectiveness of vision transformers in object detection tasks and highlighting the potential of self-supervised pre-training strategies.

## Model Examination

## Model Card for hustvl/yolos-tiny

### Experimental Section: Explainability/Interpretability

The hustvl/yolos-tiny model is a compact version of the YOLOS (You Only Look One-level Series) model family, which is designed to adapt the Vision Transformer (ViT) architecture for object detection tasks. In this section, we delve into the explainability and interpretability aspects of the YOLOS-Tiny model.

#### Attention Visualization

To understand how the model makes predictions, we have studied the self-attention mechanisms within the YOLOS-Tiny model. By visualizing the attention maps, we can observe that different self-attention heads focus on various patterns and locations across the input image. These visualizations can be both interpretable and non-interpretable, providing insights into the model's focus areas during detection.

For instance, we have compared the attention maps of two YOLOS models pre-trained on ImageNet-1k for different epochs but achieving the same average precision (AP). The visualizations suggest that even with the same AP, the models may attend to different features or regions for a given predicted object.

#### Self-Attention of [DET] Tokens

The [DET] tokens, which are crucial for object detection in YOLOS models, have been specifically inspected for their self-attention patterns. By following the visualization pipeline from Caron et al., we have generated visual results that help us conclude the behavior of these tokens in the context of object detection.

#### Correlation Study

We have conducted a study to measure the linear correlation between the proximity of [DET] tokens (in terms of cosine similarity) and their corresponding predictions' proximity (in terms of Euclidean distance). The result, with a correlation coefficient of ρ X,Y = −0.80, indicates a strong inverse relationship, suggesting that [DET] tokens closer to each other tend to predict objects that are also nearby in the image.

#### Model Design and Performance

The YOLOS-Tiny model is intentionally designed to be simple, closely following the original ViT architecture. This design choice is not aimed at optimizing detection performance but rather at revealing the characteristics and transferability of Transformer models in object detection tasks. Despite this, YOLOS-Tiny demonstrates impressive performance when compared to tiny-sized CNN detectors, being strong in AP and competitive in terms of FLOPs and FPS.

#### Interpreting the Results

The results obtained from YOLOS-Tiny should be interpreted with the understanding that the model is not optimized for peak performance. Instead, it serves as a proof of concept for the application of ViT in object detection and as a starting point for model scaling within the Transformer family.

In summary, the YOLOS-Tiny model provides valuable insights into the application of Transformer architectures in computer vision, particularly in object detection. Its design allows for the study of explainability and interpretability within this domain, contributing to the broader understanding of how such models perceive and process visual information.

## Environmental Impact

- **Hardware Type:** The model hustvl/yolos-tiny is trained on a single node with 8 × 12G GPUs.
- **Software Type:** The model hustvl/yolos-tiny is trained on a single node with 8 × 12G GPUs.
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture of `hustvl/yolos-tiny` (YOLOS-Ti) is based on the vanilla Vision Transformer (ViT) with minimal modifications to adapt it for the task of object detection. The key characteristics of the YOLOS-Ti architecture are as follows:

1. **Detector Heads**: YOLOS-Ti simplifies the detector head compared to traditional object detection models. It uses a single multilayer perceptron (MLP) with two hidden layers and ReLU activation functions for both classification and bounding box regression tasks.

2. **Inductive Bias**: The architecture is designed with minimal additional inductive biases. The inductive biases that are inherent from ViT include patch extraction at the network stem and resolution adjustment for position embeddings. YOLOS-Ti does not add any non-degenerate convolutions (e.g., 3x3) on top of the ViT architecture.

3. **[DET] Tokens**: Instead of using a [CLS] token as in image classification tasks, YOLOS-Ti uses one hundred [DET] tokens for object detection. These tokens serve as proxies for object representations, following a similar approach to DETR (DEtection TRansformer).

4. **Position Embedding (PE)**: Due to variations in image resolutions and aspect ratios in object detection tasks, YOLOS-Ti adapts the position embedding from ViT to accommodate these changes, which could affect the model size and performance.

5. **Model Design**: YOLOS-Ti closely follows the original ViT architecture and is optimized for object detection. It is designed to be easily adapted to various Transformer architectures and to reveal the characteristics of the Transformer family in object detection tasks.

6. **Objective**: The objective of YOLOS-Ti is to perform 2D object detection from a pure sequence-to-sequence perspective with minimal knowledge about the 2D spatial structure. It uses a bipartite matching loss for set prediction, following the approach by Carion et al., to perform object detection without manually injecting heuristics or prior knowledge of object spatial structure.

7. **Performance**: Despite not being designed to optimize for detection performance, YOLOS-Ti achieves impressive performance compared to tiny-sized CNN detectors, being strong in average precision (AP) and competitive in FLOPs & FPS.

In summary, `hustvl/yolos-tiny` is a compact and efficient object detection model that leverages the strengths of the Vision Transformer architecture with a focus on simplicity and minimal inductive biases. It demonstrates the potential of Transformers in computer vision tasks, particularly object detection, without relying on complex and task-specific architectural modifications.

### Compute Infrastructure

The compute infrastructure used for training the model hustvl/yolos-tiny involved a single node with 8 × 12G GPUs. The training utilized a batch size of 8 and a learning rate of 2.5 × 10^-5 with cosine learning rate decay. The model parameters were initialized from ImageNet-1k pre-trained weights, except for the MLP heads for classification and bounding box regression, as well as one hundred [DET] tokens, which were randomly initialized. Weight decay was also applied during the training process, although the specific value for weight decay is not mentioned in the provided references and would require [More Information Needed].

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

