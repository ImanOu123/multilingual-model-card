# Model Card for hustvl/yolos-small-300

The model hustvl/yolos-small-300 is a lightweight object detection model based on the Vision Transformer (ViT) architecture, designed to perform 2D object detection in a sequence-to-sequence manner with minimal 2D spatial inductive biases and region priors. It demonstrates the adaptability of Transformer models from image recognition to object detection tasks, serving as a benchmark for evaluating different pre-training strategies for Transformers in vision.

## Model Details

### Model Description

Model Card for hustvl/yolos-small-300

## Model Architecture
The hustvl/yolos-small-300 model is a small variant of the YOLOS (You Only Look One-level Series) family, which is designed to perform object detection tasks. The architecture of YOLOS is inspired by the DETR (Detection Transformer) model and follows a pure sequence-to-sequence approach. The model is built upon the Vision Transformer (ViT) architecture and uses [DET] tokens as proxies for object representations, avoiding inductive biases about 2D structures during label assignment. The detector head is simple, consisting of a Multi-Layer Perceptron (MLP) with two hidden layers and ReLU activation functions for both classification and bounding box regression.

## Training Procedures
The model is pre-trained on the ImageNet-1k dataset using a data-efficient training strategy with AdamW optimizer, cosine learning rate decay, and a weight decay of 0.05. Data augmentation techniques such as Rand-Augment and random erasing are employed during pre-training. For fine-tuning, the model is trained on the COCO object detection benchmark with parameters initialized from ImageNet-1k pre-trained weights, except for the MLP heads and [DET] tokens which are randomly initialized. Training is conducted on a single node with 8 × 12G GPUs, with a learning rate of 2.5 × 10^−5, a batch size of 8, cosine learning rate decay, and weight decay.

## Parameters
The model utilizes the parameters and architectural choices of the original ViT, with minimal additional inductive biases. The [DET] tokens and MLP heads are the only components not initialized from the pre-trained ViT weights. The model is optimized for object detection without additional 2D inductive biases or task-specific heuristics.

## Important Disclaimers
- The YOLOS architecture is designed to minimize inductive biases and reveal the versatility and transferability of pre-trained Transformers from image recognition to object detection. It does not include performance-oriented designs such as pyramidal feature hierarchy or region-wise pooling operations.
- The model requires 150 epochs of transfer learning to adapt a pre-trained ViT for object detection, and the detection results may not be on par with highly-optimized CNN object detectors.
- The tiny-sized YOLOS model, while not the same as hustvl/yolos-small-300, demonstrates competitive performance in terms of average precision (AP) and efficiency (FLOPs & FPS) compared to CNN detectors, suggesting that the small-sized YOLOS could also be a promising starting point for model scaling.

For further inquiries or updates regarding the model, please contact the project organizer responsible for the model card.

- **Developed by:** Yuxin Fang; Bencheng Liao; Xinggang Wang; Jiemin Fang; Jiyang Qi; Rui Wu; Jianwei Niu; Wenyu Liu
- **Funded by:** The project for the model hustvl/yolos-small-300 is in part supported by NSFC (No. 61876212, No. 61733007, and No. 61773176) and the Zhejiang Laboratory under Grant 2019NB0AB02.
- **Shared by:** The contributors who made the model `hustvl/yolos-small-300` available online as a GitHub repo are Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, and Wenyu Liu, as mentioned in reference 3.
- **Model type:** The model hustvl/yolos-small-300 is a Transformer-based object detection model fine-tuned on the COCO benchmark with ImageNet-1k pre-trained weights, using a sequence-to-sequence approach with minimal modifications for adapting to 2D object detection tasks.
- **Language(s):** The model hustvl/yolos-small-300 does not process any natural human language; instead, it is an object detection model based on the Vision Transformer architecture designed for image recognition tasks.
- **License:** [More Information Needed]
- **Finetuned from model:** The model `hustvl/yolos-small-300` is fine-tuned from a base model that uses the Vision Transformer (ViT) architecture. Specifically, the references mention the use of DeiT (Data-efficient Image Transformers) and DINO (self-supervised learning method) for pre-training strategies. However, the exact base model name and link are not directly provided in the given references.

For a precise answer, more information would be needed to identify the specific base model and provide a link to it. Therefore, my response is:

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/hustvl/YOLOS
- **Paper:** https://arxiv.org/pdf/2106.00666.pdf
- **Demo:** [More Information Needed]
## Uses

### Direct Use

The model `hustvl/yolos-small-300` is a pre-trained object detection model based on the YOLOS (You Only Look One-level Series) architecture, which is a variant of the Vision Transformer (ViT) adapted for object detection tasks. The model has been pre-trained on the ImageNet-1k dataset and fine-tuned on the COCO object detection benchmark.

Using the model without fine-tuning, post-processing, or plugging it into a pipeline implies that we would be using the model in an "out-of-the-box" manner, leveraging its pre-trained and fine-tuned weights to make predictions on new images.

However, to use the model for object detection, some form of post-processing is typically required to convert the raw outputs of the model (e.g., class probabilities and bounding box coordinates) into a human-interpretable format, such as drawing bounding boxes around detected objects. This post-processing step often includes non-maximum suppression to filter out overlapping boxes and thresholding to remove low-confidence detections.

Since the reference does not provide a direct code block for using the model without any post-processing or integration into a pipeline, and considering that object detection inherently requires some post-processing to interpret the raw model outputs, we cannot provide a code snippet for using the model without these steps.

Therefore, the answer to the question is: [More Information Needed]. To properly use the `hustvl/yolos-small-300` model for object detection, post-processing steps are necessary to interpret the model's output.

### Downstream Use

The `hustvl/yolos-small-300` model is a fine-tuned version of the YOLOS (You Only Look One-level Series) model, which is based on the Vision Transformer (ViT) architecture. This particular model has been fine-tuned on the COCO object detection benchmark for 300 epochs, which suggests that it has been optimized for the task of object detection in images.

When fine-tuned for a specific task, `hustvl/yolos-small-300` can be used to detect and localize objects within images. This can be particularly useful in applications that require the identification of items within a scene, such as in surveillance systems, autonomous vehicles, retail analytics, and many other areas where visual object recognition is necessary.

To plug `hustvl/yolos-small-300` into a larger ecosystem or app, you would typically load the pre-trained model and use it to make predictions on new data. The model expects input images to be pre-processed in a certain way (e.g., resized to a specific resolution) before being passed through the network.

Here is a code snippet that demonstrates how to use the `hustvl/yolos-small-300` model for inference, assuming that the model has been made available through the Hugging Face Model Hub and that you have an image pre-processed and ready to be input to the model:

```python
from transformers import DetrForSegmentation, DetrImageProcessor
from PIL import Image
import requests

# Load the model and processor from Hugging Face
model = DetrForSegmentation.from_pretrained("hustvl/yolos-small-300")
processor = DetrImageProcessor.from_pretrained("hustvl/yolos-small-300")

# Load an image from the web or local file system
image = Image.open(requests.get("image_url", stream=True).raw)

# Preprocess the image and prepare for the model
inputs = processor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Process the outputs (e.g., extract bounding boxes and labels)
# [More Information Needed] for the exact post-processing steps as they are not provided in the references.

# Note: The above code is a general example and may require adjustments based on the actual implementation details of the `hustvl/yolos-small-300` model.
```

Please note that the actual usage of the model might require additional steps or different functions, depending on the final implementation and the library's API. The code snippet provided is a general example and assumes that the model follows a similar interface to other Hugging Face object detection models. If the actual usage pattern differs, the code would need to be adjusted accordingly.

### Out-of-Scope Use

As a sociotechnic on the deep learning model development team, it is my responsibility to consider the potential societal impacts and misuse of the model hustvl/yolos-small-300. Based on the references provided, here are some considerations regarding how the model may be misused and what users should avoid doing with the model:

1. **Surveillance and Privacy Violations**: The model's object detection capabilities could be misused for unauthorized surveillance purposes, leading to invasions of privacy. Users should not deploy this model in scenarios where it could be used to track, monitor, or analyze individuals without their consent.

2. **Biased Decision-Making**: While the references do not directly address bias in the training data, it is a common issue in machine learning models. If the model has been trained on datasets that are not representative of the diversity of the real world, it may exhibit biased behavior. Users should not use the model in contexts where biased outputs could lead to discriminatory practices, such as in law enforcement or hiring.

3. **Unintended Use in Critical Systems**: The model is not designed for high-stakes scenarios and is primarily a research tool to study the transferability of ViT in object detection. Users should not use this model in critical systems where failure or inaccuracies could result in harm, such as in autonomous vehicles or medical diagnosis.

4. **Manipulation and Deepfakes**: The object detection capabilities could potentially be used to create or propagate manipulated media, such as deepfakes. Users should not use the model to generate or contribute to the spread of deceptive content.

5. **Intellectual Property Violations**: The model should not be used to detect and exploit copyrighted material without permission.

6. **Weaponization**: The model should not be used to develop or enhance autonomous weapons systems.

It is important for users to adhere to ethical guidelines and legal standards when using the model. The development team encourages responsible use and urges users to consider the broader societal implications of deploying this technology in real-world applications.

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the model hustvl/yolos-small-300, based on the provided references, can be categorized into technical limitations and sociotechnical considerations:

Technical Limitations:
1. Transfer Learning Efficiency: As mentioned in reference 1, YOLOS requires 150 epochs of transfer learning to adapt a pre-trained Vision Transformer (ViT) to perform object detection, which indicates a significant computational cost and time investment for training.
2. Supervision Requirement: Reference 2 highlights that, unlike state-of-the-art language models that can perform few-shot or zero-shot learning, pre-trained computer vision models like YOLOS still require substantial supervision to transfer to downstream tasks.
3. Model Size and Positional Encodings: Reference 3 points out that a considerable portion of the model's weights is dedicated to providing positional information, which makes the model heavy. This could be a limitation in resource-constrained environments.
4. Inductive Biases: Reference 5 and 6 discuss the balance between performance-oriented architectural designs and the desire to maintain the generality of Transformers. YOLOS aims to minimize additional inductive biases, which might limit its performance compared to models with task-specific optimizations.
5. Model Scaling: Reference 10 suggests that while YOLOS-Ti shows promise as a model scaling start point, there may be challenges in scaling the model effectively to maintain or improve performance.

Sociotechnical Considerations:
1. Societal Impact and Misuse: The model's ability to detect objects in images could potentially be misused for surveillance or other privacy-invasive applications. There is a need for ethical guidelines and regulations to prevent misuse.
2. Accessibility and Inclusivity: The computational resources required for training and adapting the model (as indicated in references 1 and 2) may limit accessibility to those with sufficient resources, potentially exacerbating the digital divide.
3. Bias and Fairness: While not explicitly mentioned in the references, any object detection model, including YOLOS, may inherit biases present in the training data. This could lead to unfair or discriminatory outcomes if not carefully addressed.
4. Transparency and Explainability: The complexity of Transformer-based models can make them less interpretable, which could lead to challenges in understanding model decisions, particularly in high-stakes scenarios.

In conclusion, while the YOLOS model demonstrates promising results in object detection, there are technical challenges related to efficiency, supervision, model size, and scaling. Additionally, sociotechnical issues such as potential misuse, accessibility, bias, and transparency need to be considered and addressed to ensure responsible deployment and use of the technology.

### Recommendations

Based on the provided references, here are some recommendations with respect to the foreseeable issues about the model hustvl/yolos-small-300:

1. **Transfer Learning Efficiency**: As noted in the references, YOLOS still requires a significant amount of transfer learning (150 epochs) to adapt a pre-trained Vision Transformer (ViT) to perform object detection (Reference 1). It is recommended to explore methods to reduce the number of epochs needed for transfer learning, possibly by investigating more efficient fine-tuning techniques or by leveraging larger pre-trained models that may require less adaptation.

2. **Supervision Requirements**: Unlike state-of-the-art language models that can perform few-shot or zero-shot learning, YOLOS and other pre-trained computer vision models still require substantial supervision to transfer to downstream tasks (Reference 2). It is recommended to investigate approaches that could reduce the supervision needed, such as semi-supervised or unsupervised learning methods, to enhance the model's adaptability and ease of use in various scenarios.

3. **Performance Expectations**: The references indicate that YOLOS is not designed for optimal performance but rather to reveal the transferability of ViT in object detection (Reference 7). Users should be aware that while the model provides valuable insights into the capabilities of Transformers in computer vision, it may not deliver the highest possible accuracy in object detection tasks. Continuous research and development are recommended to improve the model's performance while maintaining its interpretability.

4. **Model Design and Inductive Biases**: The model intentionally follows a simple setup to reveal characteristics of the Transformer family in object detection (Reference 6). It is recommended to consider the right inductive biases and priors for target tasks when designing future iterations of the model to potentially enhance performance without compromising the model's core purpose (Reference 5).

5. **Interpretation of Results**: Users should interpret the results with the understanding that YOLOS is designed to precisely reveal the transferability of ViT in object detection, and not necessarily to achieve state-of-the-art performance (Reference 7). It is recommended to provide clear documentation and guidelines on how to interpret the results and under what conditions the model performs best.

6. **Future Research Directions**: The references suggest that there is potential in exploring different positional encoding (PE) settings, as some tiny-sized models use a promising type of PE that warrants further study (Reference 4). It is recommended to continue research in this area to understand the impact of PE on model performance and transferability.

7. **Ethical Considerations and Societal Impact**: As a sociotechnic, it is crucial to consider the ethical implications and potential societal impact of deploying object detection models like YOLOS. This includes addressing issues such as privacy, surveillance, and biases in the training data that could lead to unfair outcomes. It is recommended to conduct thorough ethical reviews and bias assessments, and to develop guidelines for responsible use.

In summary, while YOLOS-small-300 provides a valuable framework for understanding the application of Transformers in computer vision, there are several areas for improvement and consideration, particularly in terms of efficiency, supervision requirements, performance optimization, and ethical use.

## Training Details

### Training Data

The training data for the model `hustvl/yolos-small-300` consists of images from the COCO object detection benchmark, which have been fine-tuned following the procedure outlined by Carion et al. The images are pre-processed to ensure the shortest side is at least 480 pixels and at most 800 pixels, with the longest side not exceeding 1333 pixels, and random crop augmentations are applied during training.

### Training Procedure

#### Preprocessing

The preprocessing steps for the data used in the model `hustvl/yolos-small-300` include the following:

1. **Tokenization**: The images are tokenized by reshaping them into a sequence of flattened 2D image patches. This is mentioned in reference 3, where the image `x ∈ R H×W ×C` is reshaped into `x_PATCH ∈ R N ×(P^2 •C)`. The resolution of each image patch is `(P, P)`, and the number of patches `N` is determined by the formula `N = HW / P^2`.

2. **Resizing**: For small and base models, which would include `hustvl/yolos-small-300`, the input images are resized such that the shortest side is at least 480 pixels and at most 800 pixels, while the longest side is at most 1333 pixels. This is specified in reference 7.

3. **Random Crop Augmentations**: During training, random crop augmentations are applied following the method of Carion et al. [10], as mentioned in reference 7.

4. **Data Augmentation**: Additional data augmentation techniques used during pre-training include Rand-Augment and random erasing, as stated in reference 8. These augmentations are implemented by the timm library.

5. **[DET] Tokens**: One hundred randomly initialized learnable [DET] tokens are appended to the [PATCH] tokens to bind objects for final predictions, as described in reference 1.

6. **Position Embeddings**: Position embeddings are added to all the input tokens to retain positional information, using standard learnable 1D position embeddings following Dosovitskiy et al. [21], as mentioned in reference 1.

7. **No Dropout or Stochastic Depth**: During fine-tuning, dropout or stochastic depth is not applied, as these regularization methods were found to hurt performance, as per reference 7.

The above steps outline the preprocessing pipeline for the `hustvl/yolos-small-300` model based on the provided references. If there are additional preprocessing steps specific to the `hustvl/yolos-small-300` variant that are not covered in the references, then [More Information Needed].

#### Training Hyperparameters

The training hyperparameters for the model `hustvl/yolos-small-300` are as follows:

- **Initialization**: The model parameters are initialized from ImageNet-1k pre-trained weights, except for the MLP heads for classification & bounding box regression and the one hundred [DET] tokens, which are randomly initialized.
- **Training Resources**: The model is trained on a single node with 8 × 12G GPUs.
- **Learning Rate**: The initial learning rate is set to 2.5 × 10^-5.
- **Batch Size**: The batch size during training is 8.
- **Learning Rate Decay**: A cosine learning rate decay schedule is used.
- **Weight Decay**: The specific weight decay value is not mentioned in the provided references, so [More Information Needed] for this detail.
- **Input Image Size**: For small and base models, input images are resized such that the shortest side is at least 480 and at most 800 pixels, while the longest side is at most 1333 pixels.
- **Data Augmentation**: Random crop augmentations are applied during training.
- **Number of [DET] Tokens**: There are 100 [DET] tokens used.
- **Loss Function and Weights**: The loss function and loss weights are kept the same as DETR.
- **Regularization Methods**: Dropout or stochastic depth are not applied during fine-tuning as these methods were found to hurt performance.

Please note that some specific details such as the exact weight decay value used during training are not provided in the references, and thus more information would be needed to provide those details.

#### Speeds, Sizes, Times

The model `hustvl/yolos-small-300` is a variant of the YOLOS (You Only Look One-level Series) designed for object detection, closely related to the DETR (Detection Transformer) and ViT (Vision Transformer) architectures. Here are the details based on the provided references:

- Throughput (FPS): The Frames Per Second (FPS) for the model is not explicitly mentioned for the `YOLOS-Small` variant in the provided references. Therefore, for the exact FPS of `hustvl/yolos-small-300`, [More Information Needed].

- Start or End Time: The references do not provide specific start or end times for the training or inference processes of the `hustvl/yolos-small-300` model. [More Information Needed].

- Checkpoint Sizes: The size of the checkpoints for the `YOLOS-Small` model is not directly stated in the provided references. Checkpoint sizes typically depend on the model's architecture complexity, number of parameters, and precision of the weights stored. [More Information Needed].

Additional information that can be inferred from the references includes:

- Input Resolution: The `YOLOS-Small` model uses an input resolution size of 300x300 for inference, as implied by the model name `hustvl/yolos-small-300`.

- Pre-training: The model is pre-trained on the ImageNet-1k dataset using a data-efficient training strategy with parameters initialized using a truncated normal distribution and optimized using AdamW. The learning rate during pre-training is 1 × 10^-3 with a batch size of 1024.

- Fine-tuning: The model is fine-tuned on the COCO object detection benchmark with a learning rate of 2.5 × 10^-5 and a batch size of 8. The learning rate follows a cosine decay schedule, and the weight decay is not specified for the fine-tuning phase.

- Model Design: `YOLOS-Small` follows the original ViT architecture and is optimized for object detection. It is designed to reveal the characteristics of the Transformer family in object detection.

- Detector Heads: The detector head of `YOLOS-Small` is simple, with the classification and bounding box regression heads implemented by one MLP with two hidden layers and ReLU activation functions.

For the most accurate and detailed information regarding throughput, start/end times, and checkpoint sizes, it would be necessary to refer to the actual training logs, model checkpoints, or additional documentation that is not provided in the references above.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model hustvl/yolos-small-300 evaluates on the COCO object detection benchmark.

#### Factors

The foreseeable characteristics that will influence how the model hustvl/yolos-small-300 behaves are as follows:

1. **Transferability of ViT**: As mentioned in reference 1, YOLOS is designed to reveal the transferability of the Vision Transformer (ViT) in object detection tasks. This means that the model's behavior is influenced by the underlying architecture's ability to adapt from natural language processing (NLP) tasks to visual tasks. The model's performance should be interpreted with this design goal in mind, rather than expecting state-of-the-art object detection results.

2. **Specialization of [DET] Tokens**: Reference 2 indicates that [DET] tokens within the model specialize in detecting objects of certain regions and sizes. This suggests that the model may perform differently across various domains and contexts where object sizes and spatial distributions vary. For instance, in scenes with predominantly small objects, the model's detection tokens might behave differently than in scenes with larger objects.

3. **Comparison with Other Models**: The model's behavior can also be compared to other models, such as DETR and tiny-sized CNN detectors, as mentioned in references 3 and 4. The performance of YOLOS-small-300 in terms of accuracy, FLOPs, and FPS should be considered in the context of these comparisons. It is important to note that while YOLOS-Ti is competitive, the small variant may have different trade-offs.

4. **Pre-training and Fine-tuning**: Reference 5 and 6 highlight the sensitivity of object detection results to the pre-training scheme. The model's behavior will be influenced by the specifics of the pre-training on ImageNet-1k and the subsequent fine-tuning on COCO. Different pre-training strategies may lead to variations in detection performance.

5. **Model Design**: As per reference 7, YOLOS closely follows the original ViT architecture and is optimized for object detection. The model's behavior is expected to reflect the characteristics of the Transformer family in object detection tasks. This includes the model's adaptability to various canonical Transformer architectures.

6. **Population Subgroups**: The model card does not provide explicit information on the performance of the model across different population subgroups. [More Information Needed] to evaluate whether there are disparities in performance when detecting objects that are culturally or contextually specific to certain subgroups.

7. **Domain and Context**: The model's performance may vary depending on the domain and context in which it is deployed. For example, the model might behave differently in urban surveillance scenarios compared to natural environments due to variations in object types and distributions. [More Information Needed] to provide a detailed analysis of performance across different domains and contexts.

In summary, the behavior of the hustvl/yolos-small-300 model is influenced by its design to showcase the transferability of ViT, the specialization of detection tokens, the comparison with other models, the pre-training and fine-tuning process, and the inherent characteristics of the Transformer architecture. However, more information is needed to assess performance disparities across different population subgroups and specific domains or contexts. Evaluation should be disaggregated across these factors to uncover any potential disparities in performance.

#### Metrics

The evaluation of the model hustvl/yolos-small-300 will primarily focus on the Average Precision (AP) metric, as this is a standard measure for object detection performance, particularly on the COCO benchmark mentioned in reference 8. AP evaluates the tradeoff between precision and recall across different object detection thresholds, making it a comprehensive metric for assessing detection accuracy.

Additionally, the model will be compared in terms of FLOPs (Floating Point Operations Per Second) and FPS (Frames Per Second), as indicated in references 1 and 2. These metrics are important for understanding the computational efficiency and real-time inference capabilities of the model. FLOPs provide a measure of the computational complexity, while FPS indicates the speed at which the model can process images, which is crucial for applications requiring real-time detection.

The model's sensitivity to object locations and sizes, as well as its insensitivity to object categories, will also be considered as part of the qualitative analysis of detection tokens, as mentioned in reference 5. This analysis will help in understanding the model's behavior in detecting objects of various sizes and its generalization across different categories.

In summary, the evaluation of hustvl/yolos-small-300 will involve:
- Average Precision (AP) for accuracy assessment.
- FLOPs for computational complexity evaluation.
- FPS for real-time inference capability.
- Qualitative analysis of detection tokens for understanding model behavior with respect to object locations, sizes, and categories.

If there were specific error tradeoffs mentioned in the references, such as between different types of detection errors (e.g., false positives, false negatives), they would be considered as well. However, since such details are not provided in the references, we can only assume that the standard object detection metrics (AP, FLOPs, FPS) will be used for evaluation.

### Results

The evaluation results of the model `hustvl/yolos-small-300` based on the provided references are as follows:

- **Architecture Transferability**: The YOLOS architecture demonstrates that the kind of architecture used in YOLOS can be successfully transferred to the COCO object detection benchmark, achieving an Average Precision (AP) of 42.0, which showcases the versatility and generality of the Transformer model in computer vision tasks.

- **Pre-training and Fine-tuning**: The model was pre-trained for 300 epochs on ImageNet-1k and then fine-tuned on the COCO dataset for 150 epochs. This extensive training regimen is indicative of the model's ability to learn robust features from a large dataset and then adapt to the specific task of object detection.

- **Comparisons with DETR**: While specific quantitative comparisons between YOLOS and DETR (DEtection TRansformer) are mentioned, the exact figures for `hustvl/yolos-small-300` are not provided in the references. [More Information Needed]

- **Comparisons with Tiny-sized CNN Detectors**: The tiny-sized YOLOS model, which may refer to `hustvl/yolos-small-300`, is noted to achieve impressive performance in comparison to established CNN object detectors. It is strong in AP and competitive in terms of FLOPs (floating-point operations per second) & FPS (frames per second), even though the Transformer was not specifically designed to optimize these factors.

- **Interpreting the Results**: The performance of YOLOS models, including `hustvl/yolos-small-300`, is meaningful as it demonstrates the transferability of the Vision Transformer (ViT) to object detection tasks. The model is not designed for peak performance but to reveal the capabilities of Transformer architectures in vision tasks.

- **Fine-tuning Details**: The model was fine-tuned on the COCO object detection benchmark with all parameters initialized from ImageNet-1k pre-trained weights, except for the MLP heads and [DET] tokens. The training was conducted on a single node with 8 × 12G GPUs, with a learning rate of 2.5 × 10^-5 and a batch size of 8. The learning rate followed a cosine decay schedule, and weight decay was applied.

- **Performance on COCO**: The YOLOS-Base model, which is closely related to `hustvl/yolos-small-300`, achieved a 42.0 box AP on the COCO validation set. This indicates that the small variant of the model would also be expected to show competitive performance, although specific numbers for `hustvl/yolos-small-300` are not provided. [More Information Needed]

- **Model Design**: YOLOS follows the original ViT architecture and is optimized for object detection. It can be easily adapted to various Transformer architectures, and the simple setup is intended to reveal the characteristics of Transformers in object detection tasks.

In summary, while the exact evaluation results for `hustvl/yolos-small-300` are not explicitly stated in the provided references, the model is part of a family that shows promising transferability and competitive performance in object detection benchmarks, particularly COCO. For specific metrics such as AP, FLOPs, and FPS for `hustvl/yolos-small-300`, [More Information Needed].

#### Summary

The evaluation results for the model `hustvl/yolos-small-300` indicate that it demonstrates promising performance when compared to tiny-sized CNN object detectors. Specifically, the YOLOS-Ti model, which is likely similar in size to the `yolos-small-300`, is noted for its strong Average Precision (AP) and competitive computational efficiency in terms of FLOPs (floating-point operations per second) and FPS (frames per second), despite Transformers not being inherently optimized for these metrics.

The YOLOS model family, including `yolos-small-300`, is designed to explore the transferability of the Vision Transformer (ViT) to object detection tasks, rather than to achieve state-of-the-art performance. The architecture is adapted from NLP models like BERT-Base, with minimal modifications, to show that Transformers can be effectively applied to computer vision challenges such as the COCO object detection benchmark. The `yolos-small-300` model, as part of this family, likely shares these characteristics and demonstrates the versatility and generality of the Transformer architecture.

In terms of qualitative analysis, the YOLOS models use [DET] tokens to represent detected objects, and these tokens are found to be sensitive to object locations and sizes but less so to object categories. This indicates a specialization in the model's ability to detect objects based on their spatial characteristics.

When compared to DETR (Detection Transformer), which is another Transformer-based object detection model, YOLOS, including `yolos-small-300`, follows a simpler design philosophy. This simplicity is intentional, aiming to provide an unbiased view of how Transformer architectures perform in object detection tasks.

Lastly, the `yolos-small-300` model's performance on object detection is sensitive to the pre-training scheme used, suggesting that it can serve as a benchmark for evaluating different pre-training strategies for Transformers in vision tasks.

For more detailed results specific to `hustvl/yolos-small-300`, such as exact AP scores, FPS, and FLOPs, [More Information Needed] as they are not provided in the given references.

## Model Examination

### Model Card: hustvl/yolos-small-300

#### Explainability/Interpretability

In our efforts to enhance the transparency and understanding of the hustvl/yolos-small-300 model, we have conducted a series of studies focusing on the model's self-attention mechanisms and their role in object detection tasks.

Our investigations, as referenced in the provided materials, have shown that different self-attention heads within the YOLOS model concentrate on various patterns and spatial locations. This diversity in attention allows the model to capture a wide range of features relevant to the detection task. However, it is important to note that while some of these attention patterns are interpretable, others remain elusive to our current methods of analysis.

Specifically, we have compared the attention maps of two YOLOS models: one trained for 200 epochs and the other for 300 epochs on the ImageNet-1k dataset. Despite both models achieving the same Average Precision (AP) of 36.1, our visualizations have revealed distinct attention behaviors between them. This suggests that the model's internal representations can vary significantly even when performance metrics do not reflect such differences.

To visualize the self-attention of the [DET] tokens, which are crucial for predictions, we employed the visualization pipeline as described by Caron et al. The results, illustrated in Figures 4 and 5, provide insights into how the [DET] tokens relate to the model's predictions and highlight the interpretability of the attention mechanism in the context of object detection.

Our approach aligns with the broader vision of adapting methodologies from NLP to computer vision, particularly in the use of Transformers. The YOLOS design philosophy emphasizes the transferability of the Vision Transformer (ViT) to downstream vision tasks with minimal modifications, following the spirit of NLP where pre-trained models adapt to new tasks with little additional training.

In conclusion, while the performance of hustvl/yolos-small-300 may not be the primary focus, the insights gained from our interpretability studies are valuable. They not only contribute to our understanding of the model's internal workings but also serve as a step towards unifying methodologies across NLP and computer vision domains.

## Environmental Impact

- **Hardware Type:** The model hustvl/yolos-small-300 was trained on a single node with 8 × 12G GPUs.
- **Software Type:** The model hustvl/yolos-small-300 is trained on software that includes the following:

1. The training strategy and optimization are based on the methods described by Carion et al. [10] and Touvron et al. [58], suggesting the use of AdamW optimizer [40] and a cosine learning rate decay schedule.

2. Data augmentation techniques such as Rand-Augment [14] and random erasing [70] are utilized, which are implemented by the timm library [65].

3. The model architecture is based on the Vision Transformer (ViT) [21], and the object detection setup follows the approach by Carion et al. [10].

4. The model is fine-tuned and pre-trained on the COCO object detection benchmark and ImageNet-1k dataset, respectively.

5. The detector heads are implemented using multi-layer perceptrons (MLPs) with ReLU activation functions [41].

Based on the references provided, the specific software frameworks or libraries (like PyTorch, TensorFlow, etc.) used for training the model are not explicitly mentioned. However, given the use of the timm library [65] for data augmentation, it is likely that the model was trained using PyTorch, as timm is a PyTorch library. Without explicit confirmation in the references, the exact software type cannot be definitively provided.

[More Information Needed]
- **Hours used:** The amount of time used to train the model hustvl/yolos-small-300 is not explicitly stated in the provided references. Therefore, the answer is "[More Information Needed]".
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture of `hustvl/yolos-small-300` is based on the You Only Look at One Sequence (YOLOS) design, which is a series of object detection models that closely follow the canonical Vision Transformer (ViT) architecture with minimal modifications. The YOLOS design philosophy is to maintain the simplicity and generality of the Transformer architecture, avoiding complex and heavy detector head designs. The detector head in YOLOS is as straightforward as the image classification layer of ViT, with both the classification and bounding box regression heads implemented by a single multilayer perceptron (MLP) with two hidden layers and ReLU activation functions.

The objective of the `hustvl/yolos-small-300` model is not to achieve the highest possible performance on object detection tasks but to investigate the versatility and transferability of pre-trained Transformers from image recognition to object detection. The model aims to perform 2D object and region-level recognition from a pure sequence-to-sequence perspective with minimal knowledge about the 2D spatial structure. This is achieved by replacing the [CLS] token used in ViT for image classification with one hundred [DET] tokens for object detection, and by adapting the position embedding to accommodate changes in input resolutions and aspect ratios.

The YOLOS architecture is designed with minimal additional inductive biases injection. The inductive biases inherent from ViT come from the patch extraction at the network stem part and the resolution adjustment for position embeddings. YOLOS adds no non-degenerated convolutions upon ViT, and it uses [DET] tokens to bind objects for final predictions, avoiding additional 2D inductive biases and task-specific heuristics.

In summary, `hustvl/yolos-small-300` is a YOLOS model that leverages the simplicity of the ViT architecture to explore the fundamental capabilities of Transformers in the object detection domain, with an emphasis on understanding the transferability of pre-trained models to new tasks with minimal task-specific modifications.

### Compute Infrastructure

The compute infrastructure used for training the model hustvl/yolos-small-300 involved a single node with 8 × 12G GPUs. The training leveraged a fine-tuning process on the COCO object detection benchmark, with parameters initialized from ImageNet-1k pre-trained weights, except for the MLP heads and [DET] tokens which were randomly initialized. The learning rate was set to 2.5 × 10^−5 and the batch size was 8. A cosine learning rate decay and weight decay were applied during the training process.

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

