# Model Card for SenseTime/deformable-detr

The SenseTime/deformable-detr model is an efficient and fast-converging end-to-end object detector that improves upon the original DETR by incorporating deformable attention modules, which focus on a small set of key sampling points. This approach allows for better performance, particularly with small objects, and requires significantly fewer training epochs compared to its predecessor.

## Model Details

### Model Description

### Model Card for SenseTime/deformable-detr

#### Model Architecture
Deformable DETR is an end-to-end object detection model that enhances the original DETR (Detection Transformer) by incorporating deformable attention modules. These modules are designed to attend to a small set of key sampling points around a reference, which allows for efficient processing of convolutional feature maps. The architecture replaces the standard Transformer attention modules with multi-scale deformable attention modules, maintaining linear complexity with respect to the spatial size of the feature maps. The encoder extracts multi-scale feature maps from the output feature maps of stages C3 through C5 in a ResNet backbone, which are then processed by the deformable attention mechanism.

#### Training Procedures
- The model is trained with a total batch size of 32.
- Training and inference speeds are benchmarked on an NVIDIA Tesla V100 GPU.
- The model is trained for 50 epochs with the learning rate decayed at the 40th epoch by a factor of 0.1.
- Adam optimizer is used with a base learning rate of 2 × 10^−4, β1 = 0.9, β2 = 0.999, and weight decay of 10^−4.
- Learning rates for certain linear projections are multiplied by a factor of 0.1.
- The model uses an ImageNet pre-trained ResNet-50 as the backbone.
- Focal Loss with a loss weight of 2 is used for bounding box classification.
- The number of object queries is increased from 100 to 300.

#### Parameters
- The deformable attention modules are set with M = 8 and K = 4 by default.
- Parameters of the deformable Transformer encoder are shared among different feature levels.
- Variants such as "Deformable DETR (single scale)" and "DC5" are mentioned, indicating different configurations of the input feature maps and the backbone network.

#### Important Disclaimers
- The complexity analysis provided in the references indicates that the model is designed to be computationally and memory efficient.
- The model's fast convergence and efficiency are highlighted, but it is noted that only the core ideas and improvements are introduced due to limited space, with implementation details available in the appendix.
- The performance of the model is reported in comparison to DETR-DC5 with similar modifications for fairness.
- The model is positioned as a starting point for exploring various variants of end-to-end object detectors.

#### Contact for Model Card Updates
For any updates or inquiries regarding the model card, please contact the project organizer at [More Information Needed].

#### Visualizations
Two figures are referenced, likely illustrating the architecture and convergence properties of the model, but the actual images are not included in the text provided.

#### Additional Notes
- The model card should be updated with actual contact information for the project organizer.
- The visualizations referenced should be included in the final version of the model card for better understanding of the model's architecture and performance.
- Further details on the performance metrics and comparison with other models would be beneficial for users evaluating the model's effectiveness for their specific use cases.

- **Developed by:** Xizhou Zhu; Weijie Su; Lewei Lu; Bin Li; Xiaogang Wang; Jifeng Dai; Sensetime Research
- **Funded by:** The project for the model SenseTime/deformable-detr is funded by the following entities:

1. National Key R&D Program of China (2020AAA0105200)
2. Beijing Academy of Artificial Intelligence
3. National Natural Science Foundation of China under grant No.U19B2044 and No.61836011
- **Shared by:** The contributors who made the model SenseTime/deformable-detr available online as a GitHub repo are Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai from SenseTime Research.
- **Model type:** The SenseTime/deformable-detr model is an end-to-end object detector trained with a total batch size of 32 on a NVIDIA Tesla V100 GPU, utilizing a novel deformable attention mechanism within a Transformer-based architecture, and operates on single-image modality.
- **Language(s):** The model SenseTime/deformable-detr does not use or process natural human language; it is an object detection model that processes image data.
- **License:** The model SenseTime/deformable-detr is released under the Apache 2.0 license. The link to the license is not directly provided in the text above, but it is mentioned that the project is released under the [Apache 2.0 license](./LICENSE).
- **Finetuned from model:** The model SenseTime/deformable-detr does not explicitly mention being fine-tuned from another specific model in the provided references. However, it is based on the DETR (DEtection TRansformer) framework, which is a known model in the field of object detection. Since the references do not provide a direct link to the DETR model or any other base model from which SenseTime/deformable-detr was fine-tuned, I would say:

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/fundamentalvision/Deformable-DETR
- **Paper:** https://arxiv.org/pdf/2010.04159.pdf
- **Demo:** [More Information Needed]
## Uses

### Direct Use

The model SenseTime/deformable-detr is designed as an end-to-end object detector, which means it can directly output the detection results without the need for additional fine-tuning, post-processing, or integration into a larger pipeline. This is possible due to the model's architecture and training process, which have been optimized to perform well on object detection tasks straight out of the box.

The deformable DETR model uses a novel deformable attention mechanism that allows it to efficiently process image feature maps and predict bounding boxes relative to reference points. This reduces the optimization difficulty and allows the model to converge faster during training, as mentioned in reference 6. As a result, the model achieves better performance, especially on small objects, with significantly fewer training epochs compared to the original DETR model.

To use the model without fine-tuning or post-processing, you would typically load the pre-trained model and pass an image through it to obtain the detection results. However, since no direct code block reference is provided in the text above, I cannot provide a specific code snippet. If you have access to the Huggingface model repository, you would typically find usage instructions there, which would include code examples on how to use the model for inference.

In summary, the model can be used as follows (hypothetical code snippet, as no direct code is provided in the references):

```python
from transformers import DeformableDETRModel, DeformableDETRConfig

# Load pre-trained model
model = DeformableDETRModel.from_pretrained('SenseTime/deformable-detr')

# Prepare your image (assuming you have a way to convert it to the required format)
input_image = ... # Your code to process the image

# Perform object detection
outputs = model(input_image)

# The outputs contain the predicted bounding boxes and class labels
predicted_boxes = outputs.pred_boxes
predicted_labels = outputs.pred_labels
```

Please note that the above code is hypothetical and meant to illustrate how one might use the model without fine-tuning or post-processing. For actual usage, you would need to refer to the official documentation or the Huggingface model repository for the correct code snippet and any additional requirements for the input image format. If the model is not available on Huggingface or if there are no usage instructions, then [More Information Needed] to provide a precise code example.

### Downstream Use

The SenseTime/deformable-detr model is a state-of-the-art object detection model that can be fine-tuned for various object detection tasks across different domains. When fine-tuning this model for a specific task, users can leverage its efficient and fast-converging properties to achieve high performance, particularly in scenarios where detecting small objects is crucial.

For instance, in a retail environment, the model could be fine-tuned to detect and identify products on shelves, enabling automated inventory management. In the field of autonomous driving, the model could be adapted to detect pedestrians, vehicles, and traffic signs, contributing to the safety and reliability of the system.

When integrating SenseTime/deformable-detr into a larger ecosystem or application, it can serve as a core component for real-time object detection functionalities. For example, in a smart city surveillance system, the model can be deployed to monitor and analyze video feeds for security and traffic management purposes.

To fine-tune or evaluate the model on a new dataset, users can follow the instructions provided in the reference to obtain the config file and pretrained model. They can then run the evaluation command on their dataset, potentially adjusting the command to accommodate their specific setup:

```bash
<path to config file> --resume <path to pre-trained model> --eval
```

For distributed evaluation, which might be necessary for large-scale datasets or when using multiple GPUs, users can use the provided scripts:

```bash
./tools/run_dist_launch.sh
```
or
```bash
./tools/run_dist_slurm.sh
```

Please note that the actual fine-tuning process would require additional steps, such as adjusting the config file to the new task, preparing the dataset in the required format, and potentially modifying the loss function or other hyperparameters to suit the specific needs of the task. However, as the exact code for fine-tuning is not provided in the references, [More Information Needed] for a complete fine-tuning code snippet.

### Out-of-Scope Use

As the sociotechnic on the development team for the SenseTime/deformable-detr model, it is my responsibility to consider the potential societal impacts and misuse of our technology. The Deformable DETR model is an advanced object detection system that has shown improved performance over its predecessor, DETR, especially in terms of training efficiency and the detection of small objects. While the technical advancements are significant, we must also be aware of how this technology could be misused.

Potential misuses of the SenseTime/deformable-detr model include:

1. **Surveillance without Consent**: The model's enhanced ability to detect objects, including small objects, could be used for unauthorized surveillance purposes. Users should not employ this model to monitor individuals without their consent, which would be a violation of privacy rights.

2. **Military Applications**: The technology could be adapted for military use in automated targeting systems or reconnaissance, which raises ethical concerns regarding autonomous weaponry and the potential for collateral damage.

3. **Bias and Discrimination**: If the training data is not representative of diverse populations or scenarios, the model could inadvertently perpetuate biases, leading to discriminatory practices. Users should not use the model in contexts where biased results could lead to unfair treatment of individuals or groups.

4. **Illegal Activities**: The model's capabilities could be exploited for illegal activities such as theft, where object detection could be used to identify and target valuable items or to circumvent security systems.

To address these concerns, users of the SenseTime/deformable-detr model should:

- Ensure that the use of the technology complies with all applicable laws and regulations, including those related to privacy and surveillance.
- Avoid using the model in any military or weaponized applications that could lead to harm or ethical violations.
- Be vigilant about the potential for bias in the model's predictions and take steps to mitigate this by using diverse and representative datasets.
- Refrain from using the model for any illegal or unauthorized purposes.

It is also important for us as developers to continue to engage with ethicists, legal experts, and other stakeholders to understand the broader implications of our technology and to develop guidelines that help prevent misuse. We encourage users to report any unethical or illegal use of the model so that appropriate actions can be taken.

### Bias, Risks, and Limitations

The SenseTime/deformable-detr model, as an advanced object detection system, presents several known and foreseeable issues that can be categorized into technical and sociotechnical limitations:

1. **Technical Limitations:**
   - **Model Complexity and Computational Resources:** The deformable attention module, while efficient, still requires significant computational resources, as indicated by the complexity analysis in references 7 and 11. This could limit the model's deployment in resource-constrained environments.
   - **Memory Access and Runtime Speed:** As mentioned in reference 9, the model has unordered memory access, which makes it slightly slower than traditional convolution-based methods. This could affect real-time applications where speed is crucial.
   - **Training Epochs and Convergence:** Although Deformable DETR requires fewer training epochs than DETR, as per reference 10, it still necessitates a substantial amount of training, which could be a barrier for rapid prototyping or for researchers with limited computational resources.
   - **Detection Performance on Small Objects:** While Deformable DETR improves the detection of small objects compared to DETR, as stated in reference 10, there may still be challenges in detecting very small or densely packed objects, which is a common issue in object detection tasks.

2. **Sociotechnical Limitations:**
   - **Bias and Fairness:** The model's performance is dependent on the training data. If the training data is biased, the model may inherit and perpetuate these biases, leading to unfair or discriminatory outcomes. [More Information Needed] to assess the specific biases in the training dataset.
   - **Transparency and Explainability:** The complexity of the deformable attention mechanism may make it difficult for users to understand how the model makes its predictions, which is important for applications in sensitive areas such as healthcare or law enforcement.
   - **Misuse and Dual Use:** As with any object detection technology, there is potential for misuse, including surveillance without consent or in violation of privacy rights. The model's capabilities could be repurposed for harmful or unethical applications.
   - **Regulatory Compliance:** Depending on the jurisdiction and application, the deployment of object detection models may be subject to regulatory scrutiny, particularly regarding privacy and data protection. [More Information Needed] to determine specific legal compliance requirements.
   - **Accessibility and Inclusivity:** The model's design and outputs need to be accessible and inclusive to diverse user groups. There may be challenges in ensuring the model works equitably across different demographics and does not exclude or disadvantage any group.

In conclusion, while the SenseTime/deformable-detr model represents a significant advancement in object detection, it is important to consider and address these technical and sociotechnical issues to ensure responsible and equitable use.

### Recommendations

Given the information provided and my role as a sociotechnic, the recommendations with respect to the foreseeable issues about the model SenseTime/deformable-detr would include:

1. **Bias and Fairness**: It is important to ensure that the model does not perpetuate or exacerbate biases present in the training data. Since the model is trained on a specific dataset (not mentioned in the references but typically COCO for object detection models), it may not perform equally well across different demographics or scenes. We recommend conducting thorough bias audits and ensuring diverse and representative datasets for training and evaluation.

2. **Transparency and Explainability**: The model card should clearly explain how the model works, including the deformable attention mechanism and its implications. This is crucial for users to understand the model's decision-making process, which is particularly important when the model is used in sensitive applications.

3. **Privacy Concerns**: As an object detection model, Deformable DETR may be used in surveillance or other applications that raise privacy concerns. We recommend establishing clear guidelines for ethical usage and ensuring compliance with privacy regulations such as GDPR.

4. **Robustness and Security**: The model should be tested against adversarial attacks and other forms of manipulation to ensure its robustness in real-world applications. We recommend including information on known vulnerabilities and mitigation strategies.

5. **Environmental Impact**: The training and inference processes for deep learning models can be energy-intensive. We recommend documenting the model's carbon footprint and suggesting ways to mitigate environmental impact, such as using more efficient hardware or optimizing model architecture.

6. **Accessibility**: The model should be accessible to a wide range of users, including those with limited computational resources. We recommend providing a lighter version of the model if possible and documenting the model's performance across different hardware configurations.

7. **Legal and Ethical Use Cases**: The model card should include recommendations for legal and ethical use cases, discouraging applications that may lead to harm or violate ethical standards.

8. **Performance Limitations**: The model card should clearly state the limitations of the model's performance, including potential accuracy drops in different conditions or with different object classes, as indicated by the ablation studies in the references.

9. **Reproducibility**: Since there are slight differences in accuracy and running time due to platform switch, we recommend providing detailed documentation to ensure reproducibility of the results, including the exact environment and settings used.

10. **Maintenance and Updates**: The model card should outline the process for maintaining and updating the model, including how users can report issues or contribute to its improvement.

By addressing these recommendations, the model card for SenseTime/deformable-detr will help ensure that the model is used responsibly and effectively, while also considering the broader societal implications of its deployment.

## Training Details

### Training Data

The training data for the SenseTime/deformable-detr model consists of the COCO 2017 dataset, which includes a diverse set of images for object detection tasks. The model is trained on the train set of COCO 2017 and evaluated on the validation and test-dev sets. [More Information Needed] on data pre-processing or additional filtering as it is not explicitly mentioned in the provided references.

### Training Procedure

#### Preprocessing

The preprocessing steps for the SenseTime/deformable-detr model involve preparing the input image data to be compatible with the model's architecture. Based on the references provided, here are the preprocessing details:

1. **Feature Map Extraction**: The input images are first processed through a CNN backbone, such as ResNet (He et al., 2016), to extract feature maps. This is mentioned in reference 8, where it states that input feature maps `x ∈ R C×H×W` are extracted by a CNN backbone.

2. **Multi-Scale Feature Maps**: The model utilizes multi-scale feature maps, as described in reference 2 and 6. Feature maps `{x l } L−1 l=1` are extracted from the output feature maps of stages C3 through C5 in ResNet, transformed by a `1 × 1` convolution. These feature maps are of different resolutions, with each subsequent level `C l` being of resolution `2^l` lower than the input.

3. **Normalization**: The coordinates of the reference points for each query element are normalized, as mentioned in reference 6. The normalized coordinates `p q ∈ [0, 1] 2` are used for the multi-scale deformable attention module.

4. **Detection Head Preprocessing**: In the first stage of the Two-Stage Deformable DETR, a detection head is applied to each pixel of the output feature maps from the encoder. This involves a 3-layer FFN for bounding box regression and a linear projection for bounding box binary classification (foreground/background), as described in reference 5.

5. **Object Queries**: For the Transformer decoder, N object queries are represented by learnable positional embeddings, as stated in reference 9. These object queries are used in both cross-attention and self-attention modules within the decoder.

6. **Bounding Box Normalization**: The bounding box coordinates predicted by the FFN are normalized, as indicated by the notation `b ∈ [0, 1]` in reference 8.

The preprocessing steps do not include explicit tokenization as this is an object detection model dealing with image data rather than text. Resizing of images to fit the input size expected by the CNN backbone may be required, but this is not explicitly mentioned in the provided references. If resizing is needed, it would typically involve scaling the images while maintaining their aspect ratio, followed by padding to reach the required input dimensions.

For any specific preprocessing code or further details on the resizing of images, [More Information Needed] as it is not provided in the references.

#### Training Hyperparameters

The training hyperparameters for the SenseTime/deformable-detr model are as follows:

1. **Backbone**: The model uses a ResNet-50 pre-trained on ImageNet as the backbone for feature extraction.
2. **Feature Maps**: Multi-scale feature maps are extracted without using FPN.
3. **Deformable Attention Parameters**: By default, M (the number of attention heads) is set to 8, and K (the number of sampled keys in each attention head) is set to 4. The parameters of the deformable Transformer encoder are shared among different feature levels.
4. **Loss Function**: Focal Loss with a loss weight of 2 is used for bounding box classification.
5. **Object Queries**: The number of object queries is increased from 100 to 300.
6. **Training Epochs**: Models are trained for 50 epochs with the learning rate decayed at the 40th epoch by a factor of 0.1.
7. **Optimizer**: Adam optimizer is used with a base learning rate of 2 × 10^-4, β1 = 0.9, β2 = 0.999, and weight decay of 10^-4.
8. **Learning Rate for Projections**: Learning rates of the linear projections, used for predicting object query reference points and sampling offsets, are multiplied by a factor of 0.1.
9. **Batch Size**: All models are trained with a total batch size of 32.
10. **Training and Inference Speed**: Measured on NVIDIA Tesla V100 GPU.
11. **Dataset**: The models are trained on the COCO 2017 train set and evaluated on the val set and test-dev set.

These hyperparameters are based on the modifications and settings described in the provided references.

#### Speeds, Sizes, Times

The model SenseTime/deformable-detr has been trained on the COCO 2017 dataset and demonstrates significant improvements in object detection tasks, particularly for small objects, with considerably fewer training epochs compared to the original DETR model. Here are the details regarding the model's throughput, timing, and checkpoint sizes:

- **Throughput**: The models of Deformable DETR, including SenseTime/deformable-detr, are trained with a total batch size of 32. The training and inference speeds are measured on an NVIDIA Tesla V100 GPU. For inference, the "Batch Infer Speed" is reported with a batch size of 4 to maximize GPU utilization. However, specific throughput metrics such as images per second are not provided in the references, so [More Information Needed] for exact numbers.

- **Start or End Time**: The models are trained for 50 epochs with the learning rate decayed at the 40th epoch by a factor of 0.1. The references do not provide exact start or end times for the training process, so [More Information Needed] for precise timing details.

- **Checkpoint Sizes**: The references do not specify the checkpoint sizes for the SenseTime/deformable-detr model. Therefore, [More Information Needed] regarding the size of the model checkpoints.

For further details and updates, users are directed to check the [changelog.md](./docs/changelog.md) file. Additionally, the config file and pretrained model can be obtained from the "Main Results" section, and the model can be evaluated on the COCO 2017 validation set using the provided commands.

It's important to note that the original implementation is based on an internal codebase, and there might be slight differences in accuracy and running time due to details in the platform switch. The code for Deformable DETR is available at the provided GitHub repository, which may contain additional information and updates beyond the knowledge cutoff date.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model SenseTime/deformable-detr evaluates on the COCO (Common Objects in Context) benchmark.

#### Factors

The SenseTime/deformable-detr model exhibits several characteristics that will influence its behavior across different domains, contexts, and population subgroups. Here are some of the key factors to consider:

1. **Training Data and Domain**: The model is trained on ImageNet pre-trained ResNet-50 as the backbone, which suggests that the model's performance is influenced by the distribution and characteristics of the ImageNet dataset. If the dataset has underrepresented certain classes or demographics, the model may exhibit biases or reduced performance on those underrepresented groups.

2. **Model Architecture and Features**: The use of deformable attention modules and the absence of FPN (Feature Pyramid Networks) indicate that the model's performance may vary across different object scales. The model is reported to perform better on small objects compared to the original DETR model, which suggests that it may be more suitable for domains where detecting small objects is crucial.

3. **Hyperparameters and Training Strategy**: The model uses specific hyperparameters such as M=8 and K=4 for deformable attentions, and the number of object queries is increased from 100 to 300. These settings are likely to affect the model's sensitivity and precision in detecting objects, which could lead to disparities in performance across different scenarios where objects vary in number and density.

4. **Inference Speed**: The inference speed is measured on NVIDIA Tesla V100 GPU, which means that the model's performance in terms of speed is optimized for this particular hardware. In real-world applications, the performance may vary significantly on different hardware, potentially affecting its suitability for deployment in resource-constrained environments.

5. **Performance Metrics**: The model's performance is measured using Average Precision (AP) and particularly notes improvements on small objects. However, performance on other object sizes or in different contexts (e.g., crowded scenes, varying lighting conditions) is not explicitly mentioned, which could indicate potential disparities in performance across these factors.

6. **Convergence and Training Efficiency**: Deformable DETR is noted to achieve better performance with significantly fewer training epochs compared to Faster R-CNN + FPN and DETR. This characteristic implies that the model may be more efficient to train, but it does not provide information on how the model performs with different amounts of training data or in transfer learning scenarios.

7. **Visualization and Interpretability**: The model provides visualizations of what it looks at to give the final detection result, which can be useful for understanding model behavior. However, without specific information on how these visualizations vary across different population subgroups or domains, it is difficult to assess whether there are disparities in how the model interprets different types of images.

In summary, while the SenseTime/deformable-detr model shows promising improvements in certain areas, such as detecting small objects and training efficiency, there is a need for more detailed evaluation across a variety of factors to uncover potential disparities in performance. This includes testing the model on diverse datasets that represent different population subgroups and environmental conditions, as well as assessing its performance on a range of hardware platforms.

#### Metrics

The evaluation metrics for the SenseTime/deformable-detr model will primarily be based on the COCO benchmark, which is a standard dataset for object detection evaluation. The key metric used will be Average Precision (AP), which is a common metric for object detection models that measures the precision of the model at different recall levels. The model card should highlight the following points regarding evaluation metrics:

1. **Average Precision (AP)**: The model's performance is quantified using AP, which is a standard metric for object detection tasks. The AP metric is particularly important as it captures the precision of the model across different levels of recall, providing a comprehensive measure of the model's detection capabilities.

2. **AP on Small Objects (AP S)**: Given that the Deformable DETR model is specifically noted for its improved performance on small objects, it is crucial to report AP for small objects separately. This metric will demonstrate the model's enhanced ability to detect smaller objects compared to the baseline DETR model.

3. **AP across Different Scales**: The model uses multi-scale inputs and deformable attention modules, which are designed to improve detection accuracy across different object sizes. Therefore, reporting AP across different scales (small, medium, and large objects) will be important to showcase the effectiveness of these design choices.

4. **Convergence Curves**: While not a direct evaluation metric, the convergence curves (as mentioned in reference 2) can provide insights into the training efficiency of the model. These curves can help in understanding the trade-offs between training time and model performance.

5. **Comparison with Baselines**: The model card should include a comparison of the Deformable DETR model's performance with that of the baseline DETR and DETR-DC5 models, as well as with Faster R-CNN + FPN. This comparison will help users understand the trade-offs in terms of training efficiency and detection accuracy.

6. **Ablation Studies**: The results from ablation studies (as mentioned in reference 3) should be included to show the impact of various design choices on the model's performance. This includes the use of multi-scale deformable attention, the number of sampling points (K), and the effect of cross-level feature exchange.

7. **Runtime Performance**: Reporting the runtime on a standard hardware configuration, such as the NVIDIA Tesla V100 GPU (as mentioned in reference 5), will be useful for users to understand the trade-offs between computational efficiency and accuracy.

In summary, the model card should focus on AP as the primary metric, with additional details on AP S, AP across different scales, convergence efficiency, comparison with baselines, ablation study results, and runtime performance to provide a comprehensive evaluation of the SenseTime/deformable-detr model.

### Results

Evaluation results of the model SenseTime/deformable-detr are as follows:

Factors:
1. Training Epochs: Deformable DETR requires significantly fewer training epochs to converge compared to DETR. While DETR needs 500 epochs to converge on the COCO benchmark, Deformable DETR achieves better performance with 10× fewer training epochs.
2. Object Detection Performance: Deformable DETR delivers improved performance in detecting small objects compared to both Faster R-CNN + FPN and the original DETR model.
3. Model Variants: The "Deformable DETR (single scale)" variant uses only the res5 feature map as input for the Deformable Transformer Encoder. The "DC5" variant involves removing the stride in the C5 stage of ResNet and adding a dilation of 2. "DETR-DC5+" indicates DETR-DC5 with modifications such as using Focal Loss for bounding box classification and increasing the number of object queries to 300.
4. Training and Inference Speed: Training and inference speeds are measured on an NVIDIA Tesla V100 GPU. Batch inference speed refers to inference with a batch size of 4 to maximize GPU utilization.

Metrics:
1. Convergence Curves: Detailed convergence curves for Deformable DETR are shown in Fig. 3 of the provided references, indicating the model's learning progress over epochs.
2. Object Queries: The number of object queries is increased from 100 to 300 to improve detection performance.
3. Learning Rate and Optimizer: Models are trained using the Adam optimizer with a base learning rate of 2 × 10^-4, β1 = 0.9, β2 = 0.999, and a weight decay of 10^-4. Learning rates for specific linear projections are multiplied by a factor of 0.1.
4. Batch Size: All models of Deformable DETR are trained with a total batch size of 32.
5. Performance on COCO Benchmark: Extensive experiments on the COCO benchmark demonstrate the effectiveness of Deformable DETR, especially in terms of improved detection accuracy for small objects.

For more detailed quantitative results such as specific precision, recall, mAP scores, or comparisons with other models, [More Information Needed] as these are not provided in the given references.

#### Summary

The evaluation results for the SenseTime/deformable-detr model indicate that it achieves better performance compared to the original DETR model, particularly in detecting small objects, while requiring significantly fewer training epochs (10× less). When compared with Faster R-CNN + FPN, Deformable DETR also shows improved performance, although DETR requires more epochs to converge and generally underperforms in detecting small objects.

The Deformable DETR model introduces modifications such as a weight of 2 for bounding box classification and an increase in the number of object queries from 100 to 300. These changes, along with others, contribute to its enhanced performance. The model is trained using the Adam optimizer with a base learning rate of 2 × 10^−4, β1 = 0.9, β2 = 0.999, and a weight decay of 10^−4. Learning rates for specific linear projections are reduced by a factor of 0.1.

All models of Deformable DETR are trained with a total batch size of 32 on NVIDIA Tesla V100 GPUs. The training and inference speeds are measured on the same GPU model. The "Deformable DETR (single scale)" variant uses only the res5 feature map as input for the Deformable Transformer Encoder, and the "DC5" variant involves a modification to the C5 stage of ResNet.

In terms of computational efficiency, Deformable DETR has comparable FLOPs to Faster R-CNN + FPN and DETR-DC5 but offers a faster runtime speed, being 1.6× faster than DETR-DC5 and only 25% slower than Faster R-CNN + FPN. This speed advantage is attributed to the deformable attention mechanism, which reduces the memory access overhead associated with Transformer attention, albeit with some trade-off in terms of unordered memory access.

Extensive experiments on the COCO benchmark have demonstrated the effectiveness of the Deformable DETR approach. Additionally, the model's attention to specific image pixels for final detection results has been studied, revealing which pixels are most influential in the model's predictions.

The code for Deformable DETR has been made publicly available, facilitating further research and development in the field.

## Model Examination

In the experimental section dedicated to the explainability and interpretability of the SenseTime/deformable-detr model, we explore how the model makes its predictions and which parts of the input images are most influential in the decision-making process.

Our Deformable DETR model extends the capabilities of the original DETR model by incorporating deformable attention mechanisms. This allows the model to focus on a sparse set of key sampling points rather than the entire image, which enhances its ability to handle objects of various scales and improves computational efficiency.

1. Gradient Norm Visualization: To understand what the model looks at when making predictions, we visualize the gradient norm of each prediction component with respect to each pixel in the image. This visualization, as shown in Fig. 5, helps us identify which pixels significantly impact the model's output. The gradient norm reflects the sensitivity of the output to changes in pixel values, indicating the importance of specific image regions in the prediction process.

2. Attention to Object Extremes and Interior: Unlike the original DETR, which focuses on the extreme points of objects, our Deformable DETR also considers the interior points of objects to determine their categories. This is evident from the visualizations of the partial derivatives of the category score with respect to the input image (∂c/∂I in Fig. 5). The model attends to the left/right boundaries for the x coordinate and width, and the top/bottom boundaries for the y coordinate and height, while also considering pixels inside the object for category prediction.

3. Multi-Scale Deformable Attention Modules: We provide visualizations of the sampling points and attention weights from the last layer of the encoder and decoder to better understand the learned multi-scale deformable attention modules (Fig. 6). These visualizations combine information from feature maps of different resolutions, offering insights into how the model processes information across scales.

4. Model Complexity: The deformable attention module is designed to process convolutional feature maps efficiently. With a complexity of O(HW C^2) when applied in the DETR encoder, the module maintains linear complexity with respect to the spatial size of the feature maps. This efficiency is crucial for the model's scalability and practicality in real-world applications.

In summary, our Deformable DETR model provides a more nuanced understanding of object detection by considering both the boundaries and interior points of objects. The visualizations and analyses of attention mechanisms and gradient norms contribute to the interpretability of the model, offering insights into its decision-making process. This understanding is vital for improving the model's reliability and trustworthiness in deployment scenarios.

## Environmental Impact

- **Hardware Type:** The model SenseTime/deformable-detr is trained on NVIDIA Tesla V100 GPU.
- **Software Type:** The model SenseTime/deformable-detr is trained on a software environment that includes Linux with CUDA version 9.2 or higher and GCC version 5.4 or higher. It also requires Python version 3.7 or higher, PyTorch version 1.5.1 or higher, and torchvision version 0.6.1 or higher. The use of Anaconda to create a conda environment is recommended for setting up the software dependencies.
- **Hours used:** The model SenseTime/deformable-detr was trained for 50 epochs, with the learning rate decayed at the 40-th epoch by a factor of 0.1. However, the exact amount of time used to train the model is not provided in the references given. Therefore, to answer the question about the specific amount of time used to train the model, [More Information Needed].
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture of SenseTime/deformable-detr is based on the Deformable DETR framework, which is an end-to-end object detector. The architecture consists of a deformable transformer encoder and decoder, which replace the standard transformer attention modules in DETR with deformable attention modules.

The deformable transformer encoder processes multi-scale feature maps extracted from the output feature maps of stages C3 through C5 in a ResNet backbone. These feature maps are transformed by a 1x1 convolution and maintain the same resolutions. The encoder uses a multi-scale deformable attention module, which has a linear complexity with the spatial size of the feature maps.

The deformable transformer decoder includes cross-attention and self-attention modules, where the query elements are object queries. In the cross-attention modules, object queries extract features from the feature maps output by the encoder. In the self-attention modules, object queries interact with each other. The deformable attention module in the decoder attends to a small set of key sampling points around a reference point, which reduces the complexity and allows for faster convergence.

The objective of the SenseTime/deformable-detr model is to provide an efficient and fast-converging object detection system. It addresses the issues of slow convergence and limited feature spatial resolution found in the original DETR model by introducing a novel sampling-based efficient attention mechanism. This allows Deformable DETR to achieve better performance and explore more practical variants of end-to-end object detectors.

### Compute Infrastructure

The compute infrastructure used for the model SenseTime/deformable-detr includes training and inference conducted on NVIDIA Tesla V100 GPUs. The models were trained with a total batch size of 32. The specific details regarding the compute infrastructure, as mentioned in the references, are as follows:

1. Multi-scale feature maps are extracted and used in the deformable Transformer encoder, which shares parameters among different feature levels.
2. The models are trained for 50 epochs with the learning rate decayed at the 40th epoch by a factor of 0.1.
3. The Adam optimizer is used for training with a base learning rate of 2 × 10^-4, β1 = 0.9, β2 = 0.999, and weight decay of 10^-4.
4. Learning rates of the linear projections for predicting object query reference points and sampling offsets are multiplied by a factor of 0.1.
5. Training and inference speed benchmarks are specifically measured on the NVIDIA Tesla V100 GPU.

For any additional details regarding the compute infrastructure that are not covered in the provided references, [More Information Needed].

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

