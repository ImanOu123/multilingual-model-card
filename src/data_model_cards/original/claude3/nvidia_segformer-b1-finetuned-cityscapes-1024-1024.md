# Model Card for nvidia/segformer-b1-finetuned-cityscapes-1024-1024

The model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 is a semantic segmentation model based on the SegFormer architecture, which consists of a hierarchical Transformer encoder and a lightweight All-MLP decoder. It has been fine-tuned on the Cityscapes dataset at a resolution of 1024x1024 pixels to segment driving scenes into 19 categories.

## Model Details

### Model Description

Model Architecture:
- SegFormer consists of a hierarchical Transformer encoder to generate multi-scale features and a lightweight All-MLP decoder to fuse these features and produce the final segmentation mask.
- The hierarchical Transformer encoder, called Mix Transformer (MiT), has a larger effective receptive field than traditional CNN encoders, enabling the use of a simple MLP decoder.
- nvidia/segformer-b1-finetuned-cityscapes-1024-1024 likely uses the MiT-B1 encoder size.

Training Procedures:
- Trained on the Cityscapes dataset, which consists of 5000 fine-annotated high resolution images with 19 categories.
- Data augmentation: random resize (ratio 0.5-2.0), random horizontal flipping, random cropping to 1024x1024.
- Trained using AdamW optimizer for 160K iterations.
- Batch size of 8, initial learning rate of 0.00006 with "poly" LR schedule.
- During evaluation, inference is done using sliding window test.

Parameters:
[More Information Needed]

Important Disclaimers:
[More Information Needed]

Additional Notes:
- SegFormer predicts masks with fine details near object boundaries due to the high-resolution features captured by the Transformer encoder.
- It reduces long-range errors compared to DeepLabV3+ due to the larger effective receptive field of the Transformer encoder.

For more details on specific model parameters and important disclaimers, additional information from the model developers would be needed.

- **Developed by:** Enze Xie; Wenhai Wang; Zhiding Yu; Anima Anandkumar; Jose M Alvarez; Ping Luo;  Caltech
- **Funded by:** Based on the provided references, there is no clear information about the specific people or organizations that funded the SegFormer model project. The references mention NVIDIA Research Licensing for business inquiries, which suggests NVIDIA may have been involved in the development of SegFormer. However, without more explicit details, the most appropriate response is:

[More Information Needed]

To fully answer this question for the model card, additional details would need to be provided by the model development team regarding the funding sources and organizations involved in supporting the SegFormer project.
- **Shared by:** Based on the provided references, the main contributors who made the model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 available are:

Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo from Caltech.

The references also thank Ding Liang, Zhe Chen and Yaojun Liu for insightful discussions, without which this work would not have been possible.

[More Information Needed] on the specific individuals who published the model to the Hugging Face Hub as a GitHub repo.
- **Model type:** The model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 is a fine-tuned version of the SegFormer semantic segmentation model, which is a supervised deep learning model that takes images as input and outputs segmentation masks.
- **Language(s):** The model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 processes images and does not use or process natural human language.
- **License:** The license for the model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 is specified in the LICENSE file. Based on the information provided in the References:

SegFormer may be used non-commercially, meaning for research or evaluation purposes only. For business inquiries, please visit the NVIDIA Research Licensing website at https://www.nvidia.com/en-us/research/inquiries/ and submit the form.
- **Finetuned from model:** The model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 is likely fine-tuned from the SegFormer-B1 model described in the paper "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers". However, the exact base model used for fine-tuning is not explicitly mentioned in the provided references.

[More Information Needed] on the specific base model and link.
### Model Sources

- **Repository:** https://github.com/NVlabs/SegFormer/blob/master/LICENSE
- **Paper:** https://arxiv.org/pdf/2105.15203.pdf
- **Demo:** Based on the provided references, here is the relevant information I could find to answer the question:

The model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 appears to be a SegFormer model fine-tuned on the Cityscapes dataset for semantic segmentation. The references mention that:

- SegFormer is a simple, efficient and powerful semantic segmentation method using a hierarchical Transformer encoder and lightweight All-MLP decoder. 
- It was evaluated on the Cityscapes dataset, which consists of 5000 fine-annotated high resolution images with 19 categories for driving scenes.
- SegFormer provided better details and smoother predictions compared to other methods like SETR and DeeplabV3+ on the Cityscapes test set.

However, the provided references do not contain a direct link to a demo of the specific nvidia/segformer-b1-finetuned-cityscapes-1024-1024 model. 

[More Information Needed] for the demo link. The model card would need to be updated with this information from another source.
## Uses

### Direct Use

The model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 can be used for semantic segmentation on Cityscapes images without requiring any further fine-tuning or post-processing. It takes as input images of size 1024x1024 and outputs the predicted segmentation mask.

To use the model for inference:

1. Rescale the short side of the input image to 1024 while keeping the aspect ratio. 
2. Do inference using sliding window test by cropping 1024x1024 windows.
3. The model will output the predicted segmentation mask at 1/4 of the input resolution, with the number of channels equal to the number of semantic classes in Cityscapes (19).

The following code snippet shows how to run inference using the trained model checkpoint:

```
./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM>
```

[More Information Needed] on how to exactly load the model and run inference in Python. The model should provide smooth and detailed predictions on Cityscapes images as shown in the qualitative results.

### Downstream Use

The nvidia/segformer-b1-finetuned-cityscapes-1024-1024 model can be used for semantic segmentation tasks, especially in the autonomous driving domain since it has been fine-tuned on the Cityscapes dataset. Key points on its usage:

- It takes an input image and outputs a segmentation mask at 1/4 of the original resolution, with each pixel classified into one of the 19 Cityscapes categories
- The model is efficient and robust, making it suitable for real-time applications 
- It can be plugged into larger autonomous driving perception systems that require accurate scene understanding
- Fine-tuning the model on a custom driving dataset could adapt it to specific use cases

Here's a code snippet showing how to run inference on a single image using the MMSegmentation library:

```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${DEVICE_NAME}] [--palette-thr ${PALETTE}]
```

[More Information Needed] on integrating it into a full autonomous driving stack or other applications. Additional experimentation and engineering would be required depending on the specific system architecture and requirements.

### Out-of-Scope Use

Based on the provided references, here are some potential misuses of the nvidia/segformer-b1-finetuned-cityscapes-1024-1024 model that users ought to avoid:

The model should not be used for safety-critical autonomous driving applications without extensive further testing and validation. While the references mention evaluating robustness on corrupted Cityscapes data, more real-world testing would be needed before deploying this for driving. [More Information Needed] on the specific robustness results.

Users should not assume the model will generalize well to data distributions significantly different from the Cityscapes, ADE20K and COCO-Stuff datasets it was trained on. Applying it to very different image domains may lead to unreliable results.

The model appears to still have limitations in the level of detail and smoothness of its segmentation predictions compared to other approaches like DeepLabV3+. Users should be aware of these limitations and not rely on it for applications requiring very fine-grained, pixel-perfect segmentation.

[More Information Needed] on the exact computational requirements, but the references suggest the smallest variant may still be too large for very memory-constrained edge devices. Users should test the model's efficiency for their specific deployment scenario.

In general, as an AI system, the model's outputs should be treated as predictions rather than ground truth. Human oversight and judgment is still recommended when using the model's segmentation maps to inform important decisions. Users should be aware of potential failure modes and biases in the model's behavior.

### Bias, Risks, and Limitations

Based on the provided references about the SegFormer model, here are some potential issues and limitations:

Technical limitations:
- It's unclear if the smallest 3.7M parameter model can work well on edge devices with only 100k memory. More research is needed. (Reference 1)
- [More Information Needed] on the specific performance and efficiency of the B1 variant used in this finetuned model.

Potential misuse and harms:
- As a semantic segmentation model finetuned on the Cityscapes driving dataset, misuse of the model outputs could potentially lead to unsafe autonomous driving decisions. Robustness to corruptions and perturbations is important for safety-critical tasks. (References 5, 8) 
- [More Information Needed] on more specific potential harms and misuse cases.

Sociotechnical considerations:
- Deploying the model in real-world systems like self-driving cars requires carefully considering the societal implications, including safety, liability, and impact on jobs. Collaboration between technical and non-technical stakeholders is important.
- [More Information Needed] on bias and fairness analysis of the model outputs across different demographics and contexts.

To responsibly publish this model, I recommend clearly documenting the intended use cases, limitations, potential risks, and societal considerations in the model card. Inviting multidisciplinary discussion and feedback can help surface additional issues to address. Ongoing monitoring and maintenance is also important as the model is used in the real world.

### Recommendations

Based on the provided references about the SegFormer model nvidia/segformer-b1-finetuned-cityscapes-1024-1024, here are some recommendations regarding foreseeable issues:

1. Model robustness: The SegFormer model shows excellent zero-shot robustness compared to other models like DeepLabV3+ when tested on various corruptions and perturbations (References 1, 3, 5). However, further testing may be needed to ensure robustness in real-world safety-critical applications like autonomous driving.

2. Model size and efficiency: While the smallest SegFormer model (B0) is compact and efficient for real-time applications, it's unclear if it can work well on edge devices with limited memory (Reference 6). More testing is needed to determine suitability for resource-constrained environments.

3. Performance on different resolutions: SegFormer avoids using positional encoding, which can lead to decreased performance when testing resolution differs from training (Reference 10). However, [More Information Needed] on how the model performs on resolutions not seen during training.

4. Societal impact: [More Information Needed] The references don't provide information on potential societal impacts, biases, or fairness considerations of the SegFormer model. These aspects should be carefully evaluated before deployment.

5. Limitations and future work: The provided references acknowledge the need for further research to address limitations like edge device performance (Reference 6). Continued work is recommended to improve the model's applicability and performance.

In summary, while SegFormer shows promising robustness and efficiency, more testing and analysis is advised, especially for deployment in safety-critical applications and resource-constrained environments. Societal impact and fairness should also be thoroughly assessed.

## Training Details

### Training Data

The model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 was finetuned on the Cityscapes dataset, which is a driving dataset for semantic segmentation consisting of 5000 fine-annotated high resolution images with 19 categories. During training, data augmentation was applied through random resize with ratio 0.5-2.0, random horizontal flipping, and random cropping to 1024x1024 for Cityscapes.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about the preprocessing for the data of the model nvidia/segformer-b1-finetuned-cityscapes-1024-1024:

Tokenization:
Given an image of size H × W × 3, it is first divided into patches of size 4 × 4. These patches are then used as input to the hierarchical Transformer encoder.

Resizing:
During training, data augmentation was applied through:
- Random resize with ratio 0.5-2.0 
- Random horizontal flipping
- Random cropping to 1024 × 1024 for the Cityscapes dataset

During evaluation:
For Cityscapes, inference is done using sliding window test. [More Information Needed] on the exact resizing details during evaluation.

Other preprocessing details:
[More Information Needed]

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the model nvidia/segformer-b1-finetuned-cityscapes-1024-1024:

- Dataset: Cityscapes, which consists of 5000 fine-annotated high resolution images with 19 categories
- Batch size: 8
- Learning rate: Initial value of 0.00006 with a "poly" LR schedule and factor 1.0
- Optimizer: AdamW
- Training iterations: [More Information Needed]
- Data augmentation: 
  - Random resize with ratio 0.5-2.0
  - Random horizontal flipping
  - Random cropping to 1024x1024
- Encoder pre-training: Pre-trained on ImageNet-1K dataset
- Decoder initialization: Randomly initialized

To train the model, the following command can be used:
[More Information Needed]

The validation images were merged with the training set for the final results on the Cityscapes test set. The model was also evaluated using pre-training on the Mapillary Vistas dataset in addition to ImageNet-1K.

#### Speeds, Sizes, Times

Here are the details about the model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 based on the provided references:

Model Size:
- The SegFormer-B1 model has 15.9M parameters. [More Information Needed] on the exact parameter count for the finetuned Cityscapes model.

Training:
- Batch size of 8 was used for training on Cityscapes. 
- Initial learning rate was set to 0.00006 with a "poly" LR schedule using factor 1.0.
- [More Information Needed] on exact training time.

Inference:
- For Cityscapes, inference is done using sliding window test.
- [More Information Needed] on exact inference throughput and latency.

Checkpoint:
To test the trained checkpoint:
```
./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM>
```
- [More Information Needed] on exact checkpoint size.

Performance:
- SegFormer-B1 achieves state-of-the-art performance on Cityscapes. Exact mIoU metrics [More Information Needed].
- Compared to DeepLabV3+, SegFormer provides smoother predictions and reduces long-range errors.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 was evaluated on the Cityscapes dataset. Specifically:

- The model was evaluated on the Cityscapes test set, following the common setting of merging the validation images into the training set (Reference 1). 

- Cityscapes is a dataset for semantic segmentation of driving scenes, consisting of 5000 finely-annotated high resolution images with 19 categories (Reference 3).

- The model achieves 84.0% mIoU on the Cityscapes validation set (Reference 7).

- The model's robustness was also evaluated on Cityscapes-C, an expanded version of the Cityscapes validation set with 16 types of algorithmically generated corruptions (Reference 2).

[More Information Needed] on the exact training/validation/test splits and evaluation metrics used for this specific nvidia/segformer-b1-finetuned-cityscapes-1024-1024 model.

#### Factors

Based on the provided references about the SegFormer model fine-tuned on the Cityscapes dataset, here are some key characteristics that may influence the model's behavior:

Domain and Context:
- The model is designed for semantic segmentation tasks, specifically in the domain of autonomous driving and urban scene understanding (Refs 1, 2, 3, 4).
- It was trained and evaluated on the Cityscapes dataset, which consists of high-resolution images captured from a driving perspective in urban environments (Refs 3, 4, 5, 8).

Population Subgroups:
- The Cityscapes dataset covers 19 semantic categories relevant to urban scenes, such as roads, buildings, vehicles, pedestrians, etc. (Ref 3). The model's performance across these different categories should be evaluated to uncover any disparities.
- [More Information Needed] on how the model performs on specific subgroups within the Cityscapes categories, such as different types of vehicles, pedestrians of various ages or genders, etc.

Other Factors:
- The model's robustness to common corruptions and perturbations, such as noise, blur, weather conditions, etc., is important for safety-critical applications like autonomous driving (Ref 2). The Cityscapes-C dataset can be used to evaluate this (Ref 2).
- The model's performance may vary depending on the encoder size used (Refs 6, 7, 9). Larger encoders generally yield better results but with increased computational cost.
- The model's ability to capture fine details and handle long-range dependencies is influenced by its architecture, specifically the hierarchical Transformer encoder and lightweight All-MLP decoder (Refs 10, 11).

To comprehensively evaluate the model's behavior, its performance should be disaggregated across the mentioned factors, such as semantic categories, population subgroups (if applicable), and different types of corruptions or perturbations. This can help uncover potential disparities or biases in the model's predictions.

#### Metrics

Based on the provided references, the main evaluation metric used for the nvidia/segformer-b1-finetuned-cityscapes-1024-1024 model on the Cityscapes dataset is mean Intersection over Union (mIoU). Specifically, reference 1 states:

"We report semantic segmentation performance using mean Intersection over Union (mIoU)."

The references do not provide details on tradeoffs between different error types for this specific model. [More Information Needed] on what specific error tradeoffs were considered or reported for the nvidia/segformer-b1-finetuned-cityscapes-1024-1024 model.

### Results

Based on the provided references, here are the evaluation results for the model nvidia/segformer-b1-finetuned-cityscapes-1024-1024:

Factors:
- Dataset: The model was evaluated on the Cityscapes dataset. It was trained using the Cityscapes train set merged with the validation set, following the common setting. (Reference 1)
- Pretraining: The model used ImageNet-1K pretraining. (Reference 1)

Metrics:
- mIoU: The model achieves 84.0% mIoU on the Cityscapes validation set. (Reference 10)
- Robustness: SegFormer shows excellent zero-shot robustness on Cityscapes-C, a dataset with algorithmically generated corruptions. (References 9, 10)
- [More Information Needed] on specific metrics like parameters, FLOPs, and latency for the SegFormer-B1 variant.

The references provide qualitative results showing that SegFormer provides better details than SETR and smoother predictions than DeepLabV3+ on Cityscapes. (Reference 1)

Overall, the nvidia/segformer-b1-finetuned-cityscapes-1024-1024 model demonstrates strong performance and robustness on the Cityscapes dataset, but more specific quantitative metrics for this particular variant would be helpful to include in the evaluation results.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model nvidia/segformer-b1-finetuned-cityscapes-1024-1024:

The SegFormer model was evaluated on the Cityscapes dataset. On the Cityscapes test set, the common setting of merging the validation images into the training set was followed. Results were reported using ImageNet-1K pre-training as well as pre-training on Mapillary Vistas.

Qualitative results on Cityscapes show that SegFormer provides better details compared to SETR and smoother predictions than DeepLabV3+. 

[More Information Needed] on the specific quantitative performance metrics for the SegFormer-B1 variant finetuned on Cityscapes at 1024x1024 resolution.

The robustness of SegFormer was evaluated on Cityscapes-C, which expands the Cityscapes validation set with 16 types of algorithmically generated corruptions. SegFormer was compared to variants of DeepLabV3+ and other methods. [More Information Needed] on the specific robustness results for SegFormer-B1.

In summary, SegFormer demonstrates strong semantic segmentation performance on Cityscapes, with fine details and smooth predictions. However, more specific information is needed on the exact evaluation metrics for the SegFormer-B1 model variant that was finetuned on Cityscapes at 1024x1024 resolution.

## Model Examination

Based on the provided references, here are some key points about the interpretability and explainability of the SegFormer model nvidia/segformer-b1-finetuned-cityscapes-1024-1024:

- SegFormer predicts masks with finer details near object boundaries compared to SETR, because the Transformer encoder captures higher resolution features preserving more detailed texture information. (Reference 1)

- SegFormer reduces long-range errors compared to DeepLabV3+, benefiting from the larger effective receptive field of the Transformer encoder. (References 1, 3)

- The effective receptive field (ERF) of SegFormer is more sensitive to the context of the image compared to DeepLabV3+. SegFormer's ERF learns patterns of roads, cars, buildings, while DeepLabV3+'s ERF shows a relatively fixed pattern. This indicates SegFormer's Transformer encoder has a stronger feature extraction ability than ConvNets. (Reference 3)

- [More Information Needed] on specific code or techniques used to analyze explainability/interpretability of the SegFormer model.

In summary, the SegFormer model demonstrates improved boundary details, reduced long-range errors, and more contextually sensitive feature extraction compared to previous models like SETR and DeepLabV3+. However, more specific information would be needed to fully characterize the explainability techniques applied to this particular fine-tuned SegFormer model.

## Environmental Impact

- **Hardware Type:** The model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 was trained on a server with 8 Tesla V100 GPUs, according to the implementation details provided in the references.
- **Software Type:** Based on the provided references, the model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 is trained on the Cityscapes dataset, as evidenced by these excerpts:

"On Cityscapes test set, we follow the common setting [20] and merge the validation images to the train set and report results using Imagenet-1K pre-training and also using Mapillary Vistas [76]."

"In Figure 5, we present more qualitative results on Cityscapes, ADE20K and COCO-Stuff, compared with SETR and DeepLabV3+."

However, the specific software type or deep learning framework (such as PyTorch or TensorFlow) used for training the model is not explicitly mentioned in the provided references. Therefore, for the software type, [More Information Needed].
- **Hours used:** Based on the provided references, the SegFormer models were trained for 160K iterations on the Cityscapes dataset using an AdamW optimizer (Reference 4). However, there is no specific information provided about the training time for the nvidia/segformer-b1-finetuned-cityscapes-1024-1024 model.

[More Information Needed] on the exact training time for the nvidia/segformer-b1-finetuned-cityscapes-1024-1024 model.
- **Cloud Provider:** Based on the implementation details provided in the references, the model nvidia/segformer-b1-finetuned-cityscapes-1024-1024 was trained on a server with 8 Tesla V100 GPUs. However, the specific cloud provider is not mentioned. [More Information Needed] about the cloud provider to conclusively answer the question.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the model nvidia/segformer-b1-finetuned-cityscapes-1024-1024. To accurately report the carbon emissions, more details would be needed such as the hardware used for training, total training time, energy consumption, and the carbon intensity of the electricity grid used.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
SegFormer consists of two main modules:
1. A hierarchical Transformer encoder that generates high-resolution coarse features and low-resolution fine features. The encoder is based on the Mix Transformer (MiT) design, which is tailored and optimized for semantic segmentation.
2. A lightweight All-MLP decoder that fuses the multi-level features from the encoder to produce the final semantic segmentation mask. The decoder consists only of MLP layers, avoiding hand-crafted and computationally demanding components.

The hierarchical Transformer encoder outputs multi-scale features at {1/4, 1/8, 1/16, 1/32} of the original image resolution. The input image is divided into patches of size 4x4, which are then passed through the encoder.

[More Information Needed] on the specific architecture details of the nvidia/segformer-b1-finetuned-cityscapes-1024-1024 model.

Objective:
The objective of the nvidia/segformer-b1-finetuned-cityscapes-1024-1024 model is to perform semantic segmentation on the Cityscapes dataset. SegFormer aims to provide a simple, efficient, and powerful framework for semantic segmentation by unifying Transformers with lightweight MLP decoders.

The model is designed to have the following advantages:
1. It does not require positional encoding, avoiding the interpolation of positional codes that can lead to decreased performance when the testing resolution differs from training.
2. It avoids complex decoders, using a simple and lightweight MLP decoder that aggregates information from different layers, combining both local and global attention for powerful representations.

[More Information Needed] on the specific performance metrics and results of the nvidia/segformer-b1-finetuned-cityscapes-1024-1024 model on the Cityscapes dataset.

### Compute Infrastructure

Based on the provided references, here is the information about the compute infrastructure used for the model nvidia/segformer-b1-finetuned-cityscapes-1024-1024:

The model was trained on a server with 8 Tesla V100 GPUs, as mentioned in reference 3:

"We used the mmsegmentation codebase and train on a server with 8 Tesla V100."

For the Cityscapes dataset specifically, a batch size of 8 was used, as stated in reference 2:

"used a batch size of 16 for ADE20K and COCO-Stuff, and a batch size of 8 for Cityscapes."

The training was run for 160K iterations on Cityscapes, according to reference 4:

"We trained the models using AdamW optimizer for 160K iterations on ADE20K, Cityscapes, and 80K iterations on COCO-Stuff."

[More Information Needed] on the exact training script or command used for the Cityscapes dataset with this specific model.

## Citation

```
@misc{enze-segformer,
    author = {Enze Xie and
              Wenhai Wang and
              Zhiding Yu and
              Anima Anandkumar and
              Jose M Alvarez and
              Ping Luo and
              Caltech},
    title  = {SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
    url    = {https://arxiv.org/pdf/2105.15203.pdf}
}
```

