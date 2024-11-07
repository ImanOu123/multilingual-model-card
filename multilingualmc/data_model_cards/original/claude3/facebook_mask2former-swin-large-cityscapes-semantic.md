# Model Card for facebook/mask2former-swin-large-cityscapes-semantic

The facebook/mask2former-swin-large-cityscapes-semantic model is a Mask2Former model with a Swin-L backbone that performs semantic segmentation on the Cityscapes dataset. It outperforms the state-of-the-art SegFormer model with a Swin-B backbone on the Cityscapes semantic segmentation task.

## Model Details

### Model Description

Model Architecture:
- Mask2Former adopts a meta architecture with a backbone, pixel decoder, and a Transformer decoder
- The Transformer decoder uses a masked attention operator that constrains cross-attention to within the foreground region of the predicted mask for each query
- An efficient multi-scale strategy is used to utilize high-resolution features from the pixel decoder by feeding one scale to one Transformer decoder layer at a time
- The order of self and cross-attention is switched, query features are learnable, and dropout is removed in the Transformer decoder

Training Procedures:
- Trained on the Cityscapes dataset for semantic segmentation
- Uses a learning rate multiplier of 0.1 applied to both CNN and Transformer backbones 
- Swin-L backbone uses an initial learning rate of 0.0001 and weight decay of 0.05
- [More Information Needed] on other training hyperparameters and procedures

Model Parameters:
- Uses a Swin-L (Swin-Large) backbone
- [More Information Needed] on the total number of parameters

Important Disclaimers:
- [More Information Needed]

Let me know if you need any other details about the model! I'll do my best to find the relevant information or indicate where more details are required.

- **Developed by:** Bowen Cheng; Ishan Misra; Alexander G Schwing; Alexander Kirillov; Rohit Girdhar; Facebook Ai Research
- **Funded by:** [More Information Needed]

The provided references do not contain any information about the funding sources for the facebook/mask2former-swin-large-cityscapes-semantic model specifically. More details would be needed from the model developers or associated documentation to determine what organizations or individuals, if any, provided funding for this particular model's development.
- **Shared by:** Based on the provided references, the contributors who made the model facebook/mask2former-swin-large-cityscapes-semantic available online are:

Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar from Facebook AI Research.

The references do not specify if the model was made available as a GitHub repo, so [More Information Needed] for that part of the question.
- **Model type:** The model facebook/mask2former-swin-large-cityscapes-semantic is a supervised deep learning model for semantic segmentation on images, trained on the Cityscapes dataset using the Mask2Former architecture with a Swin-L backbone.
- **Language(s):** The model facebook/mask2former-swin-large-cityscapes-semantic does not use or process natural human language, as it is an image segmentation model trained on the Cityscapes dataset for semantic segmentation.
- **License:** The model facebook/mask2former-swin-large-cityscapes-semantic uses a combination of licenses:

The majority of Mask2Former is licensed under the MIT License.

However, portions of the project are available under separate license terms:
- Swin-Transformer-Semantic-Segmentation is licensed under the MIT license
- Deformable-DETR is licensed under the Apache-2.0 License
- **Finetuned from model:** Based on the provided references, the model facebook/mask2former-swin-large-cityscapes-semantic likely uses the Swin-L (Swin Large) backbone, as mentioned in this excerpt:

"For panoptic segmentation, Mask2Former with Swin-L backbone outperforms the state-of-the-art Panoptic-DeepLab [11] with SWideRnet [9] using single-scale inference."

However, there is no explicit information provided about whether this model was fine-tuned from another pre-existing model. To confidently answer this question, [More Information Needed] about the specific training process and initialization of the facebook/mask2former-swin-large-cityscapes-semantic model.
### Model Sources

- **Repository:** https://github.com/facebookresearch/Mask2Former/
- **Paper:** https://arxiv.org/pdf/2112.01527.pdf
- **Demo:** [More Information Needed]

The provided references do not contain a direct link to a demo of the model facebook/mask2former-swin-large-cityscapes-semantic. The references discuss training details, results on various datasets, and comparisons to other methods, but do not mention a publicly available demo. More information would be needed from the model developers to determine if a demo link exists that could be included in the model card.
## Uses

### Direct Use

The model facebook/mask2former-swin-large-cityscapes-semantic can be used for semantic segmentation on the Cityscapes dataset without requiring fine-tuning, post-processing, or plugging into a pipeline. 

Mask2Former is an effective architecture for universal image segmentation that achieves state-of-the-art results on standard benchmarks like Cityscapes. This specific model uses a Swin-L backbone and was trained on the Cityscapes dataset for the semantic segmentation task.

To use the model, you can run the demo provided using Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uIWE5KbGFSjrxey2aRd5pWkKNY1_SaNq)

The model is also integrated into Huggingface Spaces using Gradio. You can try out the web demo here:  
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/Mask2Former)

A Replicate web demo and docker image is available as well:
[![Replicate](https://replicate.com/facebookresearch/mask2former/badge)](https://replicate.com/facebookresearch/mask2former)

[More Information Needed] for providing a direct code snippet to use the model.

### Downstream Use

The Mask2Former model facebook/mask2former-swin-large-cityscapes-semantic can be fine-tuned for semantic segmentation on the Cityscapes dataset. It achieves state-of-the-art performance on this task, outperforming other specialized architectures like SegFormer.

To use this model in a larger ecosystem or app, it can be integrated via the Hugging Face Transformers library. A demo of the model is available on Hugging Face Spaces using Gradio: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/Mask2Former)

[More Information Needed] for a specific code snippet on fine-tuning or inference. The model card should provide those details or link to further documentation.

In summary, this powerful semantic segmentation model can be a valuable component in computer vision applications requiring pixel-wise classification of images into semantic categories. Its strong performance on the Cityscapes benchmark demonstrates its effectiveness on urban scene understanding tasks.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the model facebook/mask2former-swin-large-cityscapes-semantic may foreseeably be misused and what users ought not do with it:

The model should undergo ethical review to ensure the predictions do not propagate problematic stereotypes, and the approach should not be used for applications including but not limited to illegal surveillance (Reference 8). Users should not deploy the model for unethical purposes without proper review.

[More Information Needed] on other specific ways the model could be misused, as the references do not go into further detail. The model card should elaborate more on foreseeable misuse cases and provide clear guidance to users on unacceptable uses of the model to mitigate potential harms. Additional ethical and sociological analysis would help identify risks.

In summary, users ought not use the model for unethical purposes like illegal surveillance or in ways that could propagate harmful stereotypes, but more information is needed in the references to fully address potential misuse cases. The model card should provide clearer use guidelines based on deeper sociotechnical analysis.

### Bias, Risks, and Limitations

Based on the provided references about the Mask2Former model (facebook/mask2former-swin-large-cityscapes-semantic), here are some known or foreseeable issues:

Ethical considerations:
- While the technical innovations do not appear to have inherent biases, models trained on real-world datasets should undergo ethical review to ensure predictions do not propagate problematic stereotypes and the approach is not used for illegal surveillance or other harmful applications. [Reference 2]

Technical limitations:
- Mask2Former struggles with segmenting small objects and is unable to fully leverage multiscale features. Better utilization of the feature pyramid and designing losses for small objects are believed to be critical improvements needed. [Reference 5]

Generalization and training:
- Although Mask2Former can generalize to different tasks, it still needs to be trained for those specific tasks to achieve optimal performance. Developing a model that can be trained only once for all image segmentation tasks is a future goal. [Reference 1]

[More Information Needed] on other potential misunderstandings or sociotechnical limitations, as the provided references do not cover those aspects in detail.

### Recommendations

Based on the provided references about the Mask2Former model, here are some recommendations regarding foreseeable issues:

1. Generalization to different tasks: While Mask2Former can generalize to different segmentation tasks, it still needs to be trained specifically for those tasks to achieve optimal performance. Developing a single model that can be trained only once for all image segmentation tasks remains a future goal. [More Information Needed] on how to achieve this.

2. Segmenting small objects: Mask2Former struggles with segmenting small objects and is unable to fully leverage multiscale features. Better utilization of the feature pyramid and designing losses for small objects are critical to address this issue.

3. Computational efficiency: [More Information Needed] on the computational efficiency of Mask2Former compared to other models, as this was not the primary focus of the provided references.

4. Ethical considerations: [More Information Needed] on potential ethical issues arising from the use of Mask2Former, such as bias in the training data or misuse of the model for surveillance purposes.

5. Societal impact: [More Information Needed] on the long-term societal impact of advanced image segmentation models like Mask2Former, such as their effect on job displacement or privacy concerns.

To address these issues, it is recommended to conduct further research on improving the model's generalization capabilities, handling small objects, computational efficiency, and ethical and societal implications. Engaging with stakeholders and experts from various fields can help identify and mitigate potential risks associated with the deployment of Mask2Former.

## Training Details

### Training Data

The model facebook/mask2former-swin-large-cityscapes-semantic was trained on the Cityscapes dataset, which is an urban egocentric street-view dataset containing 2975 high-resolution images (1024 × 2048 pixels) for training, with a total of 19 semantic classes. [More Information Needed]

### Training Procedure

#### Preprocessing

Based on the provided references, the preprocessing details for the data used in the facebook/mask2former-swin-large-cityscapes-semantic model are not explicitly mentioned. The references discuss model architecture, training settings, and evaluation metrics, but do not go into specifics about data preprocessing such as tokenization or resizing.

[More Information Needed] on the exact preprocessing steps applied to the data before being fed into the Mask2Former model for training and inference on the Cityscapes semantic segmentation task.

#### Training Hyperparameters

Here are the training hyperparameters for the model facebook/mask2former-swin-large-cityscapes-semantic based on the provided references:

Backbone and pixel decoder:
- Swin-Large backbone
- Multi-scale deformable attention Transformer (MSDeformAttn) pixel decoder with 6 layers applied to feature maps with resolution 1/8, 1/16 and 1/32
- Upsampling layer with lateral connection on final 1/8 feature map to generate 1/4 resolution feature map as per-pixel embedding

Transformer decoder: 
- 9 layers total (L=3)
- 100 queries
- Auxiliary loss added to every intermediate layer and learnable query features before decoder

Optimizer and schedule:
- AdamW optimizer 
- Initial learning rate of 0.0001
- Weight decay of 0.05 for all backbones
- Learning rate multiplier of 0.1 applied to both CNN and Transformer backbones
- Learning rate decayed at 0.9 and 0.95 fractions of total training steps by factor of 10

Training:
- Trained for 50 epochs 
- Batch size of 16
- Large-scale jittering (LSJ) augmentation with random scale from 0.1 to 2.0 and fixed size crop to 1024x1024

Losses:
- Mask loss: binary cross-entropy (weight 5.0) + dice loss (weight 5.0)  
- Classification loss weight: 2.0 for matched predictions, 0.1 for "no object"
- Final loss: L_mask + λ_cls * L_cls, with λ_cls = 2.0

[More Information Needed] on exact learning rate schedule, inference settings, and data-specific training details for Cityscapes.

#### Speeds, Sizes, Times

Here are the details about the model facebook/mask2former-swin-large-cityscapes-semantic based on the provided references:

Throughput: 
The model achieves a runtime of 0.11 seconds per image on a V100 GPU with a batch size of 1, taking the average runtime on the entire validation set including post-processing time. (Reference 9)

Start or end time: [More Information Needed]

Checkpoint sizes: [More Information Needed]

Additional details:
- It uses the Swin-L backbone. (Reference 8) 
- The model is trained for semantic segmentation on the Cityscapes dataset. (Reference 8)
- It uses a learning rate multiplier of 0.1 applied to both CNN and Transformer backbones, an initial learning rate of 0.0001, and a weight decay of 0.05. (Reference 9)
- The model uses 100 queries. (Reference 8)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model facebook/mask2former-swin-large-cityscapes-semantic evaluates on the Cityscapes dataset.

Specifically, reference 8 states:

"Cityscapes is an urban egocentric street-view dataset with high-resolution images (1024 × 2048 pixels). It contains 2975 images for training, 500 images for validation and 1525 images for testing with a total of 19 classes."

The model is trained and evaluated on the Cityscapes dataset for semantic segmentation.

#### Factors

Based on the provided references, here are some foreseeable characteristics that may influence the behavior of the facebook/mask2former-swin-large-cityscapes-semantic model:

Domain and Context:
- The model is trained and evaluated on the Cityscapes dataset, which contains urban street scenes (References 2, 3, 9). Performance on other domains like indoor scenes or rural areas is unclear.
- The model aims to perform semantic segmentation, assigning a class label to each pixel (References 6, 8). Its performance on other tasks like instance or panoptic segmentation in the same domain is not fully specified.

Population Subgroups:
- The model struggles with segmenting small objects (Reference 4), so it may perform worse on scenes with many small objects or on subgroups of objects that tend to be small.

[More Information Needed] on other relevant population subgroups and how performance may vary across them.

Disaggregated Evaluation:
- The provided references do not contain disaggregated evaluation results across different subgroups or attributes.
[More Information Needed] to uncover any performance disparities.

In summary, while the model achieves strong results on the Cityscapes semantic segmentation task, more information is needed to fully characterize its behavior across different domains, subgroups, and evaluation factors. Targeted testing on more diverse data could help uncover potential limitations and biases.

#### Metrics

Based on the provided references, the key evaluation metrics for the Mask2Former model on the Cityscapes semantic segmentation task would be:

mIoU (mean Intersection-over-Union) - This is the standard metric for evaluating semantic segmentation performance, as mentioned in reference 1. It measures the overlap between the predicted and ground truth masks.

The references do not provide specific information on other metrics used for evaluating the Mask2Former model on the Cityscapes semantic segmentation dataset. Metrics like pixel accuracy are mentioned for ADE20K in reference 7, but not specifically for Cityscapes.

[More Information Needed] on any additional metrics beyond mIoU used for evaluating the facebook/mask2former-swin-large-cityscapes-semantic model, and details on tradeoffs between different error types for this specific model and dataset.

### Results

Based on the provided references, the evaluation results for the model facebook/mask2former-swin-large-cityscapes-semantic on the Cityscapes dataset are:

Semantic Segmentation:
- mIoU: Mask2Former with Swin-B backbone outperforms the state-of-the-art SegFormer [59]. The exact mIoU value for the Swin-L backbone is [More Information Needed].

Instance Segmentation:
- AP: [More Information Needed]

Panoptic Segmentation:
- PQ: [More Information Needed]
- AP_Th_pan: [More Information Needed]
- mIoU_pan: [More Information Needed]

The references mention that Mask2Former with Swin-L backbone outperforms the state-of-the-art Panoptic-DeepLab [11] with SWideRnet [9] using single-scale inference for panoptic segmentation on Cityscapes. However, the exact metric values are not provided.

#### Summary

Based on the provided references, the evaluation results for the model facebook/mask2former-swin-large-cityscapes-semantic on the Cityscapes dataset can be summarized as follows:

For semantic segmentation on Cityscapes, Mask2Former with the Swin-B backbone outperforms the state-of-the-art SegFormer model. 

[More Information Needed] on the specific mIoU metric values achieved by the Mask2Former model with the Swin-Large backbone on the Cityscapes semantic segmentation task.

The references also mention that Mask2Former with the Swin-L backbone achieves state-of-the-art performance on Cityscapes for panoptic segmentation, outperforming Panoptic-DeepLab with SWideRnet using single-scale inference. However, the exact panoptic segmentation metrics (PQ, AP Th pan, mIoU pan) for the Swin-Large backbone are [More Information Needed].

## Model Examination

Based on the provided references about the Mask2Former model, here is what I can say about the explainability and interpretability of the facebook/mask2former-swin-large-cityscapes-semantic model:

The Mask2Former architecture uses masked attention to constrain the cross-attention within predicted mask regions. As mentioned in reference 5, this helps focus the attention on the object of interest, compared to standard cross-attention where the attention spreads over the entire image and can be dominated by background regions. Visualizing the attention maps (as shown in Figure Ia in the paper) can provide insights into what the model is attending to when making predictions.

The model uses learnable object queries which are fed into the Transformer decoder (references 3 and 4). Analyzing these learned query features and how they evolve through the decoder layers could shed light on how the model represents and reasons about the objects in the image.

However, more specific details would be needed to provide a comprehensive overview of the explainability and interpretability of this particular Mask2Former variant trained on the Cityscapes semantic segmentation dataset. The references do not go into detail on analyzing the behavior and failure modes of the facebook/mask2former-swin-large-cityscapes-semantic model. [More Information Needed]

## Environmental Impact

- **Hardware Type:** Based on the provided references, the model facebook/mask2former-swin-large-cityscapes-semantic was likely trained on V100 GPUs, as mentioned in reference 1:

"a V100 GPU with a batch size of 1 by taking the average runtime on the entire validation set including post-processing time."

However, more specific details on the hardware setup used for training this particular model on the Cityscapes dataset are not provided. To confidently state the exact hardware used, [More Information Needed].
- **Software Type:** Based on the provided references, the model facebook/mask2former-swin-large-cityscapes-semantic is trained using code that is largely based on MaskFormer (https://github.com/facebookresearch/MaskFormer), as mentioned in reference 7:

"Code is largely based on MaskFormer (https://github.com/facebookresearch/MaskFormer)."

Therefore, the software type that the model is trained on is the codebase from the MaskFormer repository on GitHub.
- **Hours used:** Based on the provided references, the model facebook/mask2former-swin-large-cityscapes-semantic was trained for 50 epochs with a batch size of 16, as stated in reference 2:

"we train our models for 50 epochs with a batch size of 16."

However, the exact amount of time used for training is not provided in the given references. To determine the total training time, more information would be needed, such as:

- The hardware used for training (GPU type, number of GPUs, etc.)
- The exact model architecture and size
- Any additional data preprocessing or augmentation steps performed during training

Without these details, it is not possible to accurately estimate the total training time. Therefore, the answer for the specific training time is: [More Information Needed].
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the model facebook/mask2former-swin-large-cityscapes-semantic. More information would be needed from the model developers or training logs to determine the cloud provider utilized.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the model facebook/mask2former-swin-large-cityscapes-semantic. To accurately estimate the carbon emissions, additional details would be needed, such as:

- Total training time
- Power consumption of the hardware used for training
- Carbon intensity of the electricity grid where the training was performed

Without these specifics, it is not possible to provide a reliable estimate of the carbon emissions associated with training this particular model.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
Mask2Former adopts a meta architecture with a backbone, pixel decoder, and a Transformer decoder. The key components are:
- Masked attention operator in the Transformer decoder, which extracts localized features by constraining cross-attention to within the foreground region of the predicted mask for each query
- Efficient multi-scale strategy to utilize high-resolution features from the pixel decoder, by feeding successive feature maps into successive Transformer decoder layers
- Switching the order of self and cross-attention (masked attention), making query features learnable, and removing dropout for more effective computation

The model uses a Swin-Large backbone.

Objective:
The facebook/mask2former-swin-large-cityscapes-semantic model is trained for semantic segmentation on the Cityscapes dataset. It aims to group pixels based on their category membership.

[More Information Needed] on the specific training details and hyperparameters used for this model.

### Compute Infrastructure

Based on the provided references, here is the compute infrastructure information I could find about the model facebook/mask2former-swin-large-cityscapes-semantic:

The model's runtime performance was measured on a V100 GPU with a batch size of 1 by taking the average runtime on the entire validation set including post-processing time. (Reference 1)

The model was trained with a batch size of 16 for 50 epochs. For data augmentation, large-scale jittering (LSJ) augmentation was used with a random scale sampled from range 0.1 to 2.0 followed by a fixed size crop to 1024×1024. (Reference 4)

[More Information Needed] on the exact compute infrastructure used for training the model, such as number and type of GPUs, CPU, RAM, etc.

## Citation

```
@misc{bowen-maskedattention,
    author = {Bowen Cheng and
              Ishan Misra and
              Alexander G Schwing and
              Alexander Kirillov and
              Rohit Girdhar and
              Facebook Ai Research},
    title  = {Masked-attention Mask Transformer for Universal Image Segmentation},
    url    = {https://arxiv.org/pdf/2112.01527.pdf}
}
```

