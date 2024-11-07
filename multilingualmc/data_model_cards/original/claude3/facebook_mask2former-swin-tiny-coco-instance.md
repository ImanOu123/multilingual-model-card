# Model Card for facebook/mask2former-swin-tiny-coco-instance

The facebook/mask2former-swin-tiny-coco-instance model is an instance of the Mask2Former architecture, which is a universal image segmentation model that can handle panoptic, instance, and semantic segmentation tasks. It outperforms specialized architectures on popular datasets like COCO, setting a new state-of-the-art for instance segmentation with 50.1 AP on the COCO dataset.

## Model Details

### Model Description

Model Architecture:
- Mask2Former adopts a meta architecture with a backbone, pixel decoder, and a Transformer decoder
- The Transformer decoder uses a masked attention operator that constrains cross-attention to within the foreground region of the predicted mask for each query
- Utilizes high-resolution features efficiently by feeding successive feature maps from the pixel decoder's feature pyramid into successive Transformer decoder layers

Training Procedures:
- Trained on the COCO dataset for instance segmentation
- Uses AdamW optimizer and step learning rate schedule
- Initial learning rate of 0.0001 and weight decay of 0.05 for the Swin-Tiny backbone
- Learning rate multiplier of 0.1 applied to the backbone
- Learning rate decayed at 0.9 and 0.95 fractions of total training steps by a factor of 10
- Trained with a batch size of 1 on a V100 GPU

Model Parameters:
- Uses a Swin-Tiny backbone
- [More Information Needed] on the exact number of parameters

Important Disclaimers:
- [More Information Needed]

Please note that this model card provides a high-level overview based on the given references. For more detailed information or to report any issues, please contact the project organizer.

- **Developed by:** Bowen Cheng; Ishan Misra; Alexander G Schwing; Alexander Kirillov; Rohit Girdhar; Facebook Ai Research
- **Funded by:** Based on the provided references, there is no explicit mention of the people or organizations that funded the development of the model facebook/mask2former-swin-tiny-coco-instance. The references focus on the technical details, datasets used, and comparisons to other models, but do not specify funding sources.

[More Information Needed]

To properly answer this question for the model card, additional information would need to be provided by the team members or organizations involved in the project, specifying any funding sources or grants that supported the work.
- **Shared by:** Based on the provided references, the contributors who made the model facebook/mask2former-swin-tiny-coco-instance available online are:

Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar from Facebook AI Research.

The references mention that Mask2Former was developed by these authors and published in CVPR 2022. However, there is no explicit information about who specifically made the model available on GitHub. [More Information Needed] on that aspect.
- **Model type:** The facebook/mask2former-swin-tiny-coco-instance model is an instance segmentation model trained on the COCO dataset using the Mask2Former architecture with a Swin-Tiny backbone, which is a type of Transformer-based deep learning model for computer vision tasks.
- **Language(s):** The model facebook/mask2former-swin-tiny-coco-instance does not directly use or process natural human language based on the given references, which focus on the model's architecture and performance on various image segmentation tasks without mentioning any natural language processing capabilities. [More Information Needed]
- **License:** The model facebook/mask2former-swin-tiny-coco-instance uses a combination of licenses:

The majority of Mask2Former is licensed under the MIT License.

However, portions of the project are available under separate license terms:
- Swin-Transformer-Semantic-Segmentation is licensed under the MIT license
- Deformable-DETR is licensed under the Apache-2.0 License

MIT License link: https://opensource.org/licenses/MIT

[More Information Needed] for the specific Apache-2.0 License link used by Deformable-DETR.
- **Finetuned from model:** Based on the provided references, there is no clear indication that the model facebook/mask2former-swin-tiny-coco-instance is fine-tuned from another base model. The references discuss various components and settings used in the Mask2Former architecture, but do not explicitly mention fine-tuning from a pre-existing model.

[More Information Needed] to determine if the model is fine-tuned from another base model, and if so, the name and link to that base model.
### Model Sources

- **Repository:** https://github.com/facebookresearch/Mask2Former/
- **Paper:** https://arxiv.org/pdf/2112.01527.pdf
- **Demo:** Here are the relevant demo links for the model facebook/mask2former-swin-tiny-coco-instance based on the provided references:

Hugging Face Spaces demo using Gradio: https://huggingface.co/spaces/akhaliq/Mask2Former

Replicate web demo and docker image: https://replicate.com/facebookresearch/mask2former

Colab demo: https://colab.research.google.com/drive/1uIWE5KbGFSjrxey2aWRd5pWkKNY1_SaNq

The references don't specify which exact model variant is used in each demo. More information would be needed to confirm they are all using the facebook/mask2former-swin-tiny-coco-instance model specifically.
## Uses

### Direct Use

The model facebook/mask2former-swin-tiny-coco-instance can be used for instance segmentation without requiring fine-tuning, post-processing or plugging into a pipeline. It is integrated into Huggingface Spaces using Gradio, and a web demo is available to try out the model directly:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/Mask2Former)

The model can also be run using a Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uIWE5KbGFSjrxey2aRd5pWkKNY1_SaNq)

Additionally, a Replicate web demo and docker image is available:

[![Replicate](https://replicate.com/facebookresearch/mask2former/badge)](https://replicate.com/facebookresearch/mask2former)

[More Information Needed] for providing a direct code snippet to use the model.

In summary, the model can be easily accessed and used for instance segmentation through the provided web demos and Colab notebook, without needing any additional fine-tuning, post-processing or integration steps.

### Downstream Use

The facebook/mask2former-swin-tiny-coco-instance model can be fine-tuned for instance segmentation tasks on custom datasets. It can also be integrated into larger applications that require instance segmentation capabilities, such as object detection and tracking systems.

To use the model, first prepare your custom dataset in the format required by Mask2Former. See [Preparing Datasets for Mask2Former](datasets/README.md) for instructions.

Then follow the steps in [Getting Started with Mask2Former](GETTING_STARTED.md) to fine-tune the model on your dataset.

The fine-tuned model can be used for inference in your application. Mask2Former has also been integrated into the Huggingface Spaces platform using Gradio for easy testing and demonstration:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/Mask2Former)

A Replicate web demo and Docker image are available as well:  

[![Replicate](https://replicate.com/facebookresearch/mask2former/badge)](https://replicate.com/facebookresearch/mask2former)

For more advanced usage, refer to [Advanced Usage of Mask2Former](ADVANCED_USAGE.md).

### Out-of-Scope Use

Based on the provided references, here are some potential ways the facebook/mask2former-swin-tiny-coco-instance model could be misused and what users should not do:

The model should undergo ethical review to ensure its predictions do not propagate problematic stereotypes when trained on real-world datasets. Users should not apply the model for unethical purposes like illegal surveillance. 

While the technical innovations themselves may not have inherent biases, using the model on certain datasets or for certain applications without proper review could lead to biased or harmful outcomes. Users have an obligation to carefully consider the impacts and not deploy the model in ways that could enable discrimination, violate privacy rights, or cause other societal harms.

[More Information Needed] on other specific misuse cases to avoid, as the provided references focus more on the model's capabilities and limitations rather than an in-depth ethical analysis. A thorough assessment by an ethics review board is recommended before releasing the model to mitigate risks of foreseeable misuse.

### Bias, Risks, and Limitations

Based on the provided references about the Mask2Former model, here are some known or foreseeable issues:

Technical limitations:
- Mask2Former struggles with segmenting small objects and is unable to fully leverage multiscale features. Better utilization of the feature pyramid and designing losses for small objects are critical areas for improvement. (Reference 4)
- [More Information Needed] on other specific technical limitations.

Sociotechnical issues and potential harms: 
- While the technical innovations in Mask2Former do not appear to have any inherent biases, the models trained on real-world datasets should undergo ethical review to ensure the predictions do not propagate problematic stereotypes. (Reference 5)
- Mask2Former should not be used for applications including but not limited to illegal surveillance. (Reference 5)
- [More Information Needed] on other specific sociotechnical issues and potential harms.

Generalizability and future work:
- Even though Mask2Former can generalize to different segmentation tasks, it still needs to be trained for those specific tasks for optimal performance. Future work aims to develop a model that can be trained only on panoptic segmentation but still perform well on instance and semantic segmentation. (Reference 1)
- [More Information Needed] on plans and timeline to address the known limitations and expand the model's capabilities.

In summary, while Mask2Former achieves state-of-the-art performance on multiple segmentation tasks, it has some known technical limitations especially in segmenting small objects. Ethical considerations around dataset biases and responsible usage also need to be addressed. More information is needed on other specific limitations, potential harms, and future development plans.

### Recommendations

Based on the provided references about the Mask2Former model, here are some recommendations regarding foreseeable issues:

1. Generalization to different tasks: While Mask2Former can generalize to different segmentation tasks, it still needs to be trained specifically for those tasks to achieve optimal performance. Developing a more versatile model that can be trained only once for multiple tasks remains a future goal. [More Information Needed] on the specific implications and challenges of this limitation.

2. Performance on small objects: Mask2Former struggles with segmenting small objects and may not fully leverage multiscale features. Improving the utilization of the feature pyramid and designing losses tailored for small objects could be critical to address this issue. [More Information Needed] on the potential societal impact of suboptimal performance on small objects.

3. Compute-performance trade-off: Mask2Former has a better compute-performance trade-off compared to its predecessor, MaskFormer. However, [More Information Needed] on the specific computational requirements and their implications for accessibility and environmental impact.

4. Training data and parameters: The choice of training data, augmentation techniques, and parameters significantly impacts Mask2Former's performance. [More Information Needed] on the potential biases and limitations introduced by the training data and the need for transparency in reporting these details.

5. Comparison to specialized architectures: While Mask2Former outperforms specialized architectures on several datasets, [More Information Needed] on its performance across a wider range of datasets and real-world applications to assess its robustness and generalizability.

To comprehensively address the foreseeable issues and their societal implications, more information is needed on the specific use cases, deployment contexts, and potential risks associated with the Mask2Former model. Engaging with domain experts, stakeholders, and affected communities could provide valuable insights to guide the responsible development and deployment of this technology.

## Training Details

### Training Data

The model facebook/mask2former-swin-tiny-coco-instance is trained on the COCO dataset for instance segmentation. The training data is augmented using large-scale jittering (LSJ) with random scales from 0.1 to 2.0 followed by a fixed size crop of 1024x1024.

### Training Procedure

#### Preprocessing

For the preprocessing of the data for the model facebook/mask2former-swin-tiny-coco-instance, based on the provided references:

Data augmentation:
We use the large-scale jittering (LSJ) augmentation [18,23] with a random scale sampled from range 0.1 to 2.0 followed by a fixed size crop to 1024×1024.

Resizing during inference:
We use the standard Mask R-CNN inference setting where we resize an image with shorter side to 800 and longer side up-to 1333.

[More Information Needed] for details on tokenization.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters used for the model facebook/mask2former-swin-tiny-coco-instance:

Optimizer:
- AdamW optimizer

Learning rate schedule:
- Initial learning rate: 0.0001
- Learning rate multiplier of 0.1 applied to both CNN and Transformer backbones
- Decay learning rate at 0.9 and 0.95 fractions of the total number of training steps by a factor of 10

Weight decay:
- 0.05 for all backbones

Training epochs and batch size:
- 50 epochs with a batch size of 16

Data augmentation:
- Large-scale jittering (LSJ) augmentation with a random scale sampled from range 0.1 to 2.0
- Fixed size crop to 1024×1024

[More Information Needed] on other hyperparameters such as specific CNN and Transformer backbone architectures used, pixel decoder architecture, and Transformer decoder architecture.

#### Speeds, Sizes, Times

Here are the details about the model facebook/mask2former-swin-tiny-coco-instance based on the provided references:

Throughput: [More Information Needed]

Start or end training time: [More Information Needed] 

Checkpoint sizes: [More Information Needed]

The references mention some architectural details of Mask2Former such as using a multi-scale deformable attention Transformer pixel decoder, a Transformer decoder with 9 layers and 100 queries, and training settings like learning rates and weight decay values for ResNet and Swin backbones. However, they do not provide specific information about the throughput, training time, or checkpoint sizes for the facebook/mask2former-swin-tiny-coco-instance model.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model facebook/mask2former-swin-tiny-coco-instance evaluates on the following datasets:

- COCO dataset for instance segmentation (80 "things" categories)
- Cityscapes dataset
- ADE20K dataset (100 "things" and 50 "stuff" categories) 
- Mapillary Vistas dataset

The model is also evaluated on the panoptic and semantic segmentation tasks in addition to instance segmentation, showing its universality. Specific metrics reported include PQ (panoptic quality), AP (average precision) for instance segmentation, and mIoU (mean Intersection-over-Union) for semantic segmentation.

#### Factors

Based on the provided references about the Mask2Former model (facebook/mask2former-swin-tiny-coco-instance), here are some key characteristics that may influence the model's behavior:

Datasets and Domains:
- The model was trained and evaluated on multiple image segmentation datasets including COCO, ADE20K, Cityscapes, and Mapillary Vistas (References 2, 4). Performance may vary across these different domains.
- The model can generalize to different image segmentation tasks (panoptic, instance, semantic), but still needs task-specific training for optimal results (Reference 1).

Object Sizes and Types: 
- The model struggles with segmenting small objects, suggesting potential performance disparities based on object size (Reference 3).
- COCO dataset contains 80 "thing" categories and 53 "stuff" categories. Instance segmentation is only evaluated on "thing" categories. (Reference 7). There may be performance differences between "things" and "stuff".

[More Information Needed] on other relevant population subgroups and factors that performance should be disaggregated across to uncover potential disparities. The references do not provide sufficient information to comment on this.

In summary, key influencing characteristics include the training datasets/domains, object sizes, and thing vs. stuff categories. More disaggregated evaluation across relevant factors would help uncover any other performance disparities in the model. Let me know if you need any clarification or have additional information to provide.

#### Metrics

Based on the provided references, the main evaluation metric for the instance segmentation task performed by the facebook/mask2former-swin-tiny-coco-instance model is:

AP (Average Precision) [2, 9]: This is the standard metric used for evaluating instance segmentation performance on the COCO dataset, which focuses on the "things" categories.

The references also mention additional metrics that can be derived from a model trained on panoptic segmentation annotations:

AP_Th_pan [1, 9]: AP evaluated on the "thing" categories using instance segmentation annotations from a model trained on panoptic segmentation.

However, since the facebook/mask2former-swin-tiny-coco-instance model is specifically trained for instance segmentation, AP is the primary metric for evaluating its performance and tradeoffs between different errors.

[More Information Needed] on the specific tradeoffs and error types considered for this model.

### Results

Based on the references provided, here are the evaluation results for the model facebook/mask2former-swin-tiny-coco-instance:

Datasets:
- The model was evaluated on the COCO dataset for instance segmentation. COCO has 80 "things" categories which are used for instance segmentation evaluation.

Metrics:
- For instance segmentation, the standard AP (average precision) metric is used.

Results:
[More Information Needed]

The references mention evaluation results for Mask2Former with various backbones on the COCO val2017 set in Table III. However, there are no specific results provided for the Swin-Tiny backbone variant. More information would be needed from the Mask2Former paper or supplementary materials to report the exact AP numbers achieved by the facebook/mask2former-swin-tiny-coco-instance model.

#### Summary

Based on the provided references, the model facebook/mask2former-swin-tiny-coco-instance was likely evaluated on the COCO instance segmentation dataset. The references mention that for instance segmentation, the standard AP (average precision) metric is used.

However, the provided references do not contain the specific evaluation results for the facebook/mask2former-swin-tiny-coco-instance model. The references discuss the general performance of Mask2Former with various backbones on the COCO dataset, but do not provide the AP score for the specific Swin-Tiny backbone variant.

To summarize the evaluation results for the facebook/mask2former-swin-tiny-coco-instance model, [More Information Needed] on the specific AP score achieved by this model on the COCO instance segmentation dataset.

## Model Examination

Based on the provided references about the Mask2Former model, here is what I can say about the explainability and interpretability of the facebook/mask2former-swin-tiny-coco-instance model:

The references suggest that object queries play an important role in Mask2Former. The number of queries can be tuned for different image segmentation tasks to achieve the best performance, depending on the number of segments per image for a particular task or dataset (Reference 3, 5). This provides some insight into how the model works internally.

However, the references also note that Mask2Former struggles with segmenting small objects and is unable to fully leverage multiscale features (Reference 4). Better utilization of the feature pyramid and designing losses for small objects are seen as critical areas for improvement. 

To make the model more explainable and interpretable, further analysis could be done to understand:

- How the object queries attend to and represent different regions and objects in the image
- What the model has difficulty with in terms of small objects, and why
- How the feature pyramid is being utilized and how that impacts predictions

Visualizations of attention maps and intermediate feature representations could help provide more interpretability. Ablation studies on model components could clarify their contributions.

Overall though, more specific information would be needed to comprehensively explain the inner workings and decision making process of the facebook/mask2former-swin-tiny-coco-instance model. The provided references give some high-level insights but lack details on the explainability and interpretability of this specific variant of Mask2Former.

[More Information Needed]

## Environmental Impact

- **Hardware Type:** The model facebook/mask2former-swin-tiny-coco-instance is trained on a V100 GPU with a batch size of 16, as mentioned in the references:

"we train our models for 50 epochs with a batch size of 16."

"Frames-per-second (fps) is measured on a V100 GPU with a batch size of 1 by taking the average runtime on the entire validation set including post-processing time."
- **Software Type:** Based on the provided references, the model facebook/mask2former-swin-tiny-coco-instance appears to be trained on the COCO dataset for the instance segmentation task. The references mention training Mask2Former on various datasets like COCO, Cityscapes, ADE20K, and Mapillary Vistas, but do not explicitly state the software or framework used for training.

[More Information Needed] regarding the specific software or deep learning framework (e.g., PyTorch, TensorFlow) used to train the facebook/mask2former-swin-tiny-coco-instance model.
- **Hours used:** Based on the references provided, the model facebook/mask2former-swin-tiny-coco-instance was trained for 50 epochs with a batch size of 16 (Reference 1).
- **Cloud Provider:** [More Information Needed]

The provided references do not specify the cloud provider used for training the model facebook/mask2former-swin-tiny-coco-instance. More information would be needed from the model developers or training logs to determine the specific cloud provider utilized.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the model facebook/mask2former-swin-tiny-coco-instance. To accurately estimate the carbon emissions, additional details would be needed, such as:

- The total training time
- The hardware used for training (GPU model, number of GPUs, etc.)
- The energy consumption of the hardware during training
- The carbon intensity of the electricity grid where the training was performed

Without these specifics, it is not possible to provide a reliable estimate of the carbon emissions associated with training this particular model based solely on the given references.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
Mask2Former adopts a meta architecture with a backbone, pixel decoder, and a new Transformer decoder. The key components of the Transformer decoder include:
- Masked attention operator that extracts localized features by constraining cross-attention within the predicted mask regions for each query
- Efficient multi-scale strategy that feeds successive feature maps from the pixel decoder's feature pyramid into successive Transformer decoder layers to handle small objects
- Additional optimizations like switching the order of self and cross-attention, making query features learnable, and removing dropout

The model uses a Swin-Tiny backbone pretrained on ImageNet.

Objective:
Mask2Former is a new architecture capable of addressing any image segmentation task, including instance, panoptic, and semantic segmentation. By constraining cross-attention within predicted mask regions using masked attention, Mask2Former extracts localized features effectively. The model aims to reduce research efforts by unifying architectures across different segmentation tasks while outperforming specialized architectures on popular benchmarks like COCO and ADE20K.

[More Information Needed] on specific training details, hyperparameters, and dataset used for the facebook/mask2former-swin-tiny-coco-instance model.

### Compute Infrastructure

Based on the provided references, the compute infrastructure details for the model facebook/mask2former-swin-tiny-coco-instance are:

- GPU: V100
- Batch size: 1 (for inference)
- Frames-per-second (fps) is measured on a V100 GPU with a batch size of 1 by taking the average runtime on the entire validation set including post-processing time.

[More Information Needed] for the following:
- Specific training batch size
- Exact training infrastructure (GPUs, machines, etc.)
- Training time

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

