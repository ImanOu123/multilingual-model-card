# Model Card for microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft

The model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is a Swin Transformer V2 model with 197M parameters, pre-trained on ImageNet-22K at 192x192 to 256x256 resolution with a window size of 12 to 16, and fine-tuned on ImageNet-1K for image classification tasks.

## Model Details

### Model Description

Model Architecture:
Swin Transformer V2 (SwinV2) is a hierarchical Transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. SwinV2 introduces techniques to scale up the model capacity and window resolution, including res-post-norm and scaled cosine attention, as well as a log-spaced continuous relative position bias approach.

Training Procedures:
[More Information Needed]

Parameters:
The SwinV2-L model is a large-capacity variant of Swin Transformer V2.

Important Disclaimers:
- Larger models are more data hungry. SwinV2 exploits a self-supervised pre-training method, SimMIM, to alleviate the demands on labeled data.
- If exceeding a certain model size, more labeled data, stronger regularization, or advanced self-supervised learning methods may be required for further performance gains.

Additional Notes:
- The model was pre-trained on ImageNet-22K and fine-tuned on ImageNet-1K.
- It uses an input image size (window size) of 192×192 to 256x256 (12×12 to 16x16).

For more detailed information on specific hyperparameters and training settings, please refer to the original paper or contact the model developers.

- **Developed by:** Ze Liu; Han Hu; Yutong Lin; Zhuliang Yao; Zhenda Xie Yixuan; Wei Jia; Ning Yue; Cao Zheng; Zhang Li; Dong Furu; Wei Baining Guo
- **Funded by:** Based on the provided references, it appears that Microsoft funds the project of the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft. The references mention:

- Thanking colleagues at Microsoft for their help, discussion, and GPU resources (Reference 1)
- The project may contain Microsoft trademarks or logos (Reference 4)

No other specific funding organizations are mentioned in the given references. To fully answer this question, [More Information Needed] on whether there are any additional funding sources beyond Microsoft.
- **Shared by:** Based on the references provided, the main contributors that made the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft available online as a GitHub repo are:

Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie Yixuan, Wei Jia, Ning Yue, Cao Zheng, Zhang Li, Dong Furu, Wei Baining Guo

The references also thank many colleagues at Microsoft for their help, in particular, Eric Chang, Lidong Zhou, Jing Tao, Aaron Zhang, Edward Cui, Bin Xiao, Lu Yuan, Peng Cheng, Fan Yang for useful discussion and the help on GPU resources and datasets.

[More Information Needed] on the specific roles and contributions of each individual in making the model available on GitHub.
- **Model type:** The microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft model is a large-scale vision Transformer model that uses self-supervised pre-training with SimMIM on ImageNet-22K, followed by supervised fine-tuning on ImageNet-1K for image classification.
- **Language(s):** The model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is an image classification model and does not use or process natural human language.
- **License:** [More Information Needed]

The references provided do not contain any information about the specific license being used for the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft. More details would be needed from the model developers or repository to determine the license that applies to this particular model.
- **Finetuned from model:** Based on the information provided in the references, the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is likely fine-tuned from the SwinV2-L* model pretrained on ImageNet-22K with an input resolution of 192x192, as mentioned in note 6:

"SwinV2-B* (SwinV2-L*) with input resolution of 256x256 and 384x384 both fine-tuned from the same pre-training model using a smaller input resolution of 192x192."

The pretrained model details and download links are provided in reference 9:

| Model | Pretrain Dataset | Resolution | Window Size | acc@1 | acc@5 | #Params | FLOPs | FPS | Weights |
| SwinV2-L* | ImageNet-22K | 384x384 | 24x24 | 87.6 | 98.3 | 197M | 115.4G | 33 | [github](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth)/[baidu](https://pan.baidu.com/s/11PhCV7qAGXtZ8dXNgyiGOw?pwd=swin)/[config](configs/swinv2/swinv2_large_patch4_window12_192_22k.yaml) |

So the base model is likely SwinV2-L* pretrained on ImageNet-22K with 192x192 resolution, and then fine-tuned on ImageNet-1K with increased resolution from 192x192 to 256x256.
### Model Sources

- **Repository:** https://github.com/microsoft/Swin-Transformer
- **Paper:** https://arxiv.org/pdf/2111.09883.pdf
- **Demo:** [More Information Needed]

The references provided do not contain a direct link to a demo of the specific model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft. More information would be needed to locate or create a demo for this particular fine-tuned Swin Transformer V2 large model.
## Uses

### Direct Use

The model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft can be used for image classification tasks without requiring fine-tuning, post-processing, or plugging into a pipeline. As mentioned in the references, detailed instructions for using Swin Transformer models for image classification can be found in the get_started.md file.

Here's a code snippet showing how to use the model for image classification (based on the information provided in the references):

[More Information Needed]

The model has been pre-trained using a self-supervised learning method called SimMIM, which reduces the need for vast labeled images. This allows the model to achieve state-of-the-art performance on various visual benchmarks while using only a fraction of the labeled data compared to other large vision models.

To further improve the model's performance and training stability, techniques such as residual-post-norm method combined with cosine attention, log-spaced continuous position bias method, and activation check-pointing have been employed.

For more details on using the model for other tasks like object detection, instance segmentation, semantic segmentation, self-supervised learning, and video recognition, please refer to the respective repositories mentioned in the references.

If you have any further questions or need assistance with using the model, feel free to reach out to me, as I serve as the contact person for model card updates.

### Downstream Use

Based on the provided references, the microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft model can be used for the following tasks when fine-tuned:

1. ImageNet-1K image classification: The model can be fine-tuned for 30 epochs using an AdamW optimizer, cosine decay learning rate scheduler, and data augmentation to achieve strong performance on the ImageNet-1K dataset.

2. COCO object detection: The model can be used as a backbone in the Cascade Mask R-CNN framework for object detection on the COCO dataset. Fine-tuning settings include multi-scale augmentation, AdamW optimizer, and a 3x scheduler.

3. ADE20K semantic segmentation: The model can be fine-tuned for semantic segmentation on the ADE20K dataset. [More Information Needed] for specific fine-tuning settings.

4. Kinetics-400 video action recognition: The model can be evaluated on the Kinetics-400 dataset for video action classification. [More Information Needed] for specific fine-tuning settings.

The model has also been used in various larger ecosystems and applications, such as:

1. Audio Classification: Hierarchical Token Semantic Audio Transformer
2. StyleGAN: StyleSwin
3. Face Recognition: FaceX-Zoo
4. Image Restoration: SwinIR

Code snippet for inference using TorchServe:
[More Information Needed]

The model's performance can be further improved by using a different window size at test time, as suggested in the references. Scaling up the model size and using more labeled data or advanced self-supervised learning methods like SimMIM can also lead to better performance on downstream tasks.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft may be misused and what users ought not do:

- The model should not be used in a way that causes confusion or implies Microsoft sponsorship without following Microsoft's Trademark & Brand Guidelines. Any use of Microsoft or third-party trademarks/logos must adhere to the respective policies. (Reference 2)

- Users should not engage in conduct that violates the Microsoft Open Source Code of Conduct when using or contributing to this model. (Reference 3) 

- When making contributions to the model, users must agree to the Contributor License Agreement and only contribute what they have the rights to. (Reference 5)

- The model may exhibit degraded performance when transferring across different window resolutions. Users should be aware of this limitation and not assume the model will maintain accuracy in such scenarios without further testing/validation. (Reference 7)

[More Information Needed] on other specific technical ways the model could be misapplied or abused, as the references do not go into detail on this. More information would be needed from the model developers on risks they foresee.

In general, users have an ethical obligation to use the model responsibly, respect intellectual property rights, follow applicable laws and terms of use, and consider potential societal impacts. They should not employ the model for deceptive, harmful or illegal purposes.

### Bias, Risks, and Limitations

Based on the provided references about the Swin Transformer V2 model (microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft), here are some potential issues and limitations:

Technical limitations:
- The model may have issues scaling up capacity and window resolution further without techniques like res-post-norm and scaled cosine attention to stabilize training (ref 1, 2, 6). Exceeding a certain model size may require more labeled data, stronger regularization, or advanced self-supervised learning methods (ref 8).
- The log-spaced continuous relative position bias approach helps the model transfer across window resolutions, but there are still accuracy drops with large resolution changes between pre-training and fine-tuning (ref 5).

Potential misuse or misunderstandings: 
- [More Information Needed] The references do not cover potential misuse cases or misunderstandings of the model.

Societal impact and potential harms:
- [More Information Needed] The references focus on technical details and do not discuss broader societal impacts or potential harms from the model or its applications.

In summary, the key known limitations relate to challenges in stably scaling the model to even larger capacities and resolutions. More information would be needed to assess potential misuse risks and societal harms. The model achieves strong results on benchmarks, but responsible deployment would require further analysis of its capabilities and limitations in real-world contexts.

### Recommendations

Based on the provided references about the Swin Transformer V2 model (microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft), here are some recommendations regarding foreseeable issues:

1. Scaling up model capacity and window resolution: The original Swin Transformer faced issues with exploding activation values in deeper layers when scaled up to larger sizes (References 2, 8). The res-post-norm and scaled cosine attention techniques introduced in Swin Transformer V2 help mitigate this issue and stabilize training for larger models (References 1, 2, 3). However, it's important to monitor and validate the model's behavior as it is scaled up further.

2. Transferring models across window resolutions: The original Swin Transformer showed degraded performance when transferring pre-trained models to larger image resolutions and window sizes (Reference 9). The log-spaced continuous relative position bias approach in Swin Transformer V2 aims to address this issue and improve transferability (References 1, 4). However, it's recommended to thoroughly test and validate the model's performance when transferring across different resolutions.

3. Ethical considerations: [More Information Needed] The provided references do not discuss ethical considerations or potential misuse of the model.

4. Societal impact: [More Information Needed] The references do not provide insights into the long-term societal impact of the Swin Transformer V2 model.

5. Legal and rights aspects: The model adopts the Microsoft Open Source Code of Conduct (Reference 6), but [More Information Needed] regarding other legal or rights-related aspects.

To comprehensively address foreseeable issues, it is recommended to gather more information on the model's ethical considerations, potential societal impact, and legal aspects. Collaboration with domain experts in these areas can help identify and mitigate risks associated with the model's deployment and use.

## Training Details

### Training Data

The model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft was pre-trained on the ImageNet-22K dataset with an input image size of 192x192, then fine-tuned on ImageNet-1K with image sizes of 192x192 to 256x256. [More Information Needed] on any additional data filtering or pre-processing steps used.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about the preprocessing for the data of the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft:

Resizing:
- The model was pre-trained on ImageNet-22K using an input image size (window size) of 192×192 (12×12). (Reference 1)
- For ImageNet-1K fine-tuning, if the input image resolution is larger than that in pre-training, an additional fine-tuning step is conducted. (Reference 6)
- The model name suggests fine-tuning was done with an increased resolution of 256×256 (16×16 window size).

Data Augmentation and Regularization:
During pre-training, the following augmentation and regularization strategies were used (Reference 1):
- RandAugment
- Mixup
- Cutmix
- Random erasing
- Stochastic depth

[More Information Needed] on the specific augmentation parameters used.

The same data augmentation and regularizations were used in the ImageNet-1K fine-tuning step. (Reference 6)

Tokenization:
[More Information Needed] The references do not provide details on the image tokenization method used.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters I could find for the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft:

ImageNet-22K pre-training stage:
- Input image size (window size): 192x192 (12x12) 
- Optimizer: AdamW
- Number of epochs: 90
- Learning rate scheduler: Cosine with 5-epoch linear warm-up
- Batch size: 4096
- Initial learning rate: 0.001
- Weight decay: 0.1
- Gradient clipping with max norm of 5.0
- Data augmentation and regularization: RandAugment, Mixup, Cutmix, Random Erasing, Stochastic Depth (ratio not specified for large model)

ImageNet-1K fine-tuning stage:
- Input image size (window size): 256x256 (window size 12 to 16)
- [More Information Needed] on other hyperparameters

Additional techniques used:
- Activation check-pointing to reduce GPU memory usage
- Zero-Redundancy Optimizer (ZeRO) for better memory efficiency 
- Sequential self-attention computation for very large image resolutions

[More Information Needed] on several other training details like exact stochastic depth ratio used, fine-tuning hyperparameters, etc.

#### Speeds, Sizes, Times

Here are the details I can provide about the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft based on the given references:

Model architecture:
- The model is a Swin Transformer V2 Large variant
- C = 192, #blocks = {2, 2, 18, 2} 

Training details:
- Trained on image resolutions from 192x192 to 256x256
- Uses window sizes from 12x12 to 16x16
- Pre-trained on ImageNet-22k dataset and fine-tuned on ImageNet-1k

Optimizations used:
- Gradient checkpointing to reduce memory usage
- Activation checkpointing to reduce memory usage of feature maps in Transformer layers
- Sequential self-attention computation to handle very large image resolutions
- Zero-Redundancy Optimizer (ZeRO) to split model parameters and optimization states across GPUs

[More Information Needed] for the following:
- Exact model size in number of parameters 
- Throughput metrics
- Training start and end times
- Checkpoint sizes

The references do not contain code blocks directly relevant to this specific model. More details would be needed from the model developers to fill in the missing information.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft evaluates on the following benchmarks or datasets:

1. ADE20K semantic segmentation (Reference 1)
2. ImageNet V2 image classification (Reference 2)
3. ImageNet-1K V1 image classification (Reference 2)
4. COCO object detection (Reference 5)
5. Kinetics-400 video action recognition (Reference 4)

The model achieves state-of-the-art performance on these benchmarks, suggesting that scaling up vision models is beneficial for various vision tasks such as semantic segmentation, image classification, object detection, and video action recognition.

#### Factors

Based on the provided references, here are some foreseeable characteristics that may influence the behavior of the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft:

1. Domain and context:
   - The model is trained on the ImageNet-22K dataset (References 6, 7, 8, 10), which covers a wide range of image categories. However, its performance on specific domains or contexts outside of ImageNet-22K is not explicitly mentioned and may require further evaluation.

2. Image resolution and window size:
   - The model is trained with image resolutions of 192x192 to 256x256 and window sizes of 12x12 to 16x16 (References 6, 7).
   - Transferring the model to different image resolutions and window sizes may lead to degraded performance (Reference 11).

3. Population subgroups:
   - [More Information Needed] The provided references do not explicitly mention the model's performance across different population subgroups or potential disparities.

4. Evaluation metrics:
   - The model's performance is evaluated using top-1 accuracy on the ImageNet-1K dataset (References 6, 7, 8, 9).
   - [More Information Needed] Disaggregated evaluation across different factors or subgroups is not provided in the references.

5. Computational requirements:
   - The model has 197M parameters and requires 47.5G to 115.4G FLOPs (References 6, 7), which may influence its deployment and usability in resource-constrained environments.

To better understand the model's behavior and potential disparities, further evaluation and disaggregation across relevant factors and subgroups would be beneficial.

#### Metrics

Based on the provided references, the following metrics will be used for evaluating the SwinV2 models:

1. Top-1 accuracy on ImageNet-1K V1 and V2 image classification benchmarks
2. mIoU (mean Intersection over Union) on ADE20K semantic segmentation benchmark
3. Top-1 accuracy on Kinetics-400 video action classification benchmark
4. [More Information Needed] for object detection metrics on COCO dataset

The references mention using different window sizes at test time to potentially improve performance, leveraging the Log-spaced CPB approach. However, more specific information would be needed to determine the exact tradeoffs and errors being considered between different model configurations.

### Results

Based on the provided references, here are the evaluation results for the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft:

Model Description:
- Architecture: SwinV2-L* (Swin Transformer V2 Large)
- Pre-trained on ImageNet-22K dataset
- Fine-tuned on ImageNet-1K dataset
- Input resolution: 192x192 to 256x256
- Window size: 12x12 to 16x16

Performance:
- ImageNet-1K top-1 accuracy (256x256 input, 16x16 window): 86.9%
- ImageNet-1K top-5 accuracy (256x256 input, 16x16 window): 98.0%
- ImageNet-1K top-1 accuracy (384x384 input, 24x24 window): 87.6%
- ImageNet-1K top-5 accuracy (384x384 input, 24x24 window): 98.3%

Efficiency:
- Number of parameters: 197M
- FLOPs (256x256 input): 47.5G
- Throughput (256x256 input): 95 images/s

[More Information Needed] for additional evaluation metrics or results.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft:

- The model achieves strong performance on the ImageNet-1K and ImageNet-V2 benchmarks. SwinV2-L* (384x384) achieves 78.31% top-1 accuracy on ImageNet-1K-V2. (Reference 4)

- Scaling up the vision model size is beneficial for dense vision recognition tasks like object detection. (Reference 2) 

- The model achieves 59.9 mIoU on the ADE20K semantic segmentation val set, which is 1.5 points higher than the previous best result. Using a larger window size at test time can bring an additional 0.2 gain. (Reference 3)

- On the Kinetics-400 action classification benchmark, the model achieves 86.8% top-1 accuracy, outperforming the previous best result by 1.4%. Using a larger window size at test time brings an additional 0.2% improvement. (Reference 6)

- [More Information Needed] on the specific pre-training dataset and fine-tuning procedure used for this model.

- The Log-spaced Continuous Position Bias (CPB) approach allows effectively using larger window sizes at test time for additional performance gains. (References 3, 6)

- Architectural changes like post-norm and scaled cosine attention help stabilize training at larger model sizes. (Reference 7)

## Model Examination

Based on the provided references, here is the experimental section on explainability/interpretability for the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft:

The Swin Transformer V2 model introduces several adaptations over the original Swin Transformer architecture to better scale up model capacity and window resolution:

1. A res-post-norm configuration replaces the previous pre-norm configuration
2. A scaled cosine attention replaces the original dot product attention  
3. A log-spaced continuous relative position bias approach is introduced

These changes aim to address issues observed when scaling up capacity and window resolution in the original Swin Transformer, such as degraded performance when transferring models across window resolutions.

[More Information Needed] on specific experiments or analysis done regarding the explainability and interpretability of the architectural changes in Swin Transformer V2. The provided references do not contain code blocks directly relevant to this topic.

Scaling up the model size from Swin-B to Swin-L shows diminishing gains, suggesting that exceeding this size may require more labeled data, stronger regularization, or advanced self-supervised learning methods to see further benefits. Using different window sizes at test time can provide additional improvements on dense vision recognition tasks like object detection.

## Environmental Impact

- **Hardware Type:** Based on the provided references, the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft was trained on Nvidia A100-40G GPUs, as mentioned in reference 6:

"With the above techniques, we successfully trained a 3 billion Swin Transformer model and effectively transferred it to various vision tasks with image resolution as large as 1,536×1,536, using Nvidia A100-40G GPUs."
- **Software Type:** Based on the provided references, the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is likely trained using PyTorch deep learning framework, as evidenced by the following:

1. The references mention using Nvidia A100 GPUs for training, which are commonly used with PyTorch.

2. Techniques like activation check-pointing and sequential self-attention computation are mentioned, which are typically implemented in PyTorch for memory optimization during training.

3. The Swin Transformer architecture is a popular computer vision backbone that has been implemented and trained using PyTorch in various research papers and open-source repositories.

However, the references do not explicitly state the software framework used for training this specific model. To provide a definitive answer, more information would be needed, such as the model's training code or documentation.

[More Information Needed]
- **Hours used:** [More Information Needed]

The provided references do not contain specific information about the training time for the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft. The references mention some training details like number of epochs, batch size, learning rate, etc. for SwinV2-B and SwinV2-L models in general, but do not specify the exact training time for this particular model.
- **Cloud Provider:** [More Information Needed]

The provided references do not contain information about the specific cloud provider used for training the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft. More details would be needed from the model development process to determine the cloud provider.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft. To accurately report the carbon emissions, details such as the hardware used for training, total training time, energy consumption, and the carbon intensity of the electricity grid would be required.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
Swin Transformer V2 is a hierarchical Transformer architecture that computes representations using shifted windows. Key adaptations from the original Swin Transformer (V1) include:
1. Res-post-norm configuration to replace the previous pre-norm 
2. Scaled cosine attention to replace the original dot product attention
3. Log-spaced continuous relative position bias approach to more effectively transfer across window resolutions

The model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is a Swin Transformer V2 variant with the following configuration:
- Large model size 
- Patch size: 4x4
- Window size: 12x12 to 16x16
- Input image resolution: 192x192 to 256x256
- Pre-trained on ImageNet-22K dataset and fine-tuned on ImageNet-1K

Model Objective:
The key objective of Swin Transformer V2 is to serve as a general-purpose backbone for computer vision tasks. By introducing visual priors like hierarchy, locality, and translation invariance into the Transformer architecture, it aims to combine the strengths of CNNs and Transformers.

The specific model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is pre-trained on the larger ImageNet-22K dataset and then fine-tuned on the ImageNet-1K dataset for the image classification task. The goal is to leverage the power of large-scale pre-training and transfer learning to achieve state-of-the-art performance on downstream vision tasks.

[More Information Needed] on the exact training hyperparameters, computational requirements, and evaluation metrics for this specific model variant.

### Compute Infrastructure

Based on the provided references, the compute infrastructure used for training the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft likely included:

- A100 GPUs with 40GB memory, as mentioned in reference 3 for training large models on very high resolutions
- Activation check-pointing to reduce GPU memory consumption during training, as described in reference 2
- Sequential self-attention computation to alleviate memory bottlenecks for large resolutions, per reference 3
- Zero-Redundancy Optimizer (ZeRO) to split model parameters and optimization states across multiple GPUs to reduce memory usage, as noted in reference 5

The exact number of GPUs, batch sizes, and other specific training hyperparameters [More Information Needed] from the given references.

## Citation

```
@misc{ze-swin,
    author = {Ze Liu and
              Han Hu and
              Yutong Lin and
              Zhuliang Yao and
              Zhenda Xie Yixuan and
              Wei Jia and
              Ning Yue and
              Cao Zheng and
              Zhang Li and
              Dong Furu and
              Wei Baining Guo},
    title  = {Swin Transformer V2: Scaling Up Capacity and Resolution},
    url    = {https://arxiv.org/pdf/2111.09883.pdf}
}
```

