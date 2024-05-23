# Model Card for MCG-NJU/videomae-base-finetuned-kinetics

The MCG-NJU/videomae-base-finetuned-kinetics model is a self-supervised video pre-training method called Video Masked Autoencoder (VideoMAE) that masks random cubes in videos and reconstructs the missing ones, and is fine-tuned on the Kinetics-400 dataset.

## Model Details

### Model Description

Model Description for MCG-NJU/videomae-base-finetuned-kinetics:

Model Architecture:
- Uses an asymmetric encoder-decoder architecture for video self-supervised pre-training
- Encoder: Vanilla ViT-Base with joint space-time attention 
- Decoder: [More Information Needed]
- Discards the decoder during the fine-tuning phase

Training Procedures:
Pre-training:
- Pre-trained on Kinetics-400 for 800 epochs by default
- Masking ratio: 75%
- Batch size: 192 
- Base learning rate: 3e-4
- Samples 16 frames with a temporal stride of 4

Fine-tuning:
- Performs dense sampling following Slowfast
- Trained for [More Information Needed] epochs
- Batch size: [More Information Needed]
- Base learning rate: [More Information Needed]
- Layer decay: [More Information Needed] 
- Drop path: [More Information Needed]
- Uses repeated augmentation

Evaluation:
- Inference protocol: 5 clips × 3 crops

Parameters:
[More Information Needed]

Important Disclaimers:
- Domain shift between pre-training and target datasets could impact transfer learning performance when fine-tuning the pre-trained model on other video datasets.

For any further questions or updates to this model card, please contact the project organizer at [More Information Needed].

- **Developed by:** Zhan Tong; Yibing Song; Jue Wang; Limin Wang
- **Funded by:** Based on the provided references, there is no explicit information about the funding sources for the MCG-NJU/videomae-base-finetuned-kinetics model. The references mention the authors, licenses, datasets, and technical details, but do not specify the organizations or people that funded the project.

[More Information Needed]
- **Shared by:** Based on the references provided, the contributors who made the model MCG-NJU/videomae-base-finetuned-kinetics available online as a GitHub repo are:

Zhan Tong, Yibing Song, Jue Wang, and Limin Wang

The references mention that portions of the project use code from SlowFast, pytorch-image-models, and BEiT, but it's not clear if the contributors of those projects were directly involved in creating the VideoMAE model and publishing it. More information would be needed to determine that.
- **Model type:** The MCG-NJU/videomae-base-finetuned-kinetics model is a self-supervised video pre-training model using the masked autoencoder (MAE) approach with an extremely high masking ratio (90%-95%) and tube masking strategy, which is then fine-tuned on the Kinetics-400 dataset for video classification.
- **Language(s):** The model MCG-NJU/videomae-base-finetuned-kinetics processes videos from the Kinetics-400 dataset, which contains human action videos in natural settings.
- **License:** The majority of the VideoMAE project is released under the CC-BY-NC 4.0 license as found in the LICENSE file: https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE

Portions of the project are available under separate license terms:
- SlowFast and pytorch-image-models are licensed under the Apache 2.0 license. 
- BEiT is licensed under the MIT license.
- **Finetuned from model:** Based on the provided references, the model MCG-NJU/videomae-base-finetuned-kinetics appears to be fine-tuned from a base VideoMAE model that was pre-trained using masked autoencoding on video data. However, the specific name and link to the base pre-trained VideoMAE model are not provided in the given information.

[More Information Needed] on the exact name and link to the base VideoMAE model used for fine-tuning.
### Model Sources

- **Repository:** https://github.com/MCG-NJU/VideoMAE
- **Paper:** https://arxiv.org/pdf/2203.12602.pdf
- **Demo:** [More Information Needed]

The provided references do not contain a direct link or mention of a demo for the MCG-NJU/videomae-base-finetuned-kinetics model. More information would be needed from the model development team to determine if a demo link is available to include in the model card description.
## Uses

### Direct Use

Based on the provided references, the model MCG-NJU/videomae-base-finetuned-kinetics has already been fine-tuned on the Kinetics-400 dataset. The references mention:

"If the pre-trained ViT-B is additionally fine-tuned on Kinetics-400 with labels, the transfer learning performance can further increase about 5 mAP (from 26.7 to 31.8)."

Since the model is already fine-tuned, it should be able to be used for video classification tasks on the Kinetics-400 dataset or similar domains without further fine-tuning.

However, the references do not provide specific details on how to load and use the model for inference without additional processing or pipelines. A code snippet demonstrating this is not provided.

[More Information Needed] on the exact model architecture, input format, and inference process to provide a complete answer and code snippet for using the model directly.

### Downstream Use

The MCG-NJU/videomae-base-finetuned-kinetics model can be fine-tuned on downstream tasks like action recognition on datasets such as UCF101, HMDB51, and Something-Something V2. It can also be used for action detection on the AVA dataset.

For example, to use the pre-trained VideoMAE model for action detection on AVA:

1. Fine-tune the VideoMAE ViT-B model that was pre-trained on Kinetics-400 using the AVA training set. 
2. Evaluate the fine-tuned model on the AVA validation set using mean Average Precision (mAP) at IoU threshold 0.5.

[More Information Needed] for code snippet.

The pre-trained VideoMAE model demonstrates strong transferability, achieving 26.7 mAP on AVA with a ViT-B backbone. Performance can be further improved by:

- Additionally fine-tuning the pre-trained ViT-B on Kinetics-400 with labels (+5 mAP)
- Scaling up pre-training with larger video datasets like Kinetics-700 or more powerful ViT-L/ViT-H backbones

[More Information Needed] on integrating the model into a larger ecosystem or app.

### Out-of-Scope Use

Based on the provided references, here are some potential misuses of the MCG-NJU/videomae-base-finetuned-kinetics model that users ought to avoid:

- The model should not be used for tasks or on datasets that violate the licenses of the training data, which include custom licenses for Something-Something V2 and UCF101, and CC BY-NC 4.0 for Kinetics-400, HMDB51 and AVA. In particular, the non-commercial restriction of the CC BY-NC 4.0 license should be respected.

- The model may not perform optimally when applied to video datasets that are very different from the Kinetics dataset it was fine-tuned on, due to domain shift issues. Users should be cautious about applying it to vastly different video domains without further validation.

- [More Information Needed] on other specific ways the model could be intentionally misused or abused for harmful purposes.

Additionally, the energy-intensive pre-training of the model has environmental impact concerns in terms of carbon emissions. However, the shared pre-trained model amortizes this cost across many downstream applications.

In general, users should strive to use the model only for its intended video understanding purposes, respect dataset licenses, be mindful of potential domain shift limitations, and consider the computational resource implications. More specific guidance would require additional information on intended use cases and performance characteristics of the model.

### Bias, Risks, and Limitations

Based on the provided references, here are some known or foreseeable issues with the VideoMAE model MCG-NJU/videomae-base-finetuned-kinetics:

Potential negative societal impacts:
- The pre-training phase may lead to a large amount of carbon emissions due to high energy consumption. However, the model only needs to be pre-trained once and can then be fine-tuned for various downstream tasks.
- VideoMAE's ability to improve video understanding could increase the risk of the model or its outputs being used incorrectly, such as for unauthorized surveillance.

Technical limitations:
- VideoMAE might not capture motion information from very small objects due to the extremely high masking ratio. Tokens containing small motion patterns might all be masked, making it difficult for VideoMAE to reconstruct them.
- When transferring pre-trained VideoMAE models to other video datasets (e.g., from Kinetics to Something-Something), the results are slightly worse than models directly pre-trained on the target video datasets. Domain shift between pre-training and target datasets could be an important issue.

[More Information Needed] on other potential misunderstandings or sociotechnical limitations, as the provided references do not cover these aspects in detail.

### Recommendations

Based on the provided references about the VideoMAE model MCG-NJU/videomae-base-finetuned-kinetics, here are some foreseeable issues and recommendations:

Energy Consumption and Carbon Emissions:
The pre-training phase of VideoMAE can lead to significant energy consumption and carbon emissions (reference 6, 11). While pre-training only needs to be done once, it's important to consider ways to mitigate the environmental impact, such as using renewable energy sources or optimizing the training process for efficiency.

Potential Misuse:
VideoMAE's strong video understanding capabilities could potentially be misused for unethical purposes like surveillance overreach (reference 6). It's crucial to establish guidelines and safeguards around the model's use and deployment.

Domain Shift and Generalization:
There are indications that domain shift between pre-training and target datasets can impact VideoMAE's performance when transferring to new datasets (reference 2, 5). More research is needed to improve the model's generalization and robustness to domain shifts.

Small Object Motion:
VideoMAE may struggle to capture motion information from very small objects, possibly due to the high masking ratio (reference 8). Further investigation and potential improvements to the masking strategy could help address this limitation.

[More Information Needed] on the specific pre-training dataset, model architecture details, and compute resources used, to provide more targeted recommendations.

Overall, while VideoMAE shows promising results, it's important to carefully consider the ethical implications, environmental impact, and domain generalization challenges in its development and deployment. Ongoing research and responsible practices can help address these issues.

## Training Details

### Training Data

The model MCG-NJU/videomae-base-finetuned-kinetics is fine-tuned on the Kinetics-400 dataset, which contains around 240k training videos and 20k validation videos of 10s from 400 classes. [More Information Needed] for documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the provided references about the VideoMAE model, here are the details about the preprocessing for the MCG-NJU/videomae-base-finetuned-kinetics model:

Temporal downsampling:
- One video clip consisting of t consecutive frames is randomly sampled from the original video
- Temporal sampling is used to compress the clip to T frames, each containing H × W × 3 pixels
- The stride τ is set to 4 on Kinetics dataset

Cube embedding:
- Joint space-time cube embedding is adopted
- Each cube of size 2 × 16 × 16 is treated as one token embedding
- The cube embedding layer obtains T/2 × H/16 × W/16 3D tokens and maps each token to the channel dimension D

[More Information Needed] about the specific values of t, T, H, W, and D.

Tokenization:
[More Information Needed]

Resizing/Rewriting:
[More Information Needed]

The model is pre-trained on the Kinetics-400 dataset for 800 epochs with a mask ratio ρ=90%. During fine-tuning and evaluation, dense sampling is performed, and the inference protocol uses 5 clips × 3 crops.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the model MCG-NJU/videomae-base-finetuned-kinetics:

Pre-training:
- Dataset: Kinetics-400 
- Epochs: 800
- Masking ratio: 90%

Fine-tuning:
- Sampling: Dense sampling following Slowfast [23]
- Inference protocol: 5 clips × 3 crops

[More Information Needed] for the following hyperparameters:
- Batch size 
- Learning rate
- Optimizer
- Learning rate schedule

The references do not provide code blocks with the exact hyperparameter settings used. More specific details would be needed to fully reproduce the training.

#### Speeds, Sizes, Times

Here are the details about the model MCG-NJU/videomae-base-finetuned-kinetics based on the provided references:

Model Architecture:
- Backbone: 16-frame vanilla ViT-B
- Pre-trained with mask ratio ρ=90% for 800 epochs on Kinetics-400
- Fine-tuned on Kinetics-400 for evaluation

Training Details:
- Pre-training: 800 epochs on Kinetics-400
- Fine-tuning: [More Information Needed] epochs on Kinetics-400
- Batch size: [More Information Needed]
- Learning rate: [More Information Needed]

Inference Protocol:
- 5 clips × 3 crops on Kinetics-400

[More Information Needed]:
- Throughput
- Start and end time of training
- Checkpoint sizes

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model MCG-NJU/videomae-base-finetuned-kinetics is evaluated on the Kinetics-400 dataset. Specifically, reference 1 mentions that "We evaluate our VideoMAE on five common video datasets: Kinetics-400 [34], Something-Something V2 [26], UCF101 [61], HMDB51 [35], and AVA [27]." and reference 6 provides more details on the evaluation protocol: "For evaluation, all models share the same inference protocol, i.e., 5 clips × 3 crops."

The references do not provide specific benchmark results for the MCG-NJU/videomae-base-finetuned-kinetics model on the Kinetics-400 dataset. [More Information Needed] on the exact performance metrics achieved by this specific model.

#### Factors

Based on the provided references about the VideoMAE model MCG-NJU/videomae-base-finetuned-kinetics, here are some key foreseeable characteristics that may influence the model's behavior:

Domain and Context:
- The model was evaluated on five common video datasets: Kinetics-400, Something-Something V2, UCF101, HMDB51, and AVA (Ref 1). These datasets cover different types of actions and visual cues.
- Kinetics-400 videos are mostly stationary and scene-related, so the effect of temporal modeling may not be as obvious compared to datasets like Something-Something V2 which are more motion-centric (Ref 8, 9).

Population Subgroups:
- [More Information Needed] The references do not provide a clear breakdown of population subgroups in the datasets used for training and evaluation.

Performance Disparities: 
- The model may struggle to capture motion information from very small objects, as the high masking ratio during pre-training could result in tokens containing small motions being masked out (Ref 6).
- Temporal redundancy in videos may lead the model to focus more on static or slow motions rather than capturing high-level spatiotemporal reasoning (Ref 4, 5).

To uncover potential performance disparities, ideally the model should be evaluated on disaggregated subgroups within the datasets. However, the provided references do not include this level of analysis. More information would be needed about the specific composition and characteristics of the video datasets used to fully assess how different factors may influence the model's behavior across various subpopulations.

#### Metrics

Based on the provided references, the main evaluation metrics for the VideoMAE model fine-tuned on Kinetics-400 (which I assume is the MCG-NJU/videomae-base-finetuned-kinetics model) are:

1. Top-1 accuracy on the Kinetics-400 validation set (Reference 11 mentions fine-tuning on Kinetics-400)

2. Mean Average Precision (mAP) on the AVA dataset for action detection, under IoU threshold of 0.5 (Reference 2)

The references also mention evaluating transfer learning performance of the Kinetics-400 pre-trained VideoMAE on datasets like Something-Something V2, UCF101, and HMDB51 (Reference 7). However, more specific information would be needed on the exact metrics used for these transfer learning experiments for the MCG-NJU/videomae-base-finetuned-kinetics model.

The tradeoffs between different error types are not explicitly discussed for this specific model in the given references. [More Information Needed]

### Results

Based on the provided references, here are the evaluation results for the model MCG-NJU/videomae-base-finetuned-kinetics:

Dataset: The model was fine-tuned and evaluated on the Kinetics-400 (K400) dataset.

Sampling: Dense sampling was used during fine-tuning on K400.

Inference Protocol: For evaluation, 5 clips × 3 crops were used.

[More Information Needed] on the specific accuracy metrics achieved by the model on Kinetics-400.

The model was pre-trained using VideoMAE for 800 epochs on Kinetics-400 before fine-tuning. The default pre-training and fine-tuning settings are:

[More Information Needed]

In summary, the VideoMAE pre-training allowed the model to capture useful spatiotemporal structures, leading to strong fine-tuning performance on Kinetics-400. However, more specific accuracy metrics would be needed to fully quantify the model's performance.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the MCG-NJU/videomae-base-finetuned-kinetics model:

- The model uses a 16-frame vanilla ViT-B backbone, pre-trained with a mask ratio of 90% for 800 epochs on Kinetics-400, and then fine-tuned for evaluation. (Reference 1, 11)

- For inference on Kinetics-400, it uses 5 clips × 3 crops. (Reference 1, 11) 

- Transferring the learned representation from Kinetics-400 to Something-Something V2, UCF101, and HMDB51 showed the VideoMAE pre-trained models outperformed MoCo v3 pre-training, indicating VideoMAE learns more transferable representations. (Reference 4)

- However, the transferred representation performed worse on Something-Something V2 compared to models trained directly on that dataset. (Reference 2)

- Fine-tuning the Kinetics-400 pre-trained ViT-B model on labeled Kinetics-400 data improved transfer learning performance by about 5 mAP. (Reference 5)

- [More Information Needed] on the specific top-1 accuracy results of the MCG-NJU/videomae-base-finetuned-kinetics model on Kinetics-400.

In summary, the MCG-NJU/videomae-base-finetuned-kinetics model demonstrates strong transfer learning capabilities from its Kinetics-400 pre-training, but may have limitations on complex temporal datasets like Something-Something V2. Fine-tuning with labels can further boost performance.

## Model Examination

Based on the provided references, here is what I can say about the explainability and interpretability of the VideoMAE model MCG-NJU/videomae-base-finetuned-kinetics:

The analysis of the model's results on the Something-Something V2 (SSV2) dataset shows that VideoMAE brings significant gains for most categories compared to ImageMAE and ImageNet-21k supervised pre-trained models. This implies that VideoMAE can capture more spatiotemporal structure representations (Reference 1). 

However, VideoMAE performs slightly worse on some categories. Looking at specific validation examples, it appears VideoMAE might not capture motion information well from very small objects, possibly because tokens containing small motion are all masked due to the high masking ratio (Reference 2).

On the other hand, VideoMAE can capture object deformation and movement in other examples that image pre-training cannot discriminate (Reference 5). Even under extremely high masking ratios, VideoMAE produces satisfying reconstructed results, implying it learns representative features that capture holistic spatiotemporal structure (References 3-4).

[More Information Needed] on detailed interpretability analysis, e.g. visualizations of learned features, or attribution of predictions to input regions. The current references focus more on reconstruction examples and aggregate performance metrics.

In summary, VideoMAE shows capabilities in learning spatiotemporal representations, with some limitations in capturing small motions. More detailed interpretability work could help further explain its behavior and representations. The model's strong performance gains imply it is learning useful video features during pre-training.

## Environmental Impact

- **Hardware Type:** [More Information Needed]

The provided references do not contain information about the specific hardware used to train the MCG-NJU/videomae-base-finetuned-kinetics model. More details would be needed from the model developers to determine the hardware type, such as GPU models, number of GPUs used, etc.
- **Software Type:** Based on the provided references, the model MCG-NJU/videomae-base-finetuned-kinetics is fine-tuned on the Kinetics-400 dataset, as mentioned in reference 7:

"Kinetics-400. Our VideoMAE is pre-trained for 800 epochs on Kinetics-400 by default. During the fine-tuning phase, we perform the dense sampling following Slowfast [23]."

However, the specific software or framework used for training is not explicitly mentioned in the given references. Therefore, for the software type, the answer would be:

[More Information Needed]
- **Hours used:** Based on the provided references, the VideoMAE model was pre-trained for 800 epochs on the Kinetics-400 dataset by default (Reference 6). The model was then fine-tuned for evaluation, but the specific number of epochs for fine-tuning is not provided in the given references.

Therefore, the answer to the question "Provide the amount of time used to train the model MCG-NJU/videomae-base-finetuned-kinetics" is:

The VideoMAE model was pre-trained for 800 epochs on the Kinetics-400 dataset. [More Information Needed] on the number of epochs used for fine-tuning the model on the Kinetics-400 dataset after pre-training.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the MCG-NJU/videomae-base-finetuned-kinetics model. More information would be needed from the model development team to determine the cloud provider details.
- **Carbon Emitted:** [More Information Needed]

Based on the provided references, there is no specific information about the amount of carbon emitted during the training of the MCG-NJU/videomae-base-finetuned-kinetics model. The references discuss various aspects of the model, such as the pre-training and fine-tuning settings, evaluation protocols, and performance analysis, but do not mention the carbon footprint of the training process. To accurately report the carbon emissions, more details would be needed about the computing infrastructure, energy consumption, and duration of the training runs.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
VideoMAE uses an asymmetric encoder-decoder architecture for video self-supervised pre-training. The encoder and decoder use the vanilla ViT with joint space-time attention to better capture high-level spatio-temporal information. The specific architectural design for the ViT-Base encoder and decoder is:
[More Information Needed]

During fine-tuning, the decoder is discarded and only the encoder is used.

Pre-training Objective:
VideoMAE performs masked autoencoding, where a high proportion of video tokens (e.g. 90-95%) are masked and the objective is to reconstruct the original video from the remaining tokens. This encourages learning of high-level video representations.

Kinetics-400 Pre-training and Fine-tuning:
- Pre-training is done for 800 epochs with a masking ratio of 75%. 
- For fine-tuning, dense frame sampling is used following the SlowFast approach. The model is fine-tuned for [More Information Needed] epochs.
- For evaluation, 5 clips x 3 crops are used.

In summary, the VideoMAE model uses an asymmetric ViT encoder-decoder with a high masking ratio to learn video representations via reconstruction, and is fine-tuned on the Kinetics-400 dataset for video recognition.

### Compute Infrastructure

Based on the provided references, here is the information I could find about the compute infrastructure for the MCG-NJU/videomae-base-finetuned-kinetics model:

The model was pre-trained on the Kinetics-400 dataset for 800 epochs with a masking ratio of 90%. For fine-tuning and evaluation on Kinetics-400, dense sampling was performed following the approach in the SlowFast paper [23]. The inference protocol used 5 clips × 3 crops for evaluation.

[More Information Needed] on the specific compute hardware (GPUs, machines) and software (deep learning framework, libraries) used for pre-training and fine-tuning the model. The references do not provide these details.

## Citation

```
@misc{zhan-videomae,
    author = {Zhan Tong and
              Yibing Song and
              Jue Wang and
              Limin Wang},
    title  = {VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
    url    = {https://arxiv.org/pdf/2203.12602.pdf}
}
```

