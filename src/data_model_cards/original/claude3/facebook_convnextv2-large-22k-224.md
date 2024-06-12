# Model Card for facebook/convnextv2-large-22k-224

The facebook/convnextv2-large-22k-224 model is a large-sized variant of the ConvNeXt V2 architecture, which is a pure convolutional neural network that achieves state-of-the-art performance on various visual recognition tasks. It is pre-trained on the ImageNet-22K dataset and fine-tuned on the ImageNet-1K dataset, using a resolution of 224x224 pixels.

## Model Details

### Model Description

Model Architecture:
- Encoder: ConvNeXt V2 Large model (198M parameters)
- Channel dimension: 192
- Block configuration: (3,3,27,3)
- Decoder: Single ConvNeXt block with dimension 512

Training Procedure:
- Pre-training using the Fully Convolutional Masked Autoencoder (FCMAE) framework
- Intermediate fine-tuning on ImageNet-22K dataset
- Final fine-tuning on ImageNet-1K dataset
- Uses 384x384 resolution images for pre-training and fine-tuning

Parameters: 198 million

Important Disclaimers:
[More Information Needed]

- **Developed by:** Sanghyun Woo; Shoubhik Debnath; Ronghang Hu; Xinlei Chen; Zhuang Liu; In So Kweon; Saining Xie; Meta Ai
- **Funded by:** Based on the provided references, the model facebook/convnextv2-large-22k-224 appears to be funded by Meta AI, as evidenced by the author affiliations listed in Reference 1:

Sanghyun Woo; Shoubhik Debnath; Ronghang Hu; Xinlei Chen; Zhuang Liu; In So Kweon; Saining Xie; Meta Ai

No other specific funding sources or organizations are mentioned in the given references.
- **Shared by:** Based on the references provided, the main contributors who made the model facebook/convnextv2-large-22k-224 available online are:

Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie from Meta AI.

The references also mention that the repository borrows from the work of Ross Wightman on the initial design of small-compute ConvNeXt model variants and the associated training recipe. Kaiming He provided helpful discussions and feedback as well.
- **Model type:** The facebook/convnextv2-large-22k-224 model is a self-supervised ConvNeXt model pre-trained on ImageNet-22K using a masked autoencoder approach for computer vision tasks.
- **Language(s):** The facebook/convnextv2-large-22k-224 model processes natural images and does not use or process natural human language.
- **License:** The ImageNet pre-trained and fine-tuned models of ConvNeXt V2 are licensed under a CC-BY-NC license, while the rest of the project is released under the MIT license. The link to the license file is provided in the references, but the full URL is not given.

[More Information Needed] on the exact URL of the license file.
- **Finetuned from model:** Based on the provided references, the model facebook/convnextv2-large-22k-224 is likely fine-tuned from a self-supervised pre-trained ConvNeXt V2 model, as mentioned in this excerpt:

"Our ConvNeXt V2 model, which is powered by self-supervised learning, provides a simple way to upgrade existing models and achieve a significant boost in"

However, the specific pre-trained model used as the base is not directly mentioned. The closest relevant information is in reference 3:

"| ConvNeXt V2-L | 224x224 | 87.3 | 198M  | 34.4G   | [model](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt)|"

This points to a ConvNeXt V2-L model trained on ImageNet-22K at 224x224 resolution, which could potentially be the base model. But without more definitive information, the safest response is:

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/facebookresearch/ConvNeXt-V2
- **Paper:** https://arxiv.org/pdf/2301.00808.pdf
- **Demo:** [More Information Needed]

Based on the provided references, there is no specific information about a demo link for the model facebook/convnextv2-large-22k-224. The references discuss the ConvNeXt V2 architecture, training details, and performance on various benchmarks, but do not mention a publicly available demo. More information would be needed from the model developers to determine if a demo link exists.
## Uses

### Direct Use

The facebook/convnextv2-large-22k-224 model can be used for image classification without requiring any fine-tuning, post-processing or plugging into a pipeline. It has been pre-trained on the ImageNet-22K dataset and can directly output predicted class probabilities for 22,000 categories given an input image.

To use the model for evaluation, you can run the following example commands:

Single-GPU
```
python main_finetune.py \
--model convnextv2_base \
--eval true \
--resume /path/to/checkpoint \
--input_size 224 \
--data_path /path/to/imagenet-1k \
```

Multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--model convnextv2_base \
--eval true \
--resume /path/to/checkpoint \
--input_size 224 \
--data_path /path/to/imagenet-1k \
```

The model takes as input an image of size 224x224 and outputs predicted probabilities for the 22K ImageNet classes. No additional fine-tuning or processing is needed to use the model for classification out-of-the-box.

[More Information Needed] on details about using the large variant of the model or plugging it into an application pipeline without fine-tuning. The example commands only show evaluating the base model on ImageNet-1K.

### Downstream Use

The ConvNeXt V2-L model, pre-trained on ImageNet-22K and fine-tuned on 224x224 resolution images, can be used for various downstream tasks such as object detection, semantic segmentation, and fine-tuning on other datasets. Here are a few examples:

1. Object detection and segmentation: The model can be used as a backbone for frameworks like Mask R-CNN to improve performance on datasets like COCO. [More Information Needed]

2. Semantic segmentation: ConvNeXt V2-L can be integrated into semantic segmentation frameworks like UperNet to enhance performance on datasets such as ADE20K. [More Information Needed]

3. Fine-tuning on other datasets: The model can be fine-tuned on specific datasets for various classification tasks. Here's a code snippet for evaluating the model:

Single-GPU:
```
python main_finetune.py \
--model convnextv2_large \
--eval true \
--resume /path/to/checkpoint \
--input_size 224 \
--data_path /path/to/dataset \
```

Multi-GPU:
```
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
--model convnextv2_large \
--eval true \
--resume /path/to/checkpoint \
--input_size 224 \
--data_path /path/to/dataset \
```

The ConvNeXt V2-L model, with its improved architecture and self-supervised pre-training, offers significant performance gains compared to its V1 counterpart, making it a powerful choice for various computer vision tasks when fine-tuned or integrated into larger ecosystems.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the ConvNeXt V2 large model pre-trained on ImageNet-22K could be misused and what users ought not do:

- The model is licensed under CC-BY-NC for the ImageNet pre-trained and fine-tuned versions. This means it cannot be used for commercial purposes. Users should not deploy this model in any commercial applications without obtaining the appropriate commercial license.

- As a powerful image classification model, it could potentially be misused for harmful surveillance applications that violate privacy and civil liberties if fine-tuned on sensitive data like facial images. Users should carefully consider the ethical implications and potential for abuse before using this model on any data that could enable tracking or identification of individuals without consent.

- [More Information Needed] on specific failure modes or biases the model may exhibit on certain demographics or image types that users should be aware of and avoid relying on the model for.

- [More Information Needed] on whether the model training data contained any inappropriate, explicit, or offensive content that could be reflected in model outputs if probed in certain ways. Users should not attempt to elicit such content from the model.

In general, users should use the model only for its intended purpose of generic image classification within appropriate domains, be transparent about its usage, and consider potential negative societal impacts before deploying it in any system that could cause real-world harms if the model fails or exhibits biases. More specific information on limitations and ethical considerations would help provide clearer guidance to users.

### Bias, Risks, and Limitations

Based on the provided references about the ConvNeXt V2 model, here are some potential issues and limitations:

Foreseeable harms and misunderstandings:
- The model was trained on the ImageNet-22K dataset, which may contain biases present in the training data. This could lead to the model exhibiting biased or unfair behavior when applied to real-world tasks. [More Information Needed] on specific biases and harms.

- As a large-scale image classification model, it could potentially be misused for unintended purposes like surveillance or generating deepfakes if released without proper safeguards. [More Information Needed] on specific misuse cases.

Technical limitations: 
- The sparse convolution libraries used for pre-training are not highly optimized for efficiency yet (Reference 6). This impacts computational and memory efficiency.

- [More Information Needed] on model performance and failure modes on specific downstream tasks and datasets beyond ImageNet, COCO, and ADE20K which were evaluated.

Sociotechnical limitations:
- Releasing such a powerful model publicly requires carefully considering the potential societal impacts and implementing appropriate responsible AI practices. [More Information Needed] on the specific practices followed.

- Transparent communication is needed to users about the model's capabilities, limitations, and intended use cases to prevent misunderstandings and misuse. [More Information Needed] on the specific communication and documentation provided.

In summary, while ConvNeXt V2 represents an advancement in ConvNet-based models for computer vision, more information is needed to fully characterize the potential issues and limitations from both a technical and sociotechnical perspective before a public release. A thorough model card covering these points would be beneficial.

### Recommendations

Based on the provided references, here are some recommendations regarding foreseeable issues with the facebook/convnextv2-large-22k-224 model:

1. Feature collapse at the MLP layer: The references mention identifying a potential issue of feature collapse at the MLP layer when training ConvNeXt directly on masked input. This was addressed by adding a Global Response Normalization (GRN) layer to enhance inter-channel feature competition. It's important to ensure that the model architecture incorporates this change to mitigate the feature collapse issue.

2. Transferability of learned features: The class selectivity index analysis suggests that ConvNeXt V2 tends to include more class-generic features compared to V1. While class-agnostic features are more transferable, it's crucial to evaluate the model's performance on a diverse range of downstream tasks to ensure the learned features are indeed transferable and effective.

3. Comparison with contrastive self-supervised learning: The references mention comparing the performance of the proposed FC-MAE framework with contrastive learning methods like MoCoV3. [More Information Needed] on the specific results and implications of this comparison for the facebook/convnextv2-large-22k-224 model.

4. Efficiency and scalability: ConvNeXt V2 models are designed to cover a broader range of complexity and efficiency. While the model is said to excel in scenarios requiring lower complexity, [More Information Needed] on the specific efficiency metrics and scalability characteristics of the facebook/convnextv2-large-22k-224 variant.

5. Societal impact and fairness: [More Information Needed] on any analyses or considerations regarding the potential societal impact, biases, or fairness aspects of the facebook/convnextv2-large-22k-224 model. It's important to assess and mitigate any unintended consequences or disparate impacts the model might have on different demographics or user groups.

To summarize, while the provided references offer insights into some architectural improvements and performance aspects of ConvNeXt V2 models, more specific information is needed to comprehensively address the foreseeable issues and provide detailed recommendations for the facebook/convnextv2-large-22k-224 model. Further evaluation, benchmarking, and analysis of the model's efficiency, transferability, fairness, and societal impact would be beneficial to inform its responsible deployment and use.

## Training Details

### Training Data

The ConvNeXt V2-L model was pre-trained on ImageNet-22K dataset, and fine-tuned on ImageNet-1K at 224x224 resolution. [More Information Needed] for details on data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the facebook/convnextv2-large-22k-224 model:

Resizing:
The model uses 384x384 resolution images for pretraining and fine-tuning, as mentioned in reference 7: "We use 384 2 resolution images for pretraining and fine-tuning [38]."

Tokenization:
[More Information Needed] The references do not provide specific details about the tokenization process for this model.

Other preprocessing details:
[More Information Needed] The references do not provide additional information about other preprocessing steps, such as normalization or data augmentation techniques, for the facebook/convnextv2-large-22k-224 model.

#### Training Hyperparameters

Based on the provided references, the ConvNeXt V2 Large model (facebook/convnextv2-large-22k-224) was trained using the following steps and hyperparameters:

1. Pre-training using the FCMAE (Fully Convolutional Masked Auto-Encoder) framework on the ImageNet-22K dataset. [More Information Needed] on the specific hyperparameters used during pre-training.

2. Intermediate fine-tuning on the ImageNet-22K dataset. The setups are summarized in Table 12 and 13 from the references. Using larger layer-wise learning rate decay values for small models is helpful. [More Information Needed] on the exact values used for the Large model.

3. Final fine-tuning on the ImageNet-1K dataset using 384x384 resolution images, as mentioned in reference 2.

[More Information Needed] on the specific hyperparameters used during the ImageNet-1K fine-tuning stage, such as learning rate, batch size, optimizer, and number of epochs.

The model achieves state-of-the-art performance on various recognition benchmarks, including ImageNet classification, COCO detection, and ADE20K segmentation, as stated in reference 8.

#### Speeds, Sizes, Times

Based on the provided references, here are the details I could find about the model facebook/convnextv2-large-22k-224:

Model size: 198M parameters (from reference 8)
Throughput: 34.4G (likely referring to FLOPs, from reference 8)
ImageNet-1K top-1 accuracy: 85.8% (from reference 8)

[More Information Needed] for the following:
- Start or end training time 
- Checkpoint sizes

The references do not contain specific information about the training time or checkpoint sizes for this particular model.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the ConvNeXt V2 Large model, which is likely the facebook/convnextv2-large-22k-224 model, evaluates on the following benchmarks and datasets:

1. ImageNet-1K classification (References 4, 5, 7, 8)
- Achieves state-of-the-art 88.9% top-1 accuracy using only publicly available data (Reference 4)

2. COCO object detection and segmentation (Reference 3)
- Fine-tunes Mask R-CNN on COCO dataset
- Reports detection mAP (box) and segmentation mAP (mask) on COCO val2017 set

3. ADE20K segmentation (Reference 7)

The model is pre-trained using the FCMAE framework and fine-tuned on the ImageNet-22K dataset (References 4, 6, 8).

#### Factors

Based on the provided references, here are some foreseeable characteristics that may influence the behavior of the facebook/convnextv2-large-22k-224 model:

Domain and Context:
- The model was pretrained on ImageNet-22K dataset and fine-tuned on ImageNet-1K (Ref 1, 4, 8). This suggests it is primarily designed for image classification tasks in the domain of natural images similar to those found in ImageNet.

Population Subgroups:
[More Information Needed] - The references do not provide specific information about performance on different population subgroups or potential disparities.

Other Factors:
- The model uses a convolution-based architecture (ConvNeXt V2) co-designed with the FCMAE pre-training framework (Ref 3, 5, 7, 9). This unique architecture and training approach may influence its performance characteristics compared to other models.
- The model exhibits strong scaling behavior, with improved performance as model size increases (Ref 2, 7). The "large" variant is likely to have different performance than smaller or larger versions.
- The model's feature representations evolve distinctly across layers compared to supervised ConvNeXt and MAE pretrained ViT models (Ref 10). This may affect its suitability for different downstream tasks or fine-tuning approaches.

To fully understand potential performance disparities across population subgroups, further disaggregated evaluation would be needed. The current references focus primarily on the model's architecture, training approach, and benchmark results on ImageNet.

#### Metrics

Based on the provided references, the main evaluation metric used for the ConvNeXt V2 Large model pre-trained on ImageNet-22K is top-1 accuracy on the ImageNet-1K dataset. Specifically:

- The ConvNeXt V2 architecture, when pre-trained using the FCMAE framework and fine-tuned on ImageNet-22K, achieves a state-of-the-art 88.9% top-1 accuracy on ImageNet-1K, using only publicly available data (References 4, 6).

- The references consistently highlight top-1 accuracy improvements on ImageNet-1K across various model sizes, from the efficient 3.7M-parameter Atto model to the 650M Huge model (References 6, 7).

However, the references do not provide detailed information about the tradeoffs between different types of errors or alternative evaluation metrics used for this specific model. [More Information Needed] regarding other potential metrics or error tradeoffs considered during the model's development and evaluation process.

### Results

Based on the provided references, here is the evaluation information for the model facebook/convnextv2-large-22k-224:

Model Architecture:
- The model uses the ConvNeXt V2 architecture, which incorporates the GRN (Global Response Normalization) layer into the original ConvNeXt block.
- LayerScale is removed when GRN is applied.
- The model is part of the ConvNeXt V2 model family, which includes models of varying efficiency and capacity, from lightweight (e.g., Atto) to compute-intensive (e.g., Huge).

Pre-training:
- The model was pre-trained using the FC-MAE (Fully Convolutional Masked Autoencoder) framework.
- Pre-training was done on ImageNet-22K dataset.

Fine-tuning:
- The model was fine-tuned on ImageNet-1K dataset.
- Fine-tuning was performed using 384x384 resolution images.

Performance:
- The ConvNeXt V2 models pre-trained on FC-MAE outperform the Swin transformer counterparts across all model sizes.
- The model achieves state-of-the-art accuracy using only publicly available data (ImageNet-1K and ImageNet-22K).

[More Information Needed] for the specific performance metrics (e.g., top-1 accuracy) of the facebook/convnextv2-large-22k-224 model on ImageNet-1K dataset.

The model was also evaluated on various recognition benchmarks, including:
- ImageNet classification
- COCO detection
- ADE20K segmentation

[More Information Needed] for the specific performance metrics on these benchmarks for the facebook/convnextv2-large-22k-224 model.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the facebook/convnextv2-large-22k-224 model:

The ConvNeXt V2 model, pre-trained using the FCMAE framework and fine-tuned on the ImageNet-22K dataset, achieves a new state-of-the-art accuracy of 88.9% top-1 on the ImageNet-1K dataset, using only publicly available data. This convolution-based architecture outperforms previous state-of-the-art designs, including transformer-based and hybrid models.

The model demonstrates strong scaling behavior, with consistently improved performance over supervised baselines across all model sizes. ConvNeXt V2 pre-trained on FCMAE outperforms Swin transformer counterparts pre-trained with SimMIM across all model sizes, with the largest gap achieved in the huge model regime.

[More Information Needed] on specific performance metrics for the Large model size variant.

In transfer learning benchmarks, ConvNeXt V2 shows significant improvements over ConvNeXt V1 across various recognition tasks, including ImageNet classification, COCO detection, and ADE20K segmentation.

## Model Examination

Based on the provided references, here is an attempt at an experimental section on explainability/interpretability for the facebook/convnextv2-large-22k-224 model:

We performed qualitative analysis to gain insight into the learning behavior of the ConvNeXt architecture used in the facebook/convnextv2-large-22k-224 model. By visualizing the activations of a FCMAE pre-trained ConvNeXt-Base model, we observed an intriguing "feature collapse" phenomenon: many dead or saturated feature maps and redundant activations across channels, particularly in the dimension-expansion MLP layers of the ConvNeXt blocks (Reference 2, Figure 3).

To quantitatively validate this observation, we conducted a feature cosine distance analysis on the activation tensors. [More Information Needed] 

The results, plotted against normalized layer indexes, showed that the ConvNeXt V1 FCMAE pre-trained model exhibited severe feature collapse behavior compared to the supervised model (Reference 5, Figure 4).

These findings shed light on the learning dynamics of the ConvNeXt architecture and highlight opportunities for further research to mitigate feature collapse and improve the interpretability of the facebook/convnextv2-large-22k-224 model. [More Information Needed]

## Environmental Impact

- **Hardware Type:** The model facebook/convnextv2-large-22k-224 was trained on TPU (v3-256) pods, as mentioned in reference 7:

"The experiments in the main paper are all conducted on TPU (v3-256) pods and we release a PyTorch reproduction."
- **Software Type:** The model facebook/convnextv2-large-22k-224 is trained using PyTorch, as evidenced by the following information from the references:

1. The evaluation commands provided use Python and PyTorch libraries (e.g., `python main_finetune.py`, `python -m torch.distributed.launch`).

2. The repository borrows from timm, which is a PyTorch library for image models.

3. The references explicitly state: "This repo contains the PyTorch version of *8* model definitions (*Atto, Femto, Pico, Nano, Tiny, Base, Large, Huge*), pre-training/fine-tuning code and pre-trained weights (converted from JAX weights trained on TPU) for our ConvNeXt V2 paper."

Therefore, the software type used for training the model facebook/convnextv2-large-22k-224 is PyTorch.
- **Hours used:** [More Information Needed]

The provided references do not contain specific information about the training time for the facebook/convnextv2-large-22k-224 model. More details would be needed from the model developers or training logs to determine the exact amount of time used for training this particular model.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the facebook/convnextv2-large-22k-224 model. More information would be needed from the model developers or training logs to determine the cloud provider utilized.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the facebook/convnextv2-large-22k-224 model. To accurately report the carbon emissions, details such as the hardware used for training, training duration, energy consumption, and the carbon intensity of the energy source would be required.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The ConvNeXt V2-L (Large) model follows the ConvNeXt V1 architecture with C=192 channels and B=(3,3,27,3) blocks in each stage. It incorporates the Gated Residual Network (GRN) layer into the ConvNeXt block, replacing the LayerScale component.

Objective:
The ConvNeXt V2 models are pre-trained using a fully convolutional masked autoencoder (FCMAE) approach. The learning objective is to predict the missing parts of randomly masked input visuals given the remaining context. The models are then fine-tuned on the ImageNet-22K dataset for improved performance on downstream tasks such as classification, object detection, and segmentation.

[More Information Needed] on the specific pre-training and fine-tuning hyperparameters used for the ConvNeXt V2-L model.

### Compute Infrastructure

Based on the provided references, the compute infrastructure used for the ConvNeXt V2 large model (facebook/convnextv2-large-22k-224) is as follows:

- The experiments in the main paper were all conducted on TPU (v3-256) pods. (Reference 2)
- A PyTorch reproduction was also released. (Reference 2)
- Sparse encoding was implemented using:
  1) Sparse convolution supported by external libraries like MinkowskiEngine on the GPU environment with PyTorch framework. (Reference 2, 3) 
  2) Simulating sparse encoding with masked dense convolution. (Reference 3)

[More Information Needed] on the exact compute infrastructure used specifically for the ConvNeXt V2 large model, as the references mostly discuss the general compute setup for the ConvNeXt V2 model family.

## Citation

```
@misc{sanghyun-convnext,
    author = {Sanghyun Woo and
              Shoubhik Debnath and
              Ronghang Hu and
              Xinlei Chen and
              Zhuang Liu and
              In So Kweon and
              Saining Xie and
              Meta Ai},
    title  = {ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
    url    = {https://arxiv.org/pdf/2301.00808.pdf}
}
```

