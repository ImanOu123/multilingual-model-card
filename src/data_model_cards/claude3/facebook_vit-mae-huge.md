# Model Card for facebook/vit-mae-huge

The facebook/vit-mae-huge model is a large Vision Transformer (ViT) model that uses Masked Autoencoder (MAE) pretraining on ImageNet-1K. It achieves state-of-the-art 87.8% top-1 accuracy on ImageNet-1K fine-tuning and demonstrates strong performance on downstream tasks like object detection, instance segmentation, and semantic segmentation.

## Model Details

### Model Description

Model Architecture:
- Follows the standard Vision Transformer (ViT) architecture with a stack of Transformer blocks, each consisting of a multi-head self-attention block and an MLP block, both having LayerNorm (LN) [1, 16, 57].
- Uses an asymmetric encoder-decoder design. The encoder operates only on the visible subset of patches (without mask tokens), while the lightweight decoder reconstructs the original image from the latent representation and mask tokens [2, 7, 11].
- Adds positional embeddings (sine-cosine version) to both encoder and decoder inputs [1].
- Uses a linear projection layer after the encoder to match the different widths of the MAE encoder and decoder [1].
- Appends an auxiliary dummy token to the encoder input during pre-training, which is treated as the class token for training the classifier in linear probing and fine-tuning [5].

Training Procedures:
- Performs self-supervised pre-training on the ImageNet-1K (IN1K) training set [8, 13].
- Masks random patches of the input image (e.g., 75%) and reconstructs the missing pixels [2, 9, 11].
- Extracts features from the encoder output for fine-tuning and linear probing [5, 8].
- Evaluates the representations with (i) end-to-end fine-tuning or (ii) linear probing, reporting top-1 validation accuracy of a single 224×224 crop [8].

Parameters:
- Uses ViT-Huge as the backbone model [6, 10].
- [More Information Needed] on the exact number of parameters.

Important Disclaimers:
- The MAE decoder is only used during pre-training for the image reconstruction task; only the encoder is used to produce image representations for recognition [7].
- [More Information Needed] on potential biases or limitations of the model.

- **Developed by:** Kaiming He; Xinlei Chen; Saining Xie; Yanghao Li; Piotr Dollár; Ross Girshick
- **Funded by:** [More Information Needed]

The provided references do not contain any information about the funding sources for the facebook/vit-mae-huge model project. More details would be needed from the model developers or Facebook AI Research (FAIR) to determine what organizations or individuals financially supported this specific work.
- **Shared by:** Based on the references provided, the key contributors that made the model facebook/vit-mae-huge available online are:

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick

They authored the paper "Masked Autoencoders Are Scalable Vision Learners" which introduced the MAE approach that the model is based on.

However, more specific information is needed on:
- Who exactly implemented and open-sourced the PyTorch version of the model 
- Which individuals or teams at Facebook AI were involved in training and releasing the vit-mae-huge weights

The references focus more on the research contributions and lack details on the engineering efforts to make the model available. [More Information Needed] for those aspects to fully answer the question of all contributors involved in open-sourcing this specific model.
- **Model type:** The facebook/vit-mae-huge model is a self-supervised learning model for computer vision that uses masked autoencoding (MAE) to pre-train a large Vision Transformer (ViT) on ImageNet-1K data by masking random patches of the input image and reconstructing the missing pixels.
- **Language(s):** The model facebook/vit-mae-huge processes images, not natural human language. [More Information Needed]
- **License:** [More Information Needed]

The provided references do not contain any information about the specific license being used for the facebook/vit-mae-huge model. More details would be needed from the model creators or documentation to determine the license under which this model is being released.
- **Finetuned from model:** Based on the provided references, there is no explicit mention of the facebook/vit-mae-huge model being fine-tuned from another pre-existing model. The references discuss the general architecture and training process of the Masked Autoencoder (MAE) model, but do not specify a particular base model that was fine-tuned.

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/facebookresearch/mae
- **Paper:** https://arxiv.org/pdf/2111.06377.pdf
- **Demo:** Here is the link to the interactive visualization demo of the model facebook/vit-mae-huge using a Colab notebook (no GPU needed):

https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb
## Uses

### Direct Use

The facebook/vit-mae-huge model is a pre-trained Vision Transformer (ViT) model that has been trained using the Masked Autoencoder (MAE) approach. It can be used for extracting features from images without requiring fine-tuning or post-processing.

To use the model for feature extraction:

1. Extract features from the encoder output of the pre-trained model. The ViT architecture has a class token, so during MAE pre-training, an auxiliary dummy token is appended to the encoder input to adapt to this design. This token will be treated as the class token for training the classifier in linear probing and fine-tuning.

[More Information Needed] (No specific code snippet found in the provided references)

The extracted features can then be used for various downstream tasks such as classification, object detection, instance segmentation, and semantic segmentation. The model has shown strong performance and scalability, achieving state-of-the-art results on several benchmarks when fine-tuned.

Note: More specific information on how to load and use the pre-trained model directly for feature extraction without fine-tuning or post-processing would be needed to provide a complete answer and code snippet.

### Downstream Use

The facebook/vit-mae-huge model, pre-trained using the Masked Autoencoder (MAE) approach, can be fine-tuned for various downstream tasks to achieve state-of-the-art performance. Some potential applications include:

1. Image Classification: The model can be fine-tuned on datasets like ImageNet-1K, iNaturalist, and Places, achieving top accuracy compared to previous best results (Table 6 in the references).

2. Object Detection and Instance Segmentation: By adapting the ViT backbone for use with Feature Pyramid Networks (FPN) and fine-tuning Mask R-CNN end-to-end on the COCO dataset, the model can be used for object detection and instance segmentation tasks (Table 4 and point 5 in the references).

3. Semantic Segmentation: The model can be used with architectures like UperNet and fine-tuned on datasets such as ADE20K, significantly improving results over supervised pre-training (Table 5 and point 1 in the references).

To fine-tune the model for a specific task, you would typically add a task-specific head on top of the pre-trained model and train the entire model on the target dataset. The high capacity and generalization ability of the MAE pre-trained model make it well-suited for various computer vision tasks.

[More Information Needed] (No specific code snippet provided in the references)

For more details on fine-tuning and using the model in a larger ecosystem or app, please refer to the official repository or documentation.

### Out-of-Scope Use

Based on the provided references about the facebook/vit-mae-huge model, here are some potential ways the model may be misused and what users ought not do:

The model learns from and reflects biases present in the training data, which may include biases with negative societal impacts. Users should not assume the model is free of harmful biases or use it in contexts where those biases could cause real-world damage.

The model can generate plausible but inexistent/fake content. Users should not present model outputs as real images or use them to intentionally deceive. Further research is needed before using this model to generate images in high-stakes domains.

The model may infer semantically reasonable but incorrect content when reconstructing images with missing patches. Users should not treat model reconstructions as reliable or as a substitute for real complete images.

[More Information Needed] on other specific misuse cases, but in general users should be cautious about relying on a model that may reflect problematic biases and generate fake content. The model should be thoughtfully evaluated before deployment, especially for important decisions or in contexts that could disadvantage certain groups if the model exhibits bias.

### Bias, Risks, and Limitations

Based on the provided references about the facebook/vit-mae-huge model, here are some known or foreseeable issues:

1. Bias and negative societal impacts: The model predicts content based on learned statistics from the training data, and as such will reflect biases present in those data, including ones with potentially negative societal impacts. (Reference 1)

2. Generation of non-existent content: The model may generate content that does not actually exist. (Reference 1) 

3. Differences between images and language: Images and language are signals of a different nature, and this difference must be carefully considered. The model removes random patches from images that likely don't form semantic segments, and reconstructs pixels rather than semantic entities. More research is needed on addressing these differences. (Reference 4)

4. [More Information Needed] on the specific training data used and potential biases or underrepresentation issues in that data.

5. [More Information Needed] on how the model's outputs could potentially be misused or misinterpreted by users.

6. Technical limitation: The model is very large and data-hungry, requiring a lot of compute resources to train and use. This could limit accessibility. (References 9, 11)

In summary, key issues to consider are training data bias, generation of non-existent content, careful handling of differences between images and language, potential for misuse of outputs, and technical limitations due to model size. More specific information would be needed to fully assess societal implications and limitations.

### Recommendations

Based on the provided references about the facebook/vit-mae-huge model, here are some recommendations regarding foreseeable issues:

Biases and negative societal impacts:
The model predicts content based on the training data statistics and as such will reflect biases present in that data, including ones with potentially negative societal impacts. More research and consideration is warranted when building upon this work to generate images, to mitigate harmful biases. [More Information Needed] on specific biases present and mitigation strategies.

Inexistent content generation:
The model may generate content that does not actually exist. [More Information Needed] on how frequently this occurs and potential negative implications. Monitoring and detection methods for generated vs real content may be needed.

Generalization and robustness:
While the model shows good robustness on ImageNet variants, [More Information Needed] on performance on more diverse datasets and real-world scenarios. Continued testing of model generalization and robustness as it is applied to new tasks is recommended.

Scaling and compute requirements: 
The model benefits from scaling to high capacity, which comes with significant computational costs for training and deployment. [More Information Needed] on the specific compute resources required and implications for energy usage, environmental impact, and accessibility of the model. Responsible scaling with these considerations in mind is advised.

In summary, key issues to consider are societal impact of biases, generated content detection, generalization to real-world applications, and responsible compute scaling. Collaboration between the technical team and experts on societal implications can help navigate these challenges as the model is developed and deployed.

## Training Details

### Training Data

The model facebook/vit-mae-huge is pre-trained on the ImageNet-1K (IN1K) training set using self-supervised learning. [More Information Needed] for details on data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the provided references about the MAE (Masked Autoencoder) model, here are the details about preprocessing for the facebook/vit-mae-huge model:

Tokenization and Patch Embedding:
- The input image is split into patches, and each patch is linearly projected into an embedding vector.
- Positional embeddings are added to the patch embeddings to retain positional information.
- The MAE encoder only operates on a small subset (e.g., 25%) of the full set of patches. Masked patches are removed, and no mask tokens are used in the encoder.
- The full set of tokens, consisting of encoded visible patches and mask tokens, is handled by the lightweight decoder.

Resizing/Rewriting:
[More Information Needed]

Additional Preprocessing Details:
- An auxiliary dummy token is appended to the encoder input to serve as the class token for training the classifier in linear probing and fine-tuning.
- Data augmentation is mainly performed by random masking, which generates new training samples for each iteration.
- Different mask sampling strategies are compared, as illustrated in Figure 6 (not provided in the references).

Specific details about resizing or rewriting the input data are not provided in the given references. More information would be needed to elaborate on those aspects of preprocessing for the facebook/vit-mae-huge model.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the model facebook/vit-mae-huge:

Pre-training:
- Initialization: Xavier uniform for all Transformer blocks
- Learning rate: linear scaling rule, lr = base lr × batchsize / 256
- [More Information Needed] for other pre-training hyperparameters like optimizer, weight decay, etc.

Fine-tuning:
- Follows common practice of supervised ViT training
- Uses layer-wise learning rate decay
- Fine-tuning for 50 epochs (compared to 200 epochs when training from scratch)
- [More Information Needed] for specific fine-tuning hyperparameters like learning rate, batch size, optimizer, etc.

Architecture:
- Standard ViT architecture with a stack of Transformer blocks
- Each block has a multi-head self-attention block and an MLP block, both with LayerNorm
- Encoder ends with LayerNorm
- Linear projection layer after encoder to match MAE encoder and decoder widths
- Positional embeddings added to both encoder and decoder inputs
- Auxiliary dummy token appended to encoder input to adapt to ViT's class token design

[More Information Needed] for additional training details like masking ratio, input image size, data augmentation, regularization techniques, etc.

#### Speeds, Sizes, Times

Based on the provided references, here are the details I can provide about the model facebook/vit-mae-huge:

Architecture:
- The model uses the standard ViT (Vision Transformer) architecture, with a stack of Transformer blocks. Each block consists of a multi-head self-attention block and an MLP block, both having LayerNorm (LN).
- The model is pre-trained using the MAE (Masked Autoencoder) approach, where a high proportion (e.g., 75%) of the input image is masked.
- The MAE encoder and decoder have different widths, so a linear projection layer is used after the encoder to match the dimensions.
- Positional embeddings (sine-cosine version) are added to both the encoder and decoder inputs.

Training:
- The model is pre-trained on the ImageNet-1K dataset.
- Xavier uniform initialization is used for all Transformer blocks.
- The linear learning rate scaling rule is used: lr = base_lr * batchsize / 256.
- [More Information Needed] on specific throughput, start/end time, and checkpoint sizes.

Fine-tuning:
- The model achieves 87.8% accuracy when fine-tuned on ImageNet-1K, outperforming previous results that use only ImageNet-1K data.
- Layer-wise learning rate decay is used during fine-tuning.
- [More Information Needed] on specific fine-tuning hyperparameters and training time.

The MAE pre-training approach allows for efficient and effective training of large models like ViT-Huge, improving generalization performance and enabling strong transfer learning results on tasks like object detection, instance segmentation, and semantic segmentation.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model facebook/vit-mae-huge evaluates on the following benchmarks and datasets:

1. Object detection and instance segmentation on COCO dataset, using Mask R-CNN fine-tuned end-to-end with the ViT backbone adapted for FPN. Reports box AP for object detection and mask AP for instance segmentation.

2. Semantic segmentation on ADE20K dataset using UperNet. 

3. Transfer learning and fine-tuning on classification datasets:
   - iNaturalists (iNat) dataset
   - Places dataset
   - ImageNet-1K (IN1K) dataset, achieving 87.8% accuracy with a ViT-Huge model fine-tuned at 448 size, outperforming previous results using only IN1K data.

[More Information Needed] on the specific metrics reported for iNat, Places and IN1K classification tasks.

#### Factors

Based on the provided references about the facebook/vit-mae-huge model, here are some key characteristics that may influence the model's behavior:

Domain and Context:
- The model was pre-trained on the ImageNet-1K dataset, so it is primarily suited for general image classification tasks. Transfer learning performance was evaluated on object detection, instance segmentation, and semantic segmentation.
- [More Information Needed] on specific domains or contexts beyond ImageNet-like images.

Population Subgroups:
- The references mention that the model will reflect biases present in the training data, including ones with negative societal impacts. However, no specific details are provided on performance across different population subgroups.
- [More Information Needed] to properly evaluate performance disparities across different demographics or subgroups. Ideally results should be disaggregated to uncover any gaps.

Other Factors:
- The model may generate non-existent or semantically plausible but incorrect content in its reconstructions. This warrants further research on responsible generation.
- Increasing model size and input image size generally leads to better performance, but can also increase compute requirements and potential for overfitting.
- The references focus on English-language benchmarks. [More Information Needed] on multilingual settings.

In summary, while the provided references give useful insights into the model's general capabilities and some potential issues, more disaggregated evaluation across relevant factors would be needed to comprehensively assess its behavior for different subgroups and contexts. The model card should note these current information gaps.

#### Metrics

Based on the provided references about the facebook/vit-mae-huge model, the main evaluation metrics used are:

1. Top-1 validation accuracy on ImageNet-1K (IN1K) for end-to-end fine-tuning and linear probing of the pre-trained model representations. (Reference 4)

2. Transfer learning performance on downstream tasks like object detection, instance segmentation, and semantic segmentation. (Reference 9)

The references do not provide specific details on how different error types or tradeoffs are handled in the evaluation metrics. [More Information Needed] on any special considerations for balancing different types of errors in the model's predictions.

The key focus seems to be on achieving high accuracy on IN1K classification and strong transfer learning performance, by pre-training very large Vision Transformer models like ViT-Huge using the Masked Autoencoder (MAE) approach.

### Results

Based on the provided references about the facebook/vit-mae-huge model, here are the evaluation results I could find:

Image Classification Accuracy:
- Achieves 87.8% top-1 accuracy on ImageNet-1K when fine-tuned at 448x448 resolution, using only ImageNet-1K data for pre-training (Reference 9)
- Outperforms supervised pre-training, with the accuracy gap becoming larger for higher capacity models (Reference 10)

Transfer Learning Performance: 
- Achieves state-of-the-art transfer learning accuracy on iNaturalist and Places datasets (Reference 3)
- Significantly improves semantic segmentation results on ADE20K over supervised pre-training baselines, e.g. by 3.7 points for ViT-L (Reference 2)

[More Information Needed] on the following factors and metrics:
- Efficiency analysis (FLOPs, parameters, speed, memory usage, etc.)
- Robustness evaluation
- Fairness analysis
- Carbon footprint estimate

The references focus mainly on the model's accuracy and transfer learning capabilities compared to baselines. More information would be needed to comprehensively evaluate the model on other important factors and metrics.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model facebook/vit-mae-huge:

- Using the ViT-Huge (ViT-H) model pre-trained with MAE and fine-tuned on ImageNet-1K (IN1K) at 224 size, an accuracy of 86.9% was obtained. 

- By fine-tuning ViT-H at 448 size, an accuracy of 87.8% was achieved using only IN1K data. This outperforms all previous results that use only IN1K data.

- MAE pre-training helps scale up model sizes and enables training very large models like ViT-Huge on IN1K with improved generalization performance compared to training from scratch.

- MAE pre-training also shows benefits on transfer learning tasks like object detection, instance segmentation, and semantic segmentation. 

[More Information Needed] on the specific transfer learning results.

## Model Examination

Based on the provided references about the facebook/vit-mae-huge model, here is the experimental section on explainability/interpretability:

The facebook/vit-mae-huge model demonstrates strong scalability, with accuracy improving considerably as the model size increases. This suggests the Masked Autoencoder (MAE) pre-training approach enables learning high-capacity models that generalize well. 

A key aspect of the MAE method is masking a high proportion (e.g. 75%) of the input image during pre-training. This creates a meaningful self-supervisory task that allows efficiently training large Vision Transformer (ViT) models like ViT-Huge. The model achieves state-of-the-art 87.8% top-1 accuracy on ImageNet-1K using only ImageNet-1K data.

[More Information Needed] on explainability techniques used, such as visualizing attention maps or generating counterfactual examples, to interpret the model's predictions.

[More Information Needed] on analysis of the learned representations and semantic features captured by different layers of the model.

In summary, the MAE pre-training allows successfully scaling ViT to very large model sizes while maintaining good generalization. The strong results on both ImageNet-1K and diverse transfer learning tasks suggest the representations learned are meaningful and useful. However, more work is needed to open up the "black box" and interpret how the model arrives at its outputs.

## Environmental Impact

- **Hardware Type:** [More Information Needed]

The provided references do not specify the hardware type that the model facebook/vit-mae-huge was trained on. The references mention using PyTorch and GPU for the re-implementation, but do not clarify the hardware used for training the specific facebook/vit-mae-huge model.
- **Software Type:** Based on the references provided, the facebook/vit-mae-huge model is trained using PyTorch on GPUs. Specifically, the references state:

"This is a PyTorch/GPU re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):"

"The original implementation was in TensorFlow+TPU. This re-implementation is in PyTorch+GPU."
- **Hours used:** According to the references provided, the MAE models in Table 3 (which likely includes the facebook/vit-mae-huge model) were pre-trained for 1600 epochs. The references mention that training ViT-L on 128 TPU-v3 cores took 31 hours for 1600 epochs.

However, the exact training time for the specific model facebook/vit-mae-huge is not explicitly stated. To provide a precise answer, [More Information Needed] on the hardware used and the exact duration of training for this particular model.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the facebook/vit-mae-huge model. More information would be needed from the model developers or training logs to determine the cloud provider.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emissions generated during the training of the facebook/vit-mae-huge model. To answer this question, more specific information would be needed, such as the compute resources used, training time, energy consumption, and carbon intensity of the electricity used.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The model facebook/vit-mae-huge follows the standard Vision Transformer (ViT) architecture. It has an asymmetric encoder-decoder design:
- The encoder operates only on the visible, unmasked patches (without mask tokens). It embeds patches by a linear projection with added positional embeddings, and processes them via a series of Transformer blocks.
- The decoder is lightweight and reconstructs the original image from the latent representation and mask tokens.
- A linear projection layer is added after the encoder to match the different width of the encoder and decoder.

Model Objective:
The model is pre-trained using a masked autoencoder (MAE) approach. The key aspects are:
- Random patches of the input image are masked, and the objective is to reconstruct the missing pixels. 
- A high masking ratio (e.g. 75%) is used to create a nontrivial and meaningful self-supervisory task.
- The model is trained to map the observed signal to a latent representation (via the encoder) and then reconstruct the original signal from the latent representation (via the decoder).

The asymmetric encoder-decoder architecture and high masking ratio enable efficient and effective training of large, high-capacity models that generalize well.

[More Information Needed] on specific training details like datasets used, training duration, hyperparameters, etc.

### Compute Infrastructure

Based on the provided references, here is the available information about the compute infrastructure for the model facebook/vit-mae-huge:

[More Information Needed]

The references do not provide specific details about the compute infrastructure used for training the ViT-Huge model with the MAE (Masked Autoencoder) approach. More information would be needed from the model developers or associated research papers to determine the exact compute setup, such as GPU types, number of GPUs, distributed training framework, etc.

The references do mention that MAE allows for efficiently training large models like ViT-Huge, with 3× or more acceleration in training speed compared to standard supervised training. However, the specific compute infrastructure enabling this speedup is not detailed.

## Citation

```
@misc{kaiming-masked,
    author = {Kaiming He and
              Xinlei Chen and
              Saining Xie and
              Yanghao Li and
              Piotr Dollár and
              Ross Girshick},
    title  = {Masked Autoencoders Are Scalable Vision Learners},
    url    = {https://arxiv.org/pdf/2111.06377.pdf}
}
```
