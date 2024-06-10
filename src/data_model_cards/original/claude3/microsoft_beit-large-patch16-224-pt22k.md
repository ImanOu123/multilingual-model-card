# Model Card for microsoft/beit-large-patch16-224-pt22k

The model microsoft/beit-large-patch16-224-pt22k is a self-supervised vision representation model called BEIT (Bidirectional Encoder representation from Image Transformers), which is pretrained on ImageNet-22k using a masked image modeling task and achieves state-of-the-art performance on downstream tasks such as image classification and semantic segmentation.

## Model Details

### Model Description

Model Architecture:
- Uses the standard vision Transformer (ViT) architecture as the backbone
- Includes improvements like LayerScale and relative position bias to enhance performance
- Input is a sequence of image patches linearly projected to obtain patch embeddings
- A special token [S] is prepended to the input sequence 
- Learnable 1D position embeddings are added to the patch embeddings
- The input is fed into an L-layer Transformer encoder
- The final hidden vectors from the Transformer are used as the encoded image patch representations

Training Procedure: 
- Pretrained using the masked image modeling (MIM) task in a self-supervised manner
- Approximately 40% of image patches are randomly masked during pretraining
- The objective is to recover the original visual tokens of the masked patches based on the corrupted image
- Proper parameter initialization is important for stabilizing training, especially with large-scale models
- After pretraining, the model is fine-tuned on downstream tasks by appending task-specific layers

Parameters:
- This is a large-size model
[More Information Needed]

Important Disclaimers:
[More Information Needed]

Let me know if you need any other details about the BEIT model!

- **Developed by:** Hangbo Bao; Li Dong; Songhao Piao; Furu Wei
- **Funded by:** Based on the provided references, there is no explicit mention of the people or organizations that fund the BEiT (microsoft/beit-large-patch16-224-pt22k) project. The references only list the authors (Hangbo Bao, Li Dong, Songhao Piao, Furu Wei) and provide contact information for Li Dong and Furu Wei, who are affiliated with Microsoft.

To definitively answer this question, [More Information Needed] from the authors or Microsoft about the specific funding sources for the BEiT project.
- **Shared by:** Based on the references provided, the main contributors who made the model microsoft/beit-large-patch16-224-pt22k available online as a GitHub repo are:

Li Dong (lidong1@microsoft.com), Furu Wei (fuwei@microsoft.com), Hangbo Bao, and Songhao Piao.

The references mention that for help, issues or other communications regarding the BEiT models, one should contact Li Dong and Furu Wei. It also lists Hangbo Bao, Li Dong, Songhao Piao and Furu Wei as authors of the BEiT paper.

However, more specific information may be needed to confirm if there are any additional contributors who directly worked on making this particular model variant available on GitHub.
- **Model type:** The microsoft/beit-large-patch16-224-pt22k model is a self-supervised vision representation model that uses masked image modeling for pre-training, and follows the pre-training-then-fine-tuning paradigm for downstream vision tasks.
- **Language(s):** The model microsoft/beit-large-patch16-224-pt22k processes visual information from images, not natural human language. [More Information Needed]
- **License:** Based on the references provided, the license for the model microsoft/beit-large-patch16-224-pt22k can be found in the LICENSE file in the root directory of the source tree (Reference 2). However, the specific name and link to the license are not directly provided in the given information.

[More Information Needed] on the specific name and link to the license file.
- **Finetuned from model:** Based on the provided references, the model microsoft/beit-large-patch16-224-pt22k is not directly fine-tuned from another model. The references describe BEIT as a self-supervised vision representation model that is pre-trained using a masked image modeling task, similar to BERT in natural language processing.

The pre-training process involves "tokenizing" the original image into visual tokens, randomly masking some percentage of image patches, and then predicting the visual tokens corresponding to the masked patches. After the self-supervised pre-training, BEIT can be fine-tuned on downstream tasks like image classification and semantic segmentation.

The references do not explicitly mention that microsoft/beit-large-patch16-224-pt22k is fine-tuned from another specific base model. Therefore, [More Information Needed] to determine if this particular model is fine-tuned from another pre-existing model.
### Model Sources

- **Repository:** https://github.com/microsoft/unilm/tree/master/beit
- **Paper:** https://arxiv.org/pdf/2106.08254.pdf
- **Demo:** [More Information Needed]

The provided references do not contain a link to a demo of the model microsoft/beit-large-patch16-224-pt22k. More information would be needed to provide a demo link for this specific model in the model card.
## Uses

### Direct Use

[More Information Needed]

The provided references do not contain information on how to use the microsoft/beit-large-patch16-224-pt22k model without fine-tuning, post-processing, or plugging into a pipeline. The references mainly discuss the pre-training and fine-tuning process of the BEIT model, but do not provide specific details on using the pre-trained model directly for inference without any additional steps.

To provide a complete answer, more information would be needed on the specific use case and requirements for using the model without fine-tuning or post-processing, as well as any relevant code snippets or examples demonstrating such usage.

### Downstream Use

The BEIT (Bidirectional Encoder representation from Image Transformers) model microsoft/beit-large-patch16-224-pt22k can be used for various downstream vision tasks by fine-tuning the pre-trained model on task-specific data. Here are a couple of examples:

1. Image Classification: After pre-training BEIT, you can append a task layer (e.g., a softmax classifier) upon the Transformer and fine-tune the parameters on the target dataset. The fine-tuning process is similar to how BERT is fine-tuned in NLP tasks.

2. Semantic Segmentation: BEIT can be used as a backbone encoder for semantic segmentation tasks. You can incorporate several deconvolution layers as a decoder to produce segmentation maps. The model is end-to-end fine-tuned similar to image classification.

[More Information Needed] for a specific code snippet.

The pre-trained BEIT model can also benefit from intermediate fine-tuning on a data-rich dataset like ImageNet-1K before being fine-tuned on the target downstream tasks. This is a common practice in NLP for BERT fine-tuning.

Overall, the BEIT model provides a powerful foundation for various vision tasks and can be easily integrated into larger ecosystems or applications by following the pre-training-then-fine-tuning paradigm.

### Out-of-Scope Use

Based on the provided references about the BEIT (Bidirectional Encoder representation from Image Transformers) model, here are some potential misuses and what users ought not do with the microsoft/beit-large-patch16-224-pt22k model:

The model should not be used for tasks that require pixel-level precision, as the pre-training objective focuses on recovering visual tokens rather than exact pixel values. Using it for applications like image generation or super-resolution may lead to suboptimal results.

Users should be cautious when applying the model to sensitive domains or tasks with potential societal impact, such as facial recognition, surveillance, or biometric identification. The references do not provide information on the model's performance or biases in these areas.

The model should not be assumed to have human-level understanding or reasoning capabilities about the semantic content of images. While it can acquire knowledge about semantic regions, it is still a statistical model trained on a specific dataset.

[More Information Needed] on the specific data and images used during pre-training. Users should be aware of potential biases or limitations inherited from the training data.

[More Information Needed] regarding the model's robustness to adversarial attacks or manipulated images. Users should not assume the model is immune to such vulnerabilities without proper evaluation.

In general, users should be transparent about the model's usage, capabilities, and limitations when deploying it in real-world applications. They should also consider the potential consequences and ethical implications of using the model in their specific context.

### Bias, Risks, and Limitations

Based on the provided references about the BEIT (Bidirectional Encoder representation from Image Transformers) model, here are some potential issues and limitations:

Technical limitations:
- The ablation studies show that removing certain components like blockwise masking or visual tokens significantly degrades performance, especially on semantic segmentation tasks. This suggests the model heavily relies on these specific techniques.
- Scaling the model size and pre-training data tends to improve performance, but extremely large models may be challenging to train and deploy in practice. More information is needed on computational requirements and efficiency.

Potential misuse or misunderstanding:
- The model achieves strong results on academic benchmarks like ImageNet classification, but real-world performance on diverse tasks is unknown and would require further evaluation. Users should be cautious about claiming the model will work well for all vision applications.
- Using the model on data very different from its pre-training data (ImageNet) may lead to unexpected or degraded results. The model's robustness to distribution shift is not well characterized.

[More Information Needed] on several key points:
- Potential negative societal impacts or misuse cases of the model 
- Fairness and bias considerations in terms of performance across different demographics
- Privacy implications of the model being able to reconstruct missing patches in images
- Environmental impact of training such large-scale models
- How the model's reliance on large pre-training datasets impacts its accessibility and usability for different communities

In summary, while BEIT represents an important advance in vision transformers, users should be aware of its technical limitations and carefully evaluate it for their specific use case. More research is needed to fully characterize its societal impact and limitations. Responsible disclosure of these gaps alongside the model is important.

### Recommendations

Based on the provided references about the BEIT (Bidirectional Encoder representation from Image Transformers) model, here are some recommendations regarding foreseeable issues:

1. Scaling up the model size and pre-training data: The references suggest that BEIT benefits more from scaling up to larger model sizes, especially when labeled data is insufficient for supervised pre-training. However, more information is needed on the specific computational resources and data requirements for scaling up BEIT to even larger sizes (e.g., 1B or 10B parameters).

2. Long-term impact on society: [More Information Needed] The references do not provide insights into the potential long-term societal implications of the BEIT model. Further analysis by ethicists, sociologists, and rights advocates would be necessary to assess the model's impact on privacy, fairness, and other societal considerations.

3. Multimodal pre-training: Reference 7 mentions the future goal of conducting multimodal pre-training using similar objectives and shared architecture for texts and images. However, [More Information Needed] on the specific plans, challenges, and potential risks associated with this multimodal pre-training approach.

4. Automatically acquired knowledge: Reference 7 highlights the intriguing property of BEIT in automatically acquiring knowledge about semantic regions without human-annotated data. [More Information Needed] on the extent and limitations of this automatically acquired knowledge, as well as any potential biases or risks associated with relying on such knowledge in downstream applications.

5. Downstream task performance: While the references demonstrate strong performance of BEIT on tasks like image classification and semantic segmentation, [More Information Needed] on its performance and potential limitations on a wider range of downstream tasks, especially those with real-world implications.

## Training Details

### Training Data

The model microsoft/beit-large-patch16-224-pt22k was pretrained on the ImageNet-22k dataset with images of 224x224 resolution. [More Information Needed] on any additional data pre-processing or filtering steps used during training.

### Training Procedure

#### Preprocessing

For the model microsoft/beit-large-patch16-224-pt22k, the preprocessing steps are as follows:

1. Tokenization: The image is tokenized into a 14 x 14 grid of visual tokens using a publicly available image tokenizer [1, 5]. The vocabulary size is set to 8192 [1].

2. Patch Splitting: The 2D image is split into a sequence of patches with a resolution of 16 x 16 [2, 5]. Each 224 x 224 image is split into a 14 x 14 grid of patches [5].

3. Patch Flattening and Linear Projection: The image patches are flattened into vectors and linearly projected, similar to word embeddings in BERT [2].

4. Masked Patches: During pre-training, some image patches are randomly masked [3, 10]. The number of patches to be masked is specified by the `--num_mask_patches` argument [3].

5. Input and Output Representations: The image patches serve as the input features to the BEIT model [7, 8, 11], while the visual tokens are used as the output representations during pre-training [7, 11].

[More Information Needed] regarding the specific code or library used for the image tokenizer and the exact linear projection applied to the flattened patches.

#### Training Hyperparameters

Here are the training hyperparameters for the model microsoft/beit-large-patch16-224-pt22k based on the provided references:

Model Architecture:
- BEiT-large: #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16 (#parameters: 304M)
- The network architecture follows that of ViT
- Proper initialization is important to stabilize Transformer, especially for large-scale pretraining:
  - Randomly initialize all parameters within a small range, such as [−0.02, 0.02]
  - For the l-th Transformer layer, rescale the output matrices (last linear projection within each sub-layer) of the self-attention module and the feed-forward network by 1/sqrt(2l)

Training Data:
- Pretrained on ImageNet-22k with 224x224 resolution

Training Hardware:
[More Information Needed]

Training Hyperparameters:
[More Information Needed]

Training Code:
[More Information Needed]

#### Speeds, Sizes, Times

Here are the details about the model microsoft/beit-large-patch16-224-pt22k based on the provided references:

Model Specifications:
- BEiT-large model
- #layers=24; hidden size=1024; FFN factor=4x; #heads=16; patch size=16x16 
- Number of parameters: 304M
- Pretrained on ImageNet-22k with 224x224 resolution images

Pretrained Checkpoint:
- Checkpoint link: [beit_large_patch16_224_pt22k_ft22k](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D)
- [More Information Needed] on checkpoint size

Fine-tuning:
- Can be fine-tuned on ImageNet-22k using a DGX-2 box (16 V100-32GB GPUs)
```bash
[More Information Needed]
```

Performance:
- Achieves state-of-the-art ImageNet top-1 accuracy of 88.6% in July 2021, under the setting of using only ImageNet-22k data without extra data
- [More Information Needed] on detailed throughput and training time

Publication Timeline:
- June 2021: Released preprint "BEiT: BERT Pre-Training of Image Transformers" on arXiv 
- July 2021: Released code and pretrained models on GitHub

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model microsoft/beit-large-patch16-224-pt22k evaluates on the following benchmarks and datasets:

1. ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images (Reference 6)

2. ImageNet-1K (References 1, 5, 8)

3. ImageNet-22K (References 1, 7, 9)

The model achieves state-of-the-art ImageNet top-1 accuracy (88.6%) under the setting without extra data other than ImageNet-22k (Reference 9).

[More Information Needed] on specific performance metrics for the microsoft/beit-large-patch16-224-pt22k model on these datasets.

#### Factors

Based on the provided references about the BEIT (Bidirectional Encoder representation from Image Transformers) model microsoft/beit-large-patch16-224-pt22k, here are some foreseeable characteristics that may influence the model's behavior:

Domain and Context:
- The model was pretrained on the ImageNet-22K dataset, so it is likely to perform best on natural images similar to that domain. Performance on other visual domains like medical imaging, satellite imagery, etc. is unclear without further evaluation. [More Information Needed]

- The model uses a patch size of 16x16 and was trained on images of resolution 224x224. Behavior on significantly higher or lower resolution images is unknown. [More Information Needed]

Population Subgroups:
- [More Information Needed] The references do not provide any information on model performance disparities across different population subgroups (e.g. by age, gender, skin tone, etc). Disaggregated evaluation is needed to uncover any such performance differences.

- The pretraining data ImageNet-22K likely contains imbalances and biases present in web-scraped image data. Without further analysis, it's unknown if the model inherits and amplifies these biases.

Other Factors:
- As a large 307M parameter model, BEIT-L requires significant compute resources to use, which could limit accessibility. Smaller variants exist but with lower performance.

- The blockwise masking pretraining appears beneficial for downstream tasks like segmentation. However, predicting mean patch colors performs worse than using a visual token bottleneck.

In summary, more targeted evaluation is needed to characterize BEIT's performance across diverse domains and populations. Potential issues around bias, fairness, and compute requirements are important to analyze further before real-world deployment.

#### Metrics

Based on the provided references, the main evaluation metric used for the microsoft/beit-large-patch16-224-pt22k model is top-1 accuracy on the ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images (Reference 4 and 5).

The references mention that BEiT-large achieves state-of-the-art ImageNet top-1 accuracy (88.6%) without using extra data other than ImageNet-22k (Reference 6). However, there is no explicit discussion of tradeoffs between different types of errors.

[More Information Needed] regarding specific tradeoffs between different errors for the BEiT-large model.

### Results

Based on the provided references, here are the evaluation results for the model microsoft/beit-large-patch16-224-pt22k:

Model Performance:
- Top-1 accuracy on ImageNet-1K: 87.4
- Top-5 accuracy on ImageNet-1K: 98.3
- Image resolution: 224x224
- Number of parameters: 304M

The results show that the BEiT-large model pretrained on ImageNet-22K and fine-tuned on ImageNet-22K achieves strong performance on the ImageNet-1K benchmark. The model benefits from scaling up to a larger size compared to the base model.

[More Information Needed] for additional evaluation metrics or results on other benchmarks.

Scaling Behavior:
The references suggest that BEiT tends to benefit more from scaling up to extremely large model sizes (such as 1B or 10B parameters), especially when labeled data is insufficient for supervised pretraining of such large models. The improvements of BEiT become greater when scaling from base to large size compared to supervised pretraining with ImageNet-22K.

[More Information Needed] for specific quantitative comparisons of scaling behavior.

For further assistance or issues using the BEiT models, please submit a GitHub issue or contact the authors Li Dong (lidong1@microsoft.com) or Furu Wei (fuwei@microsoft.com).

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model microsoft/beit-large-patch16-224-pt22k:

- BEIT-large achieves state-of-the-art ImageNet top-1 accuracy of 88.6% when pretrained on ImageNet-22k, without using any extra data. (Reference 9)

- Scaling up BEIT from base to large size leads to greater improvements compared to supervised pretraining on ImageNet-22k. BEIT-L outperforms BEIT by 2.0 in top-1 accuracy. (Reference 5)

- BEIT learns to distinguish semantic regions using self-attention heads after pretraining, without any task-specific supervision. This helps explain why BEIT is beneficial for downstream tasks. (Reference 7) 

- Ablation studies show that blockwise masking and the masked image modeling pretraining task using visual tokens significantly outperform alternatives like random masking and pixel-level autoencoding. (References 8, 10)

[More Information Needed] on the specific hyperparameters and training details used for the microsoft/beit-large-patch16-224-pt22k model. The references discuss BEIT models in general but do not provide the exact settings for this particular pretrained model.

In summary, the microsoft/beit-large-patch16-224-pt22k model achieves very strong results on ImageNet classification, benefiting from the BEIT pretraining approach, scaling to large model size, and key components like blockwise masking and visual tokens. More details on its training setup would help complete the model card description.

## Model Examination

Based on the provided references, here is the experimental section on explainability/interpretability for the model microsoft/beit-large-patch16-224-pt22k:

The self-attention mechanism in BEIT can separate objects, even without relying on any manual annotations during pre-training [1, 5]. By visualizing the self-attention map for different reference points within an image, it is observed that after pre-training, BEIT learns to distinguish semantic regions using self-attention heads [1]. This property partially indicates why BEIT is able to help with downstream tasks, as the knowledge acquired during pre-training potentially improves the generalization ability of fine-tuned models [1, 4].

The code for analyzing the self-attention map is grouped in the BEiTv2 Repo [3]:
[More Information Needed]

Further experiments show that BEIT's performance improvements from base to large models are greater than supervised pre-training with ImageNet-22K [6]. This suggests that BEIT tends to help more for extremely larger models, especially when labeled data are insufficient for supervised pre-training [6].

Ablation studies demonstrate that blockwise masking is beneficial for BEIT, particularly on semantic segmentation tasks [7]. Additionally, the proposed masked image modeling task significantly outperforms naive pixel-level auto-encoding [7, 8].

## Environmental Impact

- **Hardware Type:** Based on the references provided, the BEiT-large model can be fine-tuned on ImageNet-22k using a DGX-2 box with 16 V100-32GB GPUs.
- **Software Type:** Based on the provided references, the model microsoft/beit-large-patch16-224-pt22k is likely trained using the BEIT (Bidirectional Encoder representation from Image Transformers) architecture, which is a self-supervised vision representation model inspired by BERT from natural language processing.

However, the specific software or deep learning framework (such as PyTorch or TensorFlow) used for training the model is not explicitly mentioned in the given references. Therefore, for the software type used for training, the answer would be:

[More Information Needed]
- **Hours used:** [More Information Needed]

The provided references do not contain specific information about the training time for the model microsoft/beit-large-patch16-224-pt22k. More details would be needed from the model developers or training logs to determine the exact amount of time used to train this particular model.
- **Cloud Provider:** Based on the provided references, the BEiT-large model is fine-tuned on ImageNet-22k using a DGX-2 box with 16 V100-32GB GPUs. The specific cloud provider is not mentioned in the given information.

[More Information Needed] on the exact cloud provider used for training the microsoft/beit-large-patch16-224-pt22k model.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the model microsoft/beit-large-patch16-224-pt22k. To accurately report the carbon emissions, details such as the hardware used for training, total training time, and energy consumption would be necessary.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
- The model uses the standard vision Transformer (ViT) as the backbone network, following the same architecture as in the original ViT paper for fair comparison.
- The input is a sequence of image patches, which are linearly projected to obtain patch embeddings. A special token [S] is prepended to the input sequence, and learnable 1D position embeddings are added.
- The Transformer encoder contains L layers of Transformer blocks. The output of the last layer provides the encoded representations for the image patches.
- [More Information Needed] on the specific details of the Transformer architecture, such as number of layers, hidden dimensions, etc.

Pre-training Objective:
- BEIT is pretrained using the masked image modeling (MIM) task in a self-supervised learning manner.
- The input image is "tokenized" into two views: image patches and visual tokens (discrete tokens).
- Some image patches are randomly masked, and the corrupted image is fed into the Transformer backbone.
- The pre-training objective is to recover the original visual tokens based on the corrupted image patches.
- [More Information Needed] on the specific masking strategy and loss function used for the MIM task.

Fine-tuning:
- After pre-training, the model parameters are fine-tuned on downstream tasks by appending task-specific layers upon the pretrained encoder.
- The model is evaluated on tasks such as image classification and semantic segmentation.

Additional Techniques:
- LayerScale and relative position bias are found to improve the performance of ViT models on downstream tasks, but the vanilla Transformer is used for stability when scaling up to billions of parameters.
- [More Information Needed] on the hyperparameters and training details for the specific microsoft/beit-large-patch16-224-pt22k model.

### Compute Infrastructure

Based on the provided references, the compute infrastructure used for the microsoft/beit-large-patch16-224-pt22k model is as follows:

The BEiT-large model can be fine-tuned on ImageNet-22k using a DGX-2 box (16 V100-32GB GPUs).

[More Information Needed] for the pretraining compute infrastructure of the BEiT-large model.

## Citation

```
@misc{hangbo-beit,
    author = {Hangbo Bao and
              Li Dong and
              Songhao Piao and
              Furu Wei},
    title  = {BEIT: BERT Pre-Training of Image Transformers},
    url    = {https://arxiv.org/pdf/2106.08254.pdf}
}
```

