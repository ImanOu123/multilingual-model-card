# Model Card for google/vit-large-patch16-224-in21k

The model google/vit-large-patch16-224-in21k is a Vision Transformer (ViT) model pre-trained on the ImageNet-21k dataset, which splits an image into fixed-size patches, linearly embeds each patch, adds position embeddings, and feeds the resulting sequence of vectors to a standard Transformer encoder for image classification tasks.

## Model Details

### Model Description

Model Card for google/vit-large-patch16-224-in21k

Model Architecture:
- Vision Transformer (ViT) architecture
- Splits an image into fixed-size patches (patch size 16x16), linearly embeds each patch, adds position embeddings, and feeds the resulting sequence of vectors to a standard Transformer encoder
- Uses a learnable "classification token" prepended to the sequence of embedded patches for classification tasks
- The model has much less image-specific inductive bias compared to CNNs

Training Procedure:
- Pre-trained on the ImageNet-21k dataset with 21k classes and 14M images
- Fine-tuned on downstream tasks by removing the pre-trained prediction head and attaching a zero-initialized feedforward layer
- Fine-tuning often performed at higher resolution than pre-training while keeping the patch size the same

Parameters:
[More Information Needed]

Important Disclaimers:
- The model may not saturate within the range tried, motivating future scaling efforts
- When transferring to downstream tasks, ViT attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train

Additional Notes:
- The first layer of the Vision Transformer linearly projects the flattened patches into a lower-dimensional space
- The model can handle arbitrary sequence lengths (with some restrictions based on the pre-trained position embeddings)

For more information or questions, please contact the project organizer at [More Information Needed].

- **Developed by:** Alexey Dosovitskiy; Lucas Beyer; Alexander Kolesnikov; Dirk Weissenborn; Xiaohua Zhai; Thomas Unterthiner; Mostafa Dehghani; Matthias Minderer; Georg Heigold; Sylvain Gelly; Jakob Uszkoreit; Neil Houlsby
- **Funded by:** Based on the provided references, it appears that the ViT (Vision Transformer) model google/vit-large-patch16-224-in21k was developed by researchers at Google. The work was performed in Berlin, Zürich, and Amsterdam, and many colleagues at Google helped with the project.

However, there is no explicit mention of the specific funding sources or organizations for this project. To confidently list the funders, more information would be needed.

[More Information Needed]
- **Shared by:** Based on the provided references, the main contributors who made the model google/vit-large-patch16-224-in21k available online as a GitHub repo are:

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby

The work was performed in Berlin, Zürich, and Amsterdam. The team also thanked many colleagues at Google for their help, in particular:

- Andreas Steiner for crucial help with the infrastructure and the open source release of the code
- Joan Puigcerver and Maxim Neumann for help with the large-scale training infrastructure 
- Dmitry Lepikhin, Aravindh Mahendran, Daniel Keysers, Mario Lučić, Noam Shazeer, Ashish Vaswani, and Colin Raffel for useful discussions

[More Information Needed] on the specific GitHub repository where the model was made available.
- **Model type:** The model google/vit-large-patch16-224-in21k is a Vision Transformer (ViT) model pre-trained on the ImageNet-21k dataset with 21k classes and 14M images, which splits an image into fixed-size patches, linearly embeds each of them, adds position embeddings, and feeds the resulting sequence of vectors to a standard Transformer encoder for image classification tasks.
- **Language(s):** The model google/vit-large-patch16-224-in21k is a Vision Transformer (ViT) model that processes image data, and does not use or process natural human language. [More Information Needed] on any natural language capabilities of this specific model.
- **License:** [More Information Needed]

The references provided do not contain any information about the specific license being used for the model google/vit-large-patch16-224-in21k. To answer this question, more details would be needed from the model developers or repository maintainers regarding the license they have chosen for this particular model.
- **Finetuned from model:** The model google/vit-large-patch16-224-in21k is fine-tuned from the ViT-L/16 model that was pretrained on ImageNet-21k.

From the references:

"2020-10-29: Added ViT-B/16 and ViT-L/16 models pretrained on ImageNet-21k and then fine-tuned on ImageNet at 224x224 resolution (instead of default 384x384). These models have the suffix "-224" in their name."

The model name google/vit-large-patch16-224-in21k indicates it is the ViT-L/16 variant ("-large-patch16") that was pretrained on ImageNet-21k ("in21k") and fine-tuned at 224x224 resolution ("-224").

[More Information Needed] on the specific link to the base ViT-L/16 model pretrained on ImageNet-21k.
### Model Sources

- **Repository:** https://github.com/rwightman/pytorch-image-models
- **Paper:** https://arxiv.org/pdf/2010.11929.pdf
- **Demo:** [More Information Needed]

The references provided do not contain a direct link to a demo of the model google/vit-large-patch16-224-in21k. The references mention Colab notebooks that demonstrate the usage of Vision Transformer models in general, but there is no specific demo link for the google/vit-large-patch16-224-in21k model. More information would be needed to provide a demo link for this specific model.
## Uses

### Direct Use

The model google/vit-large-patch16-224-in21k can be used without fine-tuning for image classification tasks by removing the pre-trained prediction head and attaching a zero-initialized feedforward layer with the number of output classes for the target dataset.

To use the model, you would:
1. Remove the entire pre-trained head (two linear layers)
2. Replace it with a single, zero-initialized linear layer that outputs the number of classes required by your target dataset
3. Feed images into the model, keeping the patch size the same as during pre-training. For higher resolution images, this will result in a larger effective sequence length.

The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints), but the pre-trained position embeddings may no longer be meaningful. To address this, perform 2D interpolation of the pre-trained position embeddings according to their location in the original image.

[More Information Needed] for a specific code snippet.

### Downstream Use

The google/vit-large-patch16-224-in21k model can be fine-tuned on a downstream task of interest. To do this:

1. Remove the pre-trained prediction head and attach a zero-initialized D × K feedforward layer, where K is the number of downstream classes. 

2. Fine-tune the model using a higher resolution than pre-training (e.g. 384x384) while keeping the patch size the same. This results in a larger effective sequence length that ViT can handle.

3. Use SGD with momentum 0.9 for fine-tuning and run a grid search over learning rates.

4. If transferring to another dataset, remove the whole pre-trained head (two linear layers) and replace it with a single zero-initialized linear layer outputting the number of classes in the target dataset.

5. Perform 2D interpolation of the pre-trained position embeddings according to their location in the original image to adapt them to the new resolution.

Here is example code for fine-tuning ViT-L/16 pre-trained on ImageNet-21k on the CIFAR-10 dataset:

```bash
python -m vit_jax.main --workdir=/tmp/vit-$(date +%s) \
    --config=$(pwd)/vit_jax/configs/vit.py:b16,cifar10 \
    --config.pretrained_dir='gs://vit_models/imagenet21k'
```

[More Information Needed] on integrating the fine-tuned model into a larger ecosystem or app. The fine-tuned model can likely be exported and used for inference in other environments.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the google/vit-large-patch16-224-in21k model may be misused and what users ought not do:

The model should not be used for detection or segmentation tasks without further research and validation, as the references indicate applying ViT to those tasks is still a challenge that needs more exploration.

Users should not assume the model's performance from self-supervised pre-training is on par with supervised pre-training on large datasets. The references note there is still a large gap between the two approaches.

The model should not be applied to downstream tasks or datasets significantly different from what it was originally trained and evaluated on (ImageNet, ImageNet-21k, etc.) without additional fine-tuning and testing. Transferability to very different domains is not guaranteed.

[More Information Needed] on the specific intended use cases and limitations of this model. The references focus mainly on the model architecture and training approach, but do not go into detail on appropriate and inappropriate applications.

[More Information Needed] on potential fairness, bias and ethical issues that could arise from misuse. The references do not address these important considerations.

In general, users should be cautious about relying on the model output without having a clear understanding of its capabilities and limitations for their specific use case. More testing and analysis is recommended before deploying the model in high-stakes or sensitive contexts.

### Bias, Risks, and Limitations

Based on the provided references about the Vision Transformer (ViT) model google/vit-large-patch16-224-in21k, here are some known or foreseeable issues:

1. The model still faces challenges when applied to other computer vision tasks like detection and segmentation. More research is needed to explore its effectiveness in these areas. (Reference 1)

2. There is still a large performance gap between self-supervised pre-training and large-scale supervised pre-training for the model. Further work is required to improve self-supervised methods. (Reference 1) 

3. The model's performance is modest when trained on mid-sized datasets like ImageNet without strong regularization, falling a few percentage points below comparable ResNets. This may be due to lacking some inductive biases. (Reference 11)

4. [More Information Needed] on any specific foreseeable harms or misunderstandings that could arise from using this model. The references do not cover this.

5. [More Information Needed] on the model's computational costs, efficiency, and environmental impact at scale. More details would be needed to assess this.

6. Sociotechnical limitations are not clearly addressed in the references. [More Information Needed] on how the model's usage could interact with and impact society, different demographics, and stakeholders.

In summary, the key limitations seem to be the model's generalization to other vision tasks, the self-supervised performance gap, and modest results on smaller datasets. More information is needed to thoroughly analyze foreseeable harms, misunderstandings, and sociotechnical implications. Continued research and responsible development practices will be important.

### Recommendations

Based on the provided references about the Vision Transformer (ViT) model google/vit-large-patch16-224-in21k, here are some recommendations regarding foreseeable issues:

1. Applicability to other vision tasks: While the ViT model shows promising results on image classification, its performance on other computer vision tasks like detection and segmentation still needs to be explored and validated. [Reference 1]

2. Self-supervised pre-training: Although self-supervised pre-training improves the model's performance, there is still a significant gap compared to large-scale supervised pre-training. Further research is needed to bridge this gap and improve self-supervised methods for the ViT model. [Reference 2]

3. Scaling and computational resources: Scaling up the ViT model is likely to lead to improved performance. However, this may require significant computational resources, which could be a limiting factor for some users or applications. [Reference 3, 4]

4. Inductive bias: Compared to Convolutional Neural Networks (CNNs), the ViT model has much less image-specific inductive bias. While this allows for more flexibility, it may also lead to less efficient learning on smaller datasets or tasks that benefit from strong inductive biases. [Reference 7]

5. Fine-tuning and resolution: Fine-tuning the ViT model at higher resolutions than pre-training can be beneficial, but it also increases the sequence length and computational requirements. Users should be aware of these trade-offs when fine-tuning the model for specific tasks. [Reference 8]

6. Interpretability and understanding: [More Information Needed] on how the ViT model processes image data and how its internal representations can be interpreted and understood by users and researchers.

7. Fairness, bias, and ethical considerations: [More Information Needed] on potential biases or fairness issues that may arise from the pre-training data or the model's architecture, as well as any ethical considerations related to the model's use and deployment.

## Training Details

### Training Data

The model google/vit-large-patch16-224-in21k was pre-trained on the ImageNet-21k dataset, which contains 14 million images across 21,000 classes. [More Information Needed] on any additional data pre-processing or filtering steps used during training.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about the preprocessing for the data of the model google/vit-large-patch16-224-in21k:

Resizing/Reshaping:
- The input image x ∈ R H×W×C is reshaped into a sequence of flattened 2D patches x p ∈ R N×(P2•C) 
- (H, W) is the resolution of the original image
- C is the number of channels
- (P, P) is the resolution of each image patch (16x16 for this model)
- N = HW/P2 is the resulting number of patches, which also serves as the effective input sequence length for the Transformer

Patch Embedding:
- The flattened patches are linearly projected to a lower-dimensional space of dimension D using a trainable linear projection (Eq. 1)
- The output of this projection is referred to as the patch embeddings

Position Embeddings:
- A learnable embedding is prepended to the sequence of embedded patches (z00 = xclass)
- This serves as the image representation y at the output of the Transformer encoder (z0L)

[More Information Needed] on the specific tokenization details.

The model can handle higher resolution images at fine-tuning time by keeping the patch size the same, resulting in a larger effective sequence length.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters used for the google/vit-large-patch16-224-in21k model:

Model architecture:
- ViT-L/16 configuration (ViT "Large" variant with 16x16 input patch size) [Reference 6]

Optimizer:
- Adam optimizer with β1 = 0.9, β2 = 0.999 [Reference 2]
- Weight decay of 0.1 [Reference 2]

Training:
- Batch size of 4096 [Reference 2]
- Linear learning rate warmup and decay [Reference 2, details in Appendix B.1]

Dataset:
- Pre-trained on ImageNet-21k dataset with 21k classes and 14M images [Reference 3]

Regularization:
- [More Information Needed] on specific regularization techniques like dropout and label smoothing

Training duration:
- Could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days [Reference 11]

Fine-tuning:
- Fine-tuned using SGD with momentum, batch size 512 [Reference 2]

Some key information like the exact learning rate schedule, number of training epochs/steps, and regularization parameters are not explicitly provided in the given references and would need further details from the paper or training logs.

#### Speeds, Sizes, Times

Here are the details I could find about the model google/vit-large-patch16-224-in21k based on the provided references:

Model name: ViT-L/16 (Vision Transformer Large with 16x16 patch size)

Pre-training dataset: ImageNet-21k (21k classes, 14M images)

Fine-tuning:
- Fine-tuned on ImageNet at higher resolution of 512x512
- Used Polyak & Juditsky (1992) averaging with factor 0.9999 for fine-tuning

Performance:
- Outperforms BiT-L on ImageNet and other downstream tasks while requiring 2-4x less compute
- Achieves state-of-the-art results on most recognition benchmarks compared to CNNs

Throughput: [More Information Needed]

Start/End Time: [More Information Needed] 

Checkpoint Size: [More Information Needed]

The model checkpoint can be downloaded from:
```
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz
```

Configuration details can be found in `vit_jax/configs/models.py` under `config.model_name`.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model google/vit-large-patch16-224-in21k was pre-trained on the ImageNet-21k dataset and then fine-tuned on the ImageNet dataset at 224x224 resolution.

The model was evaluated on the following benchmarks and datasets:

1. ImageNet (ILSVRC-2012) validation set with original labels and cleaned-up ReaL labels
2. CIFAR-10 and CIFAR-100
3. Oxford-IIIT Pets
4. Oxford Flowers-102
5. The 19-task VTAB classification suite, which includes:
   - Natural tasks (similar to ImageNet, CIFAR, Pets, etc.)
   - Specialized tasks
   - Structured tasks

Additionally, the model is expected to achieve 82.7% top-1 accuracy on the ImageNet validation set at 224x224 resolution.

[More Information Needed] regarding the specific performance metrics on the other datasets and tasks mentioned.

#### Factors

Based on the provided references about the Vision Transformer (ViT) model google/vit-large-patch16-224-in21k, here are some foreseeable characteristics that may influence its behavior:

Dataset Size and Domain:
The performance of the ViT model is heavily influenced by the size and domain of the pre-training dataset. References 1, 2, and 5 indicate that ViT models benefit from larger datasets like ImageNet-21k and JFT-300M compared to smaller datasets like ImageNet. The model's performance on downstream tasks will likely vary depending on the similarity between the pre-training data domain and the target domain.

Computational Cost and Model Size: 
References 4 and 7 suggest that the computational cost of pre-training and the model size impact ViT's performance. Larger ViT models tend to perform better but require more compute resources for pre-training. The ViT-L/16 model pre-trained on ImageNet-21k shows good performance while being more computationally efficient.

Fine-tuning Resolution:
Reference 3 mentions that fine-tuning ViT at a higher resolution than the pre-training resolution can be beneficial. The patch size remains the same, resulting in a larger effective sequence length. This suggests that the model's performance may vary based on the resolution used during fine-tuning.

[More Information Needed] regarding specific population subgroups or factors for disaggregated evaluation, as the provided references do not cover this aspect in detail.

In summary, the pre-training dataset size and domain, computational cost, model size, and fine-tuning resolution are key characteristics that influence the behavior of the google/vit-large-patch16-224-in21k model. More information is needed to assess potential disparities in performance across different population subgroups or factors.

#### Metrics

Based on the references provided, the key metrics used to evaluate the google/vit-large-patch16-224-in21k model and compare it to other models like ResNets include:

- Accuracy on downstream image classification tasks after pre-training, such as:
  - ImageNet 
  - CIFAR-10/100
  - Oxford-IIIT Pets
  - Oxford Flowers-102
  - The 19-task VTAB classification suite, broken down into Natural, Specialized and Structured task groups

- Few-shot linear probing accuracy is reported instead of full fine-tuning accuracy to save compute during evaluation

- Computational cost (compute resources) required for pre-training the model, to assess the performance/compute trade-off. Vision Transformers use 2-4x less compute than ResNets to attain the same downstream task performance.

The key tradeoff evaluated is between downstream task accuracy and the computational cost of pre-training. No specific information is provided on tradeoffs between different error types.

### Results

Based on the provided references, here are the evaluation results for the model google/vit-large-patch16-224-in21k:

Performance:
- The ViT-L/16 model pre-trained on JFT-300M outperforms BiT-L on all tasks, while requiring substantially less computational resources to train. (Reference 1)
- ViT performs very favorably when considering the computational cost of pre-training, attaining state of the art on most recognition benchmarks at a lower pre-training cost. (Reference 2)
- ViT-L/16 model pre-trained on the public ImageNet-21k dataset performs well on most datasets, while taking fewer resources to pre-train. It could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days. (Reference 4)
- ViT-L/16 is expected to achieve 82.7% top-1 accuracy when pretrained on ImageNet-21k and then fine-tuned on ImageNet at 224x224 resolution. (Reference 7)

Computational Efficiency:
- ViT-L/16 requires substantially less computational resources to train compared to BiT-L. (Reference 1)
- ViT performs favorably when considering the computational cost of pre-training. (Reference 2)
- ViT-L/16 takes fewer resources to pre-train and could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days. (Reference 4)

[More Information Needed] for detailed benchmark results on specific datasets like VTAB, CIFAR-100, etc.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model google/vit-large-patch16-224-in21k:

- The ViT-L/16 model, which is likely the google/vit-large-patch16-224-in21k model, was pre-trained on the public ImageNet-21k dataset. It performs well on most transfer learning datasets while requiring fewer computational resources to pre-train compared to other state-of-the-art models. It could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days. (Reference 4)

- When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train. (Reference 9) 

- The ViT-L/16 model outperforms BiT-L, which is pre-trained on the same JFT-300M dataset, on all evaluated tasks, while requiring substantially less computational resources to train. (Reference 3)

- [More Information Needed] on the specific performance metrics of the google/vit-large-patch16-224-in21k model on individual benchmarks.

In summary, the google/vit-large-patch16-224-in21k model, likely referring to ViT-L/16, achieves strong performance on various transfer learning tasks while being more computationally efficient to train compared to state-of-the-art CNN models. However, more specific performance numbers on individual benchmarks would be helpful to include in the model card.

## Model Examination

Based on the provided references, here are some key points about the interpretability and explainability of the google/vit-large-patch16-224-in21k model:

1. Self-attention allows the model to integrate information globally across the entire image, even in the lowest layers. The "attention distance" (analogous to receptive field size in CNNs) varies across attention heads, with some attending to most of the image while others focus on smaller regions. As depth increases, the attention distance increases for all heads. [References 1, 2, 3]

2. The model learns to encode distance within the image in the similarity of position embeddings. Patches that are closer together tend to have more similar position embeddings, and there is a row-column structure where patches in the same row/column have similar embeddings. [Reference 5]

3. Attention Rollout can be used to compute maps of the attention from the output token to the input space, by averaging attention weights across all heads and recursively multiplying the weight matrices of all layers. [Reference 4]

4. [More Information Needed] on the specific training scripts or configurations used for the google/vit-large-patch16-224-in21k model.

5. The model uses an extra learnable "classification token" appended to the sequence of image patches. Attempting to use only image-patch embeddings with global average pooling performed poorly, but this was due to the requirement for a different learning rate rather than the extra token or pooling operation. [Reference 8]

## Environmental Impact

- **Hardware Type:** Based on the provided references, there is no direct mention of the specific hardware type that the model google/vit-large-patch16-224-in21k was trained on. The references discuss training Vision Transformer models in general using TPUs (Reference 6), but do not specify the hardware for this particular model.

[More Information Needed] on the exact hardware type used to train google/vit-large-patch16-224-in21k.
- **Software Type:** Based on the references provided, the model google/vit-large-patch16-224-in21k was likely trained using code from the GitHub repository https://github.com/google-research/big_vision/. Specifically, the training scripts may have been similar to [configs/vit_i21k.py](https://github.com/google-research/big_vision/blob/main/big_vision/configs/vit_i21k.py) for pre-training the Vision Transformer (ViT) model.

However, more specific information would be needed to definitively state the exact software and code used to train this particular model. The references do not explicitly mention the google/vit-large-patch16-224-in21k model.
- **Hours used:** [More Information Needed]

The provided references do not contain specific information about the training time for the model google/vit-large-patch16-224-in21k. More details would be needed from the model developers or training logs to determine the exact amount of time used to train this specific model.
- **Cloud Provider:** Based on the provided references, the model google/vit-large-patch16-224-in21k was likely trained on Google Cloud TPUs. Specifically, reference 4 mentions that "the ViT-L/16 model pre-trained on the public ImageNet-21k dataset... could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days."

While the references don't explicitly state the cloud provider used for this specific model, they do mention using Google Cloud TPUs for training similar models. For example, reference 1 shows a command for creating a TPU VM on Google Cloud:

```
gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --project=$PROJECT --zone=$ZONE \
    --accelerator-type v3-8 \
    --version tpu-vm-base
```

Therefore, it is reasonable to infer that the google/vit-large-patch16-224-in21k model was also trained on Google Cloud TPUs, but more specific information would be needed to confirm this definitively.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted when training the google/vit-large-patch16-224-in21k model. To accurately report the carbon emissions, more details would be needed such as the specific compute hardware used, total training time, energy consumption, and the carbon intensity of the electricity grid where the training was performed.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The Vision Transformer (ViT) model splits an image into fixed-size patches, linearly embeds each patch, adds position embeddings, and feeds the resulting sequence of vectors to a standard Transformer encoder. The model prepends a learnable embedding to the sequence of embedded patches, whose state at the output of the Transformer encoder serves as the image representation. A classification head (MLP with one hidden layer during pre-training, single linear layer during fine-tuning) is attached to this image representation for classification tasks.

Model Objective:
The Vision Transformer model is typically pre-trained on large datasets and then fine-tuned on downstream tasks. During pre-training, the model learns to capture general image features, while during fine-tuning, it adapts to specific tasks such as image classification. The model has shown excellent results compared to state-of-the-art convolutional networks when pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks, while requiring substantially fewer computational resources to train.

[More Information Needed] on specific pre-training and fine-tuning details for the google/vit-large-patch16-224-in21k model.

### Compute Infrastructure

Based on the provided references, here is the information about the compute infrastructure used for the google/vit-large-patch16-224-in21k model:

The ViT-L/16 model pre-trained on the public ImageNet-21k dataset could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days (Reference 9).

The models were originally trained using more advanced code (e.g. multi-host training) in the https://github.com/google-research/big_vision/ repository (Reference 10).

[More Information Needed] on the specific compute infrastructure details like the exact number of TPUs or GPUs used, memory configurations, etc. for training the google/vit-large-patch16-224-in21k model.

## Citation

```
@misc{alexey-title,
    author = {Alexey Dosovitskiy and
              Lucas Beyer and
              Alexander Kolesnikov and
              Dirk Weissenborn and
              Xiaohua Zhai and
              Thomas Unterthiner and
              Mostafa Dehghani and
              Matthias Minderer and
              Georg Heigold and
              Sylvain Gelly and
              Jakob Uszkoreit and
              Neil Houlsby},
    title  = {None},
    url    = {https://arxiv.org/pdf/2010.11929.pdf}
}
```

