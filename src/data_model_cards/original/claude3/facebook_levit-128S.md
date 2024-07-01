# Model Card for facebook/levit-128S

The facebook/levit-128S model is a hybrid neural network architecture called LeViT that combines convolutional and transformer components for fast and accurate image classification. LeViT uses a pyramid structure with pooling, similar to LeNet, and replaces the uniform structure of a transformer to learn convolutional-like features efficiently.

## Model Details

### Model Description

Model Card for facebook/levit-128S

Model Details:
The facebook/levit-128S model is based on the LeViT (LeVision Transformer) architecture, which integrates ResNet stages within the transformer architecture. Key components include:

- Normalization layers and activations: Each convolution is followed by batch normalization. 
- Patch embedding: A small convnet is applied on input to the transformer stack.
- No classification token: Average pooling on the last activation map produces an embedding used in the classifier.
- Downsampling: Between LeViT stages, a shrinking attention block reduces activation map size.

Training Procedure:
LeViT is trained with distillation driven training similar to DeiT, using two classification heads with cross-entropy loss. The first head receives supervision from groundtruth classes, the second from a RegNetY-16GF model trained on ImageNet. 

To train LeViT-256 on ImageNet with hard distillation on 8 GPUs:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model LeViT_256 --data-path /path/to/imagenet --output_dir /path/to/save
```

[More Information Needed] on specific training details for the LeViT-128S variant.

Model Parameters:
[More Information Needed]

Important Disclaimers:
[More Information Needed]

For more details or questions, please contact the model development team at [More Information Needed].

- **Developed by:** Benjamin Graham; Alaaeldin El-Nouby; Hugo Touvron; Pierre Stock; Armand Joulin; Hervé Jégou; Matthijs Douze
- **Funded by:** Based on the provided references, it appears that the LeViT-128S model was developed by researchers at Facebook. The paper cited in reference 5 lists the authors as Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou and Matthijs Douze, who are likely part of the Facebook research team.

However, there is no explicit mention of the specific organizations or people that funded this project. To definitively answer this question, [More Information Needed] from the authors or Facebook.
- **Shared by:** Based on the references provided, the contributors who made the model facebook/levit-128S available online as a GitHub repo are:

Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, and Matthijs Douze from Facebook Research.

They published the paper "LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference" with details about the LeViT models, and released the code at https://github.com/facebookresearch/LeViT.
- **Model type:** The facebook/levit-128S model is a hybrid neural network that combines convolutional and transformer architectures, trained on the ImageNet dataset using distillation-driven training with a cross-entropy loss for image classification. [More Information Needed] on the specific training hyperparameters, hardware used, and evaluation metrics.
- **Language(s):** The facebook/levit-128S model processes image data, not natural human language. [More Information Needed]
- **License:** The model facebook/levit-128S is released under the Apache 2.0 license, as stated in the reference:

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
- **Finetuned from model:** Based on the provided references, the facebook/levit-128S model builds upon the vision transformer (ViT) architecture and uses the Data-efficient Image Transformer (DeiT) training method, as mentioned in reference 7:

"LeViT builds upon the ViT [2] architecture and DeiT [3] training method."

However, the exact base model that facebook/levit-128S is fine-tuned from is not explicitly stated in the given references. More specific information about the base model and a link to it would be needed to fully answer this question.

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/facebookresearch/LeViT
- **Paper:** https://arxiv.org/pdf/2104.01136.pdf
- **Demo:** [More Information Needed]

The provided references do not contain a link to a demo of the facebook/levit-128S model. More information would be needed to provide a demo link in the model card description.
## Uses

### Direct Use

To evaluate the pre-trained facebook/levit-128S model on the ImageNet validation set using a single GPU, you can run the following code snippet:

```
python main.py --eval --model LeViT_128S --data-path /path/to/imagenet
```

This should give the following results:
```
* Acc@1 [More Information Needed] Acc@5 [More Information Needed] loss [More Information Needed]
```

The model can be used for inference without any fine-tuning or post-processing steps. Simply pass an input image through the model to obtain the predicted class probabilities.

To use the model, first clone the LeViT repository:
```
git clone https://github.com/facebookresearch/levit.git
```

Then, install the required dependencies:
```
conda install -c pytorch pytorch torchvision
pip install timm
```

[More Information Needed] on any additional steps or code required to use the model for inference.

For more details on the model architecture and design choices, please refer to the information provided in the "References" section above.

### Downstream Use

The LeViT-128S model can be fine-tuned for various image classification tasks by training it on a smaller dataset specific to the desired task. It can also be integrated into larger applications or systems that require efficient and accurate image classification.

To evaluate a pre-trained LeViT model on the ImageNet validation set using a single GPU, you can run:

```
python main.py --eval --model LeViT_256 --data-path /path/to/imagenet
```

This should give results like:

```
* Acc@1 81.636 Acc@5 95.424 loss 0.750
```

To fine-tune LeViT-128S for a specific task, you would need to modify the training code to use your custom dataset instead of ImageNet.

[More Information Needed] on detailed steps for fine-tuning and integrating LeViT-128S into a larger ecosystem or application. Additional code snippets and guidance would be required based on the specific use case.

### Out-of-Scope Use

Based on the provided references, there is not enough information to comprehensively address how the LeViT-128S model may be misused or what users should not do with it. The references focus mainly on technical details of the model architecture, training process, and ablation studies.

To properly answer this question, more information would be needed, such as:

1. The intended use cases and applications of the LeViT-128S model
2. Potential negative consequences or harm that could result from misuse of the model
3. Ethical considerations and societal implications of deploying the model
4. Any built-in safeguards or limitations to prevent misuse

Without these additional details, it is difficult to provide a complete answer. The sociotechnic team member would need to collaborate with the rest of the development team to gather this information and conduct a thorough analysis of the model's potential misuse and establish clear guidelines for what users should not do with the model.

[More Information Needed]

### Bias, Risks, and Limitations

Based on the provided references about the LeViT-128S model, here are some potential issues and limitations:

Technical limitations:
- The pyramid structure with reducing resolution is key to controlling computational complexity. Removing this pyramid shape and using a straight stack of attention and MLPs significantly reduces accuracy. [Ref 1, 4]
- The patch convolutions used for preprocessing the input are important for model performance in this low-capacity regime. Removing them reduces accuracy. [Ref 2, 5]
- Using more traditional transformer block configurations with larger keys and MLP expansion factors hurts performance compared to LeViT's relatively wide blocks. [Ref 3, 6]

Sociotechnical limitations: 
[More Information Needed] The provided references focus on technical ablation experiments and model architecture details. They do not discuss broader sociotechnical implications.

Potential misunderstandings:
[More Information Needed] The references do not cover how the model could be misunderstood or misused.

Foreseeable harms: 
[More Information Needed] No information is given on risks or harms that could arise from using this model.

In summary, the references highlight some key technical design choices that enable LeViT-128S to achieve strong speed-accuracy tradeoffs, but more information would be needed to assess the broader sociotechnical limitations, potential for misunderstanding, and foreseeable harms. The provided details focus narrowly on model architecture experiments.

### Recommendations

Based on the provided references about the LeViT-128S model, here are some recommendations regarding foreseeable issues:

[More Information Needed] on the specific training data and intended use cases of the model. This information is crucial for identifying potential biases or misuse risks.

The pyramid structure and PatchConv preprocessing are important for LeViT's performance (References 3-5). Removing these components significantly degrades accuracy. When using or fine-tuning LeViT-128S, preserving this architecture is recommended to maintain effectiveness.

[More Information Needed] on testing LeViT-128S for robustness to distribution shift, adversarial examples, or other failure modes. Additional testing is prudent before deployment in high-stakes applications.

The model card should clearly document LeViT-128S's capabilities and limitations to set appropriate expectations and boundaries for users, per standard responsible AI practices (Reference 7).

[More Information Needed] on the energy and compute requirements of training and inference for LeViT-128S. Efficiency considerations are relevant to the model's societal and environmental impact.

In summary, key issues to address are training data documentation, robustness testing, use case guidance, and resource footprint assessment. Preserving the core LeViT architecture is also advised. Additional context on the model's development and intended scope of use would allow for a more comprehensive assessment.

## Training Details

### Training Data

The model facebook/levit-128S is trained on the ImageNet-2012 dataset and evaluated on its validation set. No additional training data is used in this work.

[More Information Needed] for links to documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about data preprocessing for the facebook/levit-128S model:

Tokenization:
[More Information Needed] - The references do not provide specific details about the tokenization process used.

Resizing/Rewriting: 
The model expects input images to be in the standard layout for the torchvision datasets.ImageFolder, with the training data in a train/ folder and validation data in a val/ folder. The directory structure should be:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

Other Preprocessing:
[More Information Needed] - The references do not specify other preprocessing steps like normalization, data augmentation, etc. used for training the model.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters I can provide for the facebook/levit-128S model:

Training hardware:
- 32 GPUs were used to perform 1000 training epochs, which took 3 to 5 days.

Regularization:
- Distillation driven training was used, similar to DeiT. LeViT was trained with two classification heads with a cross entropy loss. The first head received supervision from the groundtruth classes, the second one from a RegNetY-16GF model trained on ImageNet.

Dataset:
- The model was trained on the ImageNet-2012 dataset and evaluated on its validation set.

[More Information Needed] for the following hyperparameters:
- Exact learning rate schedule 
- Optimizer
- Batch size
- Weight decay
- Exact data augmentation and regularization techniques

The provided code snippets are for training LeViT-256, not the specific LeViT-128S model, so I cannot include those details without more information.

#### Speeds, Sizes, Times

Based on the provided references, here are the details I can provide about the model facebook/levit-128S:

Throughput and Inference Speed:
LeViT-128S is on-par with DeiT-Tiny in accuracy but uses 4× fewer FLOPs, indicating it has much higher throughput and faster inference speed.

Checkpoint Size: [More Information Needed]

Training Dataset: 
The model was trained on the ImageNet-2012 dataset.

Training Hardware: [More Information Needed]

Training Time: [More Information Needed] 

Model Architecture:
LeViT-128S is a hybrid neural network architecture that combines convolutional layers and vision transformers. The number 128 in the model name indicates it has 128 channels input to the first transformer stage.

Accuracy:
The exact accuracy numbers for LeViT-128S are not provided. However, the references state that in general, the LeViT models outperform convolutional and transformer architectures in the speed-accuracy tradeoff, especially at faster operating points.

Code:
[More Information Needed]

To summarize, LeViT-128S is a fast and efficient model that provides a strong speed-accuracy tradeoff, but more specific details on checkpoint size, training hardware and time, and code are needed to fully describe the model. The references focus more on the general LeViT architecture and tradeoffs rather than specifics of the 128S variant.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the LeViT model evaluates on the following benchmark and dataset:

Dataset: The model is trained and evaluated on the ImageNet-2012 dataset, using the validation set for evaluation (Reference 3).

Benchmarks:
- The model is compared against EfficientNet as a strong convolutional baseline, and DeiT as a strong transformer-only architecture baseline. The comparison is made in terms of speed-precision tradeoffs (Reference 2).
- LeViT-128S is compared to DeiT-Tiny, showing that it achieves similar accuracy while using 4× fewer FLOPs (Reference 5).

[More Information Needed] on specific benchmark results or metrics for the LeViT-128S model.

#### Factors

Based on the provided references about the LeViT (facebook/levit-128S) model, here are some key characteristics that may influence its behavior:

Domain and Context:
- The model is designed for image classification tasks, building upon the Vision Transformer (ViT) architecture. It is adapted for high-throughput inference and aims to improve efficiency.

Population Subgroups and Potential Disparities:
[More Information Needed]
The references do not provide specific information about performance across different population subgroups or potential disparities. To properly evaluate this, the model would need to be tested on diverse datasets representing various demographics, and results should be disaggregated and analyzed for fairness.

Other Notable Characteristics:
- Uses a multi-stage transformer architecture with attention for downsampling
- Employs a computationally efficient patch descriptor to reduce features in early layers  
- Incorporates a learned, per-head translation-invariant attention bias
- Has a redesigned Attention-MLP block for improved capacity given compute constraints
- Borrows concepts from convolutional architectures like ResNet

To fully assess the model's real-world impact, more information is needed on training data composition, evaluation across diverse populations, and testing for potential biases or disparities in performance. The architectural choices are clearly outlined, but societal implications require further analysis.

#### Metrics

Based on the provided references, the key metrics used to evaluate the LeViT-128S model and compare it against other architectures are:

1. ImageNet top-1 accuracy
2. FLOPs (floating point operations) as a measure of computational efficiency
3. Inference speed on different hardware platforms like CPU

The references highlight that LeViT models aim to achieve a good speed-accuracy tradeoff, providing competitive accuracy while being much faster and more efficient in terms of FLOPs compared to other transformer and convolutional architectures.

For example, reference 3 states that "LeViT-128S is on-par with DeiT-Tiny and uses 4× fewer FLOPs." Reference 10 mentions that "at 80% ImageNet top-1 accuracy, LeViT is 5 times faster than EfficientNet on CPU."

So in summary, the main evaluation metrics considering the tradeoffs are ImageNet accuracy, FLOPs, and inference speed on different hardware. The model aims to optimize the speed-accuracy tradeoff.

### Results

Based on the provided references, here is the evaluation information I could find for the facebook/levit-128S model:

Datasets and Evaluation:
The model was trained on the ImageNet-2012 dataset and evaluated on its validation set, following the approach used in the DeiT paper. No additional training data beyond ImageNet was used.

Speed-Accuracy Tradeoff:
Table 3 shows the speed-accuracy tradeoffs obtained with LeViT models, including LeViT-128S, compared to competitive architectures like EfficientNet and DeiT. The exact accuracy and speed numbers for LeViT-128S are [More Information Needed] as they are not explicitly provided in the given references.

Model Variants and Ablations:
The LeViT-128S model was trained for 100 epochs to evaluate the impact of design changes relative to ViT/DeiT. The conclusions drawn from these experiments are expected to hold for larger models and longer training schedules. Specific results for the LeViT-128S ablations are [More Information Needed].

Model Scaling:
The LeViT models can provide a range of speed-accuracy tradeoffs by varying the size of the computation stages. LeViT-128S is one of the models in this family, with 128 channels input to the first transformer stage. The exact stage design details for LeViT-128S are [More Information Needed].

Training Code:
[More Information Needed]

#### Summary

Based on the provided references, here is a summary of the evaluation results for the facebook/levit-128S model:

- LeViT-128S is designed for the high-throughput regime, aiming to be fast and efficient. (Reference 1)

- In experiments, LeViT-128S was trained for 100 epochs to magnify differences and reduce training time compared to ViT/DeiT variants. The conclusions hold for larger models and longer training schedules. (Reference 3) 

- LeViT-128S achieves accuracy on-par with DeiT-Tiny while using 4x fewer FLOPs, demonstrating a superior speed-accuracy tradeoff. (Reference 5)

- The pyramid structure and reduction of resolution in LeViT-128S are key to keeping computational complexity under control. Ablation studies show that removing the pyramid shape and keeping FLOP count similar results in very low accuracy. (Reference 6)

- [More Information Needed] on specific runtime measurements of LeViT-128S.

- LeViT-128S is part of the LeViT model family that can provide a range of speed-accuracy tradeoffs by varying the size of computation stages. The 128 in LeViT-128S indicates 128 channels input to the first transformer stage. (Reference 11)

In summary, the LeViT-128S model demonstrates strong performance in the speed-accuracy tradeoff, outperforming comparable ViT/DeiT models. Its pyramid structure and resolution reduction are key architectural choices for efficiency.

## Model Examination

Based on the provided references, here is the experimental section on explainability/interpretability for the facebook/levit-128S model:

The attention bias maps from Equation 1 in the main paper provide insights into how the LeViT-128S model attends to different regions of the input image. By visualizing these 2D maps (Figure 6), we can observe the following:

- Some attention heads are quite uniform, while others specialize in nearby pixels (e.g., most heads of the shrinking attention). 
- Certain heads are directional, such as heads 1 and 4 of Stage 2/block 1 that handle pixels adjacent vertically and horizontally, respectively.
- Head 1 of stage 2, block 4 exhibits a specific period-2 pattern, possibly due to its output being fed to a subsampling filter in the next shrinking attention block.

These visualizations shed light on how the LeViT-128S model processes and attends to different spatial regions and patterns in the input images.

[More Information Needed] on other aspects of explainability and interpretability, such as feature visualizations, saliency maps, or ablation studies specifically for the LeViT-128S model.

## Environmental Impact

- **Hardware Type:** Based on the provided references, the LeViT model is trained on 32 GPUs for 1000 epochs, which takes 3 to 5 days. The specific GPU model is not mentioned.

For inference, the performance is evaluated on three hardware platforms:
1. One 16GB NVIDIA Volta GPU (peak performance is 12 TFLOP/s), which is mentioned as a typical training accelerator.
2. An Intel Xeon 6138 CPU at 2.0GHz, described as a typical server in a datacenter for feature extraction on streams of incoming images.
3. An ARM Graviton2 CPU (Amazon C6g instance), representing the type of processors found in mobile phones and other edge devices.

However, there is no specific information provided about the exact hardware used for training the facebook/levit-128S model. More details would be needed to accurately answer this question.
- **Software Type:** The model facebook/levit-128S is trained using PyTorch 1.7.0+ and torchvision 0.8.1+, as mentioned in the references:

"Then, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm
```"
- **Hours used:** Based on the provided references, the LeViT-128S model was trained for 100 epochs in the experiments to evaluate design changes relative to ViT/DeiT (Reference 6). The full 1000-epoch training schedule mentioned in Reference 1 was not specifically stated to be used for LeViT-128S.

Therefore, for the training time of the facebook/levit-128S model, [More Information Needed] on the exact number of epochs and hardware used in its final training.
- **Cloud Provider:** [More Information Needed]

The provided references do not specify the cloud provider used for training the facebook/levit-128S model. More information would be needed from the model development team or training logs to determine the specific cloud provider utilized.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the carbon emissions from training the LeViT-128S model. To answer this question, we would need details on the compute resources used for training (GPUs, CPUs, etc.), the duration of training, and the energy mix of the electricity used. Without this information, it is not possible to estimate the carbon footprint of the model training process.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
LeViT-128S integrates ResNet stages within the transformer architecture. Inside the stages, the architecture has a residual structure with alternating MLP and activation blocks. The attention blocks are modified compared to the classical setup. Each convolution is followed by batch normalization, with the batch norm weights joining residual connections initialized to zero. The batch norms can be merged with the preceding convolution for inference efficiency. A small convnet is applied to the input before the transformer stack to improve accuracy. The classification token is removed, using average pooling on the last activation map to produce the embedding for the classifier instead. Separate heads are trained for classification and distillation tasks. The non-linear activations used are Hardswish.

Objective: 
The objective of LeViT-128S is to provide a hybrid neural network architecture that achieves a strong speed/accuracy tradeoff for image classification, especially in the high-throughput inference regime. It aims to significantly outperform existing convnets and vision transformers in terms of inference speed at a given accuracy level.

[More Information Needed] on the exact details of the LeViT-128S model variant, such as the number and configuration of the computation stages.

### Compute Infrastructure

Based on the provided references, the compute infrastructure used for training the LeViT-128S model is as follows:

Hardware:
- The model was likely trained using 32 GPUs, as mentioned in reference 5 for the general LeViT training setup. However, specific details for LeViT-128S are not provided.
- Inference speed measurements were performed on three different hardware platforms (reference 8):
  1. One 16GB NVIDIA Volta GPU (peak performance is 12 TFLOP/s)
  2. An Intel Xeon 6138 CPU at 2.0GHz
  3. An ARM Graviton2 CPU (Amazon C6g instance)

Software:
- The experiments were modeled on the DeiT work and built upon PyTorch and the Timm library (reference 2).

Training:
- [More Information Needed] on the specific training details (e.g., number of epochs, batch size) for LeViT-128S.

Evaluation:
- The model was evaluated on the ImageNet-2012 validation set (reference 2).
- To evaluate a pre-trained LeViT-256 model on ImageNet val with a single GPU (reference 9):
```
python main.py --eval --model LeViT_256 --data-path /path/to/imagenet
```

[More Information Needed] on the exact compute infrastructure and training details specific to the LeViT-128S model, as the provided references mostly discuss the general LeViT architecture and training setup.

## Citation

```
@misc{benjamin-levit,
    author = {Benjamin Graham and
              Alaaeldin El-Nouby and
              Hugo Touvron and
              Pierre Stock and
              Armand Joulin and
              Hervé Jégou and
              Matthijs Douze},
    title  = {LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference},
    url    = {https://arxiv.org/pdf/2104.01136.pdf}
}
```

