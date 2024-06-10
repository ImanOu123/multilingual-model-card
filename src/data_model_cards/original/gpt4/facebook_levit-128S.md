# Model Card for facebook/levit-128S

The facebook/levit-128S is a hybrid neural network model that combines the architecture of vision transformers with convolutional-like features for fast inference image classification, achieving a significant speed/accuracy tradeoff improvement over existing models such as EfficientNet and Vision Transformers when evaluated on datasets like ImageNet.

## Model Details

### Model Description

Model Architecture:
The `facebook/levit-128S` model is a hybrid neural network that builds upon the Vision Transformer (ViT) architecture and the DeiT training method. It incorporates elements from convolutional architectures to enhance performance. The model removes the classification token used in ViT and instead applies average pooling on the last activation map to produce an embedding for the classifier. LeViT integrates ResNet stages within the transformer architecture, maintaining a residual structure with alternated MLP and activation blocks. The attention blocks in LeViT have been modified to include downsampling, where a shrinking attention block reduces the size of the activation map between stages. Each convolution in LeViT is followed by batch normalization, which can be merged with the preceding convolution for inference, offering a runtime advantage over layer normalization.

Training Procedures:
LeViT was trained using 32 GPUs for 1000 epochs, which takes approximately 3 to 5 days. This extended training period is typical for visual transformers and has been shown to improve performance. The model employs distillation-driven training with two classification heads, one receiving supervision from the ground truth classes and the other from a RegNetY-16GF model trained on ImageNet. The training time is primarily dominated by the teacher model's inference time.

Parameters:
The specific number of parameters for the `facebook/levit-128S` model is not provided in the references. However, it is implied that the model is designed to balance speed and precision, making it suitable for a range of hardware platforms and application scenarios.

Important Disclaimers:
The references do not provide explicit disclaimers about the `facebook/levit-128S` model. However, it is important to note that the performance of the model may vary based on the hardware used for inference and the specific application scenario. Additionally, the model's accuracy benefits significantly from the DeiT-like distillation when trained on ImageNet alone, which may not generalize to other datasets without similar training procedures.

[More Information Needed]:
For a complete model card, additional information such as the exact number of parameters, specific training hyperparameters, and any potential biases in the model would be required.

- **Developed by:** Benjamin Graham; Alaaeldin El-Nouby; Hugo Touvron; Pierre Stock; Armand Joulin; Hervé Jégou; Matthijs Douze
- **Funded by:** [More Information Needed]
- **Shared by:** The contributors that made the model facebook/levit-128S available online as a GitHub repo are Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, and Matthijs Douze.
- **Model type:** The model facebook/levit-128S is a hybrid neural network for fast inference image classification that utilizes a vision transformer architecture with distillation-driven training, suitable for supervised learning, and operates on the image modality.
- **Language(s):** The model facebook/levit-128S does not process natural human language; instead, it is designed for fast inference image classification tasks.
- **License:** The model facebook/levit-128S is released under the Apache 2.0 license. The link to the license can be found in the [LICENSE](LICENSE) file.
- **Finetuned from model:** The model facebook/levit-128S is fine-tuned from a RegNetY-16GF model trained on ImageNet. However, a direct link to the base model is not provided in the references above, so [More Information Needed] for the link to the base model.
### Model Sources

- **Repository:** https://github.com/facebookresearch/LeViT
- **Paper:** https://arxiv.org/pdf/2104.01136.pdf
- **Demo:** The link to the demo of the model facebook/levit-128S is not explicitly provided in the references above. However, there is a link to download the model weights:

[model](https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth)

For a demo, users would typically need to load these weights into a compatible deep learning framework and run inference on their data. Since a direct link to an interactive demo is not provided, the answer is:

[More Information Needed]
## Uses

### Direct Use

The model `facebook/levit-128S` can be used for image classification tasks without the need for fine-tuning, post-processing, or plugging it into a pipeline if you have a pre-trained version of the model. This is because the model has already been trained on the ImageNet 2012 dataset, which is a large and diverse dataset suitable for general image classification tasks.

To use the `facebook/levit-128S` model for classifying images, you would load the pre-trained model and pass your input images through it to obtain predictions. The model outputs the probabilities of each class, and you can take the class with the highest probability as the prediction.

Here is a code snippet that demonstrates how to use the `facebook/levit-128S` model for inference on a single image:

```python
from PIL import Image
import requests
from torchvision import transforms
from timm.models import create_model
import torch

# Load the pre-trained LeViT-128S model
model = create_model('LeViT_128S', pretrained=True)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),  # resize the image to 256x256
    transforms.CenterCrop(224),               # crop the image to 224x224
    transforms.ToTensor(),                    # convert the image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # normalize the image
])

# Load an image from the web
img_url = 'https://example.com/image.jpg'  # Replace with your image URL
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))

# Preprocess the image and add batch dimension
img = transform(img).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    outputs = model(img)

# Get the top-1 prediction
_, predicted = outputs.max(1)

# Print the predicted class index
print(f'Predicted class index: {predicted.item()}')
```

Please note that you need to replace `'https://example.com/image.jpg'` with the URL of the image you want to classify. Also, the model expects the input image to be preprocessed in a certain way (resize, crop, tensor conversion, normalization) as shown in the `transform` code block.

This code snippet assumes that you have the necessary libraries installed (`PIL`, `requests`, `torchvision`, `timm`, and `torch`) and that you are using a Python environment where these libraries are available. The `timm` library is used here to create the model with the `create_model` function, which should be set to `'LeViT_128S'` to load the correct model architecture with pre-trained weights.

### Downstream Use

The LeViT-128S model, as part of the LeViT family of models, is a vision transformer designed for efficient image classification tasks. When fine-tuning LeViT-128S for a specific task, users can leverage its pre-trained weights as a starting point, which can lead to better performance and faster convergence compared to training from scratch, especially when the available dataset for the new task is relatively small.

Fine-tuning the model involves the following steps:
1. Loading the pre-trained LeViT-128S model.
2. Replacing the classification head with a new head suitable for the target task (if the number of classes differs).
3. Optionally, freezing some of the earlier layers to retain learned features and only train the later layers or the new classification head.
4. Training the model on the new dataset with an appropriate loss function and optimization algorithm.

When integrating LeViT-128S into a larger ecosystem or app, the model can serve as a feature extractor for downstream tasks such as object detection, segmentation, or even in multimodal scenarios where image features are combined with text or other data types. The extracted features can be fed into additional layers or models designed for the specific application.

Here's a conceptual code snippet for fine-tuning the LeViT-128S model on a new dataset with a different number of classes (assuming the necessary libraries and the new dataset are already set up):

```python
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from levit import LeViT_128S  # Assuming LeViT_128S is a callable class from the cloned repository

# Load the pre-trained LeViT-128S model
model = LeViT_128S(pretrained=True)

# Replace the classification head with a new one for the target task
num_classes = 10  # Example: 10 classes for the new task
model.head = torch.nn.Linear(model.head.in_features, num_classes)

# Prepare the new dataset with appropriate transformations
transform = transforms.Compose([
    # Add necessary transformations here
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
train_dataset = datasets.ImageFolder('/path/to/train_dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Fine-tune the model
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
torch.save(model.state_dict(), '/path/to/save/fine_tuned_model.pth')
```

Please note that the actual implementation details may vary, and the above code is a simplified example. The actual `LeViT_128S` class should be imported from the repository you have cloned, and the dataset paths, transformations, and training loop details should be adjusted according to the specific task and dataset.

### Out-of-Scope Use

As the sociotechnic on the deep learning model development team, it is my responsibility to consider the potential for misuse of the facebook/levit-128S model and to advise users on practices to avoid. Here are some foreseeable misuses and guidance on what users should not do with the model:

1. **Biased Decision-Making**: The LeViT-128S model, like any machine learning model, may inadvertently learn and perpetuate biases present in the training data. Users should not employ this model in contexts where biased outputs could lead to discrimination or unfair treatment of individuals, particularly in sensitive areas such as employment, lending, and law enforcement.

2. **Surveillance**: Given the model's capabilities in image processing and its efficiency on CPUs as mentioned in reference 4, there is a risk of it being used in surveillance systems. Users should not use LeViT-128S for mass surveillance or any form of invasive monitoring that violates privacy rights and ethical norms.

3. **Deepfakes and Misinformation**: The model's efficiency in handling images could potentially be exploited to create deepfakes or to manipulate images in a way that spreads misinformation. Users should not use the model to create or disseminate deceptive media content.

4. **Intellectual Property Violations**: Users should not use the model to analyze or generate content in ways that infringe on copyrights, trademarks, or other intellectual property rights.

5. **Unethical Research**: While the model is made available for research purposes, as indicated by the distributed training available via Slurm and submitit, users should not use it for research that is not in compliance with ethical standards, including but not limited to research involving non-consensual data or studies lacking proper institutional review board (IRB) approval.

6. **Security Risks**: Users should not use the model in security-critical applications without a thorough evaluation of its robustness and reliability, as machine learning models can be susceptible to adversarial attacks and other forms of manipulation.

7. **Environmental Impact**: Users should be mindful of the environmental impact of training and running large deep learning models. While the LeViT-128S model is designed to be efficient, as noted in reference 6, users should not use it in a way that disproportionately contributes to carbon emissions without considering greener alternatives or offsets.

In conclusion, while the LeViT-128S model offers significant advantages in terms of speed and accuracy, it is crucial that users employ the model responsibly, adhering to ethical guidelines, respecting privacy, and avoiding applications that could cause harm or infringe upon the rights of individuals. Users are encouraged to consider the broader societal implications of their use of this technology and to engage with the model in a manner that promotes fairness, transparency, and accountability.

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the model facebook/levit-128S can be categorized into technical and sociotechnical limitations:

1. **Technical Limitations:**
   - **Reduced Model Complexity and Accuracy:** As mentioned in reference 2, reducing the pyramid shape of the network to a straight stack (like DeiT) and adjusting the network width to control computational complexity can lead to a significant drop in accuracy. This suggests that there is a trade-off between model complexity and performance, and the model may not perform well in all scenarios.
   - **Impact of Design Choices on Performance:** Reference 3 indicates that removing certain preprocessing convolutions can decrease the number of flops but also has a strong negative impact on accuracy. This highlights the importance of certain architectural components for the model's performance, and changes to these components can lead to suboptimal results.
   - **Hardware and Software Dependencies:** As per reference 7, the performance of LeViT is influenced by the underlying hardware and software. This means that the model's efficiency and effectiveness can vary across different platforms, which could limit its applicability in certain environments.

2. **Sociotechnical Limitations:**
   - **Potential for Misunderstanding Model Capabilities:** Users may have unrealistic expectations of the model's performance due to the trade-offs mentioned in references 2 and 3. It is important to communicate clearly about the model's limitations and the conditions under which it performs best.
   - **Ethical and Societal Impact:** While not directly mentioned in the references, any deep learning model, including LeViT-128S, can potentially perpetuate biases present in the training data. This can lead to unfair or discriminatory outcomes if not carefully monitored and addressed.
   - **Accessibility and Inclusivity:** The dependency on specific hardware and software configurations (reference 7) may limit the accessibility of the model to researchers and practitioners with fewer resources. This could exacerbate existing inequalities in the field of AI research and development.
   - **Transparency and Reproducibility:** The model card should include clear information about the training data, model architecture, and performance metrics to ensure transparency and reproducibility. Without this, there is a risk of misunderstandings about the model's capabilities and limitations.

In conclusion, while the LeViT-128S model shows promise in terms of speed-precision tradeoffs (reference 10), it is important to be aware of and communicate its technical and sociotechnical limitations to potential users and stakeholders. This includes acknowledging the trade-offs in model complexity and accuracy, the impact of hardware and software on performance, and the broader ethical and societal implications of deploying such a model.

### Recommendations

Based on the provided references, here are some recommendations with respect to the foreseeable issues about the model facebook/levit-128S:

1. **Computational Efficiency**: LeViT-128S is designed to maintain a balance between speed and accuracy. It is important to consider the computational resources available when deploying this model, as it uses fewer FLOPs compared to some other architectures (Reference 4). Users should ensure that their hardware is capable of efficiently running the model to leverage its design for fast inference.

2. **Model Generalization**: The model's performance is evaluated on specific datasets and under certain training conditions (Reference 1). Before deploying the model in different contexts or with different data distributions, it is advisable to conduct additional testing to ensure that the model generalizes well to new scenarios.

3. **Hardware Optimization**: The model's efficiency may vary across different hardware platforms (Reference 5). It is recommended to optimize the model for the specific hardware it will be deployed on to ensure the best speed-accuracy trade-off.

4. **Positional Information**: The LeViT architecture incorporates positional information, which is a significant factor in vision transformers (Reference 6). Users should be aware of how positional information is handled in the model and consider this when applying the model to tasks where positional context is crucial.

5. **Dataset Considerations**: The model's performance is benchmarked on certain datasets, and its accuracy is contingent on the quality and size of the pre-training dataset (Reference 7 and 8). Users should consider the characteristics of the datasets they intend to use and may need to fine-tune the model accordingly.

6. **Ethical and Societal Impact**: As a sociotechnic, it is important to consider the broader implications of deploying this model. This includes assessing the potential for biased outcomes if the training data is not representative of the application domain, and ensuring that the use of the model aligns with ethical guidelines and does not infringe on privacy rights.

7. **Transparency and Documentation**: To facilitate responsible use, it is recommended to provide thorough documentation on the model's capabilities, limitations, and appropriate use cases. This includes clear guidelines on how to interpret the model's predictions and any necessary steps to mitigate potential biases or errors.

8. **Continual Monitoring and Updating**: After deployment, the model should be continually monitored for performance degradation or unexpected behavior, especially as it encounters new data. Regular updates may be necessary to maintain its effectiveness and to address any issues that arise.

In summary, while the LeViT-128S model shows promising results in terms of efficiency and performance, it is important to consider the model's deployment context, optimize for specific hardware, ensure ethical use, and maintain transparency and ongoing monitoring to address any foreseeable issues.

## Training Details

### Training Data

The training data for the model facebook/levit-128S is the ImageNet-2012 dataset. The dataset is structured according to the standard layout expected by the torchvision `datasets.ImageFolder`, with separate `train/` and `val/` folders containing subfolders for each class and corresponding images. No additional data pre-processing or filtering steps are specified beyond the standard ImageNet preparation.

### Training Procedure

#### Preprocessing

The preprocessing for the data of the model facebook/levit-128S involves several steps to prepare the images for training the model. Here's a detailed description based on the provided references:

1. **Tokenization**: In the context of the LeViT model, tokenization refers to the process of converting input images into patches. This is analogous to tokenization in NLP, but instead of words, we deal with image patches. According to reference 1, ViT's patch extractor, which LeViT builds upon, uses a 16x16 convolution with a stride of 16 to extract patches from the input images. These patches are then linearly embedded to form the sequence of tokens that serve as input to the transformer model.

2. **Resizing/Rewriting**: The model does not use a traditional classification token as mentioned in reference 9. Instead, it relies on average pooling on the last activation map to produce an embedding for the classifier. This implies that the input images are resized to fit the model's expected input dimensions and then processed through a small convolutional network before being fed into the transformer stack. The exact resizing dimensions are not specified in the provided references, so [More Information Needed] for the specific resizing dimensions.

3. **Additional Preprocessing**: Reference 4 mentions that the smoothness in the attention masks is likely caused by data augmentation. This suggests that data augmentation techniques such as random translations might be part of the preprocessing pipeline to ensure the model learns spatial smoothness. However, the specific data augmentation techniques used are not detailed in the provided references, so [More Information Needed] for the exact data augmentation methods.

4. **Dataset Structure**: As per reference 8, the ImageNet dataset is used for training and validation, with the directory structure compatible with the torchvision `datasets.ImageFolder`. This indicates that the images are organized into subfolders representing each class for both training and validation sets.

5. **Distillation**: Reference 7 mentions that the model uses distillation during training, with two classification heads. One head is supervised with ground truth classes, and the other is supervised with outputs from a pre-trained RegNetY-16GF model. This distillation process is part of the training but not directly related to the initial preprocessing of the data.

In summary, the preprocessing for the LeViT-128S model involves extracting image patches using a 16x16 convolution with stride 16, possibly resizing the images to fit the model's input dimensions, applying data augmentation techniques for regularization, and organizing the dataset according to the expected directory structure for training and validation. Specific details on resizing dimensions and data augmentation methods are not provided in the references, so further information would be needed to complete those aspects of the preprocessing description.

#### Training Hyperparameters

The training hyperparameters for the model `facebook/levit-128S` are not fully detailed in the provided references. However, I can infer some of the training setup based on the information given:

1. **Number of GPUs**: The model is trained using 32 GPUs.
2. **Training Epochs**: The model undergoes 1000 training epochs, which is a longer schedule than usual for convolutional networks but is beneficial for visual transformers like LeViT.
3. **Regularization**: Distillation-driven training is used for regularization, similar to DeiT. This involves training with two classification heads, both using cross-entropy loss. One head is supervised by the ground truth classes, and the other is supervised by a RegNetY-16GF model trained on ImageNet.
4. **Training Command**: The command provided for training LeViT-256 with hard distillation suggests that the training is distributed across 8 GPUs on a single node. Although this command is for LeViT-256, it might be similar for LeViT-128S, with adjustments for the model specification.
5. **Activation Function**: For DeiT-Tiny, the GELU activation is replaced with Hardswish to reduce runtime. It is not explicitly stated that the same is done for LeViT-128S, but it could be inferred that similar optimizations might be considered.
6. **Training Variants**: Experiments with LeViT-128S involve training variants for 100 epochs to evaluate design changes, but this is not the full training schedule.

The exact values for other hyperparameters such as learning rate, batch size, weight decay, optimizer, and learning rate schedule are not provided in the references. Therefore, for those specifics, I must say: [More Information Needed].

#### Speeds, Sizes, Times

The LeViT-128S model is a part of the LeViT family, which is designed to offer a range of speed-accuracy tradeoffs. The model is characterized by its efficiency and performance, as highlighted by the following details:

- **Throughput and Efficiency**: LeViT-128S is designed to be highly efficient, offering a significant reduction in FLOPs compared to other models. It is on-par with DeiT-Tiny in accuracy but uses 4× fewer FLOPs, making it a very efficient choice for applications where computational resources are limited.

- **Speed-Accuracy Tradeoff**: The model achieves a favorable speed-accuracy tradeoff. For instance, at 80% ImageNet top-1 accuracy, LeViT is 5 times faster than EfficientNet on CPU. This demonstrates the model's capability to deliver high accuracy at a much faster rate, which is crucial for real-time applications.

- **Training Time**: The training of LeViT models, including LeViT-128S, is quite intensive. Using 32 GPUs, the 1000 training epochs can be completed in 3 to 5 days. This is longer than the usual training schedule for convolutional networks, but it is necessary for visual transformers to achieve high precision.

- **Checkpoint Sizes**: [More Information Needed]

- **Runtime Measurements**: The runtime of LeViT-128S is optimized, although it is noted that float32 operations are not as well optimized compared to Intel. Despite this, LeViT maintains a favorable speed-accuracy tradeoff.

- **Design and Runtime Analysis**: Detailed runtime analysis comparing LeViT blocks with DeiT blocks indicates that LeViT is 33% wider in terms of channels (C = 256 vs C = 192) at the first stage, which is the most computationally expensive part of the model. This suggests that LeViT-128S has been optimized for speed without compromising on width.

- **Training Details**: LeViT-128S, like other models in the LeViT family, uses distillation-driven training with two classification heads and a cross-entropy loss. The first head receives supervision from ground truth classes, and the second from a RegNetY-16GF model trained on ImageNet.

- **Evaluation**: To evaluate a pre-trained LeViT model, a command is provided, but it is specific to LeViT-256. For LeViT-128S, a similar command would be used with the appropriate model flag. The exact accuracy metrics for LeViT-128S are not provided in the references, so [More Information Needed] for the specific accuracy and loss values.

- **Code Availability**: The code for the LeViT models is available on GitHub, which would include the LeViT-128S model as well.

In summary, the LeViT-128S model is a highly efficient and fast model within the LeViT family, designed for applications requiring a good balance between speed and accuracy. However, for some specific details such as checkpoint sizes and exact evaluation metrics for LeViT-128S, more information is needed.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model facebook/levit-128S is evaluated on the ImageNet-2012 dataset.

#### Factors

The model facebook/levit-128S is a deep learning architecture designed for image classification tasks, building upon the Vision Transformer (ViT) and Data-efficient image Transformer (DeiT) methodologies. The following characteristics are foreseeable in influencing how the model behaves:

1. **Domain and Context**: The model has been trained on the ImageNet-2012 dataset, which is a large-scale dataset consisting of a wide variety of images across different categories. The performance of the model is therefore likely to be optimized for domains that have similar characteristics to the ImageNet dataset. For contexts that significantly deviate from the types of images or the distribution found in ImageNet, such as medical imaging or satellite imagery, the model may not perform as well without further fine-tuning or domain adaptation.

2. **Population Subgroups**: Since the model is evaluated on the ImageNet validation set, its performance may be biased towards the demographic distribution of subjects within that dataset. If ImageNet has underrepresentation or overrepresentation of certain population subgroups, this could lead to disparities in model performance across different demographics. [More Information Needed] to make specific claims about the representation of population subgroups within ImageNet and how that might affect the model's performance.

3. **Evaluation Disaggregation**: The references do not provide detailed information on disaggregated evaluation across factors such as age, gender, or ethnicity, which would be necessary to uncover disparities in performance. [More Information Needed] to determine if such disaggregated evaluation has been conducted and what the results were.

4. **Speed-Precision Tradeoffs**: The model is designed to offer a range of speed-accuracy tradeoffs, which means that its performance can be tuned according to the computational resources available and the requirements of the task at hand. For example, LeViT-128S is designed to be faster but may trade off some accuracy compared to larger models like LeViT-384.

5. **Design Choices**: The model's architecture incorporates certain design choices such as the removal of the classification token and the use of average pooling on the last activation map. These choices are likely to influence the model's behavior, particularly in how it processes and classifies images.

6. **Computational Complexity**: The model's design aims to control computational complexity, which is a critical factor in deployment contexts where resources are limited. The use of a pyramid shape and the reduction of resolution in LeViT are tools to keep computational complexity under control, which may affect the model's performance on high-resolution images or tasks requiring fine-grained detail.

In summary, the model's behavior is influenced by the domain and context of the application, the characteristics of the population subgroups present in the training data, and the specific design and architectural choices made during development. To fully understand the model's performance across different subgroups and factors, further evaluation and analysis would be required.

#### Metrics

The evaluation of the model `facebook/levit-128S` will primarily focus on the following metrics:

1. **Accuracy**: This is measured by the top-1 and top-5 accuracy on the ImageNet validation set, as indicated by the reference to the pre-trained LeViT-256 model's performance. Accuracy is a direct measure of the model's ability to correctly classify images.

2. **FLOPs (Floating Point Operations Per Second)**: The model's efficiency is evaluated in terms of the number of floating-point operations required for a single forward pass. LeViT-128S is highlighted for its efficiency, using 4× fewer FLOPs compared to DeiT-Tiny for a similar level of accuracy.

3. **Speed**: The inference speed is considered, especially in terms of the speed-accuracy tradeoff. The model's runtime measurements are important, and LeViT is noted to be faster than competitive architectures like EfficientNet and DeiT when achieving similar accuracy levels.

4. **Loss**: The model's performance is also evaluated using the loss metric, as shown in the example command for evaluating a LeViT-256 model, which outputs accuracy and loss values.

These metrics are used to demonstrate the tradeoffs between different errors and to position LeViT-128S in the context of the speed-precision tradeoff landscape. The model aims to achieve high accuracy with fewer computational resources and faster inference times, making it suitable for applications where both precision and speed are critical.

### Results

The evaluation results of the model facebook/levit-128S are not explicitly detailed in the provided references. However, we can infer some information based on the context given:

1. The model is trained on the ImageNet-2012 dataset and evaluated on its validation set, following the experimental setup of the DeiT work.
2. The model benefits from DeiT-like distillation, which significantly improves its accuracy when trained solely on ImageNet.
3. The LeViT-128S model is part of a series of experiments designed to evaluate design changes relative to ViT/DeiT, with a focus on the speed/accuracy tradeoff.
4. The model is specialized in the high-throughput regime, aiming for fast inference times while maintaining competitive accuracy.
5. The model is compared with other transformer-based architectures, emphasizing its efficiency in terms of FLOPs and inference speed.
6. The model is designed to be faster than EfficientNet at comparable accuracy levels, specifically being 5 times faster at 80% ImageNet top-1 accuracy on CPU.

For specific metrics such as top-1 accuracy, top-5 accuracy, loss, number of parameters, FLOPs, and inference speed, the references do not provide direct figures for the LeViT-128S model. Therefore, for these exact metrics, [More Information Needed].

#### Summary

The LeViT-128S model is a part of the LeViT family, which is a transformer-based architecture designed for efficient image classification. The model has been trained on the ImageNet-2012 dataset and evaluated on its validation set. The evaluation results for LeViT-128S specifically are not directly provided in the references, so [More Information Needed] for the exact performance metrics of LeViT-128S.

However, we can infer from the references that LeViT models, in general, benefit from a DeiT-like distillation process, which significantly improves their accuracy when trained solely on ImageNet. The LeViT models are also designed to offer a favorable speed/accuracy tradeoff, being 5 times faster than EfficientNet at comparable accuracy levels on CPU.

In terms of design experiments, the references mention that the LeViT-128S model and its variants were trained with only 100 training epochs to highlight the differences in design changes relative to ViT/DeiT and to reduce training time. The conclusions drawn from these experiments are expected to hold for larger models and longer training schedules.

For a more detailed evaluation of the LeViT-128S model, including specific accuracy and speed metrics, [More Information Needed].

## Model Examination

In the development of the LeViT-128S model, our team has prioritized explainability and interpretability to ensure that users can understand and trust the model's predictions. Here are some key points regarding the explainability of the LeViT-128S model:

1. **Convergence Analysis**: As observed in Figure 3, the convergence behavior of LeViT-128S provides insights into the learning dynamics of the model. Initially, the model converges similarly to a convolutional neural network (convnet), benefiting from the strong inductive biases of convolutional layers. This suggests that the early layers of the model are efficient at capturing low-level features. As training progresses, the convergence rate shifts to resemble that of DeiT-S, indicating a transition to the learning characteristics of transformers. This dual convergence pattern can be interpreted as the model leveraging both convolutional and transformer strengths throughout training.

2. **Ablation Studies**: By conducting ablation studies where we modify one component at a time, we have gained a deeper understanding of the contributions of different architectural elements to the model's performance. For instance, the removal of the pyramid shape in favor of a straight stack of attention and MLPs, akin to DeiT, resulted in a significant drop in accuracy. This highlights the importance of resolution reduction in LeViT for controlling computational complexity without sacrificing performance.

3. **Grafted Architecture**: The LeViT-128S model incorporates a grafted architecture that combines elements of both DeiT and ResNet-50. This design choice is validated by the improved results over the individual architectures, as summarized in Table 1. The hybrid nature of LeViT allows for a more interpretable structure where the contributions of convolutional and transformer components can be separately analyzed.

4. **Patch Embedding and Positional Information**: The model's use of patch embedding and the removal of the classification token in favor of average pooling on the last activation map provide a more traditional BCHW tensor format. This design choice aligns with common practices in convolutional networks and may offer a more intuitive understanding of how the model processes spatial information.

5. **Efficiency Measures**: LeViT-128S has been evaluated for efficiency on different hardware platforms, reflecting a range of application scenarios. This consideration of efficiency across platforms contributes to the interpretability of the model's performance in real-world settings.

6. **Distillation Heads**: For distillation during training, separate heads for classification and distillation tasks are employed. This allows for a clearer interpretation of how knowledge is transferred within the model and how it impacts the learning process.

Overall, the LeViT-128S model's design and evaluation have been conducted with an emphasis on explainability, from its convergence behavior and ablation studies to its architectural choices and efficiency considerations. These aspects contribute to a better understanding of the model's operation and decision-making process.

## Environmental Impact

- **Hardware Type:** The model facebook/levit-128S is trained on a hardware setup that includes 32 GPUs. The specific type of GPU used is an NVIDIA Volta GPU with 16GB of memory and a peak performance of 12 TFLOP/s. This is mentioned as a typical training accelerator. Additionally, the CPUs used in the training process include an Intel Xeon 6138 CPU at 2.0GHz, which is typical for a server in a datacenter, and an ARM Graviton2 CPU, which is a model for processors in mobile phones and other edge devices.
- **Software Type:** The model facebook/levit-128S is trained on a software environment that includes PyTorch, as indicated by the Python command provided for training the model (`python -m torch.distributed.launch`). Additionally, distributed training is facilitated by Slurm and `submitit`, a Python 3 package for submitting jobs to Slurm.
- **Hours used:** The training time for the model facebook/levit-128S is not explicitly stated in the provided references. However, it is mentioned that 32 GPUs perform the 1000 training epochs in 3 to 5 days for the LeViT models in general. Since the LeViT-128S is a variant of the LeViT models, it could be inferred that its training time would be within a similar range if it also underwent 1000 epochs of training. However, for the specific experiments to evaluate design changes, it is mentioned that only 100 training epochs are used to magnify differences and reduce training time. Therefore, without explicit information on the exact number of epochs and the specific hardware configuration used for the facebook/levit-128S model, we cannot provide an accurate training time. 

[More Information Needed]
- **Cloud Provider:** The cloud provider that the model facebook/levit-128S is trained on is not explicitly mentioned in the provided references. However, reference 4 mentions an ARM Graviton2 CPU, which is associated with Amazon Web Services (AWS) as the Amazon C6g instance. This suggests that AWS might be the cloud provider used for training the model.
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model `facebook/levit-128S` is a hybrid neural network architecture designed for fast inference in image classification tasks. It builds upon the Vision Transformer (ViT) architecture and incorporates training methods from Data-efficient Image Transformer (DeiT). The model integrates ResNet stages within the transformer architecture, with modifications to the attention blocks to suit visual processing.

Key architectural features of LeViT include:

1. **Residual Structure**: Similar to a visual transformer, LeViT employs a residual structure with alternating Multi-Layer Perceptron (MLP) and activation blocks.

2. **Normalization and Activations**: Unlike ViT, which uses layer normalization, LeViT uses batch normalization after each convolution. The batch normalization weights that connect with a residual connection are initialized to zero, following previous research. This allows for merging batch normalization with the preceding convolution during inference, offering a runtime advantage.

3. **Patch Embedding**: LeViT applies a small convolutional network to the input before it enters the transformer stack, which has been shown to improve accuracy. It does not use a classification token; instead, it uses average pooling on the last activation map to produce an embedding for the classifier.

4. **Downsampling**: LeViT uses shrinking attention blocks between stages to reduce the size of the activation map. This involves subsampling before the Q transformation in the attention block, resulting in a reduced spatial dimension of the output tensor.

5. **Attention and MLP Blocks**: Each stage in the LeViT model consists of several pairs of Attention and MLP blocks. The model uses drop path regularization with a certain probability on each residual connection.

6. **Positional Information**: LeViT addresses the need for positional information in vision transformers, which is crucial for image classification tasks.

7. **Efficiency**: LeViT is designed with efficiency in mind, offering a significant speed/accuracy tradeoff advantage. For instance, at 80% ImageNet top-1 accuracy, LeViT is reported to be five times faster than EfficientNet on CPU.

The objective of the `facebook/levit-128S` model is to provide a highly efficient architecture for image classification that can deliver fast inference speeds while maintaining high accuracy, making it suitable for a wide range of hardware platforms and application scenarios. The model leverages the strengths of both convolutional networks and transformers to achieve this goal.

### Compute Infrastructure

The compute infrastructure used for training the model facebook/levit-128S includes the following:

1. GPUs: The training was performed using NVIDIA Volta GPUs, with each GPU having 16GB of memory and a peak performance of 12 TFLOP/s. This type of GPU is typical for training deep learning models.

2. CPUs: An Intel Xeon 6138 CPU at 2.0GHz was used, which is a common server configuration in data centers for performing feature extraction on streams of incoming images. The setup is optimized for PyTorch, utilizing MKL and AVX2 instructions.

3. Distributed Training: The model was trained using distributed training across multiple GPUs. Specifically, 32 GPUs were used to complete 1000 training epochs in 3 to 5 days. Distributed training was facilitated by Slurm and the `submitit` Python package.

4. ARM CPU: An ARM Graviton2 CPU, which is representative of the processors found in mobile phones and other edge devices, was also mentioned as a reference hardware for timing inference.

5. Software: The experiments were conducted using PyTorch, and the model was trained on the ImageNet-2012 dataset. The Timm library was also utilized in the training process.

6. Training Command: To train a similar model, LeViT-256, on a single node with 8 GPUs, the following command was provided:

   ```
   python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model LeViT_256 --data-path /path/to/imagenet --output_dir /path/to/save
   ```

   Although this command is for LeViT-256, it gives an idea of the training setup that might have been used for LeViT-128S.

7. Evaluation: The evaluation of the pre-trained LeViT-256 model on the ImageNet validation set was performed using a single GPU, which achieved an accuracy of 81.636% (top-1) and 95.424% (top-5).

Please note that while the references provided discuss the LeViT-256 model, the question asks about the LeViT-128S model. The infrastructure details for LeViT-128S are expected to be similar to those provided for LeViT-256, but specific details about LeViT-128S are not provided in the references. Therefore, for the exact compute infrastructure of LeViT-128S, [More Information Needed].

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

