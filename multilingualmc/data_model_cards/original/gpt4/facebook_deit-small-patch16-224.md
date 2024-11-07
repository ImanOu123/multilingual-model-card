# Model Card for facebook/deit-small-patch16-224

The model facebook/deit-small-patch16-224 is a data-efficient image transformer (DeiT) that achieves competitive image classification performance using only ImageNet for training, without the need for large-scale private datasets. It incorporates a novel teacher-student strategy with a distillation token, enabling it to learn effectively from a convolutional network teacher and close the performance gap with state-of-the-art convolutional networks.

## Model Details

### Model Description

Model Name: facebook/deit-small-patch16-224

### Model Architecture
The facebook/deit-small-patch16-224 model is based on the Vision Transformer (ViT) architecture, which processes images as sequences of fixed-size patches. Specifically, the input RGB image is divided into 14x14 patches, each of 16x16 pixels, resulting in 196 patches. Each patch is linearly projected to a 768-dimensional embedding space. The model includes a class token, similar to the approach used in NLP, which is appended to the patch tokens and processed through the transformer layers. The final prediction is made using this class token after it is projected with a linear layer. The model also incorporates a distillation token, which is used in conjunction with a teacher model to improve performance through knowledge distillation.

### Training Procedures
The DeiT model is trained using PyTorch and the timm library. It employs a data-efficient training strategy, which includes the use of a lower training resolution and subsequent fine-tuning at a higher resolution. This approach speeds up training and enhances accuracy with common data augmentation techniques. The model can be trained on 4 GPUs in less than 3 days, with optional fine-tuning at a larger resolution taking an additional 20 hours on 8 GPUs.

### Parameters
The DeiT-small model is a smaller variant of the DeiT architecture, with fewer parameters compared to larger versions like DeiT-base. The exact number of parameters for the DeiT-small model is not specified in the provided references, but it is designed to be more efficient in terms of memory and computational requirements while maintaining competitive accuracy.

### Important Disclaimers
The DeiT model is designed to work well even with limited data, challenging the notion that transformers require large datasets to generalize effectively. However, it is important to note that the performance of the model may still depend on the nature and quality of the training data. Additionally, while the model aims to be on par with convolutional neural networks (CNNs) in terms of accuracy and has a lower memory footprint, users should consider the specific requirements and constraints of their applications.

### Additional Information
The open-source implementation of the DeiT model is available on GitHub, allowing users to access the code and pre-trained models for their own use and further development.

[More Information Needed] on the exact number of parameters for the DeiT-small model and other specific hyperparameters not detailed in the provided references.

- **Developed by:** Hugo Touvron; Matthieu Cord; Matthijs Douze; Francisco Massa; Alexandre Sablayrolles; Hervé Jégou; Facebook Ai
- **Funded by:** The project for the model facebook/deit-small-patch16-224 appears to be funded by Facebook AI, as indicated by the authors' affiliations and acknowledgments in the references provided. Specifically, Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou are mentioned in conjunction with Facebook AI. Additionally, colleagues at Facebook, such as Ross Wightman, Vinicius Reis, Mannat Singh, Ari Morcos, Mark Tygert, Gabriel Synnaeve, Ross Girshick, and Piotr Dollar, are thanked for their contributions and feedback, which implies their involvement and support from the organization. There is no mention of external funding sources in the provided references.
- **Shared by:** The contributors that made the model facebook/deit-small-patch16-224 available online as a GitHub repo include Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou from Facebook AI. Additionally, the acknowledgments thank Ross Wightman for sharing his ViT code and bootstrapping training method, as well as providing valuable feedback. Vinicius Reis, Mannat Singh, Ari Morcos, Mark Tygert, Gabriel Synnaeve, and other colleagues at Facebook were also mentioned for brainstorming and exploration. Ross Girshick and Piotr Dollar were acknowledged for constructive comments.
- **Model type:** The model facebook/deit-small-patch16-224 is a data-efficient image transformer trained using a distillation strategy from a strong teacher model, employing both hard and soft label distillation, and is designed for image classification tasks.
- **Language(s):** The model facebook/deit-small-patch16-224 does not process natural human language; instead, it is designed for image classification tasks.
- **License:** The model `facebook/deit-small-patch16-224` is released under the Apache 2.0 license. The license can be found in the [LICENSE](https://github.com/facebookresearch/deit/blob/main/LICENSE) file in the repository.
- **Finetuned from model:** The model `facebook/deit-small-patch16-224` is fine-tuned from a base model that is not explicitly named in the provided references. However, it is mentioned that DeiT models benefit from distillation from a "relatively weaker RegNetY" to produce `DeiT⚗`, which suggests that a RegNetY model may have been used as a teacher model for distillation. Unfortunately, without a specific name or link to the exact base model used for fine-tuning `facebook/deit-small-patch16-224`, I can only provide the general information that a RegNetY model was involved in the process.

For the exact base model name and link, [More Information Needed].
### Model Sources

- **Repository:** https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/datasets.py
- **Paper:** https://arxiv.org/pdf/2012.12877.pdf
- **Demo:** The link to the demo of the model facebook/deit-small-patch16-224 is not explicitly provided in the references above. Therefore, the answer is "[More Information Needed]".
## Uses

### Direct Use

The model `facebook/deit-small-patch16-224` can be used without fine-tuning, post-processing, or plugging into a pipeline for image classification tasks. It has been pre-trained on ImageNet and can be directly used to classify images into 1000 categories that were part of the ImageNet dataset. Here's how you can use the model in Python with the help of the `transformers` library from Hugging Face:

```python
from transformers import DeiTFeatureExtractor, DeiTForImageClassification
from PIL import Image
import requests

# Load the feature extractor and model from Hugging Face
feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-small-patch16-224')
model = DeiTForImageClassification.from_pretrained('facebook/deit-small-patch16-224')

# Load an image from the web or local file system
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess the image and prepare for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# Forward pass, get the logits/predictions
outputs = model(**inputs)
logits = outputs.logits

# Retrieve the highest probability class
predicted_class_idx = logits.argmax(-1).item()

# Convert the predicted index to label
predicted_label = model.config.id2label[predicted_class_idx]

print("Predicted class:", predicted_label)
```

This code snippet will download an image from the internet, preprocess it using the `DeiTFeatureExtractor`, and then use the `DeiTForImageClassification` model to predict the class of the image. The model outputs the logits, and the class with the highest logit is considered the predicted class. The `id2label` attribute of the model configuration is used to convert the predicted index to a human-readable label.

Please note that the actual performance of the model may vary depending on the similarity of the input images to the ImageNet dataset. For images that are significantly different from the ImageNet classes or for specialized domains, fine-tuning or additional post-processing might be necessary to achieve optimal results.

### Downstream Use

The `facebook/deit-small-patch16-224` model is a vision transformer that has been trained on ImageNet and can be fine-tuned for various image classification tasks. When fine-tuning the model for a specific task, you would typically start with the pre-trained weights and continue training on a dataset that is specific to your task. This allows the model to transfer the knowledge it has gained from the large and diverse ImageNet dataset to the new task, which can be beneficial especially when the new dataset is relatively small or when the task is similar to general image classification.

In a larger ecosystem or app, `facebook/deit-small-patch16-224` can be used as a feature extractor where the representations learned by the model are used as inputs for downstream tasks, such as object detection or segmentation. Alternatively, it can be integrated as an end-to-end solution for image classification where users can input an image and receive a classification result.

The model has been shown to perform well when fine-tuned on other datasets, as mentioned in reference 1, and it has been trained in a data-efficient manner with strong data augmentation strategies, as described in references 6 and 7. This suggests that the model should generalize well to new tasks and datasets.

When fine-tuning or using the model in an application, you would typically adjust the resolution of the input images according to the task at hand, as discussed in references 8, 9, and 10. The model has been trained at a resolution of 224x224 and fine-tuned at 384x384, but it can be fine-tuned at different resolutions if needed.

Here is a conceptual code snippet for fine-tuning the `facebook/deit-small-patch16-224` model using PyTorch and the `timm` library, assuming you have a dataset ready for fine-tuning:

```python
import timm
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load the pre-trained DeiT model
model = timm.create_model('deit_small_patch16_224', pretrained=True)

# Modify the classifier head for the new task
num_classes = 100  # Example: 100 classes for the new task
model.head = nn.Linear(model.head.in_features, num_classes)

# Prepare your dataset and dataloader
transform = transforms.Compose([
    # Add any specific transformations for your dataset
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # ... other transformations like normalization
])
train_dataset = datasets.ImageFolder('path_to_train_dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Fine-tuning loop
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the fine-tuned model
torch.save(model.state_dict(), 'deit_small_finetuned.pth')
```

Please note that this code is a simplified example and does not include all the necessary components such as validation, learning rate scheduling, or advanced data augmentation techniques that were mentioned in the references. Additionally, the actual hyperparameters and training strategies should be adapted based on the specific requirements of the task and the dataset being used.

### Out-of-Scope Use

As a sociotechnic member of the deep learning model development team, it is my responsibility to consider the potential societal impacts and misuse of the technology we develop. Regarding the model facebook/deit-small-patch16-224, there are several areas where misuse could occur:

1. **Privacy Violations**: The model has the capability to process and classify images, which could include personal or sensitive content. Users should not use this model to analyze images without the consent of the individuals depicted, especially in jurisdictions where privacy laws govern the use of personal data.

2. **Surveillance**: The technology could be employed in surveillance systems to monitor individuals without their knowledge or consent, which could lead to a breach of privacy rights and civil liberties.

3. **Deepfakes and Misinformation**: The model's ability to understand and process visual information could potentially be used to create or propagate deepfakes or other forms of visual misinformation, which could have serious implications for personal reputation, security, and democratic processes.

4. **Bias and Discrimination**: While not explicitly stated in the references, any machine learning model, including image classifiers, can inadvertently perpetuate or amplify societal biases if not carefully trained and evaluated. Users should not use the model in applications where biased predictions could lead to discriminatory outcomes.

5. **Intellectual Property Infringement**: The model should not be used to analyze or classify images in ways that infringe on intellectual property rights, such as automated content recognition for copyrighted material without proper authorization.

6. **Security**: Given the open-source nature of the model, as mentioned in reference 4, users should ensure that they follow best practices for security to prevent unauthorized access and misuse of the model.

7. **Compliance with Laws and Regulations**: Users must comply with all applicable laws and regulations, including those related to data protection, such as GDPR, and other sector-specific regulations when using the model.

In conclusion, while the model facebook/deit-small-patch16-224 represents a significant advancement in image classification technology, it is crucial that users employ the model ethically and responsibly. Misuse of the technology can have far-reaching negative consequences, and it is the responsibility of all users to ensure that their applications of the model do not harm individuals or society at large. Users should refer to the [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) and [CONTRIBUTING.md](.github/CONTRIBUTING.md) for more information on ethical guidelines and best practices.

### Bias, Risks, and Limitations

The model card for the `facebook/deit-small-patch16-224` should address the following known and foreseeable issues:

1. **Data Efficiency and Hyperparameters**: The DeiT model has been designed to learn vision transformers in a data-efficient manner. However, as indicated in reference 3, if the model does not train well, it could be due to hyperparameters not being adapted. This suggests that the model might require careful tuning of hyperparameters for optimal performance, which could be a technical limitation in scenarios where such tuning is not feasible.

2. **Distillation Strategy**: The model employs a distillation strategy as mentioned in reference 2. While this can improve performance, it also introduces complexity in the training process. If not properly understood or implemented, it could lead to suboptimal results or even training failures.

3. **Resolution Sensitivity**: As per reference 4, the model is trained at a resolution of 224×224 and fine-tuned at 384×384. This indicates that the model's performance might be sensitive to the input resolution, and using resolutions other than those specified could lead to decreased accuracy or require additional fine-tuning.

4. **Weight Initialization**: Reference 5 mentions following a recommendation to initialize weights with a truncated normal distribution. Incorrect initialization could lead to convergence issues, which is a technical limitation that needs to be managed.

5. **Generalization to Other Datasets**: Reference 10 highlights the importance of evaluating the model on other datasets with transfer learning to measure its generalization power. While DeiT performs well on ImageNet, its performance on other datasets may vary, and this could be a limitation for applications requiring robustness across diverse data.

6. **Comparison with Other Architectures**: According to reference 9, DeiT is slightly below EfficientNet in performance, indicating that there may still be a gap in certain aspects of performance compared to state-of-the-art models. Users should be aware of this when choosing a model for their specific needs.

7. **Sociotechnical Considerations**: The model card should also include sociotechnical considerations such as the potential for misuse, biases in the training data, and the ethical implications of deployment in various contexts. However, the provided references do not give specific information on these aspects, so [More Information Needed] to address these concerns comprehensively.

8. **Licensing and Contribution**: Reference 8 mentions that the repository is released under the Apache 2.0 license, which allows for broad use with few restrictions. Reference 7 encourages contributions, which implies that the model is expected to evolve over time, potentially addressing some of its current limitations.

9. **Acknowledgments and Collaborative Efforts**: Reference 6 acknowledges the collaborative efforts and contributions from various individuals and highlights the importance of community feedback in improving the model. This suggests that the model's development is an ongoing process and may continue to evolve, potentially addressing some of the known issues.

In summary, while the `facebook/deit-small-patch16-224` model shows promising results, there are technical limitations related to hyperparameter tuning, distillation strategy, resolution sensitivity, and weight initialization that could affect its performance. Additionally, there are foreseeable issues related to generalization across different datasets and a slight performance gap compared to some other architectures. Sociotechnical considerations are not explicitly discussed in the provided references, indicating a need for further information to fully understand the potential impacts of the model's deployment in society.

### Recommendations

Based on the provided references, here are some recommendations with respect to the foreseeable issues about the model `facebook/deit-small-patch16-224`:

1. **Data Efficiency and Generalization**: While DeiT models have shown promising results on ImageNet, it is crucial to evaluate their performance on a diverse set of datasets to ensure robust generalization. Researchers and practitioners should consider fine-tuning the model on domain-specific datasets to assess its transfer learning capabilities and to mitigate potential overfitting to ImageNet-specific features.

2. **Distillation Strategy**: The use of a transformer-specific distillation strategy has been beneficial for the DeiT models. However, the choice of teacher model can significantly impact the performance of the distilled student model. It is recommended to carefully select appropriate teacher models that align with the desired characteristics of the student model, such as accuracy and inference speed.

3. **Inference Efficiency**: DeiT models have demonstrated a good trade-off between accuracy and inference time on GPU. However, the throughput can vary based on the implementation. Users should ensure that the deployment environment is optimized for the model to achieve the reported throughput rates. Additionally, for applications where inference speed is critical, further optimization techniques may be necessary.

4. **Memory Footprint**: The lower memory footprint of image transformers compared to convnets is highlighted as an advantage. For applications with memory constraints, it is recommended to consider the memory efficiency of the DeiT model and potentially explore model quantization or pruning techniques to further reduce the memory requirements without significantly compromising accuracy.

5. **Open Source Implementation**: The model and its training strategy are open-sourced, which is beneficial for transparency and reproducibility. Users are encouraged to refer to the official implementation and documentation for guidance on training, fine-tuning, and deploying the model.

6. **Ethical and Societal Considerations**: While not explicitly mentioned in the references, it is important to consider the ethical implications of deploying machine learning models. Users should be aware of potential biases in the training data and the impact of these biases on the model's predictions. Efforts should be made to ensure that the model is used responsibly and does not perpetuate or exacerbate existing societal inequalities.

7. **Continuous Monitoring and Updating**: As with any machine learning model, the performance of DeiT models may change over time as new data becomes available and as the model is exposed to different environments. It is recommended to continuously monitor the model's performance and update it as necessary to maintain its accuracy and relevance.

In conclusion, while the `facebook/deit-small-patch16-224` model shows promising results, it is important to consider these recommendations to address potential issues and ensure the model's effectiveness and responsible use in real-world applications.

## Training Details

### Training Data

The training data for the model facebook/deit-small-patch16-224 consists of image datasets augmented with strong data augmentation techniques such as Rand-Augment and random erasing, as transformers require extensive data to achieve data-efficient training. The model is initially trained at a resolution of 224×224 and then fine-tuned at a resolution of 384×384. [More Information Needed] on the specific datasets used for training as it is not explicitly mentioned in the provided references.

### Training Procedure

#### Preprocessing

The preprocessing for the data of the model `facebook/deit-small-patch16-224` involves the following steps:

1. **Tokenization of Images**: The input images are treated as a sequence of tokens. Each fixed-size RGB image is decomposed into a batch of N patches, each of size 16x16 pixels. This results in N = 14x14 patches for an input image of size 224x224 pixels. Each patch is flattened and linearly projected to a dimension of D=768, which is the dimensionality of the transformer model's input tokens.

2. **Class Token**: A trainable class token is appended to the sequence of patch tokens before the first transformer layer. This class token is used to predict the class after being processed by the transformer layers.

3. **Positional Encoding**: Since the transformer architecture is permutation-invariant and does not inherently consider the order of the input tokens, positional information is added to the patch tokens. This is done using either fixed or trainable positional embeddings, which are added to the patch tokens before they are fed into the transformer blocks.

4. **Data Augmentation**: To train the model in a data-efficient manner, extensive data augmentation is employed. Techniques such as Rand-Augment and random erasing are used to improve the robustness and generalization of the model. These augmentations are applied to the input images before tokenization.

5. **Resolution Handling**: When training at a lower resolution and fine-tuning at a higher resolution, the positional encodings are adapted accordingly. The patch size remains the same, but the number of patches N changes with the resolution. Dosovitskiy et al. interpolate the positional encoding when changing the resolution to accommodate the increased number of patches.

6. **Exclusion of Dropout**: Dropout is not used in the training procedure, as it was found to be less beneficial for the transformer model in the context of the data augmentation strategies employed.

The preprocessing steps are designed to adapt the transformer architecture, originally used for NLP tasks, to handle image data efficiently. The model benefits from strong data augmentation and careful handling of positional information to achieve competitive performance in image classification tasks.

#### Training Hyperparameters

The training hyperparameters for the model `facebook/deit-small-patch16-224` are as follows:

- **Image Resolution**: The model is trained with an image resolution of 224 and fine-tuned at a resolution of 384, as per the default setup similar to ViT [Reference 2].
- **Data Augmentation**: The model employs strong data augmentation techniques, including Rand-Augment and random erasing, as confirmed by ablation studies to be useful for transformer models. Auto-Augment was considered but Rand-Augment was chosen instead [Reference 4].
- **Weight Initialization**: The weights are initialized with a truncated normal distribution, following the recommendation of Hanin and Rolnick [Reference 5].
- **Distillation Parameters**: For distillation, the typical values of τ = 3.0 and λ = 0.1 are used for soft distillation, following the recommendations from Cho et al. [Reference 5].
- **Positional Embeddings**: During fine-tuning at different resolutions, the positional embeddings are interpolated using classical image scaling techniques like bilinear interpolation [Reference 3].
- **Optimization and Regularization**: The schedule, regularization, and optimization procedure are identical to that of FixEfficientNet, but with the training-time data augmentation maintained [Reference 3].

For more specific hyperparameters such as learning rate, batch size, optimizer type, and other details, [More Information Needed] as they are not explicitly mentioned in the provided references. The open-source implementation provided at the GitHub repository may contain the exact configuration used for training.

#### Speeds, Sizes, Times

The model `facebook/deit-small-patch16-224` is a vision transformer that has been trained to efficiently process images with high throughput and accuracy. Here are the details based on the provided references:

Throughput: The throughput of our models, including `facebook/deit-small-patch16-224`, is measured in terms of the number of images processed per second on a 16GB V100 GPU. While the exact throughput for the `deit-small-patch16-224` model is not provided in the references, it is mentioned that throughput can vary according to the implementation and is calculated based on the largest possible batch size for the model's resolution, averaged over 30 runs (Reference 1). For a direct comparison of throughput, please refer to the specific benchmark results, which are not included in the provided references. [More Information Needed]

Start or End Time: The references mention that a typical training of 300 epochs for the larger DeiT-B model takes 37 hours with 2 nodes or 53 hours on a single node (Reference 5). However, the exact start or end time for the training of the `deit-small-patch16-224` model is not specified. [More Information Needed]

Checkpoint Sizes: The references do not provide specific information about the checkpoint sizes for the `facebook/deit-small-patch16-224` model. [More Information Needed]

Additional details about the model include its training strategy, which builds upon PyTorch and the timm library, and the model's ability to be fine-tuned at different resolutions (Reference 6, 7). The model also benefits from a distillation method that allows it to achieve a competitive trade-off between accuracy and throughput (Reference 10). The `deit-small-patch16-224` model processes input images as a sequence of 16x16 pixel patches, with each patch projected with a linear layer to maintain its dimension (Reference 11).

For more specific details regarding throughput, training times, and checkpoint sizes, one would need to refer to the actual benchmark results or the model's repository, which may contain this information.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model facebook/deit-small-patch16-224 evaluates on the following benchmarks or datasets:

1. ImageNet: The model is compared against state-of-the-art convolutional networks (convnets) and other vision transformer models on the ImageNet dataset for top-1 classification accuracy (Reference 1, 2, 3, 5, 10).

2. ImageNet V2 and ImageNet Real: These datasets have a test set distinct from the ImageNet validation set, which helps in reducing overfitting on the validation set. The model's performance on these datasets is also reported (Reference 1).

3. Transfer learning tasks: The generalization power of DeiT is measured by fine-tuning on additional datasets, which are listed in Table 6 of the references. However, the specific datasets used for transfer learning tasks are not mentioned in the provided references, so [More Information Needed] for the exact names of these datasets (Reference 3).

4. The throughput of the model is measured on a 16GB V100 GPU, which is an indication of the model's efficiency, although this is not a dataset but rather a performance metric (Reference 4, 10).

#### Factors

The model facebook/deit-small-patch16-224 is a vision transformer that has been trained on the ImageNet1k dataset. Based on the references provided, several characteristics can be anticipated to influence the model's behavior:

1. **Domain and Context**: The model has been evaluated on ImageNet and has shown competitive performance with state-of-the-art convolutional networks when fine-tuned on other datasets (Ref. 1). However, its generalization power to domains significantly different from ImageNet has not been explicitly mentioned. Therefore, the model's performance may vary when applied to images from different domains or contexts that are not well-represented in the ImageNet dataset.

2. **Transfer Learning**: DeiT has been evaluated on transfer learning tasks and is on par with competitive convolutional models (Ref. 1). This suggests that the model can adapt to new tasks with fine-tuning, but the degree of performance may depend on the similarity of the new task to the original ImageNet training data.

3. **Population Subgroups**: The references do not provide specific information on the model's performance across different population subgroups. Since ImageNet contains a wide variety of objects and scenes, it is not clear how well the model will perform on images that predominantly feature underrepresented groups or non-Western contexts. [More Information Needed] to determine if there are disparities in performance across different demographic or cultural subgroups.

4. **Evaluation Metrics**: The model's performance is measured in terms of top-1 accuracy and throughput (Ref. 3, 5). While these metrics provide a general idea of the model's effectiveness and efficiency, they may not capture performance disparities across different subgroups or specific use-case scenarios.

5. **Data and Training**: The model's training on ImageNet1k only and its comparison to models trained on larger datasets like JFT-300M suggest that while DeiT models can perform well without extensive data, their performance might still be influenced by the quantity and diversity of the training data (Ref. 2, 6).

6. **Inference Time and Throughput**: The model's throughput, or the number of images processed per second, is a key characteristic that influences its practical deployment (Ref. 5). The model's efficiency is competitive, but actual throughput can vary depending on the implementation and hardware used.

7. **Distillation**: The use of transformer-specific distillation indicates that the model's performance can be enhanced when it learns from a teacher model, potentially affecting its accuracy and throughput positively (Ref. 4, 8).

In summary, while the model shows promise in terms of accuracy and efficiency, a more detailed evaluation is needed to fully understand its behavior across different domains, contexts, and population subgroups. Disaggregated evaluation across these factors is essential to uncover any disparities in performance.

#### Metrics

The evaluation of the model `facebook/deit-small-patch16-224` will primarily focus on the following metrics:

1. **Top-1 Accuracy**: This is the conventional metric for image classification tasks, where the model's performance is measured based on its ability to correctly classify the image with the highest probability prediction.

2. **Throughput**: This metric measures the number of images processed per second on a V100 GPU. It is an important performance metric that reflects the model's efficiency and speed during inference.

3. **Transfer Learning Performance**: The generalization power of the model will be evaluated by fine-tuning it on different datasets and comparing the results with those of other state-of-the-art vision transformer and convolutional architectures.

4. **Trade-off between Accuracy and Inference Time**: The model's performance will be assessed based on the balance between its classification accuracy and the time it takes to infer on a GPU.

5. **Memory Footprint**: Although not explicitly mentioned as a metric, the lower memory footprint for a given accuracy is highlighted as an advantage of vision transformers over convnets, suggesting that it is an important consideration in the evaluation.

These metrics will help in understanding the trade-offs between different errors and the overall efficiency of the model. The model card should reflect these evaluation criteria to inform potential users about the strengths and limitations of `facebook/deit-small-patch16-224`.

### Results

The evaluation results of the model `facebook/deit-small-patch16-224` can be summarized based on the provided references focusing on factors such as accuracy, throughput, and generalization to other datasets:

1. **Accuracy**: The DeiT models, including `facebook/deit-small-patch16-224`, show competitive performance on ImageNet with significant improvements over previous ViT models trained only on ImageNet1k. Specifically, DeiT models with distillation (indicated by the symbol ⚗) outperform EfficientNet and ViT-B models pre-trained on the same dataset. However, the exact top-1 accuracy percentage for the `facebook/deit-small-patch16-224` model is not provided in the references, so [More Information Needed] for the specific accuracy metric.

2. **Throughput**: The throughput of DeiT models is highlighted as a key factor, with the number of images processed per second on a V100 GPU being used as a measure. DeiT models are designed to be efficient, with the ability to process a large number of images per second, which suggests high throughput for the `facebook/deit-small-patch16-224` model. However, the exact throughput figures for this specific model are not provided in the references, so [More Information Needed] for the specific throughput metric.

3. **Generalization**: DeiT models have been evaluated on transfer learning tasks to measure their generalization power. The references indicate that DeiT models are on par with competitive convolutional network (convnet) models on these tasks. This suggests that `facebook/deit-small-patch16-224` should also exhibit good generalization capabilities when fine-tuned on other datasets. However, specific results from Table 6 or Table 7 are not provided in the references, so [More Information Needed] for detailed generalization metrics.

4. **Efficiency**: The DeiT models are designed to learn in a data-efficient manner, which is particularly important for the smaller `facebook/deit-small-patch16-224` model. The references suggest that DeiT models can be trained quickly on a single machine, which implies that `facebook/deit-small-patch16-224` is likely to be efficient in terms of training time and resource usage. However, specific data regarding training efficiency for this model is not provided, so [More Information Needed] for detailed efficiency metrics.

In conclusion, while the references provide a general overview of the DeiT model's performance, specific evaluation results for the `facebook/deit-small-patch16-224` model regarding top-1 accuracy, throughput, generalization to other datasets, and training efficiency are not explicitly stated and would require further information to be accurately reported.

#### Summary

The evaluation results for the model `facebook/deit-small-patch16-224` indicate that the DeiT (Data-efficient Image Transformers) approach has achieved competitive performance compared to state-of-the-art convolutional networks (convnets) and other vision transformer models when trained solely on the ImageNet1k dataset. Specifically, DeiT shows a significant improvement over previous ViT models trained on ImageNet1k, with a +6.3% increase in top-1 accuracy in a comparable setting.

When enhanced with transformer-specific distillation from a RegNetY, the DeiT model outperforms EfficientNet, which is a highly optimized convnet. Additionally, DeiT demonstrates better performance than the ViT-B model pre-trained on the larger JFT-300M dataset at a higher resolution, while also being faster to train.

The model's generalization capabilities were also assessed through transfer learning tasks on various datasets, where DeiT's performance was found to be on par with competitive convnet models. This aligns with the conclusion drawn from the ImageNet results, suggesting that DeiT can generalize well across different datasets.

In terms of efficiency, DeiT models offer a favorable trade-off between accuracy and inference time on a GPU, with throughput measured in images processed per second. The throughput is calculated based on the largest possible batch size for the model's usual resolution and averaged over 30 runs.

Overall, the DeiT models, including `facebook/deit-small-patch16-224`, are considered to be on par with state-of-the-art convnets when trained without external data, and they are expected to become a method of choice due to their lower memory footprint for a given accuracy level. The open-source implementation of DeiT is available on GitHub, which further facilitates its adoption and evaluation by the research community.

## Model Examination

Explainability/Interpretability Section for Model Card: facebook/deit-small-patch16-224

The facebook/deit-small-patch16-224 model is a vision transformer (ViT) that has been trained using a novel distillation technique to improve its performance, especially when dealing with limited amounts of data. The model leverages a teacher-student paradigm where the teacher can be a strong image classifier, such as a convolutional neural network (convnet), or a mixture of classifiers. The distillation process involves two main components: the class embedding and the distillation embedding.

The distillation embedding is particularly interesting from an explainability perspective. It is used similarly to the class token, interacting with other embeddings through self-attention and is output by the network after the last layer. The target objective for the distillation embedding is provided by the distillation component of the loss function. This allows the model to learn from the output of the teacher model while remaining complementary to the class embedding. Through this process, we observe that the learned class and distillation tokens converge towards different vectors, which suggests that the model is internalizing information from the teacher in a distinct manner from the class prediction itself.

In terms of interpretability, our analysis in Table 4 provides insights into the decision-making process of the model. We compare the decision agreement between the convnet teacher, our image transformer DeiT learned from labels only, and our transformer DeiT with distillation. The distilled model shows a higher correlation with the convnet than with a transformer learned from scratch. This indicates that the distillation process brings the decision-making of the transformer closer to that of the convnet, potentially inheriting some of the inductive biases that facilitate training.

Furthermore, the disagreement analysis between different classifiers, including convnet, image transformers, and distilled transformers, reveals the rate of different decisions. This analysis helps us understand how the distilled models and classification heads are correlated to their respective teachers, providing a measure of the influence of the distillation process on the model's predictions.

In summary, the facebook/deit-small-patch16-224 model's distillation process not only improves its performance but also offers a window into the model's learning dynamics. The use of distillation embeddings and the analysis of decision agreements contribute to a better understanding of how the model processes information and makes predictions, which is a step towards explainability and interpretability in transformer-based image classification models.

## Environmental Impact

- **Hardware Type:** The model facebook/deit-small-patch16-224 was trained on a single node with 8 GPUs, as mentioned in reference 2, which states that fine-tuning the model at a larger resolution takes 20 hours on a single node (8 GPU).
- **Software Type:** The model facebook/deit-small-patch16-224 is trained on PyTorch and the timm library.
- **Hours used:** The model facebook/deit-small-patch16-224, referred to as DeiT-S in the references, is trained in less than 3 days on 4 GPUs.
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model `facebook/deit-small-patch16-224` is a Data-efficient image Transformer (DeiT) that builds upon the Vision Transformer (ViT) architecture introduced by Dosovitskiy et al. [15]. The architecture is designed to process images by treating them as a sequence of fixed-size patches, similar to how transformers process sequences of tokens in natural language processing (NLP).

The input RGB image is decomposed into a batch of N patches, each of size 16x16 pixels, resulting in N=14x14 total patches for an input image of size 224x224. Each patch is linearly projected to a dimension of 768, maintaining the overall dimension of 3x16x16. The transformer block, which is invariant to the order of the patch embeddings, processes these patch embeddings along with an additional trainable class token. This class token is appended to the patch tokens before the first layer and is used to predict the class after passing through the transformer layers.

The transformer block for images includes a Multihead Self-Attention (MSA) layer and a Feed-Forward Network (FFN). The FFN consists of two linear layers with a GeLu activation function in between. The first linear layer expands the dimension from D to 4D, and the second linear layer reduces it back to D. Both MSA and FFN operate as residual operators with skip-connections and layer normalization.

The DeiT model also incorporates a transformer-specific distillation token, which is shown to add to the model's performance compared to using an additional class token. This distillation token is trained with a teacher pseudo-label, which helps the model to generalize better.

The model is trained on the ImageNet dataset using a single 8-GPU node in a relatively short amount of time (53 hours of pre-training and optionally 20 hours of fine-tuning), making it competitive with convolutional networks (convnets) in terms of both parameter count and efficiency. The DeiT model achieves large improvements over previous ViT models trained only on ImageNet1k and, with the help of distillation, can outperform EfficientNets and ViT models pre-trained on larger datasets.

The objective of the `facebook/deit-small-patch16-224` model is to provide a data-efficient transformer-based approach to image classification that is competitive with traditional convnets while being trained on a smaller dataset (ImageNet1k) and with less computational resources. The model aims to close the gap between vision transformers and convnets and potentially become a method of choice due to its lower memory footprint for a given accuracy.

### Compute Infrastructure

The compute infrastructure used for training the model facebook/deit-small-patch16-224 involved training on a single 8-GPU node. The typical training duration for the larger DeiT-B model is 53 hours of pre-training, and optionally, an additional 20 hours of fine-tuning can be conducted. Although the exact times for the DeiT-Small model are not provided, it is mentioned that DeiT-S and DeiT-Ti models are trained in less than 3 days on 4 GPUs. Therefore, we can infer that the DeiT-Small model would have a similar or shorter training time compared to the DeiT-Base model, given its smaller size, and would also be trained on a similar infrastructure.

For fine-tuning at a larger resolution, it takes 20 hours on a single node with 8 GPUs to produce a FixDeiT-B model at resolution 384×384. While this information is specific to the DeiT-B model, it gives an indication of the resources required for potential fine-tuning phases for different model sizes, including the DeiT-Small model.

In summary, the DeiT-Small model is likely trained on a single node with multiple GPUs (at least 4 GPUs for smaller models like DeiT-S and DeiT-Ti), and the training is completed within a few days, which is less than the time required for the DeiT-Base model. However, for the exact training time and compute infrastructure details for the DeiT-Small model, [More Information Needed].

## Citation

```
@misc{hugo-training,
    author = {Hugo Touvron and
              Matthieu Cord and
              Matthijs Douze and
              Francisco Massa and
              Alexandre Sablayrolles and
              Hervé Jégou and
              Facebook Ai},
    title  = {Training data-efficient image transformers & distillation through attention},
    url    = {https://arxiv.org/pdf/2012.12877.pdf}
}
```

