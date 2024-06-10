# Model Card for facebook/dino-vits16

The model facebook/dino-vits16 is a self-supervised Vision Transformer (ViT) that leverages a novel method called DINO, which can be seen as a form of knowledge distillation without labels, to achieve state-of-the-art performance on image classification tasks. It demonstrates particularly strong k-NN classification capabilities and contains features that are beneficial for tasks like semantic segmentation and image retrieval.

## Model Details

### Model Description

Model Name: facebook/dino-vits16

Model Architecture:
The facebook/dino-vits16 model is based on the Vision Transformer (ViT) architecture, specifically the ViT-S/16 variant. The model takes as input a grid of non-overlapping contiguous image patches of resolution 16x16. It follows the Transformer network design with a "pre-norm" layer normalization and is composed of a sequence of self-attention and feed-forward layers, paralleled with skip connections. The model does not use batch normalizations (BN) in its architecture, making it entirely BN-free. The neural network consists of a backbone and a projection head, where the backbone is the ViT and the projection head is a 3-layer multi-layer perceptron (MLP) with a hidden dimension of 2048, followed by 2 normalization layers and a weight normalized fully connected layer with K dimensions.

Training Procedures:
The model is pretrained on the ImageNet dataset without labels using self-supervised learning techniques. The training utilizes the adamw optimizer with a batch size of 1024, distributed over 16 GPUs. The learning rate is linearly ramped up during the first 10 epochs and then decayed with a cosine schedule. Weight decay also follows a cosine schedule. The temperature parameters τ_s and τ_t are set to specific values and are subject to a linear warm-up during the initial epochs. Data augmentations include color jittering, Gaussian blur, solarization, and multi-crop with bicubic interpolation. The model is trained for 100 epochs on a single node with 8 GPUs, and the training time is approximately 1.75 days.

Parameters:
- Input resolution: 16x16 patches
- Hidden dimension of MLP: 2048
- Batch size: 1024
- Optimizer: adamw
- Learning rate schedule: Linear warmup followed by cosine decay
- Weight decay schedule: Cosine
- Temperature parameters: τ_s = 0.1, τ_t warm-up from 0.04 to 0.07
- Training epochs: 100
- GPUs: Trained on 16 GPUs, can be run on 8 GPUs

Important Disclaimers:
- The model does not use batch normalization, which is a departure from standard convnet architectures.
- The evaluation of the model shows that both linear and finetuning evaluations are sensitive to hyperparameters, and there can be a large variance in accuracy between runs when varying the learning rate.
- The model's performance and computational requirements are compared to state-of-the-art self-supervised systems based on convolutional networks of comparable sizes.
- The model's features are particularly effective for k-NN classification and weakly supervised image segmentation.
- The results suggest that self-supervised learning with ViT could be key to developing a BERT-like model for visual tasks.
- The code and models to reproduce the results are publicly available.

[More Information Needed]: For any specific values of K dimensions in the projection head, exact top-1 accuracy numbers on various datasets, and any additional disclaimers not covered by the provided references.

- **Developed by:** Mathilde Caron; Hugo Touvron; Ishan Misra; Hervé Jegou; Julien Mairal; Piotr Bojanowski; Armand Joulin; Facebook Ai Research
- **Funded by:** The project for the model facebook/dino-vits16 is funded by Facebook AI Research.
- **Shared by:** The contributors who made the model facebook/dino-vits16 available online as a GitHub repo are Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jegou, Julien Mairal, Piotr Bojanowski, and Armand Joulin from Facebook AI Research.
- **Model type:** The model facebook/dino-vits16 is a self-supervised Vision Transformer trained using a multicrop strategy for visual feature extraction, without batch normalization, and is applicable to image modality.
- **Language(s):** The model facebook/dino-vits16 does not use or process any natural human language; it is a vision transformer model pre-trained on ImageNet for image classification tasks using self-supervised learning.
- **License:** The model `facebook/dino-vits16` is released under the Apache 2.0 license. The link to the license can be found in the [LICENSE](LICENSE) file.
- **Finetuned from model:** The model facebook/dino-vits16 is not explicitly mentioned as being fine-tuned from another model in the provided references. The references discuss the Vision Transformer (ViT) architecture and the DINO framework, as well as the training details on ImageNet without labels, but they do not specify a base model from which facebook/dino-vits16 was fine-tuned. Therefore, based on the given information, the answer is:

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/facebookresearch/dino
- **Paper:** https://arxiv.org/pdf/2104.14294.pdf
- **Demo:** [More Information Needed]
## Uses

### Direct Use

The `facebook/dino-vits16` model can be used without fine-tuning, post-processing, or plugging into a pipeline for tasks such as image retrieval, copy detection, and k-NN classification directly with the features extracted from the pre-trained network. This is possible because the model has been trained using self-supervised learning, which allows it to learn rich feature representations that can be used as-is for various downstream tasks.

For instance, in the context of image retrieval and copy detection, you can use the features extracted from the model to compute cosine similarity between images. The high-dimensional feature vectors obtained from the model capture the visual content of the images, and by computing the cosine similarity, you can assess the similarity between different images.

Here's a conceptual example of how you might use the model for image retrieval or copy detection without any additional training or complex pipeline:

```python
from transformers import DinoModel, DinoProcessor
import torch

# Load pre-trained DINO model and processor
model = DinoModel.from_pretrained("facebook/dino-vits16")
processor = DinoProcessor.from_pretrained("facebook/dino-vits16")

# Process images and extract features
image1 = processor(images="path_to_image1.jpg", return_tensors="pt")
image2 = processor(images="path_to_image2.jpg", return_tensors="pt")

with torch.no_grad():
    # Obtain features from the [CLS] token and the pooled output
    features_image1 = model(**image1).pooler_output
    features_image2 = model(**image2).pooler_output

# Compute cosine similarity between the two feature vectors
cosine_similarity = torch.nn.functional.cosine_similarity(features_image1, features_image2)

print("Cosine Similarity:", cosine_similarity.item())
```

Please note that the above code is a conceptual example and assumes the existence of a `DinoProcessor` and the method `pooler_output` which may not be directly available in the Hugging Face Transformers library. The actual implementation might differ, and you would need to adapt the code to match the specific API of the model and the library.

For k-NN classification, you can use the features extracted from the model to perform k-NN search among a set of pre-computed image features. The model's features have been shown to be particularly effective for k-NN classification, achieving high top-1 accuracy on datasets like ImageNet.

In summary, `facebook/dino-vits16` is versatile and can be used directly for various tasks that rely on the quality of the learned visual features without the need for additional training or complex pipelines. However, for exact code snippets and usage, please refer to the official documentation and implementation details provided by the model's authors or the Hugging Face library. If such details are not available in the references provided, then [More Information Needed].

### Downstream Use

The `facebook/dino-vits16` model is a Vision Transformer (ViT) that has been pretrained using the DINO (self-supervised) approach on the ImageNet dataset without labels. This model can be fine-tuned for various downstream tasks, such as image classification, object detection, or even weakly supervised image segmentation. The self-supervised pretraining with DINO has been shown to provide features that transfer better to downstream tasks compared to features trained with supervision, which can lead to improved performance on these tasks.

When fine-tuning `facebook/dino-vits16` for a specific task, you would start by initializing your network with the pretrained weights and then adapt them during training to your particular dataset and task. The fine-tuning process involves training the model on your task-specific dataset, allowing the weights to adjust from the generic features learned during pretraining to features that are more relevant to the task at hand.

For integration into a larger ecosystem or app, `facebook/dino-vits16` can serve as a powerful feature extractor. The extracted features can be used as input to other machine learning models or systems within the app, enabling a variety of image-based functionalities.

Here's a general outline of how you might fine-tune the model, although a specific code snippet is not provided in the references:

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW

# Load the feature extractor and model from Hugging Face
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
model = ViTForImageClassification.from_pretrained('facebook/dino-vits16')

# Load your dataset
datasets = load_dataset('your_dataset_name')
train_dataset = datasets['train']

# Prepare the data loader
def transform(examples):
    return feature_extractor(examples['image'], return_tensors='pt')

train_dataset.set_transform(transform)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)

# Fine-tune the model
optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained('your_fine_tuned_model')
```

Please note that the actual fine-tuning process would require setting up the correct data processing, choosing appropriate hyperparameters, and potentially modifying the model architecture for the specific task. The code above is a simplified example and does not include all necessary steps such as evaluation, logging, or advanced training techniques.

For more detailed instructions on fine-tuning or integrating the model into an application, you would need to refer to the model's documentation or the code repository provided by the authors. If you require specific implementation details or code examples that are not covered in the references, you would need to say "[More Information Needed]".

### Out-of-Scope Use

As a sociotechnic involved in the development of the facebook/dino-vits16 model, it is my responsibility to consider the potential for misuse of our technology and to provide guidance to users on how to ethically and responsibly use our model. Here are some considerations regarding the misuse of the model:

1. **Privacy Concerns**: Given the model's potential in image retrieval and its ability to extract detailed features from images, there is a risk that it could be used to infringe on personal privacy. Users should not use facebook/dino-vits16 to analyze or extract information from images where individuals have not consented to such use, especially in sensitive contexts.

2. **Bias and Fairness**: While the references do not explicitly discuss the training data, models trained on datasets like ImageNet may inherit biases present in the data. Users should be cautious about deploying the model in applications where biases could lead to unfair outcomes or discrimination, such as surveillance or profiling.

3. **Deepfakes and Misinformation**: The model's strong performance in generating visual features could potentially be used to create deepfakes or to manipulate images in a way that contributes to the spread of misinformation. Users should not use facebook/dino-vits16 for creating or propagating deceptive content.

4. **Intellectual Property Violations**: The model's capabilities in image retrieval could be misused to violate intellectual property rights by extracting and using copyrighted images without permission. Users should respect copyright laws and not use the model to engage in or facilitate piracy.

5. **Security**: The model could be used to analyze and extract information from images for malicious purposes, such as reconnaissance for security breaches or cyber-attacks. Users should not use the model for any form of illegal activity, including but not limited to cyber espionage or theft.

6. **Resource Intensive Applications**: The references indicate that the model has been developed with specific versions of Python, PyTorch, CUDA, and torchvision. Users should be aware of the computational resources required to run the model and should not use it in ways that lead to unsustainable energy consumption or environmental impact.

In conclusion, while the facebook/dino-vits16 model has significant potential for advancing the field of computer vision, it is crucial that users employ the model in a manner that is ethical, respects privacy, avoids perpetuating biases, and does not infringe upon intellectual property rights or contribute to security threats. Users should also be mindful of the environmental impact of running large-scale models and strive to use computational resources responsibly.

### Bias, Risks, and Limitations

As the sociotechnic on the deep learning model development team, I would highlight the following known or foreseeable issues stemming from the model facebook/dino-vits16:

1. **Bias and Representation Issues**: Given that the model is trained on random uncurated images, there is a risk of replicating or amplifying societal biases present in the training data. This could manifest in discriminatory or unfair outcomes when the model is applied to real-world tasks, particularly in image recognition and classification.

2. **Misunderstandings of Model Capabilities**: Users may misunderstand the capabilities of the model, expecting it to perform well on tasks it was not designed for. The model is optimized for self-supervised learning and may not perform as well on supervised tasks without additional fine-tuning.

3. **Technical Limitations**: The model's performance is benchmarked on specific datasets (e.g., ImageNet, Places205, PASCAL VOC, Oxford-102 flowers). Its effectiveness on datasets with different characteristics or in different domains is not guaranteed and may require further validation.

4. **Sociotechnical Limitations**: The model's use in applications such as image retrieval and weakly supervised image segmentation may have broader implications, such as privacy concerns and the potential for misuse in surveillance systems. Ethical considerations around consent and the right to privacy must be addressed.

5. **Computational Requirements**: While the model achieves a reduction in computational requirements compared to other self-supervised systems, it still requires significant resources (two 8-GPU servers for 3 days). This may limit accessibility for researchers or organizations with fewer computational resources.

6. **Robustness and Generalization**: The model's robustness to adversarial attacks or out-of-distribution data has not been explicitly discussed. Ensuring that the model can generalize well and maintain performance in diverse and potentially adversarial environments is crucial.

7. **Environmental Impact**: The energy consumption required for training large models like facebook/dino-vits16 has an environmental impact. It is important to consider the carbon footprint and strive for more energy-efficient training methods.

8. **Future Research Directions**: The model card should clearly state that the current model is a step towards developing a BERT-like model for vision and that future work will explore the limits of visual features. This indicates that the model is part of ongoing research and may not yet be fully optimized for all potential applications.

In conclusion, while facebook/dino-vits16 shows promising results in self-supervised learning with Vision Transformers, it is important to consider the broader societal implications, ethical considerations, and technical limitations when deploying the model in real-world settings. Further research and careful consideration of these issues are necessary to ensure responsible use.

### Recommendations

Based on the provided references, here are some recommendations with respect to the foreseeable issues about the model `facebook/dino-vits16`:

1. **Computational Efficiency**: The model achieves high performance with a significant reduction in computational requirements compared to state-of-the-art self-supervised systems. However, it is important to consider the trade-off between accuracy and computational resources. For instance, using multi-crop improves the accuracy/running-time tradeoff, but it also increases memory usage. Users with limited computational resources should be aware of these trade-offs and may need to adjust the training setup accordingly.

2. **Batch Size Considerations**: The model can be trained with smaller batch sizes, but this may require re-tuning of hyperparameters such as momentum rates. Users should be prepared to experiment with these settings if they do not have access to multiple GPUs or wish to train the model with smaller batch sizes.

3. **Memory Requirements**: The use of multi-crop training increases memory usage. Users should ensure they have sufficient GPU memory to accommodate the increased requirements, or they should adjust the training settings to fit their available resources.

4. **Training Time**: The model can achieve improved performance in less time with the right settings, such as the use of multi-crop. However, users should be aware that the performance boost from multi-crop cannot be matched by simply increasing training time in a standard setting. Efficient use of time and resources should be a consideration.

5. **Momentum and Other Components**: The references indicate that momentum is crucial for the framework to work effectively. Users should ensure that momentum is properly configured during training. Additionally, the impact of other components, such as SK, should be considered in the context of whether momentum is used.

6. **Generalization and Transferability**: While the model shows promising results on benchmarks like ImageNet, it is important to consider how well the model generalizes to other datasets and tasks. Users should evaluate the model on their specific use cases and be prepared to fine-tune or adapt the model as necessary.

7. **Future Research Directions**: The references suggest that self-supervised learning could be key to developing a BERT-like model for vision tasks. Users and researchers should consider the potential of DINO for pretraining on large, uncurated datasets to push the limits of visual features.

8. **Ethical and Societal Considerations**: As a sociotechnic, it is important to consider the broader implications of deploying this model. This includes the potential for biases in the training data to be perpetuated by the model, the environmental impact of training large models, and the accessibility of the model to a wide range of users. Users should be encouraged to conduct ethical reviews and bias assessments of the model in their specific applications.

In summary, while the `facebook/dino-vits16` model shows promising results, users should be mindful of computational and memory requirements, the necessity of hyperparameter tuning, the importance of momentum, and the broader ethical and societal implications of using the model.

## Training Details

### Training Data

The training data for the model facebook/dino-vits16 consists of the ImageNet dataset without labels, which is used for self-supervised pretraining of the Vision Transformer (ViT) models. Data augmentation techniques such as color jittering, Gaussian blur, and solarization are applied, along with multi-crop strategies for adapting position embeddings to different scales. [More Information Needed] on data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

The preprocessing steps for the data used in the model `facebook/dino-vits16` include the following:

1. **Tokenization of Image Patches**: The input images are divided into a grid of non-overlapping contiguous patches of resolution N × N. For the `facebook/dino-vits16` model, we typically use N = 16, which means each image is divided into patches of 16x16 pixels [2].

2. **Embedding Formation**: These patches are then passed through a linear layer to form a set of embeddings. An extra learnable token, referred to as the [CLS] token, is added to the sequence of patch tokens to aggregate information from the entire sequence [1].

3. **Data Augmentation**: During the pretraining phase on the ImageNet dataset, we do not use labels and apply data augmentations such as random resize crops, horizontal flips, color jittering, Gaussian blur, and solarization. We also use multi-crop with bicubic interpolation to adapt the position embeddings to the scales [7, 8].

4. **Feature Concatenation**: For evaluation, the features are obtained as the concatenation of the output [CLS] token and the Generalized Mean (GeM) pooled output patch tokens [9].

5. **Feature Whitening**: Following the extraction of features, we apply whitening on these features. This transformation is learned on an extra set of 20K random images from the YFCC100M dataset, distinct from the distractors [9].

6. **Resizing**: The Vision Transformer (ViT) architecture takes as input image patches of a fixed resolution. If the input images are of different sizes, they would need to be resized to ensure that the patches extracted are of the resolution N × N (16x16 in this case) [2].

For any additional specific preprocessing steps such as normalization constants or exact resizing techniques not mentioned in the provided references, [More Information Needed].

#### Training Hyperparameters

The training hyperparameters for the model facebook/dino-vits16 are as follows:

- Temperature τ s is set to 0.1, with a linear warm-up for τ t from 0.04 to 0.07 during the first 30 epochs.
- Data augmentations include color jittering, Gaussian blur, and solarization, as well as multi-crop with bicubic interpolation for adapting position embeddings to different scales.
- The model is pretrained on the ImageNet dataset without labels.
- The optimizer used is AdamW.
- The batch size during training is 1024.
- Training is distributed over 16 GPUs when using ViT-S/16.
- The learning rate is linearly ramped up during the first 10 epochs to its base value, which follows the linear scaling rule: lr = 0.0005 * batchsize/256.
- After the warm-up period, the learning rate is decayed following a cosine schedule.
- Weight decay also follows a cosine schedule, starting from 0.04 [More Information Needed] for the exact end value.
- For linear evaluations, random resize crops and horizontal flips are used as augmentations during training, and accuracy is reported on a central crop.
- For finetuning evaluations, networks are initialized with the pretrained weights and adapted during training.
- The pretrained model is expected to reach 73.3% on k-NN eval and 76.0% on linear eval.
- Training time is approximately 2.6 days using 16 GPUs.

Please note that some specific values, such as the exact end value of the weight decay schedule, are not provided in the references and thus "[More Information Needed]" is indicated for those parts.

#### Speeds, Sizes, Times

The facebook/dino-vits16 model is a self-supervised Vision Transformer (ViT) that has been trained on the ImageNet dataset without labels. Here are the details regarding the model's throughput, training time, checkpoint sizes, and other relevant information based on the provided references:

- **Throughput**: The throughput of the model varies depending on the patch size used during training. For instance, when using 5×5 patches, the throughput falls to 44 images per second (im/s), whereas for 8×8 patches, the throughput is 180 im/s (Reference 9). However, the specific throughput for the facebook/dino-vits16 model with 16×16 patches is not provided in the references, so [More Information Needed] for the exact throughput of this configuration.

- **Training Time**: The model can be trained on a single node with 8 GPUs for 100 epochs in approximately 1.75 days (Reference 3). For the full 300 epochs training, as mentioned in Reference 1, it takes about 3 days using two 8-GPU servers.

- **Checkpoint Sizes**: The exact size of the checkpoints for the facebook/dino-vits16 model is not provided in the references. Therefore, [More Information Needed] regarding the checkpoint sizes.

- **Start or End Time**: The references do not provide specific start or end times for the training process. They only mention the duration of the training, such as 1.75 days for 100 epochs (Reference 3) and 3 days for 300 epochs (Reference 1). Therefore, [More Information Needed] for exact start or end times.

- **Additional Details**: The model uses the adamw optimizer with a batch size of 1024, distributed over 16 GPUs when using ViT-S/16 (Reference 4). The learning rate is linearly ramped up during the first 10 epochs and then decayed with a cosine schedule. The weight decay also follows a cosine schedule from 0.04 to 0.4 (Reference 6). For linear evaluations, random resize crops and horizontal flips augmentation are applied during training (Reference 5).

For further details such as the exact checkpoint sizes or throughput for the specific patch size used in facebook/dino-vits16, additional information would be required that is not provided in the references.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model facebook/dino-vits16 evaluates on the following benchmarks or datasets:

1. ImageNet: Used for the standard self-supervised benchmark, object discovery, and transfer-learning evaluations.
2. Google Landmarks: Compared performance in image retrieval using features pretrained with DINO.

[More Information Needed] for any additional benchmarks or datasets not mentioned in the provided references.

#### Factors

The foreseeable characteristics that will influence how the model facebook/dino-vits16 behaves can be inferred from the references provided:

1. **Domain and Context**: The model has been evaluated on different downstream tasks, suggesting that its performance may vary depending on the specific application. For instance, it has been used for image retrieval, object discovery, and transfer learning (Ref. 3). The model's features, pretrained with DINO, have been shown to transfer better than features trained with supervision on ImageNet for ViT architectures (Ref. 1). This indicates that the model may perform well in contexts where transfer learning is beneficial, such as when there is limited labeled data available for the target task.

2. **Population Subgroups**: The references do not provide explicit information on the performance of the model across different population subgroups. Therefore, without further evaluation, it is not possible to determine if there are disparities in performance related to demographic or other subgroup characteristics. [More Information Needed]

3. **Evaluation Disaggregation**: The references mention that both linear evaluations and finetuning evaluations are sensitive to hyperparameters, with a large variance in accuracy between runs when varying the learning rate, for example (Ref. 2). This suggests that the model's performance may not be consistent across different settings and that careful hyperparameter tuning is necessary for optimal results. However, there is no explicit mention of disaggregated evaluation across factors such as demographics, image types, or environmental conditions, which would be necessary to uncover disparities in performance. [More Information Needed]

4. **Data Augmentation and Preprocessing**: The model uses data augmentations such as color jittering, Gaussian blur, and solarization, as well as multi-crop with bicubic interpolation to adapt the position embeddings to the scales (Ref. 5). These augmentations may influence the model's robustness to variations in input data, potentially improving its generalization across different domains.

5. **Self-Supervised Learning Components**: The impact of different components from self-supervised learning on the model's performance is highlighted, with the observation that the absence of momentum in the framework leads to poor performance (Ref. 7). This indicates that certain architectural choices and training strategies are crucial for the model's behavior.

6. **Future Directions**: There is an indication that self-supervised learning could be key to developing a BERT-like model for visual tasks, and there are plans to explore pretraining on random uncurated images to push the limits of visual features (Ref. 6). This suggests that the model's current behavior and performance may evolve as new self-supervised learning techniques are incorporated.

In summary, the model's behavior is influenced by the domain and context of its application, the specific downstream tasks it is applied to, the data augmentation and preprocessing techniques used, and the architectural components and training strategies employed. However, there is a lack of information on the model's performance across different population subgroups and a need for disaggregated evaluation to uncover potential disparities.

#### Metrics

For the evaluation of the model facebook/dino-vits16, the following metrics and protocols will be used:

1. **k-NN Classification Performance**: The quality of features will be assessed using a weighted nearest neighbor classifier, with the number of nearest neighbors swept over different values to find the optimal setting. The reference indicates that using 20 nearest neighbors consistently works best for most runs.

2. **Linear Evaluation**: A linear classifier will be learned on frozen features, with data augmentation techniques such as random resize crops and horizontal flips applied during training. Accuracy will be reported on a central crop.

3. **Fine-tuning Evaluation**: The pretrained weights will be used to initialize networks, which are then adapted during training on downstream tasks. This evaluation is sensitive to hyperparameters, and a large variance in accuracy can be observed when varying the learning rate.

4. **Image Retrieval**: The performance in retrieval tasks will be compared using off-the-shelf features pretrained with supervision or with DINO on datasets like ImageNet and Google Landmarks.

5. **Object Discovery and Transfer-Learning**: The properties of the resulting features for object discovery and transfer-learning tasks will be studied.

6. **Self-Attention Maps**: The quality of self-attention maps from supervised versus self-supervised learning will be evaluated, potentially using metrics like the Jaccard similarity.

7. **Throughput**: The impact of the number of heads in the Vision Transformer (ViT) on throughput (images processed per second at inference time on a single V100 GPU) will be considered.

8. **Accuracy**: The impact of different components such as the number of heads in the ViT, patch size, and the use of different augmentations and losses on the k-NN and linear evaluation accuracy will be reported.

These metrics will help in understanding the tradeoffs between different errors and the overall performance of the model across various tasks and conditions.

### Results

Evaluation Results of the Model facebook/dino-vits16:

Factors and Metrics:

1. **Linear Evaluation Protocol**: The model was evaluated using a linear classifier on top of frozen features. During training, random resize crops and horizontal flips augmentation were applied, and accuracy was reported on a central crop. However, the evaluations are sensitive to hyperparameters, and significant variance in accuracy was observed when varying the learning rate.

2. **k-NN Classifier**: The quality of features was also assessed using a weighted nearest neighbor classifier. The pretrained model was frozen to compute and store features of the training data for the downstream task. A sweep over different numbers of nearest neighbors was performed, and it was found that using 20 nearest neighbors consistently worked best for most runs.

3. **Self-Supervised Benchmark on ImageNet**: The DINO framework was validated on the standard self-supervised benchmark on ImageNet, indicating the model's effectiveness in a self-supervised learning context.

4. **Data Augmentations**: The model used data augmentations such as color jittering, Gaussian blur, and solarization, along with multi-crop with bicubic interpolation for adapting position embeddings to different scales.

5. **Learning Rate and Weight Decay**: The model was trained with the adamw optimizer and a cosine schedule for learning rate decay after a linear warm-up phase. The weight decay also followed a cosine schedule.

6. **Comparison with Supervised Learning**: In downstream tasks, it was observed that for ViT architectures, self-supervised pretraining transfers better than features trained with supervision. This is consistent with observations made on convolutional networks.

7. **Image Retrieval**: The model's performance in image retrieval was compared using off-the-shelf features pretrained with DINO on ImageNet and Google Landmarks datasets. The comparison was made against features pretrained with supervision.

8. **Self-Attention Maps**: The self-attention maps from supervised versus self-supervised learning were evaluated, and the Jaccard similarity between the masks obtained by thresholding the self-attention maps was compared.

9. **Impact of the Number of Heads in ViT-S**: The impact of the number of heads in ViT-S on accuracy and throughput (images processed per second at inference time on a single V100 GPU) was studied.

10. **Transfer-Learning**: The pretrained features were finetuned on each downstream task following the protocol used in Touvron et al. [69]. It was found that self-supervised pretraining transfers better than supervised features for ViT architectures.

[More Information Needed]: Specific numerical results, such as accuracy percentages or throughput measurements, are not provided in the references and would be needed to complete this evaluation summary.

#### Summary

The evaluation results for the model facebook/dino-vits16 can be summarized as follows:

1. The DINO framework was validated on the standard self-supervised benchmark on ImageNet, where it was used to study feature properties for retrieval, object discovery, and transfer learning.

2. DINO was compared with other self-supervised methods using the same architecture, specifically a ResNet-50 and a ViT-small. The ViT-small was chosen due to its similarity to ResNet-50 in terms of the number of parameters and other aspects. DINO demonstrated competitive performance in image retrieval tasks when compared to features pretrained with supervision on ImageNet and Google Landmarks.

3. Training DINO with Vision Transformers (ViTs) achieved a top-1 accuracy of 76.1% on ImageNet using two 8-GPU servers over 3 days. This result surpassed state-of-the-art self-supervised systems based on convolutional networks of comparable sizes, while also reducing computational requirements. The impact of batch size on the features obtained with DINO was also studied, indicating the model's efficiency and scalability.

4. The quality of features pretrained with DINO was evaluated on different downstream tasks and compared with features from the same architectures trained with supervision on ImageNet. It was observed that for ViT architectures, self-supervised pretraining transferred better to downstream tasks than supervised features, aligning with observations made on convolutional networks.

In summary, the facebook/dino-vits16 model demonstrates strong performance in self-supervised learning benchmarks, efficient training on limited resources, and superior transferability of features to downstream tasks compared to supervised pretraining.

## Model Examination

### Model Card - facebook/dino-vits16

#### Explainability/Interpretability

Our model, facebook/dino-vits16, incorporates self-attention mechanisms that provide insights into the model's decision-making process. We have conducted extensive experiments to visualize and understand the self-attention patterns of the model.

1. **Self-Attention Visualizations**: In our experiments, we visualized the self-attention from the last layer of DINO ViT-S/8 for several reference points. These visualizations are provided in Fig. 8 and Fig. 10 of our paper. The images used for these visualizations are randomly selected from the COCO validation set and were not part of the training data. These visualizations help us understand how the model focuses on different parts of the image and can be used to interpret the model's feature extraction process.

2. **Feature Evaluation**: We have evaluated the properties of the DINO features in various contexts. For instance, we assessed the model's ability to retain information about object location and its transferability to downstream tasks. Specifically, we reported the mAP performance in copy detection on the Copydays "strong" subset, comparing it to the multigrain model trained for object retrieval. This evaluation helps us understand the robustness and applicability of the learned features.

3. **Downstream Task Transferability**: Our findings suggest that the features pretrained with DINO on the ViT architecture transfer better to downstream tasks than features trained with supervision on ImageNet. This is consistent with observations made on convolutional networks and indicates that self-supervised pretraining with DINO could be a promising approach for developing a BERT-like model for visual tasks.

4. **Projection Head Analysis**: We found that using a projection head greatly improves the accuracy of our method. The projection head consists of a multilayer perceptron (MLP) with hidden layers of 2048 dimensions and GELU activations, with the last layer of the MLP being without GELU. This design choice, along with the absence of batch normalization (BN) in ViT architectures, contributes to the model's performance and could provide insights into the importance of different architectural components for self-supervised learning.

In summary, our model demonstrates promising results in terms of explainability and interpretability, with self-attention visualizations and feature evaluations providing valuable insights into the model's inner workings. Future work will explore the potential of pretraining larger ViT models with DINO on uncurated images to further enhance the quality of visual features.

## Environmental Impact

- **Hardware Type:** The model facebook/dino-vits16 was trained on a setup with two nodes, each equipped with 8 GPUs, for a total of 16 GPUs.
- **Software Type:** The model facebook/dino-vits16 is trained on the following software type:

- Python version 3.6
- PyTorch version 1.7.1
- CUDA 11.0
- torchvision 0.8.2
- **Hours used:** The amount of time used to train the model facebook/dino-vits16 is 3 days using two 8-GPU servers.
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture for `facebook/dino-vits16` is based on the Vision Transformer (ViT) design. Specifically, it uses a grid of non-overlapping contiguous image patches of resolution N × N, typically with N = 16, as input. The architecture is composed of a backbone and a projection head. The backbone is a ViT, which processes the image patches through a series of Transformer blocks, each consisting of self-attention and feed-forward layers, with skip connections and a "pre-norm" layer normalization. The projection head is a 3-layer multi-layer perceptron (MLP) with a hidden dimension of 2048, followed by 2 normalization layers and a weight normalized fully connected layer with K dimensions.

The objective of the `facebook/dino-vits16` model is to implement a self-supervised learning method called DINO (self-DIstillation with NO labels). DINO simplifies self-supervised training by predicting the output of a teacher network, which is built with a momentum encoder, using a standard cross-entropy loss. The method also includes a centering and sharpening of the teacher output to avoid collapse. The model is trained using stochastic gradient descent and is designed to work without batch normalizations, making the system entirely BN-free.

The model has been shown to work effectively for tasks such as image classification and weakly supervised image segmentation, achieving high top-1 accuracy on the ImageNet linear classification benchmark. It also demonstrates the potential for self-supervised learning to be a key to developing BERT-like models for visual tasks.

### Compute Infrastructure

The compute infrastructure used for training the model `facebook/dino-vits16` includes the following:

1. The model was trained on the ImageNet dataset without labels.
2. The training utilized the AdamW optimizer with a batch size of 1024, distributed over 16 GPUs when using ViT-S/16 architecture.
3. The learning rate was linearly ramped up during the first 10 epochs to its base value using the linear scaling rule: `lr = 0.0005 * batchsize/256`. After the warmup period, the learning rate followed a cosine decay schedule.
4. The weight decay also followed a cosine schedule starting from 0.04.
5. The training was conducted on two 8-GPU servers for a total of 16 GPUs.
6. For a single node with 8 GPUs, the training time for 100 epochs was approximately 1.75 days, and the command provided for this setup is:
   ```
   python run_with_submitit.py --nodes 2 --ngpus 8 --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
   ```
7. The codebase was developed with Python version 3.6, PyTorch version 1.7.1, CUDA 11.0, and torchvision 0.8.2.

For more specific details on the compute infrastructure, such as the exact GPU models used or the total computational cost, [More Information Needed].

## Citation

```
@misc{mathilde-emerging,
    author = {Mathilde Caron and
              Hugo Touvron and
              Ishan Misra and
              Hervé Jegou and
              Julien Mairal and
              Piotr Bojanowski and
              Armand Joulin and
              Facebook Ai Research},
    title  = {Emerging Properties in Self-Supervised Vision Transformers},
    url    = {https://arxiv.org/pdf/2104.14294.pdf}
}
```

