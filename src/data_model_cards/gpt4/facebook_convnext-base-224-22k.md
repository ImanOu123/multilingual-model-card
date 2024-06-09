# Model Card for facebook/convnext-base-224-22k

The model facebook/convnext-base-224-22k is a pure ConvNet architecture that achieves high accuracy and efficiency in visual recognition tasks. It is pre-trained on the large-scale ImageNet-22K dataset and fine-tuned on ImageNet-1K, demonstrating competitive performance against Transformer models in various benchmarks including image classification, object detection, and segmentation.

## Model Details

### Model Description

Model Name: facebook/convnext-base-224-22k

### Model Architecture:
The ConvNeXt model is a pure ConvNet architecture that has been designed to compete with Transformer-based models in image classification tasks. It is based on the standard ConvNet design but incorporates several key design choices adapted from vision Transformers. The architecture is hierarchical, similar to vision Transformers like Swin-T, but it is constructed entirely from standard ConvNet modules. The base model has a number of blocks and channels at each stage that differ from the smaller ConvNeXt-T variant. The model also uses separate downsampling layers. For a detailed comparison of the architecture with other models like ResNet-50 and Swin-T, please refer to Table 9 in the provided references.

### Training Procedures:
The ConvNeXt model has been trained using a set of modern training techniques that enhance the performance of ConvNets. These techniques include the use of the AdamW optimizer, data augmentation methods such as Mixup, Cutmix, RandAugment, and Random Erasing, as well as regularization schemes including Stochastic Depth and Label Smoothing. The training was extended to 300 epochs, which is a significant increase from the original 90 epochs typically used for ResNets. The model uses Layer Normalization (LN) instead of Batch Normalization (BN), which has been observed to slightly improve performance, achieving an accuracy of 81.5%.

### Parameters:
The ConvNeXt model has approximately the same FLOPs as its counterparts in the same compute regime. The exact number of parameters for the facebook/convnext-base-224-22k variant is not provided in the references, so [More Information Needed] for the specific parameter count.

### Important Disclaimers:
- The exploration of the ConvNeXt model has been limited to a small scale, and while the findings are encouraging, they are not yet completely convincing. The scaling behavior of vision Transformers is a distinguishing factor that has not been fully replicated or challenged by ConvNeXt in the provided references.
- The performance of ConvNeXt on downstream tasks such as object detection and semantic segmentation is a central concern, and while the model has shown promising results, the question of whether it can compete with Swin Transformers in these areas remains open.
- Many of the design choices in ConvNeXt are not novel and have been researched separately over the last decade, but not collectively. The model's performance may challenge widely held views on the importance of convolution in computer vision, but further research and exploration are needed.

For further details and updates on the model, please refer to the official documentation and updates provided by the development team.

- **Developed by:** Zhuang Liu; Hanzi Mao; Chao-Yuan Wu; Christoph Feichtenhofer; Trevor Darrell; Saining Xie; A I Facebook;  Research; U C Berkeley
- **Funded by:** The project for the model facebook/convnext-base-224-22k is funded by Facebook AI Research and UC Berkeley.
- **Shared by:** The contributors that made the model facebook/convnext-base-224-22k available online as a GitHub repo are Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. They are affiliated with Facebook AI Research and UC Berkeley.
- **Model type:** The model facebook/convnext-base-224-22k is a convolutional neural network trained with modern techniques such as AdamW optimization and extended epochs, designed for image classification and other computer vision tasks, using a single-modality approach with visual data.
- **Language(s):** The model facebook/convnext-base-224-22k does not use or process natural human language; it is designed for computer vision tasks such as image classification, object detection, and semantic segmentation.
- **License:** The model `facebook/convnext-base-224-22k` is released under the MIT license. The link to the license can be found here: [LICENSE](LICENSE). However, since the actual URL is not provided in the reference text, I would normally need to provide a direct link to the license file in the repository. Since I can't access external content, I'm unable to provide the direct URL. 

Please note that in a real-world scenario, you would replace `(LICENSE)` with the actual URL pointing to the `LICENSE` file in the Huggingface or GitHub repository where the model is hosted.
- **Finetuned from model:** The model facebook/convnext-base-224-22k is fine-tuned from the ImageNet-22K pre-trained model. The name of the base model is ConvNeXt-B and the link to the base model is provided in reference 6:

[ConvNeXt-B 224x224](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth)
### Model Sources

- **Repository:** https://github.com/facebookresearch/ConvNeXt
- **Paper:** https://arxiv.org/pdf/2201.03545.pdf
- **Demo:** The demo of the model facebook/convnext-base-224-22k can be found on Hugging Face Spaces at the following link: [https://huggingface.co/spaces/akhaliq/convnext](https://huggingface.co/spaces/akhaliq/convnext).
## Uses

### Direct Use

The model `facebook/convnext-base-224-22k` can be used without fine-tuning, post-processing, or plugging into a pipeline for image classification tasks where the input images are of the size 224x224 pixels. Since the model has been pre-trained on ImageNet-22K and fine-tuned on ImageNet-1K, it is capable of classifying images into 1,000 different categories corresponding to the ImageNet-1K dataset.

To use the model for inference, you can load the pre-trained weights and pass an image through the model to obtain the predicted class probabilities. Here is a code snippet based on the provided references that demonstrates how to evaluate the model using a single GPU setup:

```python
import torch
from timm.models import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Load the pre-trained ConvNeXt model
model = create_model('convnext_base', pretrained=True)
model.eval()

# Resolve the data configuration for preprocessing
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

# Load and preprocess an image
img = Image.open('/path/to/image.jpg').convert('RGB')
tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Inference
with torch.no_grad():
    output = model(tensor)

# Get the top-1 prediction
pred_probabilities = torch.nn.functional.softmax(output[0], dim=0)
top1_prob, top1_catid = torch.max(pred_probabilities, dim=0)

print(f"Predicted category: {top1_catid.item()} with probability {top1_prob.item()}")
```

Please note that you need to replace `'/path/to/image.jpg'` with the actual path to the image you want to classify. Also, ensure that the necessary libraries (`torch`, `timm`, and `PIL`) are installed in your environment.

This code snippet does not require any fine-tuning or additional post-processing steps. It simply loads the model, preprocesses the input image, and performs a forward pass to get the prediction.

### Downstream Use

The `facebook/convnext-base-224-22k` model is a ConvNeXt model that has been pre-trained on the ImageNet-22K dataset and can be fine-tuned on a more specific task, such as image classification on the ImageNet-1K dataset. This model can be integrated into a larger ecosystem or application that requires image recognition capabilities, such as photo tagging, visual search, or automated content moderation.

When fine-tuning the model for a specific task, you would start with the pre-trained weights and continue training on a dataset that is relevant to your task. The fine-tuning process involves using a smaller learning rate and potentially other hyperparameters as mentioned in the references, such as layer-wise learning rate decay and specific data augmentations.

Here's an example of how you might fine-tune the `facebook/convnext-base-224-22k` model on a new dataset using PyTorch and the command line instructions provided in the references:

```python
# Single-GPU fine-tuning example
!python main.py --model convnext_base --eval false \
--resume https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/your/dataset \
--batch_size 512 --learning_rate 5e-5 --weight_decay 1e-8 \
--epochs 30 --layer_decay 0.65
```

For multi-GPU systems, you would use the distributed launch command:

```python
# Multi-GPU fine-tuning example
!python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_base --eval false \
--resume https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/your/dataset \
--batch_size 512 --learning_rate 5e-5 --weight_decay 1e-8 \
--epochs 30 --layer_decay 0.65
```

Please note that you would need to replace `/path/to/your/dataset` with the actual path to your dataset and adjust the batch size, learning rate, weight decay, and other hyperparameters as necessary for your specific task.

After fine-tuning, the model can be deployed into an application or service where it can perform image classification tasks. The application would preprocess input images to the required input size (224x224 pixels in this case), pass them through the model to obtain predictions, and then use those predictions according to the application's needs.

For more detailed instructions on fine-tuning, training, and evaluation, you should refer to the provided `TRAINING.md` and `INSTALL.md` documents. If you encounter any issues or have further questions, you can reach out to the project organizer for model card updates and support.

### Out-of-Scope Use

As a sociotechnic on the deep learning model development team, I would like to highlight potential areas of misuse for the model `facebook/convnext-base-224-22k` and provide guidance on what users should avoid doing with the model.

1. **Bias and Fairness**: The model has been pre-trained on the ImageNet-22K dataset, which, like many large-scale datasets, may contain biases. Users should not use the model in applications where these biases could lead to unfair or discriminatory outcomes without first taking steps to identify and mitigate potential biases in the data. For example, the model should not be used in surveillance or law enforcement contexts where it could exacerbate societal inequities.

2. **Robustness**: While the model has been evaluated on several robustness benchmark datasets, users should not assume that it is completely robust to all types of data corruption or adversarial attacks. It is important not to deploy the model in safety-critical systems where a lack of robustness could lead to harm, such as in autonomous vehicles or medical diagnosis, without thorough testing and validation specific to those environments.

3. **Environmental Impact**: The development of large models like `facebook/convnext-base-224-22k` requires significant computational resources, which in turn have environmental implications due to carbon emissions. Users should not use this model for trivial purposes or in a manner that requires frequent retraining or fine-tuning, which would unnecessarily increase the carbon footprint.

4. **Task Suitability**: The model has been shown to perform well on a variety of computer vision tasks, but it may not be the best choice for all applications. Users should not use the model for tasks where other architectures, such as Transformers with cross-attention modules, are known to be more suitable, especially in multi-modal learning or tasks requiring structured outputs.

5. **Intellectual Property and Privacy**: Users should not use the model to process data in ways that violate intellectual property rights or individuals' privacy. This includes unauthorized analysis of copyrighted images or the use of the model to derive sensitive information from images without consent.

6. **Misrepresentation of Capabilities**: Users should not overstate the capabilities of the model or present it as a solution to problems it has not been validated to solve. It is important to accurately represent the model's performance and limitations to avoid misleading stakeholders.

In summary, users of the `facebook/convnext-base-224-22k` model should exercise caution to ensure that their use cases are ethical, fair, and environmentally responsible, and that they respect privacy and intellectual property rights. They should also be mindful of the model's limitations and suitability for specific tasks.

### Bias, Risks, and Limitations

The model facebook/convnext-base-224-22k, as a deep learning model, presents several known and foreseeable issues that can be categorized into technical and sociotechnical limitations:

1. **Robustness and Fairness**: Reference 1 highlights the need for further investigation into the robustness behavior of ConvNeXt models. While ConvNeXt exhibits promising robustness behaviors (Reference 11), large models and datasets can introduce issues in terms of fairness. This suggests that there may be unforeseen biases or vulnerabilities in the model that could result in unfair treatment of certain groups or individuals when deployed in real-world applications.

2. **Environmental Impact**: The development of large-scale models like ConvNeXt requires significant computational resources, which in turn leads to increased carbon emissions (Reference 2). The environmental cost of training and maintaining such models is a growing concern in the field of AI, and it is important to consider the sustainability of these practices.

3. **Data Biases**: Reference 4 suggests that a responsible approach to data selection is necessary to avoid potential concerns with data biases. If the data used to train the model contains biases, the model may perpetuate or even amplify these biases, leading to unfair or inaccurate outcomes.

4. **Model Generalization**: While ConvNeXt models benefit from pre-training on large-scale datasets and demonstrate strong domain generalization capabilities (Reference 11), there is still a risk that the model may not perform equally well across all domains or in the face of distribution shifts. This could lead to reduced effectiveness in certain applications or for certain user groups.

5. **Task Suitability**: Reference 3 and 5 indicate that while ConvNeXt may perform well on a range of computer vision tasks, it may not be the best choice for all applications. For instance, tasks that require multi-modal learning or those that benefit from cross-attention modules may be better suited to Transformer models. Users may misunderstand the capabilities of ConvNeXt and attempt to apply it to unsuitable tasks, leading to suboptimal results.

6. **Complexity and Simplicity**: The pursuit of simplicity in model design is emphasized in Reference 2 and 5. However, the balance between simplicity and performance is delicate, and there may be trade-offs involved in choosing a simpler model over a potentially more complex but higher-performing one.

7. **Innovation and Research Direction**: Reference 7 and 10 suggest that the ConvNeXt model challenges widely held views about the importance of convolution in computer vision. This could lead to a shift in research focus, potentially overlooking other promising avenues of research or alternative approaches that may be more beneficial in the long term.

In summary, the ConvNeXt model, while demonstrating strong performance and robustness, brings with it concerns related to fairness, environmental impact, data biases, generalization to various domains, suitability for specific tasks, and the balance between simplicity and complexity in model design. These issues require careful consideration and ongoing research to mitigate potential harms and misunderstandings.

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model `facebook/convnext-base-224-22k`:

1. **Robustness and Fairness**: It is important to continue investigating the robustness behavior of ConvNeXt models, especially in comparison to Transformer models. This includes evaluating the model on diverse robustness benchmark datasets and considering the impact of specialized modules or additional fine-tuning procedures to enhance performance on these benchmarks.

2. **Carbon Emissions**: Given the large demands on computing resources for training models like ConvNeXt, there is a need to balance performance with environmental impact. Striving for simplicity in model design can help reduce carbon emissions. Future work should consider the carbon footprint of model training and seek ways to optimize computational efficiency.

3. **Data Selection and Bias**: A responsible approach to data selection is crucial to avoid potential concerns with data biases. When pre-training on large-scale datasets like ImageNet-22K, it is recommended to be circumspect in data curation to ensure the model does not perpetuate or amplify biases.

4. **Task Suitability**: The architecture choice should be aligned with the needs of the specific task. While ConvNeXt may excel in certain computer vision tasks, it is important to evaluate whether other architectures, such as Transformers with cross-attention modules, may be more suitable for multi-modal learning or tasks requiring structured outputs.

5. **Reevaluation of Convolution in Vision**: The ConvNeXt model challenges widely held views about the importance of convolution in computer vision. It is recommended to keep an open mind and reevaluate the role of convolutional layers in light of new evidence and research findings.

6. **Domain Generalization**: ConvNeXt models, particularly the large-scale variants, have shown strong domain generalization capabilities. It is recommended to explore and leverage these capabilities in applications where robustness to domain shifts is critical.

7. **Continued Research and Development**: The surprising observations made with the ConvNeXt model suggest that there is value in reexamining past design choices collectively. Continued research and development are recommended to further understand the interactions of these choices and their impact on model performance.

In summary, the recommendations emphasize the importance of robustness, environmental considerations, responsible data handling, task-specific architecture choices, reevaluation of convolutional approaches, domain generalization, and ongoing research to address the foreseeable issues with the `facebook/convnext-base-224-22k` model.

## Training Details

### Training Data

The training data for the model facebook/convnext-base-224-22k consists of the ImageNet-22K dataset, which includes approximately 14 million images spanning 21,841 classes, serving as a superset of the 1,000 classes found in the ImageNet-1K dataset. This pre-training on the larger dataset is followed by fine-tuning on the ImageNet-1K dataset for evaluation. [More Information Needed] on data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

The preprocessing steps for the data used in the model `facebook/convnext-base-224-22k` include several data augmentation and regularization techniques to improve the model's performance and generalization capabilities. Based on the references provided, here are the preprocessing details:

1. **Image Resizing**: The model is named `convnext-base-224-22k`, which suggests that the input images are resized to 224x224 pixels before being fed into the model. This is a common practice for models trained on ImageNet, as it standardizes the input size. [More Information Needed] for explicit mention of resizing in the provided references.

2. **Data Augmentation**: The training uses several data augmentation techniques to artificially expand the dataset and improve the model's robustness to variations in the input data. These techniques include:
   - **Mixup** [90]: This technique involves creating synthetic training examples by combining pairs of images and their labels in a weighted manner.
   - **Cutmix** [89]: This method involves cutting and pasting patches among training images where the labels are also mixed proportionally to the area of the patches.
   - **RandAugment** [14]: This is a method for automatically learning augmentation policies from data, which applies a series of random transformations to each image.
   - **Random Erasing** [91]: This technique randomly selects a rectangle region in an image and erases its pixels with random values.

3. **Regularization**: To prevent overfitting and improve the model's generalization, the following regularization techniques are applied:
   - **Stochastic Depth** [37]: This technique randomly drops a subset of layers during training to regularize the network.
   - **Label Smoothing** [69]: This technique softens the target labels, preventing the model from becoming overconfident about its predictions.

4. **Normalization and Other Preprocessing**: The references do not explicitly mention normalization or other preprocessing steps such as mean subtraction or division by standard deviation. However, these are common practices when training on ImageNet, so it is likely that such steps are included in the preprocessing pipeline. [More Information Needed] for explicit details.

5. **Layer Scale Initialization**: Layer Scale [74] with an initial value of 1e-6 is applied, although it is disabled for isotropic ConvNeXt-S/B as mentioned in reference 3.

6. **EMA Weights**: It is mentioned that Exponential Moving Average (EMA) weights are not used for fine-tuning, even if they were used during pre-training (reference 1).

7. **Resolution**: For fine-tuning at a higher resolution of 384x384, a crop ratio of 1.0 is used during testing, which implies no cropping (reference 5).

The preprocessing steps are crucial for training deep learning models as they can significantly affect the model's performance. The details provided are based on the common practices and the references given, but for a complete and precise description of the preprocessing pipeline, [More Information Needed] from the actual implementation or detailed documentation.

#### Training Hyperparameters

The training hyperparameters for the model facebook/convnext-base-224-22k are as follows:

- **Optimizer**: We used the AdamW optimizer [46], which is a variant of the Adam optimizer with a decoupled weight decay.
- **Learning Rate**: The initial learning rate was set to 4e-3.
- **Warmup**: A linear warmup was applied for the first 20 epochs.
- **Learning Rate Schedule**: After the warmup period, we employed a cosine decay schedule for the learning rate.
- **Batch Size**: The model was trained with a batch size of 4096.
- **Weight Decay**: We used a weight decay of 0.05.
- **Epochs**: The model was trained for 300 epochs.
- **Data Augmentation**: We utilized several data augmentation techniques, including Mixup [90], Cutmix [89], RandAugment [14], and Random Erasing [91].
- **Regularization**: The networks were regularized using Stochastic Depth [37] with rates of 0.1/0.2/0.5 for isotropic ConvNeXt-S/B/L (specific rates for the base model are not provided, so [More Information Needed] here), and Label Smoothing [69] with a value of 0.1.
- **Layer Scale Initialization**: Layer Scale [74] was applied with an initial value of 1e-6.
- **EMA**: Exponential Moving Average (EMA) was not used during training, as it was found to severely hurt models with BatchNorm layers [5].

Please note that some specific details, such as the stochastic depth rate for the base model, are not provided in the references and would require further information.

#### Speeds, Sizes, Times

The ConvNeXt-Base model pre-trained on ImageNet-22K and fine-tuned on ImageNet-1K has been designed to provide efficient inference throughput while maintaining a high level of accuracy. Here are the details regarding the model's throughput, timing, and checkpoint sizes:

Throughput: As per the references, the ConvNeXt models, including the ConvNeXt-Base variant, achieve faster inference throughput on A100 GPUs compared to V100 GPUs. Specifically, ConvNeXt models can be up to 49% faster than Swin Transformers when benchmarked on A100 GPUs with support for TensorFloat32 (TF32) tensor cores. The exact throughput numbers for the ConvNeXt-Base model are not provided in the references, so [More Information Needed] for the specific throughput of the facebook/convnext-base-224-22k model.

Start or End Time: The references do not provide explicit start or end times for the model inference. However, the improved throughput on A100 GPUs suggests that the model can process images more quickly than on previous hardware. For exact inference times, [More Information Needed].

Checkpoint Sizes: The references do not specify the checkpoint sizes for the ConvNeXt-Base model. Checkpoint size typically depends on the number of parameters and the precision of the weights stored. Since the ConvNeXt-Base is a variant with a similar number of parameters to the Swin-Base model, it can be inferred that the checkpoint size would be comparable to that of Swin-Base or other similar models. However, for the exact checkpoint size of the facebook/convnext-base-224-22k model, [More Information Needed].

Additional details such as the number of parameters, FLOPs, and FPS measured on an A100 GPU for the ConvNeXt models at a resolution of 224x224 are mentioned in the references, but specific numbers for the ConvNeXt-Base variant are not provided. For comprehensive details, one should refer to the full paper or the appendix mentioned in the references.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model facebook/convnext-base-224-22k has been evaluated on the following benchmarks or datasets:

1. COCO dataset for object detection and segmentation tasks, using Mask R-CNN and Cascade Mask R-CNN frameworks.
2. ADE20K dataset for semantic segmentation tasks.
3. ImageNet-1K dataset for classification robustness evaluation, including testing on robustness benchmark datasets such as ImageNet-A, ImageNet-R, ImageNet-Sketch, and ImageNet-C/C datasets.

#### Factors

The model facebook/convnext-base-224-22k is a ConvNeXt model variant that has been pre-trained on the ImageNet-22K dataset and is designed for image classification tasks. Based on the provided references, the following characteristics are likely to influence the model's behavior:

1. **Robustness to Image Corruption and Domain Shifts**: The model has demonstrated strong robustness behaviors, outperforming other state-of-the-art models on several benchmarks (Reference 2). It has been tested on robustness benchmark datasets like ImageNet-A, ImageNet-R, ImageNet-Sketch, and ImageNet-C/C, showing promising results in terms of mean corruption error and top-1 accuracy (Reference 1). This suggests that the model should perform well when faced with image corruptions or domain shifts that are similar to those present in these robustness datasets.

2. **Domain Generalization Capabilities**: With the additional ImageNet-22K data, the model has shown strong domain generalization capabilities, which is evident from its performance on ImageNet-A/R/Sketch benchmarks (Reference 2). This indicates that the model can generalize across different visual domains, which is beneficial for applications where the input data may differ from the data seen during training.

3. **Task Suitability**: While ConvNeXt models, including facebook/convnext-base-224-22k, have been shown to perform well on a range of computer vision tasks, they may be more suited for certain tasks over others (Reference 3). For example, tasks that benefit from the inductive biases of convolutional networks, such as image classification and object detection, are likely to see better performance. However, for multi-modal learning or tasks requiring structured outputs, Transformers might be more flexible (Reference 4).

4. **Scalability and Model Variants**: The ConvNeXt model comes in different variants (T/S/B/L/XL) that differ in complexity, which allows for scalability based on the requirements of the task at hand (Reference 5). The base model, which is the subject of this card, is designed to be of similar complexity to the Swin-B model, suggesting that it is suitable for tasks that require a balance between performance and computational efficiency.

5. **Data Dependency**: The model's performance benefits from pre-training on large-scale datasets (Reference 6). This implies that the quality and diversity of the pre-training data can significantly influence the model's behavior and its ability to generalize to new data.

6. **Population Subgroups**: The references do not provide specific information on the model's performance across different population subgroups. Therefore, [More Information Needed] regarding the disaggregated evaluation across factors such as age, gender, or ethnicity to uncover disparities in performance.

7. **Design Choices and Historical ConvNet Biases**: The ConvNeXt model incorporates design choices that have been examined separately over the last decade, but not collectively (Reference 7). This could mean that the model may inherit biases from historical ConvNet architectures, which could influence its performance on certain tasks or datasets.

In summary, the facebook/convnext-base-224-22k model is expected to be robust to image corruptions and domain shifts, with strong domain generalization capabilities. It is suitable for a range of computer vision tasks, particularly those that align with the strengths of convolutional networks. However, the performance across different population subgroups is not specified and would require further investigation to ensure fairness and avoid disparities.

#### Metrics

The evaluation of the model facebook/convnext-base-224-22k will utilize the following metrics:

1. Top-1 Accuracy (Acc@1): This is the standard accuracy metric where the highest probability prediction is compared to the ground truth label. The reference indicates that the model achieves an Acc@1 of 85.820.

2. Top-5 Accuracy (Acc@5): This metric checks if the true label is among the top 5 predictions made by the model. The model achieves an Acc@5 of 97.868.

3. Loss: The model's performance is also measured by the loss, which in this case is 0.563. The loss metric helps in understanding how well the model is fitting the data.

4. Mean Corruption Error (mCE) for ImageNet-C: This is a robustness metric that measures the model's performance on corrupted versions of the ImageNet dataset.

5. Corruption Error for ImageNet-C: Similar to mCE, this metric evaluates the model's robustness to various types of data corruption.

6. Robustness Benchmarks: The model is also evaluated on robustness benchmark datasets such as ImageNet-A, ImageNet-R, ImageNet-Sketch, and ImageNet-C/C datasets, with top-1 Accuracy reported for all datasets except ImageNet-C, where mCE is reported.

The model has demonstrated strong domain generalization capabilities and robustness behaviors without the need for specialized modules or additional fine-tuning procedures. These metrics will help in understanding the tradeoffs between different errors and the overall robustness of the model.

### Results

The ConvNeXt-Base model, referred to as ConvNeXt-B, is designed to be comparable in complexity to the Swin-B variant. It is a product of modernizing the architecture in the ResNet-200 regime. The model has been evaluated on the ImageNet-1K dataset and has shown competitive results when compared to other state-of-the-art models such as DeiT, Swin Transformers, RegNets, and EfficientNets.

In terms of accuracy-computation trade-off and inference throughputs, ConvNeXt-B competes favorably with strong ConvNet baselines like RegNet and EfficientNet. The model has been trained with improved settings, including longer warmup epochs, and the evaluation results at a resolution of 224x224 are presented in Table 2, which is not provided in the references above.

For robustness and out-of-domain generalization, ConvNeXt models, including the ConvNeXt-B variant, have been tested on several robustness benchmark datasets such as ImageNet-A, ImageNet-R, ImageNet-Sketch, and ImageNet-C/C datasets. The evaluation results include mean corruption error (mCE) for ImageNet-C, corruption error for ImageNet-C, and top-1 Accuracy for all other datasets. These results are detailed in Table 8, which is not included in the provided references.

The model's frames per second (FPS) performance was measured on an A100 GPU, and the FLOPs were calculated with an image size of (1280, 800). However, specific numbers for FPS and FLOPs are not provided in the references above.

In summary, the ConvNeXt-Base model demonstrates strong performance in terms of accuracy and efficiency, with promising robustness and generalization capabilities. However, for detailed numerical evaluation results, including specific accuracy figures, FPS, FLOPs, and robustness metrics, [More Information Needed] as they are not included in the provided references.

#### Summary

The ConvNeXt model, specifically the `facebook/convnext-base-224-22k` variant, has been evaluated on several benchmarks and compared with other state-of-the-art models. Here's a summary of the evaluation results:

1. **Comparison with Other Models**: The ConvNeXt model shows competitive performance when compared with recent Transformer variants such as DeiT and Swin Transformers, as well as ConvNets from architecture search like RegNets and EfficientNets. It offers a favorable accuracy-computation trade-off and inference throughputs when compared with strong ConvNet baselines like RegNet and EfficientNet.

2. **Model Variants**: The ConvNeXt model comes in different variants (T/S/B/L/XL) to match the complexities of the Swin Transformer variants. The `facebook/convnext-base-224-22k` is a product of modernizing the ResNet-50/200 regime. The variants differ in the number of channels and blocks in each stage, with the number of channels doubling at each new stage.

3. **Training and Fine-tuning**: The ConvNeXt models are trained with improved settings, including longer warmup epochs. For the `facebook/convnext-base-224-22k`, pre-training was conducted on the ImageNet-22K dataset, which contains around 14 million images across 21841 classes, followed by fine-tuning on the ImageNet-1K dataset.

4. **Robustness and Generalization**: Additional robustness evaluations were performed on datasets like ImageNet-A, ImageNet-R, ImageNet-Sketch, and ImageNet-C/C. The ConvNeXt models report mean corruption error (mCE) for ImageNet-C and top-1 accuracy for the other datasets.

5. **Performance Metrics**: The model's performance is measured in terms of top-1 accuracy on the ImageNet-1K validation set, frames per second (FPS) on an A100 GPU, and FLOPs calculated with an image size of (1280, 800).

6. **Datasets**: The ImageNet-1K dataset used for evaluation consists of 1.2 million training images across 1000 object classes. The top-1 accuracy reported is based on this validation set.

In summary, the `facebook/convnext-base-224-22k` model demonstrates strong performance across various benchmarks, showing good accuracy and robustness, with detailed training setups and configurations available in the referenced appendices.

## Model Examination

Explainability/Interpretability of facebook/convnext-base-224-22k:

Our ConvNeXt model, while primarily focused on performance across various computer vision tasks, also raises important considerations regarding model explainability and interpretability. The ConvNeXt architecture, being a pure ConvNet, leverages the well-understood mechanics of convolutional layers, which have been extensively studied over the last decade. This familiarity with convolutional operations may offer some advantages in terms of interpretability over Transformer-based models, which often require more complex mechanisms to understand their decision-making processes.

However, as our model scales in size and is pre-trained on large-scale datasets, the interpretability of the model's internal representations and predictions becomes more challenging. The performance improvements seen with ConvNeXt suggest that the model is effectively capturing complex patterns and feature interactions, but understanding these interactions at a granular level remains an open area of research.

In the context of robustness and fairness, which are closely tied to interpretability, further investigation is warranted. As noted in our references, the robustness behavior of ConvNeXt compared to Transformer models is an interesting direction for future research. This includes understanding how the model behaves under various perturbations and whether its predictions are fair and unbiased across different demographic groups.

To address these challenges, we encourage the community to explore methods that can provide insights into the model's decision-making process. Techniques such as feature visualization, saliency maps, and layer-wise relevance propagation could be adapted and applied to ConvNeXt to shed light on how the model processes visual information.

In summary, while our ConvNeXt model demonstrates strong performance, there is a need for continued research into its explainability and interpretability. We hope that the simplicity of the convolutional approach, combined with the right analytical tools, will allow researchers and practitioners to gain a better understanding of the model's inner workings and ensure its responsible use in real-world applications.

## Environmental Impact

- **Hardware Type:** The model facebook/convnext-base-224-22k was trained on A100 GPUs.
- **Software Type:** The model facebook/convnext-base-224-22k is trained using PyTorch version 1.10.
- **Hours used:** The model facebook/convnext-base-224-22k underwent several stages of training, including pre-training and fine-tuning. According to the references provided:

1. Pre-training on ImageNet-22K was conducted for 90 epochs with a warmup of 5 epochs.
2. Training on ImageNet-1K was then carried out for 300 epochs with a 20-epoch linear warmup.
3. Fine-tuning on ImageNet-1K was done for 30 epochs.

However, the references do not provide specific information about the actual wall-clock time taken for each epoch or the total training time. Therefore, to provide the amount of time used to train the model, we would need additional information such as the time per epoch or the overall training duration.

[More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model `facebook/convnext-base-224-22k` is part of the ConvNeXt family, which is a series of pure ConvNet models. The architecture of ConvNeXt models is inspired by design choices adapted from vision Transformers, but they are constructed entirely from standard ConvNet modules. The ConvNeXt models, including the `facebook/convnext-base-224-22k`, are designed to compete favorably with Transformer-based models like Swin Transformers in terms of accuracy and scalability.

The objective of the `facebook/convnext-base-224-22k` model is to provide a high-performing ConvNet for visual recognition tasks. It aims to achieve this by leveraging the simplicity and efficiency of standard ConvNets while incorporating design choices that have been researched separately over the last decade but not collectively. The model has been shown to outperform Swin Transformers on benchmarks such as ImageNet classification, COCO detection, and ADE20K segmentation.

The `facebook/convnext-base-224-22k` model specifically refers to a base-sized ConvNeXt model that has been pre-trained on an ImageNet dataset with 22k classes and is designed to process images with a resolution of 224x224 pixels. The model maintains the fully-convolutional nature for both training and testing, which makes it extremely simple to deploy and use across different application scenarios.

For the detailed architecture specifications, such as the number of blocks and channels at each stage, reference is made to Table 9 in the provided documents. However, since the actual content of Table 9 is not included in the references provided, we would need more information to give specific details about the architecture of the `facebook/convnext-base-224-22k` model. [More Information Needed] for the exact architecture specifications.

### Compute Infrastructure

The compute infrastructure used for the model facebook/convnext-base-224-22k includes training on NVIDIA A100 GPUs, which support TensorFloat32 (TF32) tensor cores. The models were benchmarked for inference throughput on both V100 and A100 GPUs, with the A100 providing a significant speedup. PyTorch version 1.10 was used to take advantage of the latest "Channel Last" memory layout for further speed improvements. The training was conducted with a batch size of 4096, and the inference throughput was measured on an A100 GPU. FLOPs were calculated with an image size of (1280, 800). No specific information about the number of GPUs used or the total computational resources involved is provided in the references, so for those details, [More Information Needed].

## Citation

```
@misc{zhuang-a,
    author = {Zhuang Liu and
              Hanzi Mao and
              Chao-Yuan Wu and
              Christoph Feichtenhofer and
              Trevor Darrell and
              Saining Xie and
              A I Facebook and
              Research and
              U C Berkeley},
    title  = {A ConvNet for the 2020s},
    url    = {https://arxiv.org/pdf/2201.03545.pdf}
}
```

