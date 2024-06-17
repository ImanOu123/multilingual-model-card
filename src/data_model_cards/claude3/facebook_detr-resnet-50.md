# Model Card for facebook/detr-resnet-50

The facebook/detr-resnet-50 model is a DEtection TRansformer (DETR) model that views object detection as a direct set prediction problem. It uses a transformer encoder-decoder architecture with a ResNet-50 CNN backbone, and achieves comparable performance to Faster R-CNN on the COCO dataset while being conceptually simpler and easier to implement.

## Model Details

### Model Description

Model Card for facebook/detr-resnet-50

Model Architecture:
The DETR (DEtection TRansformer) architecture consists of three main components:
1. A CNN backbone (ResNet-50) to extract a compact feature representation from the input image
2. An encoder-decoder transformer that reasons about the relations of the objects and the global image context
3. A simple feed forward network (FFN) that makes the final detection predictions (class and bounding box) for a fixed set of learned object queries

Training Procedure:
- Trained with AdamW optimizer
- Learning rate: 1e-4 in the transformer, 1e-5 in the backbone
- Data augmentation: Horizontal flips, scales and crops
- Images rescaled to have min size 800 and max size 1333
- Transformer trained with dropout of 0.1
- Whole model trained with gradient clipping of 0.1
- Loss function: Linear combination of L1 and GIoU losses for bounding box regression
- Trained with N = 100 decoder query slots

Parameters:
[More Information Needed]

Important Disclaimers:
- DETR is a versatile and extensible model that can be easily reproduced in any deep learning framework providing a common CNN backbone and transformer architecture implementation.
- Unlike many modern detectors, DETR doesn't require any customized layers or specialized libraries.
- DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster R-CNN baseline on the challenging COCO object detection dataset.
- DETR achieves significantly better performance on large objects than Faster R-CNN, likely thanks to the global reasoning of the transformer architecture.

For more information or updates to this model card, please contact the project organizer at [More Information Needed].

- **Developed by:** Nicolas Carion; Francisco Massa; Gabriel Synnaeve; Nicolas Usunier; Alexander Kirillov; Sergey Zagoruyko; Facebook Ai
- **Funded by:** Based on the provided references, the model facebook/detr-resnet-50 appears to be funded by Facebook AI Research (FAIR). Specifically:

- Reference 1 thanks the Facebook AI Research Paris team for discussions and advice that made the work possible.
- Reference 2 lists the authors as being from Facebook AI. 
- Reference 5 mentions that training code and pretrained models are available at a Facebook Research GitHub repo.

So in summary, the project seems to be funded and supported by the Facebook AI Research organization. However, to say this definitively, [More Information Needed] to fully confirm Facebook is the sole funder.
- **Shared by:** Based on the provided references, the main contributors to the DETR model (facebook/detr-resnet-50) are:

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko from Facebook AI Research (Reference 2).

The references also thank Sainbayar Sukhbaatar, Piotr Bojanowski, Natalia Neverova, David Lopez-Paz, Guillaume Lample, Danielle Rothermel, Kaiming He, Ross Girshick, Xinlei Chen and the whole Facebook AI Research Paris team for discussions and advice (Reference 1).

However, there is no explicit information about who made the model available online as a GitHub repository. [More Information Needed]
- **Model type:** The facebook/detr-resnet-50 model is an end-to-end object detection model that uses a ResNet-50 CNN backbone, a transformer encoder-decoder architecture, and a set-based global loss for direct set prediction, trained on the COCO dataset.
- **Language(s):** The facebook/detr-resnet-50 model does not directly use or process natural human language. It is an object detection model that predicts bounding boxes and category labels for objects of interest in images.
- **License:** The DETR model is released under the Apache 2.0 license. The link to the license file is provided in the repository:

[LICENSE](https://github.com/facebookresearch/detr/blob/main/LICENSE)
- **Finetuned from model:** Based on the provided references, the DETR model uses an ImageNet-pretrained ResNet backbone from torchvision as the base model. Specifically, reference 6 states:

"We report results with two different backbones: a ResNet-50 and a ResNet-101. The corresponding models are called respectively DETR and DETR-R101. Following [21], we also"

and 

"the backbone is with ImageNet-pretrained ResNet model [15] from torchvision with frozen batchnorm layers."

So the base model for facebook/detr-resnet-50 is an ImageNet-pretrained ResNet-50 model from the torchvision library. However, no specific link to this base model is provided in the given references, so [More Information Needed] for the link.
### Model Sources

- **Repository:** https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py
- **Paper:** https://arxiv.org/pdf/2005.12872.pdf
- **Demo:** Here are the relevant links to demos of the DETR model based on the provided references:

[Standalone Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb): This notebook demonstrates how to implement a simplified version of DETR from the ground up in 50 lines of Python, then visualize the predictions. It's a good starting point to gain a better understanding of the architecture.

[DETR's hands on Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb): Shows how to load a model from hub, generate predictions, then visualize the attention of the model (similar to the figures in the paper).

[Panoptic Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb): Demonstrates how to use DETR for panoptic segmentation and plot the predictions.

The references don't specify a demo link for the exact model facebook/detr-resnet-50, but these notebooks cover the core DETR model and architecture that facebook/detr-resnet-50 is based on. The notebooks can likely be adapted to work with the facebook/detr-resnet-50 weights.
## Uses

### Direct Use

The DETR model (facebook/detr-resnet-50) can be used for inference without requiring fine-tuning, post-processing, or plugging into a pipeline. The model simplifies the object detection pipeline by eliminating the need for hand-designed components like non-maximum suppression or anchor generation.

To use the model for inference, you can follow these steps:

1. Clone the DETR repository:
```
git clone https://github.com/facebookresearch/detr.git
```

2. Install the required dependencies (PyTorch 1.5+ and torchvision 0.6+):
```
conda install -c pytorch pytorch torchvision
```

3. Install additional dependencies (pycocotools for evaluation on COCO and scipy for training):
```
conda install cython scipy
```

The provided code snippet in the references demonstrates how to perform inference with DETR using PyTorch and Torchvision libraries. However, the complete code snippet is not available in the given references.

[More Information Needed] regarding the specific inference code snippet for the facebook/detr-resnet-50 model.

Overall, DETR's architecture is designed to be simple and easy to use, without requiring any customized layers or specialized libraries. This makes it straightforward to reproduce and use the model for inference in any deep learning framework that supports a common CNN backbone and transformer architecture.

### Downstream Use

The facebook/detr-resnet-50 model can be fine-tuned for object detection tasks or plugged into a larger ecosystem or app as follows:

1. DETR can be extended to predict segmentation masks in addition to bounding boxes, enabling it to perform panoptic segmentation. This shows DETR is a versatile and extensible model.

2. DETR has minimal package dependencies and no extra compiled components, making it simple to use. It can be installed and run with Python 3.6+, PyTorch 1.4+ and Torchvision 0.5+. Here are the key steps:

```
# Clone the DETR repository
git clone https://github.com/facebookresearch/detr.git

# Install PyTorch 1.5+ and torchvision 0.6+ 
conda install -c pytorch pytorch torchvision

# Install pycocotools (for COCO evaluation) and scipy (for training)  
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# (Optional) Install panopticapi to work with panoptic segmentation
pip install git+https://github.com/cocodataset/panopticapi.git
```

3. DETR's simple design of using a transformer encoder-decoder architecture makes it conceptually straightforward and reproducible in any deep learning framework that supports transformers. The full code is available.

4. [More Information Needed] on specific code snippets for fine-tuning DETR or plugging it into a larger system. The provided code covers installation and inference only.

In summary, DETR's versatility and simplicity make it promising to adapt for various object detection and panoptic segmentation applications. The available open-source code provides a good starting point for using DETR in practice.

### Out-of-Scope Use

Based on the provided references about the DETR (DEtection TRansformer) model, here are some potential misuses and what users ought not do with the facebook/detr-resnet-50 model:

1. The model should not be used for real-time critical applications without thorough testing and validation, as the references mention challenges regarding training, optimization, and performance on small objects that may need further improvements.

2. Users should not assume the model's predictions are always accurate, especially for small objects, as the references indicate DETR achieves significantly better performance on large objects compared to Faster R-CNN, but may have limitations with smaller objects.

3. The model should not be used for tasks beyond object detection and panoptic segmentation without proper evaluation and adaptation, as the references focus on DETR's performance and generalization specifically for these tasks.

4. [More Information Needed] regarding potential misuses related to privacy, bias, or ethical concerns, as the provided references do not cover these aspects in detail.

5. [More Information Needed] about the specific training data and any limitations or biases it may introduce, which could lead to misuse if the model is applied to significantly different data or contexts without proper consideration.

Users should be aware of these potential limitations and use the model responsibly, ensuring appropriate testing, monitoring, and adaptation for their specific use cases while considering the ethical implications and potential unintended consequences of its application.

### Bias, Risks, and Limitations

Based on the provided references about the DETR (DEtection TRansformer) object detection model facebook/detr-resnet-50, here are some known or foreseeable issues:

Technical limitations:
- DETR has challenges with training, optimization and performance on small objects compared to established detectors like Faster R-CNN (Ref 3). More work is needed to address these issues.
- By design, DETR cannot predict more objects than it has query slots, which is set to 100 in the experiments (Ref 9). This limits the maximum number of objects it can detect in an image.
- A single decoding layer of the transformer is prone to making multiple predictions for the same object, requiring post-processing like non-maximum suppression (NMS) to remove duplicates (Ref 7).

Foreseeable harms and misunderstandings:
- [More Information Needed] The provided references do not contain enough information to assess potential harms or misunderstandings.

Sociotechnical limitations: 
- [More Information Needed] The references focus on technical aspects and do not provide insights into sociotechnical limitations of the model.

To summarize, the key limitations seem to be around handling small objects, the maximum number of detectable objects, and duplicate predictions. More information would be needed to thoroughly assess potential harms, misunderstandings and sociotechnical implications. The model shows promising results but will likely require further development to address the identified challenges.

### Recommendations

Based on the provided references about the DETR (DEtection TRansformer) model facebook/detr-resnet-50, here are some foreseeable issues and recommendations:

1. Training challenges: DETR requires an extra-long training schedule and benefits from auxiliary decoding losses in the transformer. Careful tuning of training settings is crucial for optimal performance. (Reference 11)

2. Performance on small objects: DETR may face challenges in detecting small objects compared to established detectors. Future work should address improving performance on small objects. (Reference 5)

3. Specialization of object queries: Each object query slot in DETR learns different specializations, focusing on different areas and box sizes. This behavior should be further analyzed and potentially optimized. (Reference 7)

4. Positional encodings: While positional encodings contribute to performance, their exact role and optimal configuration in DETR should be further investigated. (Reference 4, 8)

5. Generalization to unseen numbers of instances: Although DETR shows promising generalization to unseen numbers of instances (Reference 6), more extensive testing on out-of-distribution scenarios is recommended.

6. Computational efficiency: [More Information Needed] on the computational efficiency of DETR compared to other detectors and potential optimizations.

7. Robustness and failure cases: [More Information Needed] on the robustness of DETR to various input perturbations and an analysis of common failure cases.

8. Fairness and bias: [More Information Needed] on evaluating the model's performance across different object categories, image types, and demographics to ensure fairness and identify potential biases.

Overall, while DETR shows promising results and simplifies the detection pipeline, further research is needed to address the identified challenges and ensure its reliable performance in real-world applications.

## Training Details

### Training Data

The model facebook/detr-resnet-50 is trained on COCO 2017 detection dataset, which contains 118k training images and 5k validation images annotated with bounding boxes. The images are resized such that the shortest side is at least 480 and at most 800 pixels while the longest side is at most 1333 pixels, and random crop augmentations are applied during training.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about the preprocessing for the data of the model facebook/detr-resnet-50:

Image Resizing:
We use scale augmentation, resizing the input images such that the shortest side is at least 480 and at most 800 pixels while the longest at most 1333. (Reference 2)

Random Crop Augmentation:
To help learning global relationships through the self-attention of the encoder, we also apply random crop augmentations during training, improving the performance by approximately 1 AP. Specifically, a train image is cropped with probability 0.5 to a random rectangular patch which is then resized again to 800-1333. (Reference 2)

[More Information Needed] for details on tokenization and other preprocessing steps.

#### Training Hyperparameters

Here are the detailed training hyperparameters for the facebook/detr-resnet-50 model based on the provided references:

Model Architecture:
- Backbone: ResNet-50 pretrained on ImageNet, imported from Torchvision. The last classification layer is discarded and batch normalization weights and statistics are frozen during training.
- Transformer: Initialized with Xavier initialization. Trained with a learning rate of 10^-4.

Losses:
- Bounding Box Regression: Linear combination of L1 and GIoU losses with weights λ_L1 = 5 and λ_iou = 2.
- Number of decoder query slots: N = 100

Optimization:
- Optimizer: AdamW 
- Transformer Learning Rate: 10^-4
- Backbone Learning Rate: 10^-5 (roughly an order of magnitude smaller than the rest of the network)
- Weight Decay: 10^-4
- Additive Dropout: 0.1 applied after every multi-head attention and FFN before layer normalization in the transformer
- Gradient Clipping: 0.1

Data Augmentation:
- Horizontal flips, scales and crops
- Images rescaled to have min size 800 and max size 1333

Training Duration:
- Number of Epochs: 300
- Time per Epoch: ~28 minutes on 8 V100 GPUs
- Total Training Time: ~6 days on a single machine with 8 V100 GPUs

Training Command:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco
```

[More Information Needed] on the exact learning rate schedule, batch size, and any other unspecified hyperparameters.

#### Speeds, Sizes, Times

Based on the provided references, here are the details I can provide about the facebook/detr-resnet-50 model:

Throughput:
- DETR with ResNet-50 backbone processes 500 object queries in 0.050 seconds per image (References 6, 8)

Checkpoint Size:
- The pretrained model file for DETR with ResNet-50 backbone is 159MB (References 6, 8)

Performance:
- DETR with ResNet-50 achieves 42.0 AP (Average Precision) on the COCO dataset (Reference 6)
- DETR demonstrates significantly better performance on detecting large objects compared to Faster R-CNN, likely due to the non-local computations of the transformer (Reference 10)

[More Information Needed] for start/end training time, as those details were not provided in the given references.

The model and training logs are publicly available at:
- Model: https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
- Logs: https://dl.fbaipublicfiles.com/detr/logs/detr-r50_log.txt
(Reference 6)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the DETR model with a ResNet-50 backbone (facebook/detr-resnet-50) is evaluated on the following datasets:

1. COCO 2017 detection dataset (References 3, 7, 9)
   - Contains 118k training images and 5k validation images
   - Each image is annotated with bounding boxes
   - There are 7 instances per image on average, up to 63 instances in a single image in the training set, ranging from small to large on the same images

2. COCO 2017 panoptic segmentation dataset (Reference 3)
   - Same image data as COCO 2017 detection dataset
   - Each image is annotated with panoptic segmentation in addition to bounding boxes

The model's performance is compared against a competitive Faster R-CNN baseline on the COCO dataset (References 1, 7, 9).

#### Factors

Based on the provided references about the DETR (DEtection TRansformer) object detection model facebook/detr-resnet-50, here are some key characteristics that may influence its behavior:

Domain and Context:
- The model was trained and evaluated on the COCO dataset, which contains common objects in natural scenes. Performance on other domains like medical imaging, satellite imagery, etc. is unclear without further testing. [More Information Needed]

- The model can generalize to unseen numbers of instances of a class in an image (e.g. detecting 24 giraffes when trained on at most 13). This suggests robustness to some distribution shift in instance counts.

Population Subgroups:
- [More Information Needed] The references do not provide a detailed breakdown of performance across different object categories, sizes, frequencies, etc. Evaluation disaggregated along these factors would help uncover any performance disparities.

- The model with a dilated C5 stage (DETR-DC5) is reported to improve small object detection at increased computational cost, indicating potential trade-offs.

Other Factors:
- Different object query slots in the decoder specialize to different object sizes and locations. The distribution of objects in the training data (COCO) likely influences this.

- The model is competitive with Faster R-CNN in quantitative evaluation on COCO. Detailed comparisons to other architectures would further characterize its relative strengths and weaknesses.

In summary, key factors are the natural images domain it was trained on, ability to generalize to unseen numbers of instances, and specialization of object query slots. However, more disaggregated evaluation is needed to characterize performance across different subgroups and uncover any disparities or trade-offs.

#### Metrics

Based on the provided references, the main evaluation metric used for the DETR ResNet-50 model on the COCO dataset is Average Precision (AP), which is the integral metric over multiple thresholds (Reference 9). Specifically, bbox AP is reported as the primary metric (Reference 10).

The model's performance is compared against a competitive Faster R-CNN baseline. DETR demonstrates significantly better performance on large objects, likely due to the non-local computations of the transformer (Reference 10).

[More Information Needed] regarding specific tradeoffs between different errors for the DETR ResNet-50 model, as this information is not explicitly provided in the given references.

To evaluate the model on COCO val5k with a single GPU, the following command can be used (Reference 2):

```
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /path/to/coco
```

Note that the performance numbers may vary depending on the batch size (number of images) per GPU (Reference 5).

### Results

Evaluation Results of facebook/detr-resnet-50:

Performance:
- Achieves competitive results compared to Faster R-CNN in quantitative evaluation on COCO (Reference 2)
- ResNet-50-based DETR model with 6 encoder, 6 decoder layers and width 256 achieves 40.6 and 42.0 AP on short and long schedules respectively (Reference 7)

Efficiency: 
- Runs at 28 FPS, similarly to Faster R-CNN (Reference 7)

Robustness:
- When approaching the limit of 100 query slots, DETR starts saturating and misses more instances as the number of instances in the image increases towards 100 (Reference 5)
- This behavior is consistent across object classes (Reference 5)

Fairness:
[More Information Needed]

Transparency:
- Code and pretrained models to reproduce experiments are provided at https://github.com/facebookresearch/detr (Reference 2)

Accountability:
[More Information Needed]

#### Summary

Here is a summary of the evaluation results for the facebook/detr-resnet-50 model based on the provided references:

- DETR achieves competitive results compared to a strong Faster R-CNN baseline on the COCO dataset, with significantly better performance on detecting large objects. This is likely enabled by the non-local computations of the transformer architecture.

- By design, DETR has an upper limit on the number of object instances it can detect (100 in the experiments). Synthetic tests show DETR generalizes well to unseen numbers of instances of a class, detecting up to 50 instances near-perfectly, more than seen during training. 

- Each decoder output slot learns to specialize in detecting objects of different sizes and locations. All slots can also predict full image-wide boxes.

- DETR can be easily extended to panoptic segmentation by training a small extension on top of the fixed DETR model, achieving competitive results.

[More Information Needed] on the exact quantitative results (e.g. AP numbers) of DETR vs Faster R-CNN.

In summary, DETR demonstrates strong object detection capabilities compared to highly optimized baselines, with advantages in global reasoning and flexibility to extend to new tasks. More details on some quantitative comparisons would help strengthen the evaluation summary.

## Model Examination

Here is a possible answer for the "Explainability/Interpretability" section of the model card for facebook/detr-resnet-50:

The DETR model provides some built-in explainability and interpretability through its attention mechanisms:

- The attention maps of the encoder can be visualized to see what image regions the model is attending to (Reference 7). This provides insight into what the model considers important for detecting objects.

- The attention maps of the decoder can be visualized for each predicted object to see the local region the model attends to when predicting that object (Reference 7). 

- By analyzing the decoder output slots, we observe that DETR learns specializations for each slot, focusing on objects of different sizes and locations. All slots can also predict image-wide boxes. This provides insight into how DETR reasons about the distribution of objects. (Reference 5)

Some example notebooks are provided to visualize the attention maps of the model:

- DETR Attention Visualization Colab Notebook: Shows how to load a model, generate predictions, and visualize the attention (Reference 3)

[More Information Needed] on any other explainability/interpretability techniques that have been applied to analyze the behavior and reasoning of the DETR model.

## Environmental Impact

- **Hardware Type:** [More Information Needed]

The provided references do not specify the hardware type that the model facebook/detr-resnet-50 was trained on. More details would be needed from the model developers or training logs to determine the specific hardware used for training this model.
- **Software Type:** Based on the provided references, the model facebook/detr-resnet-50 is trained using the following software:

- Python 3.6+
- PyTorch 1.5+
- Torchvision 0.6+

The model also requires the following additional dependencies:
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

The model is trained using the AdamW optimizer with the following settings:
- Initial transformer's learning rate: 10^-4
- Backbone's learning rate: 10^-5 
- Weight decay: 10^-4

The transformer weights are initialized with Xavier initialization, and the backbone is initialized with an ImageNet-pretrained ResNet-50 model from torchvision with frozen batchnorm layers.
- **Hours used:** Training the baseline DETR model with ResNet-50 backbone for 300 epochs on 16 V100 GPUs takes 3 days, with 4 images per GPU (hence a total batch size of 64).
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the facebook/detr-resnet-50 model. More information would be needed from the model developers or training logs to determine the cloud provider utilized.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the facebook/detr-resnet-50 model. To answer this question, we would need more specific details about the training setup, such as:

- Hardware used for training (GPU type, number of GPUs, etc.)
- Duration of the training process
- Energy consumption of the hardware during training
- Carbon intensity of the electricity grid where the training was performed

Without these details, it is not possible to accurately estimate the carbon emissions associated with training this particular model.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The DETR (DEtection TRansformer) architecture consists of three main components:
1. A CNN backbone (ResNet-50) to extract a compact feature representation of the input image
2. An encoder-decoder transformer that reasons about the global image context and object relations
3. A simple feed forward network (FFN) that makes the final detection predictions (class labels and bounding boxes) for a fixed set of N object queries

The transformer encoder takes the flattened 2D representation from the CNN backbone, supplemented with positional encodings. The decoder attends to the encoder output and N learned object query embeddings (initially set to zero), producing an output embedding for each query. These embeddings are passed through the FFN to predict the final detections in parallel.

Objective:
DETR streamlines the object detection pipeline by framing it as a direct set prediction problem. It is trained end-to-end with a set loss function that performs bipartite matching between predicted and ground-truth objects. This approach eliminates the need for hand-designed components like non-maximum suppression or anchor generation. 

The main goal is to simplify the detection architecture while achieving accuracy and run-time performance on par with well-established baselines like Faster R-CNN on challenging datasets such as COCO. By using a transformer encoder-decoder architecture, DETR can reason about the global image context and object relations to directly output the final set of predictions in parallel.

### Compute Infrastructure

Based on the provided references, here is the compute infrastructure information for the model facebook/detr-resnet-50:

The model was trained on a single node with 8 V100 GPUs for 300 epochs. Each epoch took around 28 minutes, resulting in a total training time of approximately 6 days.

To train the model, the following command was used:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```

Distributed training was also available using Slurm and submitit. To train the baseline DETR-6-6 model on 4 nodes for 300 epochs, the following command was used:

```
python run_with_submitit.py --timeout 3000 --coco_path /path/to/coco
```

[More Information Needed] on the exact hardware specifications of the nodes used for distributed training.

## Citation

```
@misc{nicolas-endtoend,
    author = {Nicolas Carion and
              Francisco Massa and
              Gabriel Synnaeve and
              Nicolas Usunier and
              Alexander Kirillov and
              Sergey Zagoruyko and
              Facebook Ai},
    title  = {End-to-End Object Detection with Transformers},
    url    = {https://arxiv.org/pdf/2005.12872.pdf}
}
```
