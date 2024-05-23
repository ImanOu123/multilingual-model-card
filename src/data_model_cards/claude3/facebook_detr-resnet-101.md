# Model Card for facebook/detr-resnet-101

The facebook/detr-resnet-101 model is a DEtection TRansformer (DETR) model that views object detection as a direct set prediction problem. It uses an encoder-decoder transformer architecture with a ResNet-101 CNN backbone, and achieves comparable performance to an optimized Faster R-CNN baseline on the COCO dataset.

## Model Details

### Model Description

Model Card for facebook/detr-resnet-101

Model Architecture:
The DETR (DEtection TRansformer) architecture consists of three main components:
1. A CNN backbone (ResNet-101) to extract a compact feature representation from the input image
2. An encoder-decoder transformer that reasons about the relations between objects and global image context
3. A simple feed forward network (FFN) that makes the final detection predictions (class labels and bounding boxes)

The transformer encoder takes the flattened 2D representation from the CNN backbone, supplements it with positional encodings, and produces an output embedding. The decoder takes a small fixed number of learned object queries, output positional encodings, and the encoder output, and produces the final set of predicted class labels and bounding boxes through the FFN.

Training Procedure:
- Trained with AdamW optimizer
- Learning rate: 1e-4 for transformer, 1e-5 for backbone
- Data augmentation: Horizontal flips, scales, and crops
- Images rescaled to have min size 800 and max size 1333
- Transformer trained with dropout of 0.1
- Whole model trained with gradient clipping of 0.1
- Loss function: Linear combination of L1 and GIoU losses for bounding box regression
- Trained with N=100 decoder query slots

Parameters:
[More Information Needed]

Important Disclaimers:
[More Information Needed]

The model simplifies the detection pipeline by dropping hand-designed components like spatial anchors and non-maximal suppression. It demonstrates accuracy and run-time performance on par with the well-established Faster R-CNN baseline on the COCO dataset.

For more details, please refer to the official DETR paper and the code repository at https://github.com/facebookresearch/detr.

- **Developed by:** Nicolas Carion; Francisco Massa; Gabriel Synnaeve; Nicolas Usunier; Alexander Kirillov; Sergey Zagoruyko; Facebook Ai
- **Funded by:** Based on the provided references, the model facebook/detr-resnet-101 appears to be funded by Facebook AI Research (FAIR). Specifically, the references mention:

- Thanking members of the Facebook AI Research Paris team for discussions and advice that made the work possible (Reference 1)
- Several authors affiliated with Facebook AI (Reference 2) 
- Code and models being available on the facebookresearch GitHub organization (Reference 5)

So in summary, the project seems to be primarily funded and supported by Facebook AI Research. However, to say this definitively, [More Information Needed] to fully confirm the funding sources and whether any other organizations were involved.
- **Shared by:** Based on the references provided, the main contributors to the DETR model (facebook/detr-resnet-101) are:

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko from Facebook AI Research (Reference 2).

The references also thank Sainbayar Sukhbaatar, Piotr Bojanowski, Natalia Neverova, David Lopez-Paz, Guillaume Lample, Danielle Rothermel, Kaiming He, Ross Girshick, Xinlei Chen and the whole Facebook AI Research Paris team for discussions and advice (Reference 1).

However, there is no explicit information on who made the model available online as a GitHub repo. [More Information Needed]
- **Model type:** The facebook/detr-resnet-101 model is an object detection model that uses a ResNet-101 backbone and a transformer encoder-decoder architecture, trained end-to-end on the COCO dataset using a set-based global loss function for direct set prediction of object bounding boxes and classes.
- **Language(s):** The facebook/detr-resnet-101 model does not use or process natural human language, as it is an object detection model that takes images as input and predicts bounding boxes and class labels for objects in the image.
- **License:** DETR is released under the Apache 2.0 license. The link to the license file is provided in the [LICENSE](LICENSE) file in the DETR repository.
- **Finetuned from model:** Based on the provided references, the DETR model uses a ResNet-101 backbone that is initialized with weights pretrained on ImageNet:

"All transformer weights are initialized with Xavier init [11], and the backbone is with ImageNet-pretrained ResNet model [15] from torchvision with frozen batchnorm layers. We report results with two different backbones: a ResNet-50 and a ResNet-101. The corresponding models are called respectively DETR and DETR-R101."

So the base model that facebook/detr-resnet-101 is fine-tuned from is a ResNet-101 pretrained on ImageNet, likely the one provided by torchvision. However, a specific link to that pretrained ResNet-101 model is not provided in the references.

[More Information Needed] for the exact link to the pretrained ResNet-101 model used.
### Model Sources

- **Repository:** https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py
- **Paper:** https://arxiv.org/pdf/2005.12872.pdf
- **Demo:** Based on the provided references, here is the relevant information for the demo link of the model facebook/detr-resnet-101:

[DETR's hands on Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb): Shows how to load a model from hub, generate predictions, then visualize the attention of the model (similar to the figures of the paper)

This Colab notebook demonstrates how to use the DETR model, including the facebook/detr-resnet-101 variant, to generate predictions and visualize the model's attention.
## Uses

### Direct Use

The facebook/detr-resnet-101 model can be used for inference without requiring fine-tuning, post-processing, or plugging into a pipeline. The model's architecture is designed to directly output the final set of predictions in parallel, streamlining the detection process.

To use the model for inference, you can follow these steps:

1. Clone the DETR repository:
```
git clone https://github.com/facebookresearch/detr.git
```

2. Install the necessary dependencies (PyTorch 1.5+ and torchvision 0.6+):
```
conda install -c pytorch pytorch torchvision
```

3. Install additional dependencies (pycocotools and scipy):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

4. Use the provided inference code snippet (Listing 1 in the references) to run the model on an input image. The code runs with Python 3.6+, PyTorch 1.4, and Torchvision 0.5. Note that the code does not support batching and is suitable only for inference or training with DistributedDataParallel with one image per GPU.

[More Information Needed] (The exact inference code snippet is not provided in the given references)

The model's simplicity allows it to be implemented in any deep learning framework that provides a common CNN backbone and a transformer architecture implementation with just a few hundred lines of code.

### Downstream Use

The DETR (DEtection TRansformer) model facebook/detr-resnet-101 can be fine-tuned for object detection tasks or plugged into larger systems for applications like panoptic segmentation. 

To use DETR for inference, the code is quite simple with PyTorch and Torchvision libraries:

```python
[More Information Needed]
```

DETR simplifies the object detection pipeline by removing the need for hand-designed components like spatial anchors or non-maximal suppression. It predicts all objects at once in a single pass through the decoder.

When fine-tuning DETR for a specific dataset or task, the key is defining the loss function to optimally match predictions to ground truth objects. The model is trained end-to-end with this set loss.

DETR can also be extended to panoptic segmentation in a unified way by training a small extension on top of the fixed base DETR model. This allows it to be plugged into larger ecosystems requiring both object detection and segmentation.

The full code and pre-trained models are available at https://github.com/facebookresearch/detr for reproducing experiments and fine-tuning DETR for new applications. Dependencies can be easily installed via conda.

For more details on fine-tuning or integrating DETR into a larger system, please refer to the full documentation and reach out with any questions. As the project organizer, I'm happy to provide additional information and keep the model card updated.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the DETR (DEtection TRansformer) model facebook/detr-resnet-101 could be misused and what users should avoid doing:

The model should not be used for surveillance or tracking of individuals without consent, as this would be unethical and potentially illegal. The object detection capabilities could enable misuse for unwanted monitoring.

Users should not rely solely on the model's outputs for high-stakes decisions, like in medical diagnosis or autonomous vehicles, without extensive testing, human oversight, and redundant safety measures. Failures could cause harm.

The model should not be applied to datasets substantially different from its training data (COCO) without further evaluation, as performance may degrade in unexpected ways. 

[More Information Needed] on specific technical failure modes or potential for generating deceptive/biased outputs. The references focus more on architecture than misuse.

In general, users should be transparent about the model's use, obtain necessary permissions, and deploy it responsibly in appropriate contexts with human oversight. They should not treat it as infallible or use it for purposes that violate laws or ethics.

### Bias, Risks, and Limitations

Based on the provided references about the DETR (DEtection TRansformer) model, here are some known or foreseeable issues:

Technical limitations:
- DETR has challenges with training, optimization, and performance on small objects compared to existing detectors. More work is needed to address these issues. [Reference 3]
- By design, DETR is limited in the maximum number of objects it can detect, based on the number of query slots (e.g. 100 in the experiments). [Reference 7]

Foreseeable harms or misunderstandings:
- [More Information Needed] The references do not directly discuss potential harms or misunderstandings from using DETR. More analysis would be needed on fairness, bias, privacy, and other sociotechnical implications.

Sociotechnical considerations:
- Deploying object detection systems like DETR in real-world applications requires carefully considering the downstream impacts and potential for misuse. Responsible development practices, transparency about limitations, and proactive mitigation of risks is important.
- [More Information Needed] More specific information on the intended use cases, deployment contexts, and end users of the model would help identify additional sociotechnical issues to address.

In summary, key limitations of DETR to be aware of are challenges with small objects and a maximum number of detectable objects. Responsible real-world use requires further analysis and risk mitigation. The provided references focus more on the technical aspects, so additional sociotechnical considerations likely remain to be examined depending on the specific application.

### Recommendations

Based on the provided references about the DETR (DEtection TRansformer) model facebook/detr-resnet-101, here are some recommendations regarding foreseeable issues:

1. Training challenges: The DETR model requires an extra-long training schedule and benefits from auxiliary decoding losses in the transformer. Careful tuning of training settings is crucial for achieving good performance. (Reference 10)

2. Performance on small objects: DETR may face challenges in detecting small objects compared to traditional detectors. Future work should address improving performance on small objects. (Reference 4)

3. Optimization difficulties: DETR introduces new challenges in terms of optimization compared to existing detectors. Addressing these optimization issues will require further research and improvements. (Reference 4)

4. Generalization to unseen numbers of instances: While DETR shows promising generalization to unseen numbers of instances (Reference 6), more extensive testing on out-of-distribution scenarios is recommended to assess its robustness.

5. Computational efficiency: [More Information Needed] regarding the computational efficiency and runtime performance of DETR compared to other object detection methods.

6. Interpretability of learned representations: [More Information Needed] on analyzing and interpreting the learned representations in the transformer components of DETR, which could provide insights into its decision-making process.

7. Fairness and bias considerations: [More Information Needed] regarding potential biases or fairness issues that may arise from the training data or model architecture of DETR.

Overall, while DETR presents a promising new approach to object detection, it is important to be aware of these potential issues and conduct further research to address them before widespread deployment of the model.

## Training Details

### Training Data

The model facebook/detr-resnet-101 is trained on COCO 2017 detection and panoptic segmentation datasets, which contain 118k training images and 5k validation images annotated with bounding boxes and panoptic segmentation. The panoptic annotations include 53 stuff categories in addition to 80 things categories.

[More Information Needed] for documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the data of the model facebook/detr-resnet-101:

We use scale augmentation, resizing the input images such that the shortest side is at least 480 and at most 800 pixels while the longest at most 1333. To help learning global relationships through the self-attention of the encoder, we also apply random crop augmentations during training, improving the performance by approximately 1 AP. Specifically, a train image is cropped with probability 0.5 to a random rectangular patch which is then resized again to 800-1333.

[More Information Needed] about tokenization and other preprocessing details.

#### Training Hyperparameters

Here are the detailed training hyperparameters for the facebook/detr-resnet-101 model based on the provided references:

Model architecture:
- Backbone: ResNet-101 pretrained on ImageNet, imported from Torchvision with the last classification layer discarded. Backbone batch normalization weights and statistics are frozen during training.
- Transformer: Initialized with Xavier initialization. Trained with a dropout of 0.1.

Losses:
- Bounding box regression: Linear combination of L1 and GIoU losses with weights λ_L1 = 5 and λ_iou = 2 respectively.
- Number of decoder query slots: N = 100

Optimization:
- Optimizer: AdamW 
- Transformer learning rate: 1e-4
- Backbone learning rate: 1e-5 
- Weight decay: 1e-4
- Gradient clipping: Max gradient norm of 0.1

Data augmentation:
- Horizontal flips, scales and crops
- Images rescaled to have min size 800 and max size 1333

Training schedule:
[More Information Needed]

The model is trained end-to-end with a set loss function which performs bipartite matching between predicted and ground-truth objects. DETR doesn't require any customized layers and can be easily reproduced in any framework that supports standard neural network modules.

#### Speeds, Sizes, Times

Based on the provided references, here are the details I can provide about the facebook/detr-resnet-101 model:

Throughput: 0.097 seconds per image (Reference 6)

Checkpoint size: 232MB (Reference 6)

The model achieves 44.9 AP (average precision) on the COCO dataset (Reference 6).

[More Information Needed] on start or end training time.

The model is based on a ResNet-101 backbone and is trained for 500 epochs (Reference 6).

It is part of the DETR (DEtection TRansformer) family of models that predict all objects at once using an encoder-decoder transformer architecture (References 9, 10).

DETR models demonstrate accuracy and run-time performance on par with highly optimized Faster R-CNN baselines, while being conceptually simpler (References 10, 11).

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model facebook/detr-resnet-101 is evaluated on the following datasets:

COCO 2017 detection dataset:
- Contains 118k training images and 5k validation images
- Each image is annotated with bounding boxes
- There are 7 instances per image on average, up to 63 instances in a single image in the training set, ranging from small to large on the same images

COCO 2017 panoptic segmentation dataset:
- Contains the same images as the COCO 2017 detection dataset
- Each image is annotated with panoptic segmentation in addition to bounding boxes

The model's performance is compared against a competitive Faster R-CNN baseline on the COCO dataset. The references mention reporting bbox AP (Average Precision) as the integral metric over multiple thresholds for evaluation.

#### Factors

Based on the provided references about the facebook/detr-resnet-101 model, here are some foreseeable characteristics that may influence the model's behavior:

Domain and Context:
- The model is trained and evaluated on the COCO dataset, so performance on other datasets or real-world domains may vary. [More Information Needed] on specific performance across different domains.
- The model can generalize to unseen numbers of instances of objects in an image, as demonstrated by finding all 24 giraffes in a synthetic out-of-distribution image (Reference 3).

Population Subgroups:
- [More Information Needed] The references do not provide a disaggregated evaluation across different population subgroups. To uncover any potential performance disparities, the model should ideally be evaluated on images representing diverse populations.

Other Factors:
- The model's performance, especially on small objects, is influenced by the resolution of the input image features. Using dilated convolutions in the backbone increases resolution but also computational cost (Reference 4).
- Each decoder output slot learns specializations for predicting boxes of different areas and sizes (Reference 5). The distribution of objects in the COCO training data likely impacts these specializations.
- Data augmentation techniques used during training, like random cropping, improve the model performance (Reference 6).
- The losses used in training, particularly the GIoU loss, have a significant impact on model performance (References 7-8).

To comprehensively characterize the model's behavior, more disaggregated evaluation across various factors would be beneficial. The current references focus primarily on technical details and COCO benchmark results.

#### Metrics

Based on the provided references, the main evaluation metric used for the DETR model (including facebook/detr-resnet-101) on the COCO dataset is Average Precision (AP). Specifically:

- Table 1 presents the results on COCO and compares DETR to Faster R-CNN baselines using the AP metric. (Reference 1)

- The model is evaluated on the COCO val5k set using a single GPU. (Reference 4)

- A gist is provided with results for all DETR detection models, but the numbers can vary depending on batch size per GPU. (References 4-5)

The references do not provide detailed information about tradeoffs between different error types or other evaluation metrics used. More information would be needed to comment on that aspect.

In summary, the key quantitative metric reported for evaluating the DETR model's object detection performance is Average Precision (AP) on COCO. Other metrics or tradeoffs are [More Information Needed] based on the given references.

### Results

Evaluation Results of facebook/detr-resnet-101:

Factors:
- DETR achieves competitive results compared to Faster R-CNN in quantitative evaluation on COCO.
- DETR simplifies the detection pipeline by dropping multiple hand-designed components that encode prior knowledge, like spatial anchors or non-maximal suppression. 
- DETR doesn't require any customized layers, and thus can be reproduced easily in any framework that provides a common CNN backbone and a transformer architecture.

Metrics:
- Results are presented in Table 1. In the top section, Faster R-CNN results from Detectron2 Model Zoo [50] for models trained with the 3x schedule are shown. In the middle section, results (with a "+") for the same models but trained with the 9x schedule (109 epochs) and described enhancements are shown, which in total adds 1-2 AP.
- A ResNet-50-based DETR model with 6 encoder, 6 decoder layers and width 256 has 41.3M parameters, achieves 40.6 and 42.0 AP on short and long schedules respectively, and runs at 28 FPS, similarly to Faster R-CNN.

[More Information Needed] for specific AP numbers on the COCO dataset for the facebook/detr-resnet-101 model. The references mention results but do not provide the exact numbers for this specific model.

#### Summary

Here is a summary of the evaluation results for the facebook/detr-resnet-101 model based on the provided references:

- DETR achieves competitive results compared to Faster R-CNN in quantitative evaluation on COCO. It demonstrates significantly better performance on large objects than Faster R-CNN, likely thanks to the encoder-decoder architecture and global computations of the transformer. (References 1, 9, 10)

- DETR does not need NMS (non-maximum suppression) by design. NMS only improves performance for predictions from the first decoder layer, but is not needed for subsequent layers due to the self-attention mechanism. (Reference 7) 

- Ablation studies show that transformer components like the global self-attention in encoder, FFN, multiple decoder layers, and positional encodings all significantly contribute to the final object detection performance of DETR. (Reference 8)

- DETR is a versatile and extensible model. It can be easily extended to panoptic segmentation by training only a small extension on top of a fixed DETR model. (Reference 1)

- [More Information Needed] on exact AP numbers achieved by the facebook/detr-resnet-101 model and runtime performance comparison to Faster R-CNN.

In summary, the facebook/detr-resnet-101 model based on the DETR architecture achieves competitive object detection performance compared to highly optimized Faster R-CNN models, with notable strengths in detecting large objects. The model is conceptually simple and extensible to other tasks like panoptic segmentation.

## Model Examination

Here is a possible experimental section on explainability/interpretability for the facebook/detr-resnet-101 model, based on the provided references:

The DETR model provides several ways to visualize and interpret what the model is learning:

- Visualizing encoder attention (Fig. 3 in [4]): The attention maps of the last encoder layer can be visualized, showing that the encoder uses global scene reasoning to disentangle objects. The attention focuses on a few key points in the image.

- Visualizing decoder attention (Fig. 6 in [1]): The attention maps for each predicted object can be visualized in different colors. The decoder attention is fairly local, mostly attending to the relevant regions for each object.

- Importance of encoder layers (Table 2 in [4]): Experiments show the encoder layers are important for performance, especially on large objects. Removing all encoder layers drops AP by 3.9 points.

- Importance of decoder layers (Fig. 4 in [5]): The decoder layers progressively refine the predictions. AP improves significantly (+8.2 points) from the first to last decoder layer, showing the importance of the iterative refinement.

- Colab notebooks ([2], [3]): Interactive Colab notebooks are provided to visualize attention and gain a better understanding of the DETR architecture.

[More Information Needed] on analyzing the model's behavior as it approaches the limit of 100 query slots, as discussed in [8]. Code for this analysis is not directly provided in the references.

Overall, the attention visualizations and layer ablations provide insight into how DETR uses global reasoning and iterative refinement to make object detection predictions. The Colab notebooks enable hands-on exploration of these interpretability techniques.

## Environmental Impact

- **Hardware Type:** [More Information Needed]

The provided references do not contain specific information about the hardware type used for training the facebook/detr-resnet-101 model. More details would be needed from the model developers or training logs to determine the exact hardware setup.
- **Software Type:** Based on the provided references, the model facebook/detr-resnet-101 is trained using the following software:

- Python 3.6+
- PyTorch 1.5+
- Torchvision 0.6+
- Cython
- SciPy
- pycocotools (for evaluation on COCO)
- submitit (for distributed training via Slurm)

The model is trained using the AdamW optimizer with the following settings:
- Initial transformer's learning rate: 10^-4
- Backbone's learning rate: 10^-5
- Weight decay: 10^-4

The transformer weights are initialized with Xavier initialization, and the backbone is an ImageNet-pretrained ResNet-101 model from torchvision with frozen batchnorm layers.

[More Information Needed] on the specific versions of Cython, SciPy, pycocotools, and submitit used for training the model.
- **Hours used:** Based on the training details provided in the references, a single epoch of training DETR on a single node with 8 GPUs takes 28 minutes. The full 300 epoch training takes around 6 days on a single machine with 8 V100 cards.

However, the specific training time for the facebook/detr-resnet-101 model variant is [More Information Needed], as the references do not provide that exact detail. The training times mentioned are for the baseline DETR model.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the facebook/detr-resnet-101 model. While the references discuss some technical details of the model architecture and training process, they do not specify the cloud infrastructure. To answer this question, more information would be needed from the model developers or documentation.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the facebook/detr-resnet-101 model. To answer this question, we would need more specific details such as:

- The compute infrastructure used for training (GPUs, CPUs, TPUs, etc.)
- The duration of the training process 
- The energy consumption of the compute resources during training
- The carbon intensity of the electricity used

Without these additional details, it is not possible to provide an accurate estimate of the carbon emissions from training this model based solely on the given references.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The DETR (DEtection TRansformer) architecture consists of three main components:
1. A CNN backbone (ResNet-101 in this case) to extract a compact feature representation from the input image
2. An encoder-decoder transformer that takes the flattened image features with positional encodings as input, and reasons about the global context and object relations
3. A simple feed forward network (FFN) that makes the final detection predictions (class labels and bounding boxes) from the decoder output embeddings

The transformer encoder takes the CNN features and positional encodings, and the decoder takes a fixed small set of learned object queries (embeddings) and attends to the encoder output. The decoder outputs are passed through the FFN to get the final parallel predictions.

Objective:
DETR streamlines the object detection pipeline by framing it as a direct set prediction problem. It eliminates the need for hand-designed components like anchor boxes and non-maximum suppression. DETR is trained end-to-end using a set loss function that performs bipartite matching between predicted and ground-truth objects. The goal is to simplify object detection while achieving accuracy and speed comparable to well-established methods like Faster R-CNN.

[More Information Needed] on specific implementation details and training hyperparameters.

### Compute Infrastructure

Based on the provided references, the compute infrastructure details for the facebook/detr-resnet-101 model are:

Training was performed on a single node with 8 V100 GPUs. A single epoch takes 28 minutes, so 300 epoch training takes around 6 days on this setup.

To train the baseline DETR model on a single node with 8 GPUs for 300 epochs, the following command was used:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```

Distributed training is also available via Slurm and submitit. To train the baseline DETR-6-6 model on 4 nodes for 300 epochs:

```
pip install submitit
python run_with_submitit.py --timeout 3000 --coco_path /path/to/coco
```

[More Information Needed] on the exact compute infrastructure used for the facebook/detr-resnet-101 variant specifically.

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

