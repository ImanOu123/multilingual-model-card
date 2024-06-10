# Model Card for vinvino02/glpn-nyu

The vinvino02/glpn-nyu model is a monocular depth estimation model that uses a global-local path network architecture and depth-specific data augmentation. It achieves state-of-the-art performance on the NYU Depth V2 dataset and shows strong generalization ability and robustness.

## Model Details

### Model Description

Model Card for vinvino02/glpn-nyu

Model Details:
- Architecture: The model uses a hierarchical transformer as the encoder to learn global dependencies from the input RGB image. The decoder restores the extracted bottleneck feature into the target depth map by constructing a local path with convolution and bilinear upsampling layers.
- Training Data: The model was trained on the NYU Depth V2 dataset.
- Training Procedure: The model was trained using a scale-invariant log scale loss function to calculate the distance between the predicted output and ground truth depth map.
- Inference: [More Information Needed]

Uses:
- The model is designed for monocular depth estimation, predicting a depth map from a single RGB image.
- [More Information Needed]

Limitations:
- [More Information Needed]

Metrics:
The model achieves state-of-the-art performance on the NYU Depth V2 dataset based on various evaluation metrics. It outperforms recently developed models such as Adabins and DPT.

Training Parameters:
[More Information Needed]

Evaluation Parameters: 
To evaluate the model on the NYU Depth V2 dataset, use the following command:
```
python ./code/train.py --dataset nyudepthv2 --data_path ./datasets/ --max_depth 10.0 --max_depth_eval 10.0
```

Quantitative Analyses:
[More Information Needed]

Ethical Considerations:
[More Information Needed]

Caveats and Recommendations:
[More Information Needed]

Contact:
For questions or concerns regarding the model, please contact [More Information Needed].

- **Developed by:** Doyeon Kim; Woonghyun Ka; Pyunghwan Ahn; Donggyu Joo; Sewhan Chun; Junmo Kim
- **Funded by:** Based on the provided references, there is no clear information about the funding sources for the vinvino02/glpn-nyu model project. The references focus on the model's performance, evaluation, and comparison to other methods, but do not mention any specific funding organizations or grants.

[More Information Needed]

To properly answer this question for the model card, the project organizer should seek additional information from the team about any funding sources that supported the development of the vinvino02/glpn-nyu model.
- **Shared by:** Based on the provided references, the main contributors to the model vinvino02/glpn-nyu are:

Doyeon Kim, Woonghyun Ga, Pyungwhan Ahn, Donggyu Joo, Sehwan Chun, and Junmo Kim

They authored the paper "Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth" in 2022, which introduces the model architecture and approach.

The code for the model is not directly linked in the references. The paper mentions "The code will be available soon", but no specific GitHub repository or link is provided. Therefore, for the code availability, [More Information Needed].
- **Model type:** The vinvino02/glpn-nyu model is a supervised deep learning model that uses a transformer-based encoder-decoder architecture to predict depth maps from RGB images.
- **Language(s):** The model vinvino02/glpn-nyu does not appear to use or process natural human language based on the given references, which focus on computer vision tasks such as depth estimation from RGB images using transformer-based architectures and techniques like selective feature fusion (SFF).
- **License:** Based on the provided references, the license information for the model vinvino02/glpn-nyu is:

The model is currently licensed for non-commercial purposes only (research, evaluation, etc). The team plans to update the license soon.

[More Information Needed] on the specific name and link to the future license that will be used.
- **Finetuned from model:** Based on the provided references, there is no clear indication that the model vinvino02/glpn-nyu is fine-tuned from another pre-existing model. The references discuss various architectures and methods used in monocular depth estimation, but do not specifically mention the base model for vinvino02/glpn-nyu.

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/vinvino02/GLPDepth
- **Paper:** https://arxiv.org/pdf/2201.07436.pdf
- **Demo:** Here is the link to the Colab demo of the model vinvino02/glpn-nyu:

<p>
<a href="https://colab.research.google.com/drive/1v6fzr4XusKdXAaeGZ1gKe1kh9Ce_WQhl?usp=sharing" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</p>

The Colab demo is provided thanks to the great work from NielsRogge.
## Uses

### Direct Use

The model vinvino02/glpn-nyu can be used for monocular depth estimation without requiring fine-tuning, post-processing, or plugging into a pipeline. It takes a single RGB image as input and directly outputs the estimated depth map.

[More Information Needed] for providing a code snippet, as there are no direct code block references in the given information about how to use the model.

The model is designed to be robust against various types of natural image corruptions, making it suitable for real-world applications where images may be corrupted to a certain degree. It achieves state-of-the-art performance on the NYU Depth V2 dataset and demonstrates strong generalization ability.

The model architecture consists of an encoder that transforms the input image into a bottleneck feature, followed by a lightweight and effective decoder to restore the bottleneck feature into the estimated depth map. The decoder utilizes a selective feature fusion (SFF) module to selectively focus on salient regions by estimating attention maps for both local and global features.

### Downstream Use

The vinvino02/glpn-nyu model can be fine-tuned and used for monocular depth estimation tasks. Some key points on how it can be applied:

- The model achieves state-of-the-art performance on the NYU Depth V2 dataset, so it is well-suited for depth estimation on similar indoor scene data. 

- To use the model, images should be resized to the largest multiple of 32 below the original size before being passed in. The predicted depth maps can then be resized back to the original image dimensions.

- The model architecture uses a global-local path design with a hierarchical transformer encoder to capture multi-scale context, and a decoder with a selective feature fusion module to produce detailed depth maps. This allows it to effectively extract features at different scales.

- The model was trained with a depth-specific data augmentation technique called vertical CutDepth to boost performance. Similar data augmentation could be applied during fine-tuning.

- Experiments show the model has good generalization ability and robustness, so it may perform well on other depth datasets beyond just NYU Depth V2.

[More Information Needed] for providing a code snippet, as there are no direct code examples in the reference text to draw from. Overall, the vinvino02/glpn-nyu model seems to be a high-performing, robust choice as a starting point for monocular depth estimation when fine-tuned and integrated into a larger system.

### Out-of-Scope Use

Based on the provided references about the depth estimation model vinvino02/glpn-nyu, here are some potential misuses to caution users against:

The model should not be used in safety-critical applications without thorough testing and validation, even though the references indicate it shows stronger robustness compared to other models. Relying solely on this model for depth perception in autonomous vehicles, robotics, or other systems where failures could cause harm is inadvisable without extensive real-world evaluation.

Users should not assume the model will perform well on images that are significantly corrupted or differ from the training data. While the model demonstrates some robustness to image corruptions, its performance will likely degrade on highly distorted or out-of-distribution images. 

The model is trained on specific datasets like NYU Depth V2 and KITTI. Applying it to radically different environments, like underwater or aerial footage, is not recommended without fine-tuning or testing in those domains. Generalization ability has limits.

[More Information Needed] on the computational requirements, latency, and hardware compatibility of the model. Users should not deploy it in resource-constrained or real-time systems without confirming it meets those performance needs.

In general, this depth estimation model is a research artifact and should not be used in production systems without extensive testing, monitoring, and human oversight. Over-reliance on its outputs or applying it outside its intended use cases and limitations may lead to suboptimal or unsafe behaviors. Depth estimation remains an active research problem and models like this are not foolproof solutions.

### Bias, Risks, and Limitations

Based on the provided references about the vinvino02/glpn-nyu model, here are some potential issues and limitations:

Foreseeable harms and misunderstandings:
- The model is designed for monocular depth estimation, but the references do not discuss potential misuse cases or unintended consequences. More information is needed on how the model could be misapplied in harmful ways.

Technical limitations: 
- The model was evaluated on specific datasets like NYU Depth V2 and KITTI. Performance may degrade on data that is very different from the training/test sets.
- Reference 1 indicates the model is more robust than baselines to image corruptions, but more details are needed on failure modes and corner cases.
- [More Information Needed] on computational requirements and latency for real-time applications.

Sociotechnical limitations:
- [More Information Needed] The references focus on technical methodology and results. There is no discussion of fairness, representation, potential biases, or societal impact.
- Deploying the model in safety-critical applications like self-driving cars (hinted at by KITTI dataset) would require further robustness testing and social considerations around responsible use.
- [More Information Needed] on if the model was tested with diverse users and environments, especially if it will be applied globally.

To comprehensively fill out the model card, more information and analysis is needed in several key areas related to responsible development and deployment. The current references are quite technical and additional sociotechnical context is required.

### Recommendations

Based on the provided references about the model vinvino02/glpn-nyu, here are some recommendations regarding foreseeable issues:

Robustness and Safety:
The model shows strong robustness to image corruptions compared to other models, making it more appropriate for safety-critical applications (Reference 1). However, more extensive testing on real-world corrupted images would help further validate its robustness and suitability for such applications.

Generalization Ability: 
While the model demonstrates good performance on the NYU Depth V2 dataset (References 2, 4, 11), its generalization ability to other datasets and real-world scenarios should be further evaluated. Testing on a wider variety of datasets and environments would provide a more comprehensive assessment of its generalization capabilities.

Computational Efficiency:
The model's decoder is designed to be computationally efficient with fewer parameters compared to other architectures (References 3, 4). However, the overall computational requirements and inference speed of the model should be clearly documented to ensure it can be deployed in resource-constrained environments.

Bias and Fairness:
[More Information Needed] The provided references do not discuss any potential biases or fairness issues in the model's performance across different demographic groups or scenarios. It is important to assess and mitigate any such biases to ensure equitable performance.

Transparency and Documentation:
The model's architecture, training process, and performance metrics should be thoroughly documented in the model card for transparency (Reference 6). This includes providing clear instructions for inference and training (Reference 6), as well as any limitations or known issues.

Ethical Considerations:
[More Information Needed] The references do not explicitly discuss ethical considerations related to the model's development or deployment. It is crucial to assess any potential misuse cases, privacy concerns, or unintended consequences of using the depth estimation model in various applications.

By addressing these aspects and providing more information where needed, the model card can help users better understand the capabilities, limitations, and considerations associated with using the vinvino02/glpn-nyu model responsibly.

## Training Details

### Training Data

The training data for the model vinvino02/glpn-nyu is the NYU Depth V2 dataset. Only the official test set of 5050 images is used for evaluating the pre-trained model.

[More Information Needed] on documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the data of the model vinvino02/glpn-nyu:

Resizing:
The image sizes are not constant in the NYU Depth V2 dataset. The images are resized to the largest multiple of 32 below the original image size before being passed to the model to predict the depth map. The predicted depth map is then resized back to the original image size.

Data Augmentation:
The following data augmentation strategies are used with 50% probability:
- Horizontal flips
- Random brightness (±0.2)
- Random contrast (±0.2)
- Random gamma (±20)
- Random hue (±20)
- Random saturation (±30)
- Random value (±20)
- Vertical CutDepth with 25% possibility and p = 0.75

[More Information Needed] about tokenization and other preprocessing details.

#### Training Hyperparameters

Here are the training hyperparameters used for the model vinvino02/glpn-nyu, based on the provided references:

- Framework: PyTorch
- Optimizer: Adam 
- Learning rate schedule: One-cycle learning rate strategy
  - Increases from 3e-5 to 1e-4 following a poly LR schedule with factor 0.9 in first half of iterations
  - Decreases from 1e-4 to 3e-5 in last half of iterations
- Number of epochs: 25
- Batch size: 12
- Pre-trained weights: MiT-b4
- Data augmentation (applied with 50% probability):
  - Horizontal flips
  - Random brightness (±0.2)
  - Random contrast (±0.2) 
  - Random gamma (±20)
  - Random hue (±20)
  - Random saturation (±30)
  - Random value (±20)
- Vertical CutDepth augmentation: 
  - Applied with 25% possibility
  - p hyperparameter set to 0.75

[More Information Needed] on the exact image resizing procedure during training and inference.

#### Speeds, Sizes, Times

Here are the details I could find about the model vinvino02/glpn-nyu based on the provided references:

Model performance on NYU Depth V2 dataset:
- State-of-the-art performance on most evaluation metrics
- Outperforms recently developed models like Adabins and DPT
- Exact metric values: [More Information Needed]

Model size:
- 0.66M parameters, which is more compact than other decoder architectures like:
  - MLPdecoder (3.19M params) 
  - BTS (5.79M params)
  - DPT (14.15M params)

Training data augmentation:
- Uses vertical CutDepth augmentation with hyperparameter p set to 0.75
- Code for CutDepth:
[More Information Needed]

Trained checkpoint files:
- Available for download: https://drive.google.com/drive/folders/17yYbLZS2uQ6UVn5ET9RhVL0y_X3Ipl5_?usp=sharing
- Exact checkpoint file sizes: [More Information Needed] 

Predicted depth maps:
- PNG files available: https://drive.google.com/drive/folders/1LGNSKSaXguLTuCJ3Ay_UsYC188JNCK-j?usp=sharing

Training throughput, start/end times: 
[More Information Needed]

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model vinvino02/glpn-nyu evaluates on the following datasets:

1. NYU Depth V2 dataset
   - The model is evaluated on the official test set of 5050 images from the NYU Depth V2 dataset.
   - The model's performance is compared with existing methods through quantitative and qualitative evaluation on this dataset.
   - The model achieves state-of-the-art performance on most evaluation metrics for the NYU Depth V2 dataset.

2. KITTI dataset
   - The model is also evaluated on the KITTI dataset, which contains outdoor depth estimation data.
   - The model is trained on approximately 23K images with a random crop of 704 × 352 and tested on 697 images.
   - The crop defined by Garg and a maximum depth value of 80m are used for evaluation on KITTI.

3. SUN RGB-D dataset
   - The model's approach is validated through experiments on the SUN RGB-D dataset.
   - [More Information Needed] on the specific details of evaluation on this dataset.

4. iBims-1 dataset
   - Additional results are provided on the iBims-1 dataset, which is an indoor dataset.
   - [More Information Needed] on the specific details of evaluation on this dataset.

The model can be evaluated on the NYU Depth V2 and KITTI datasets using the following commands:

For NYU Depth V2:
```
python ./code/eval_with_pngs.py --dataset nyudepthv2 --pred_path ./best_nyu_preds/ --gt_path ./datasets/nyu_depth_v2/ --max_depth_eval 10.0 
```

For KITTI:
```
python ./code/eval_with_pngs.py --dataset kitti --split eigen_benchmark --pred_path ./best_kitti_preds/ --gt_path ./datasets/kitti/ --max_depth_eval 80.0 --garg_crop
```

#### Factors

Based on the provided references about the model vinvino02/glpn-nyu, here are some key characteristics that may influence the model's behavior:

Domain and Context:
- The model was trained and evaluated on the NYU Depth V2 and SUN RGB-D datasets, which contain indoor scenes (References 1, 8, 10). This suggests the model is primarily designed for depth estimation in indoor environments.
- The model was also tested on the KITTI dataset (Reference 5), which contains outdoor driving scenes. This indicates some generalization ability to outdoor contexts, but more information is needed on performance compared to indoor scenes.

Population Subgroups:
- [More Information Needed] The references do not provide details on performance across different population subgroups or demographic factors.

Robustness and Corruption:
- The model's robustness was evaluated against 16 types of image corruptions at 5 intensity levels (References 2, 3, 6). The model showed stronger robustness compared to other models, suggesting it may handle corrupted or degraded images better.
- However, more information is needed on the specific corruption types and how performance was disaggregated across them.

Cross-Dataset Generalization:
- The model was able to generalize under cross-dataset validation (Reference 11), but more specifics are needed on which datasets were used and how performance compared to within-dataset evaluation.

In summary, the key factors that may influence the model's behavior are the indoor scene domain it was primarily trained on, its robustness to image corruptions, and some ability to generalize across datasets. However, more disaggregated evaluation is needed, especially across population subgroups, to uncover potential performance disparities. The references lack details in several areas to fully characterize the model's behavior.

#### Metrics

Based on the provided references about the model vinvino02/glpn-nyu, the following metrics will be used for evaluation:

1. The model's performance on the NYU Depth V2 dataset will be compared with existing methods through quantitative evaluation using various metrics (Reference 2). The specific metrics are not explicitly mentioned.

2. To evaluate the model on the NYU Depth V2 dataset, the following command can be used (Reference 3):

```
python ./code/eval_with_pngs.py --dataset nyudepthv2 --pred_path ./best_nyu_preds/ --gt_path ./datasets/nyu_depth_v2/ --max_depth_eval 10.0
```

This suggests that the model will be evaluated using the official test set of 5050 images from the NYU Depth V2 dataset, with a maximum depth of 10.0 meters.

3. The model will also be evaluated on the SUN RGB-D dataset (Reference 4), but the specific metrics for this dataset are not mentioned.

[More Information Needed] regarding the exact metrics used for quantitative evaluation on both the NYU Depth V2 and SUN RGB-D datasets, as well as any tradeoffs considered between different types of errors.

### Results

Based on the provided references, here is the evaluation information for the model vinvino02/glpn-nyu:

Datasets:
- The model was evaluated on the NYU Depth V2 and SUN RGB-D datasets.
- Only the official test set of 5050 images from NYU Depth V2 was used for evaluation.
- Additional results on the KITTI and iBims-1 datasets are provided in the supplementary material.

Evaluation Metrics:
[More Information Needed]

Results:
- On the NYU Depth V2 dataset (Table 1), the model achieved state-of-the-art performance in most evaluation metrics compared to existing methods like DPT and Adabins.
- The model showed accurate depth estimation and robustness to various illumination conditions on the NYU Depth V2 dataset (Figure 3).
- The model was also tested on the SUN RGB-D dataset, but specific results are not provided in the given references.
- Results on the KITTI dataset are shown in Table 6, but the specific metrics and values are not provided in the given references.

Model Efficiency:
- The model achieves higher performance than recent state-of-the-art models (Adabins, DPT) with fewer parameters.
- The proposed decoder has only 0.66M parameters, which is much lower than other decoder architectures like MLPdecoder (3.19M), BTS (5.79M), and DPT (14.15M).

Data Augmentation:
- The effectiveness of the vertical CutDepth data augmentation method was evaluated through an ablation study (Table 5).

Code Availability:
[More Information Needed]

Note: More specific information is needed regarding the evaluation metrics, results on the SUN RGB-D and KITTI datasets, and code availability to provide a complete evaluation summary.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model vinvino02/glpn-nyu:

The model was evaluated on the NYU Depth V2 dataset, using only the official test set of 5050 images. As shown in Table 1, the model achieved state-of-the-art performance on most evaluation metrics compared to existing methods like Adabins and DPT, which the authors attribute to their proposed architecture and enhanced depth-specific data augmentation method. The model performed well with fewer parameters than some other recent state-of-the-art models.

Qualitatively, the model produced accurate depth estimations on example images and was more robust to various illumination conditions compared to other methods, as shown in Figure 3.

The model was also evaluated on the additional indoor dataset SUN RGB-D, but [More Information Needed] on those specific results.

To reproduce the NYU Depth V2 evaluation, the following command can be run:

```
python ./code/eval_with_pngs.py --dataset nyudepthv2 --pred_path ./best_nyu_preds/ --gt_path ./datasets/nyu_depth_v2/ --max_depth_eval 10.0
```

This will save result images in ./results/test by default.

The model was also tested on the outdoor KITTI dataset and indoor iBims-1 dataset to assess generalization, but [More Information Needed] on those detailed results. The process to set up KITTI data and run evaluation was provided.

In summary, the vinvino02/glpn-nyu model achieved state-of-the-art monocular depth estimation performance on the NYU Depth V2 benchmark, with good qualitative results, while being efficient in terms of parameters. The code will be made available soon.

## Model Examination

Based on the provided references about the model vinvino02/glpn-nyu, here is the information I can provide for the experimental section on explainability/interpretability:

The model was validated through several experiments on the NYU Depth V2 dataset to demonstrate its effectiveness (Reference 2). An ablation study was conducted to show the impact of the proposed data augmentation method CutDepth. The results indicate that using vertical CutDepth with an appropriate hyperparameter p leads to improved depth estimation performance compared to basic CutDepth or no augmentation (Reference 5).

The model's robustness was evaluated by testing on corrupted images from the NYU Depth V2 test set. The results show that the model is more robust to various types of corruption compared to other models, making it more suitable for safety-critical applications (Reference 4).

[More Information Needed] on specific techniques used for explainability or interpretability of the model's predictions and internal workings. The provided references do not contain direct information on this aspect.

In summary, the experimental section validates the model's effectiveness through ablation studies on data augmentation and demonstrates its robustness on corrupted images. However, more information is needed to discuss explainability and interpretability techniques applied to this specific model.

## Environmental Impact

- **Hardware Type:** [More Information Needed]

The provided references do not contain information about the specific hardware used for training the vinvino02/glpn-nyu model. The references mention using PyTorch and some training hyperparameters, but do not specify the hardware (e.g. GPU type, number of GPUs, etc.).
- **Software Type:** The model vinvino02/glpn-nyu is trained using the PyTorch framework, as mentioned in the first reference:

"We implement the proposed network using the PyTorch framework."
- **Hours used:** Based on the provided references, the total number of epochs used to train the model vinvino02/glpn-nyu is 25 with a batch size of 12, as stated in reference 2:

"The total number of epochs is set to 25 with a batch size of 12."

However, the references do not provide the exact amount of time used for training. To determine the training time, additional information would be needed, such as the hardware specifications (GPU model, number of GPUs, etc.) and the time per epoch. Without these details, I cannot provide an accurate estimate of the total training time.

[More Information Needed] for the exact training time of the model vinvino02/glpn-nyu.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the vinvino02/glpn-nyu model. More information would be needed from the model development team to determine which cloud provider, if any, was utilized during training.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the model vinvino02/glpn-nyu. To answer this question, details about the computing resources used for training (e.g., GPU specs, training time, power consumption) would be necessary to estimate the carbon footprint.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The model vinvino02/glpn-nyu uses a hierarchical transformer as the encoder to learn global dependencies and capture multi-scale context features from the input RGB image. The encoder transforms the input image into a bottleneck feature. The decoder is lightweight and effective, using consecutive bilinear upsampling, convolution layers, and a sigmoid function to restore the bottleneck feature into the estimated depth map. The decoder also leverages skip connections and a selective feature fusion (SFF) module to focus on salient regions. The model is trained with an additional task-specific data augmentation technique.

Objective:
The objective of the model vinvino02/glpn-nyu is to predict the depth map from a given RGB image. It aims to achieve state-of-the-art performance on the challenging NYU Depth V2 dataset while demonstrating better generalization ability and robustness compared to other comparative models.

[More Information Needed] on specific implementation details, such as the exact number of layers, dimensions, or hyperparameters used in the model.

### Compute Infrastructure

Here is the information about the compute infrastructure used for the model vinvino02/glpn-nyu, based on the provided references:

Training Environment:
- Python version: 3.7.7
- PyTorch version: 1.6.0
- Other key packages:
```
h5py==3.6.0
scipy==1.7.3
opencv-python==4.5.5
mmcv==1.4.3
timm=0.5.4
albumentations=1.1.0
tensorboardX==2.4.1
gdown==4.2.1
```

Installation:
The required packages can be installed with:
```
$ pip install -r requirements.txt
```

Alternatively, a Docker image is available:
```
$ docker pull doyeon0113/glpdepth
```

Training Details:
- Optimizer: Adam 
- Learning rate schedule: One-cycle learning rate strategy, increasing from 3e-5 to 1e-4 in first half of iterations, then decreasing from 1e-4 to 3e-5 in last half
- Number of epochs: 25
- Batch size: 12
- Pre-trained weights used from MiT-b4

[More Information Needed] on the specific hardware (GPU models, number of GPUs, etc.) used for training and inference.

## Citation

```
@misc{doyeon-globallocal,
    author = {Doyeon Kim and
              Woonghyun Ka and
              Pyunghwan Ahn and
              Donggyu Joo and
              Sewhan Chun and
              Junmo Kim},
    title  = {Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth},
    url    = {https://arxiv.org/pdf/2201.07436.pdf}
}
```

