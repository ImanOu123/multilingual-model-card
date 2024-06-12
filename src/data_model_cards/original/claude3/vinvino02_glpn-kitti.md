# Model Card for vinvino02/glpn-kitti

Based on the provided references, the model vinvino02/glpn-kitti appears to be a monocular depth estimation model trained on the KITTI dataset. It uses a global-local path architecture with a hierarchical transformer encoder to capture multi-scale context features and a selective feature fusion decoder to predict depth maps from RGB images.

## Model Details

### Model Description

Model Card for vinvino02/glpn-kitti

Model Details:
- Architecture: The model uses a global-local path network (GLPN) architecture for monocular depth estimation. It consists of a hierarchical transformer encoder to learn global dependencies and a lightweight decoder to restore the bottleneck feature into the target depth map by constructing a local path.
- Training Data: [More Information Needed]
- Training Procedure: The model is trained using the one-cycle learning rate strategy with an Adam optimizer for 25 epochs with a batch size of 12. The learning rate follows a poly LR schedule, increasing from 3e-5 to 1e-4 in the first half of iterations and decreasing back to 3e-5 in the second half. The model uses pre-trained weights from MiT-b4.
- Eval Results: [More Information Needed]
- Uses: [More Information Needed]

Limitations and Bias:
[More Information Needed]

How to Get Started with the Model:
To train the model on the KITTI dataset, run:
```
$ python ./code/train.py --dataset kitti --data_path ./datasets/ --max_depth 80.0 --max_depth_eval 80.0  --garg_crop
```

Relevant Papers/References:
[More Information Needed]

If you have any questions about the model, please contact the model maintainer at [More Information Needed].

- **Developed by:** Doyeon Kim; Woonghyun Ka; Pyunghwan Ahn; Donggyu Joo; Sewhan Chun; Junmo Kim
- **Funded by:** [More Information Needed]

The provided references do not contain information about the funding sources for the vinvino02/glpn-kitti model project. More details would be needed from the model developers or associated documentation to determine what people or organizations, if any, provided financial support for this specific model's development.
- **Shared by:** Based on the provided references, there is no direct information about the specific contributors who made the model vinvino02/glpn-kitti available online as a GitHub repo. The references mention the authors "Dijk and Croon, 2019" and "Silberman et al., 2012" but it's unclear if they are the contributors for this specific model and repo.

[More Information Needed]
- **Model type:** The vinvino02/glpn-kitti model is a supervised deep learning model for monocular depth estimation trained on the KITTI dataset using a transformer encoder and a lightweight decoder with a scale-invariant log scale loss function.
- **Language(s):** The model vinvino02/glpn-kitti does not use or process natural human language. It is trained on the KITTI dataset, which contains stereo camera images and corresponding 3D LiDAR scans of various driving scenes.
- **License:** Based on the provided references, the license information for the model vinvino02/glpn-kitti is:

The model is currently for non-commercial purpose only (research, evaluation etc). The team will update the license soon.

[More Information Needed] on the specific name and link to the future license that will be used.
- **Finetuned from model:** Based on the provided references, there is no clear indication that the model vinvino02/glpn-kitti is fine-tuned from another pre-existing model. The references discuss the architecture and contributions of the proposed model, but do not mention any specific base model that was used for fine-tuning.

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/vinvino02/GLPDepth
- **Paper:** https://arxiv.org/pdf/2201.07436.pdf
- **Demo:** Here is the link to the demo of the model vinvino02/glpn-kitti:

<p>
<a href="https://colab.research.google.com/drive/1v6fzr4XusKdXAaeGZ1gKe1kh9Ce_WQhl?usp=sharing" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</p>

Thanks for the great Colab demo from NielsRogge.
## Uses

### Direct Use

The model vinvino02/glpn-kitti can be used for monocular depth estimation on the KITTI dataset without requiring any fine-tuning, post-processing, or plugging into a pipeline. To perform inference using the model on a directory of images, you can run the following command:

```
$ python ./code/test.py --dataset imagepath --data_path <dir_to_imgs> --save_visualize
```

This will load the pre-trained model and generate depth estimates for the images in the specified directory. The `--save_visualize` flag will save the visualized depth maps.

[More Information Needed] on the specific input image size and format requirements for the model.

The model is designed to be robust against various types of image corruptions commonly encountered in real-world scenarios, as demonstrated by the experiments conducted on corrupted images from the NYU Depth V2 dataset. However, [More Information Needed] on the specific performance of the model on the KITTI dataset under different corruption settings.

### Downstream Use

The vinvino02/glpn-kitti model can be fine-tuned for monocular depth estimation tasks on the KITTI dataset. It takes an RGB image as input and predicts a corresponding depth map.

To use the model in an application:
1. Resize the input RGB image to 704 x 352 pixels, which is the resolution the model was trained on for the KITTI dataset. 
2. Pass the resized image through the model to predict the depth map.
3. The predicted depth map will be the same 704 x 352 resolution. You can resize it back to the original image dimensions if needed.
4. The depth values are in meters, with a maximum range of 80m on KITTI.

[More Information Needed] for a specific code snippet.

The model's global-local architecture allows it to understand both high-level scene context and local pixel relationships for accurate depth prediction. Its data augmentation techniques also improve robustness.

The pre-trained weights can serve as a strong starting point to further fine-tune the model on custom datasets for monocular depth estimation in domains like autonomous driving, robotics, 3D reconstruction, etc. Reach out to the vinvino02 team for guidance on fine-tuning or integrating it into a larger system.

### Out-of-Scope Use

Based on the provided references about the vinvino02/glpn-kitti model, which appears to be a monocular depth estimation model trained on the KITTI dataset, here are some potential misuses and what users ought not do with the model:

The model should not be used in safety-critical applications without extensive further testing and validation. While the references mention the model shows robustness to image corruptions, relying solely on this model for depth perception in applications like self-driving vehicles could be very dangerous if the model fails on real-world data.

Users should not assume the model will generalize perfectly to data distributions very different from the KITTI dataset it was trained on. The KITTI data comes from car-mounted cameras in driving scenes, so the model may not perform as well in substantially different environments like indoor scenes, aerial imagery, etc.

The model takes monocular (single) camera images as input and should not be used with stereo camera pairs or other multi-view setups without modification and retraining. 

[More Information Needed] on the specific resolution and cropping used by this vinvino02/glpn-kitti model. The references mention a few different image sizes and crops used in various experiments. Users should be aware of the expected input resolution.

In general, the model should be used carefully and its outputs should not be blindly trusted, especially in applications where errors could cause harm. Additional testing should be done by the users to validate the model's performance and limitations in their specific use case.

### Bias, Risks, and Limitations

Based on the provided references about the vinvino02/glpn-kitti model, here are some potential issues and limitations:

Technical limitations:
- The model was trained and tested on a specific crop size of 704x352 on the KITTI dataset (Ref 1, 7). Performance on other image sizes is unknown. [More Information Needed] on how well it generalizes to other resolutions.
- Evaluation used a maximum depth value of 80m (Ref 1, 7). The model's accuracy beyond this range is unclear. [More Information Needed] on performance for farther depths.
- [More Information Needed] on computational requirements and latency for real-time inference, which is important for autonomous driving applications.

Foreseeable misuse and harms:
- If used in safety-critical applications like autonomous vehicles, any inaccuracies or failures in depth estimation could contribute to accidents. Thorough real-world testing and monitoring for edge cases would be critical.
- [More Information Needed] on fairness and bias testing. The model should be validated on diverse driving scenarios, locations, weather conditions, etc. to avoid biased or degraded performance for underrepresented groups.

Sociotechnical considerations:
- Transparency to end-users that this is an AI model with inherent limitations, not a perfect depth perception system. Misplaced overtrust could be dangerous.
- Questions of accountability in the case of model failures leading to harm. Delineation of responsibility between the model developers, deployers, and users needs to be established.
- Potential for misuse if depth data is used to identify or track individuals without consent. Usage restrictions and data governance policies should be implemented.

In summary, more real-world testing, transparency about limitations, and proactive risk mitigation strategies are recommended before deploying this model, especially in high-stakes applications. Ongoing monitoring for fairness, robustness and safety issues is also advised.

### Recommendations

Based on the provided references about the model vinvino02/glpn-kitti, here are some recommendations regarding foreseeable issues:

Robustness:
The model shows strong robustness against various types of image corruptions compared to other models, making it more suitable for safety-critical applications. However, further testing may be needed to ensure its performance under a wider range of real-world conditions.

Generalization:
The model demonstrates good generalization capabilities under cross-dataset validation. Nevertheless, it is important to continue evaluating its performance on diverse datasets to identify potential limitations and areas for improvement.

Computational Efficiency:
The proposed decoder architecture is compact and efficient, outperforming heavier decoders while using fewer parameters. This makes the model more suitable for resource-constrained environments. However, the overall computational requirements and inference time should be further assessed.

Ethical Considerations:
[More Information Needed] The provided references do not discuss ethical implications of the model, such as potential biases or misuse cases. It is crucial to conduct a thorough analysis of these aspects before deploying the model in real-world scenarios.

Societal Impact:
[More Information Needed] The references do not provide insights into the long-term societal impact of the model. It is important to consider how the model's application in various domains, such as autonomous vehicles or surveillance systems, may affect society and ensure that appropriate safeguards and regulations are in place.

Legal and Regulatory Compliance:
[More Information Needed] The references do not address legal and regulatory aspects related to the model's use. It is essential to ensure that the model complies with relevant laws, regulations, and industry standards in the target domains and regions of deployment.

In summary, while the model shows promising performance and robustness, further analysis is needed to address potential ethical, societal, and legal considerations before its widespread adoption. Continuous monitoring and updates may be necessary to maintain the model's reliability and alignment with evolving societal needs and expectations.

## Training Details

### Training Data

The training data of the model vinvino02/glpn-kitti consists of approximately 23K images from the KITTI dataset, which contains stereo camera images and corresponding 3D LiDAR scans of various driving scenes. The images were randomly cropped to 704 × 352 during training.

[More Information Needed] for documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the data of the model vinvino02/glpn-kitti:

Resizing:
The size of RGB images in the KITTI dataset is around 1224 × 368. The images are resized to the largest multiple of 32 below the original image size before being passed to the model to predict the depth map. The predicted depth map is then resized back to the original image size.

Training:
For training, a random crop of 704 × 352 is used from the resized images.

Evaluation:
For evaluation, the crop defined by Garg et al. (2016) is used, with a maximum depth value of 80m.

Data Augmentation:
The following data augmentation strategies are applied with 50% probability during training:
- Horizontal flips
- Random brightness (±0.2)
- Random contrast (±0.2) 
- Random gamma (±20)
- Random hue (±20)
- Random saturation (±30)
- Random value (±20)
- Vertical CutDepth with 25% possibility and p = 0.75

[More Information Needed] about the specific tokenization method used.

#### Training Hyperparameters

Here are the training hyperparameters used for the model vinvino02/glpn-kitti based on the provided references:

Framework: PyTorch

Optimizer: Adam 

Learning rate schedule:
- One-cycle learning rate strategy
- Learning rate increases from 3e-5 to 1e-4 following a poly LR schedule with factor 0.9 in first half of iterations
- Learning rate decreases from 1e-4 to 3e-5 in last half of iterations

Number of epochs: 25

Batch size: 12

Pre-trained weights: Started from MiT-b4 pre-trained weights

Data augmentation (applied with 50% probability):
- Vertical CutDepth (p=0.75, 25% possibility) 
- Horizontal flips
- Random brightness (±0.2)
- Random contrast (±0.2) 
- Random gamma (±20)
- Random hue (±20)
- Random saturation (±30)  
- Random value (±20)

[More Information Needed] on:
- Weight decay
- Gradient clipping
- Exact image sizes used
- Any learning rate warmup

#### Speeds, Sizes, Times

Here are the details I can provide about the model vinvino02/glpn-kitti based on the given references:

Dataset:
- The model was trained and evaluated on the KITTI dataset
- KITTI contains stereo camera images and corresponding 3D LiDAR scans of various driving scenes
- RGB image size is around 1224 × 368
- Trained on approximately 23K images with random crop of 704 × 352
- Tested on 697 images using the Garg crop and max depth of 80m

Training:
[More Information Needed] on training throughput, start/end times

Checkpoints:
- Trained checkpoint files are available for download from Google Drive:
[[Downloads]](https://drive.google.com/drive/folders/17yYbLZS2uQ6UVn5ET9RhVL0y_X3Ipl5_?usp=sharing)
[More Information Needed] on exact checkpoint file sizes

Evaluation:
- Predicted depth map PNG files for the KITTI Eigen split test set are available:  
[[Downloads]](https://drive.google.com/drive/folders/1LGNSKSaXguLTuCJ3Ay_UsYC188JNCK-j?usp=sharing)
- The model achieves state-of-the-art performance on the KITTI Eigen split benchmark:
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-local-path-networks-for-monocular/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=global-local-path-networks-for-monocular)

To reproduce the evaluation, the annotated depth maps (14GB) need to be downloaded and extracted:
```
[More Information Needed]
```

Then eval_with_pngs.py can be run on the KITTI dataset.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the model vinvino02/glpn-kitti evaluates on the following datasets:

1. KITTI: An outdoor depth estimation dataset containing stereo camera images and corresponding 3D LiDAR scans of various driving scenes. The model is trained on approximately 23K images and tested on 697 images from the KITTI dataset.

2. [More Information Needed] about evaluation on other datasets like NYU Depth V2, SUN RGB-D, or iBims-1, as the references mainly focus on KITTI for this specific model.

#### Factors

Based on the provided references about the model vinvino02/glpn-kitti, here are the foreseeable characteristics that may influence the model's behavior:

Domain and context:
- The model is trained and evaluated on the KITTI dataset, which contains stereo camera images and 3D LiDAR scans of various driving scenes acquired by car-mounted sensors. This suggests the model is primarily designed for outdoor depth estimation in the context of autonomous driving.

Population subgroups:
[More Information Needed] The references do not provide specific information about performance across different population subgroups.

Other factors influencing model behavior:
- Image corruption: The model's robustness against natural image corruptions is evaluated, showing stronger robustness compared to other models. This suggests the model may perform better on corrupted or low-quality images encountered in real-world scenarios.
- Cross-dataset generalization: The model is also evaluated on the iBims-1 dataset, an indoor depth estimation dataset, indicating some level of generalization ability to different domains.
- Data augmentation: The model is trained with task-specific data augmentation techniques like CutDepth, which can improve performance and generalization.

Disaggregated evaluation:
[More Information Needed] The references do not provide disaggregated evaluation results across different factors or subgroups to uncover potential performance disparities.

In summary, the model vinvino02/glpn-kitti is primarily designed for outdoor depth estimation in autonomous driving contexts, shows robustness to image corruptions, and exhibits some cross-dataset generalization ability. However, more information is needed regarding performance across population subgroups and disaggregated evaluation to fully assess potential biases or disparities.

#### Metrics

Based on the provided references, the following metrics will be used for evaluating the model vinvino02/glpn-kitti on the KITTI dataset:

To evaluate on the KITTI dataset, the following command can be used:
```
$ python ./code/eval_with_pngs.py --dataset kitti --split eigen_benchmark --pred_path ./best_kitti_preds/ --gt_path ./datasets/kitti/ --max_depth_eval 80.0 --garg_crop
```

The evaluation will be performed on 697 test images, using the crop defined by Garg et al. and a maximum depth value of 80m.

[More Information Needed] on the specific metrics that will be reported for the KITTI dataset evaluation, as the references do not provide details on the exact metrics used. Common depth estimation metrics include absolute relative error, squared relative error, RMSE, and threshold accuracy, but it is unclear which subset of these will be used for the KITTI evaluation of vinvino02/glpn-kitti.

The references also mention evaluating model robustness against image corruptions, but [More Information Needed] on whether this robustness evaluation will be performed on the KITTI dataset or is limited to the NYU Depth V2 dataset.

In summary, the vinvino02/glpn-kitti model will be evaluated on the KITTI dataset using the mentioned evaluation script, but more specific information is needed on the exact metrics that will be reported, in light of any tradeoffs between different types of errors.

### Results

Based on the provided references, here is the evaluation information for the model vinvino02/glpn-kitti:

Dataset:
The model was evaluated on the KITTI dataset, which contains stereo camera images and corresponding 3D LiDAR scans of various driving scenes. The dataset has around 23K training images and 697 test images. The images were randomly cropped to 704 × 352 for training and testing.

Metrics:
The model's performance was compared with previous works using the crop defined by Garg and a maximum depth value of 80m for evaluation. However, the specific quantitative metrics used for comparison are not provided in the given references.

[More Information Needed] on the exact metrics and corresponding values achieved by the model.

Evaluation Procedure:
To evaluate the model on the KITTI dataset, the following command can be used:

```
$ python ./code/eval_with_pngs.py --dataset kitti --split eigen_benchmark --pred_path ./best_kitti_preds/ --gt_path ./datasets/kitti/ --max_depth_eval 80.0 --garg_crop
```

This assumes that the annotated depth maps from the KITTI dataset have been downloaded and extracted to ./datasets/kitti/data_depth_annotated.

[More Information Needed] on the specific factors and ablation studies conducted for the model.

In summary, while the model was evaluated on the KITTI dataset using a specific crop and maximum depth value, more information is needed on the quantitative metrics, factors, and ablation studies to provide a comprehensive evaluation of the vinvino02/glpn-kitti model.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model vinvino02/glpn-kitti:

The model was evaluated on the KITTI dataset, which contains stereo camera images and corresponding 3D LiDAR scans of various driving scenes. The model was trained on approximately 23K images with a random crop of 704 × 352 and tested on 697 images. For comparison with previous works, the crop defined by Garg et al. and a maximum depth value of 80m were used for evaluation.

To evaluate the model, the following command can be used:

```
$ python ./code/eval_with_pngs.py --dataset kitti --split eigen_benchmark --pred_path ./best_kitti_preds/ --gt_path ./datasets/kitti/ --max_depth_eval 80.0 --garg_crop
```

The results on the KITTI dataset are shown in Table 6 of the referenced paper. [More Information Needed] about the specific performance metrics and how the model compares to other state-of-the-art methods on the KITTI dataset.

The model's generalization ability and robustness were also validated through extensive experiments. [More Information Needed] about the details of these experiments and their results.

In summary, the vinvino02/glpn-kitti model was evaluated on the KITTI dataset using a specific crop and maximum depth value, but more information is needed about the quantitative results and comparisons to other methods to fully assess its performance.

## Model Examination

Based on the provided references, here is what I can infer about the model vinvino02/glpn-kitti in terms of explainability and interpretability:

The model was validated through extensive quantitative and qualitative experiments on the NYU Depth V2 and SUN RGB-D datasets. The architecture and data augmentation method (vertical CutDepth) were shown to be effective in improving depth estimation performance.

An ablation study was conducted to demonstrate the impact of different components, such as the decoder design and data augmentation. The proposed decoder was found to be compact yet powerful compared to existing architectures.

The model's robustness was evaluated against various types of image corruptions, and it exhibited stronger robustness compared to other models. This suggests the model may be more suitable for safety-critical applications.

However, more specific information is needed regarding the explainability and interpretability techniques applied to the model, such as:

[More Information Needed] - Details on any visualization methods used to understand the model's predictions and internal representations.
[More Information Needed] - Information on whether any feature attribution or saliency mapping techniques were employed to identify important input regions.
[More Information Needed] - Discussion of any efforts to analyze the model's behavior and decision-making process.

In summary, while the provided references highlight the model's strong performance, robustness, and ablation studies, additional details are required to fully assess its explainability and interpretability capabilities.

## Environmental Impact

- **Hardware Type:** [More Information Needed]

The provided references do not contain information about the specific hardware used to train the vinvino02/glpn-kitti model. To answer this question, more details would be needed from the model developers or training logs regarding the GPU, CPU, TPU, or other hardware utilized during the training process.
- **Software Type:** The model vinvino02/glpn-kitti is trained using the PyTorch framework, as mentioned in the first point of the references:

"We implement the proposed network using the PyTorch framework."
- **Hours used:** Based on the provided references, there is no specific information about the training time for the model vinvino02/glpn-kitti. The references mention some training details such as the number of epochs (25) and batch size (12) for the NYU Depth V2 dataset, but there is no explicit mention of the training time for the KITTI dataset.

[More Information Needed] on the exact training time for the vinvino02/glpn-kitti model.
- **Cloud Provider:** [More Information Needed]

The provided references do not contain information about the specific cloud provider used to train the model vinvino02/glpn-kitti. More details would be needed from the model developers or training logs to determine the cloud provider.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the model vinvino02/glpn-kitti. To accurately report the carbon emissions, details such as the hardware used for training, total training time, energy consumption, and the carbon intensity of the electricity grid would be required.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The model vinvino02/glpn-kitti uses a novel global-local path architecture for monocular depth estimation. It consists of a hierarchical transformer encoder to capture global context and a lightweight yet powerful decoder to generate the estimated depth map. The encoder transforms the input RGB image into a bottleneck feature, which is then restored by the decoder to the original size through a series of convolution, upsampling, and feature fusion operations.

Key components:
- Hierarchical transformer encoder to learn global dependencies
- Lightweight decoder with fewer convolution and upsampling layers compared to previous approaches
- Global and local feature paths through the entire network

Objective:
The main objective of the model vinvino02/glpn-kitti is to predict an accurate depth map from a single RGB image. It aims to improve monocular depth estimation performance by leveraging the proposed global-local path architecture and an improved depth-specific data augmentation method.

[More Information Needed] on specific training details, dataset, and evaluation metrics.

Installation:
The model can be run using the provided Python requirements:

```
python==3.7.7
torch==1.6.0
h5py==3.6.0
scipy==1.7.3
opencv-python==4.5.5
mmcv==1.4.3
timm=0.5.4
albumentations=1.1.0
tensorboardX==2.4.1
gdown==4.2.1
```

Install the packages with:
```
$ pip install -r requirements.txt
```

Alternatively, a Docker image is available:
```
$ docker pull doyeon0113/glpdepth
```

### Compute Infrastructure

Here is the model card description for the compute infrastructure used for the vinvino02/glpn-kitti model, based on the provided references:

We implement the proposed network using the PyTorch framework. [More Information Needed] on the specific compute infrastructure such as GPU models, memory, or other hardware used for training the model.

The total number of epochs is set to 25 with a batch size of 12. 

For training, we use the one-cycle learning rate strategy with an Adam optimizer. The learning rate increases from 3e-5 to 1e-4 following a poly LR schedule with a factor of 0.9 in the first half of the total iteration, and then decreases from 1e-4 to 3e-5 in the last half.

[More Information Needed] on the dataset used for training and evaluating the vinvino02/glpn-kitti model specifically. The references mention the NYU Depth V2 and SUN RGB-D datasets but it's unclear if those were used for this particular model.

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

