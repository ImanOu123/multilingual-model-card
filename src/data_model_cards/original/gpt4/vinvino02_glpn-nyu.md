# Model Card for vinvino02/glpn-nyu

The model vinvino02/glpn-nyu is a state-of-the-art monocular depth estimation network that utilizes a novel global-local path architecture and an improved depth-specific data augmentation method to achieve high performance and robust generalization on the NYU Depth V2 dataset.

## Model Details

### Model Description

Model Name: vinvino02/glpn-nyu

### Model Architecture
The vinvino02/glpn-nyu model is a depth estimation framework designed to predict depth maps from RGB images. It features a novel architecture that incorporates both global and local feature paths throughout the network. The encoder is a hierarchical transformer that transforms the input image into a bottleneck feature, while the decoder is lightweight and effective, designed to restore the bottleneck feature to the original image size. The model also includes a fusion module that exploits local structures with fine details.

### Training Procedures
The model is implemented using the PyTorch framework and trained with a one-cycle learning rate strategy using an Adam optimizer. The learning rate varies from 3e-5 to 1e-4 following a poly LR schedule with a factor of 0.9 for the first half of the total iterations, then decreases back to 3e-5 for the second half. The model is trained for 25 epochs with a batch size of 12, using pre-trained weights from the MiT-b4. Data augmentation is applied to enhance the depth-specific features of the model.

### Parameters
- Initial Learning Rate: 3e-5
- Maximum Learning Rate: 1e-4
- Learning Rate Schedule: Poly LR with a factor of 0.9
- Epochs: 25
- Batch Size: 12
- Pre-trained Weights: MiT-b4

### Important Disclaimers
- The model has been trained and evaluated on the NYU Depth V2 dataset, and it achieves state-of-the-art performance on this dataset.
- The performance of the model is attributed to the proposed architecture and an enhanced depth-specific data augmentation method.
- The model demonstrates better generalization ability and robustness compared to other comparative models.
- The code for the model will be made available soon.
- The model uses scale-invariant log scale loss for training, which is suitable for depth estimation tasks.
- The maximum depth value used for scaling the predicted depth map is set to 10.0 meters for the NYU Depth V2 dataset.

[More Information Needed] for any additional specifics not covered by the provided references.

- **Developed by:** Doyeon Kim; Woonghyun Ka; Pyunghwan Ahn; Donggyu Joo; Sewhan Chun; Junmo Kim
- **Funded by:** [More Information Needed]
- **Shared by:** The contributors that made the model vinvino02/glpn-nyu available online as a GitHub repo are Doyeon Kim, Woonghyun Ga, Pyungwhan Ahn, Donggyu Joo, Sehwan Chun, and Junmo Kim.
- **Model type:** The model vinvino02/glpn-nyu is a supervised deep learning model for depth estimation that uses a transformer-based architecture with global and local feature paths, trained on the NYU Depth V2 dataset with scale-invariant log scale loss and enhanced depth-specific data augmentation methods, employing a one-cycle learning rate strategy with an Adam optimizer.
- **Language(s):** The model vinvino02/glpn-nyu does not process natural human language; instead, it is designed for depth estimation from RGB images using a transformer-based architecture.
- **License:** The license being used for the model vinvino02/glpn-nyu is currently not specified in the provided references. It is mentioned that the model is for non-commercial use only and that the license will be updated soon, but no specific license name or link is provided.

[More Information Needed]
- **Finetuned from model:** The model vinvino02/glpn-nyu is fine-tuned from the MiT-b4 model. Unfortunately, a direct link to the base model is not provided in the references given. However, the base model is mentioned in the context of using pre-trained weights from MiT-b4 [Xie et al., 2021]. To find the base model, one would typically search for the paper by Xie et al. from 2021 or look for the MiT-b4 model in model repositories or databases.
### Model Sources

- **Repository:** https://github.com/vinvino02/GLPDepth
- **Paper:** https://arxiv.org/pdf/2201.07436.pdf
- **Demo:** The demo of the model vinvino02/glpn-nyu can be found at the following link:

[Open In Colab](https://colab.research.google.com/drive/1v6fzr4XusKdXAaeGZ1gKe1kh9Ce_WQhl?usp=sharing)

Thanks to NielsRogge for providing the great Colab demo.
## Uses

### Direct Use

The model `vinvino02/glpn-nyu` is designed for depth estimation tasks and can be used directly for inference on images without the need for fine-tuning, post-processing, or integration into a larger pipeline. This is particularly useful for users who want to quickly evaluate the model's performance or use it for depth estimation on their own images.

To use the model for inference with an image directory, you can follow the provided code snippet for inference. Here's how you can use the model with your own images:

```bash
$ python ./code/test.py --dataset imagepath --data_path <dir_to_imgs> --save_visualize
```

Replace `<dir_to_imgs>` with the path to the directory containing your images. The command will run the model on the images in the specified directory, and the result images will be saved in the default directory `./results/test` or in the directory specified by `./args.result_dir/args.exp_name`.

For evaluating the model on the NYU Depth V2 dataset, you can use the following command:

```bash
$ python ./code/eval_with_pngs.py --dataset nyudepthv2 --pred_path ./best_nyu_preds/ --gt_path ./datasets/nyu_depth_v2/ --max_depth_eval 10.0
```

This command will evaluate the pre-trained model on the NYU Depth V2 dataset and does not require any fine-tuning or post-processing steps. The maximum depth for evaluation is set to 10.0 meters, as specified in the command.

If you have the KITTI dataset and want to evaluate the model on it, you can use the following command:

```bash
$ python ./code/eval_with_pngs.py --dataset kitti --split eigen_benchmark --pred_path ./best_kitti_preds/ --gt_path ./datasets/kitti/ --max_depth_eval 80.0 --garg_crop
```

Again, this command is for direct evaluation without any additional steps needed.

Please note that the above commands assume that you have the necessary datasets and the model's codebase set up correctly on your system. If you encounter any issues or have specific questions about using the model, you can reach out to the project organizer for assistance and model card updates.

### Downstream Use

The `vinvino02/glpn-nyu` model is a deep learning model designed for depth estimation tasks, which can be fine-tuned for specific applications or integrated into larger systems to enhance their capabilities. Here are some potential use cases and how the model can be utilized:

1. **Autonomous Vehicles and Robotics**: The model can be fine-tuned on datasets specific to the operational environment of autonomous vehicles or robots to improve their ability to perceive depth and navigate through space. This could be critical for obstacle avoidance and path planning.

2. **Augmented Reality (AR) and Virtual Reality (VR)**: In AR/VR applications, accurate depth estimation can enhance the user experience by providing more realistic interactions with virtual objects. The model can be integrated into AR/VR software to improve the depth perception in real-time.

3. **3D Reconstruction**: For applications that require 3D reconstruction from 2D images, such as in architecture or archeology, the model can be fine-tuned on images of buildings or artifacts to generate accurate depth maps, which can then be used to create 3D models.

4. **Medical Imaging**: In medical imaging, depth estimation can be useful for non-invasive exploration of body parts. The model could be fine-tuned on medical imaging data to assist in procedures like endoscopy.

To fine-tune or use the `vinvino02/glpn-nyu` model for a specific task, you would typically follow these steps:

1. Prepare your dataset for the specific task, ensuring that it is formatted similarly to the datasets the model has been trained on (e.g., NYU Depth V2 or KITTI).

2. Fine-tune the model on your dataset. This involves setting up a training script similar to the one provided for the NYU Depth V2 dataset, but with paths and parameters adjusted for your data.

3. Evaluate the model to ensure that it meets the performance requirements of your task.

4. Integrate the model into your application or ecosystem. This could involve setting up an inference pipeline where the model processes input data and provides depth estimates to be used by other components of your system.

Here is a hypothetical code snippet for fine-tuning the model on a new dataset, assuming you have a similar setup and requirements as the original training process:

```python
# Assuming you have a script similar to train.py for fine-tuning
$ python ./code/train.py --dataset yourdataset --data_path ./datasets/yourdataset/ --max_depth <your_max_depth> --max_depth_eval <your_max_depth_eval>
```

Please note that the actual code and commands would depend on the specifics of your dataset and task. If you need to adjust hyperparameters, data augmentation strategies, or other training details, you would modify the training script accordingly.

For integrating the model into an application for inference, you would use a script similar to `test.py`, specifying the path to the images you want to process:

```python
# Inference with a directory of images for your application
$ python ./code/test.py --dataset imagepath --data_path <dir_to_imgs> --save_visualize
```

Again, the specifics of how you integrate the model into your application will depend on the requirements of your system and the format of your input data. If you need to perform additional pre-processing or post-processing steps, you would include those in your application's pipeline.

### Out-of-Scope Use

As the sociotechnic on the deep learning model development team, it is my responsibility to consider the potential misuse of the model vinvino02/glpn-nyu and to advise users on what they should not do with it. Based on the references provided, here are some considerations regarding the misuse of the model:

1. **Safety-Critical Applications**: Reference 2 indicates that our model is robust and appropriate for safety-critical applications. However, users should not over-rely on the model for life-critical tasks without thorough validation specific to their use case. While the model shows strong robustness against image corruptions, it is not fail-proof. Users should not use the model as the sole decision-making tool in scenarios where depth estimation errors could lead to harm or injury.

2. **Commercial Use**: Reference 4 states that the current license is for non-commercial purposes only. Users should not use the model for commercial purposes until the license is updated to permit such use. Misuse in this context would be any commercial exploitation of the model without proper authorization or licensing.

3. **Data Augmentation and Generalization**: Reference 6 and 8 discuss the model's generalization capabilities and the effectiveness of the data augmentation method used. Users should not assume that the model will perform equally well on datasets that are significantly different from the ones it was trained and validated on (NYU Depth V2 and SUN RGB-D datasets). Misuse would include applying the model to vastly different datasets without additional training or validation.

4. **Image Corruption**: Reference 5 and 6 mention that the model has been tested against images corrupted by 16 different methods. Users should not assume that the model will be robust against all forms of corruption or novel corruption methods not included in the testing. It would be a misuse to deploy the model in environments where it is exposed to image corruptions beyond what it has been tested for without additional robustness checks.

5. **Ethical Considerations**: Users should not use the model in ways that could infringe on privacy rights or in surveillance systems without consent. The model's ability to estimate depth from monocular images could potentially be used to derive sensitive information about environments or individuals.

In summary, users of the model vinvino02/glpn-nyu should not:
- Use it as the sole decision-making entity in safety-critical applications without additional safeguards.
- Use it for commercial purposes unless the license is updated to allow such use.
- Apply it to significantly different datasets without further training or validation.
- Assume it will be robust against all forms of image corruption not explicitly tested.
- Use it in ways that could violate ethical standards or privacy rights.

[More Information Needed] for any specific legal or ethical guidelines that may apply to the use of the model in various jurisdictions or contexts.

### Bias, Risks, and Limitations

The model vinvino02/glpn-nyu, designed for depth estimation, exhibits state-of-the-art performance on the NYU Depth V2 dataset and demonstrates robustness against image corruption. However, there are several known or foreseeable issues that may arise from the deployment of this model:

1. **Generalization to Unseen Environments**: While the model generalizes well under cross-dataset validation, its performance on datasets significantly different from NYU Depth V2 and SUN RGB-D is not guaranteed. Real-world deployment could encounter diverse scenarios not represented in the training data, potentially leading to reduced accuracy or failure to generalize.

2. **Robustness to Extreme Cases**: Although the model is robust to certain types of image corruption, extreme cases or corruptions not represented in the training data could still pose challenges. The model's performance in such scenarios is unknown and could lead to errors in depth estimation.

3. **Sociotechnical Implications**: The use of the model in safety-critical applications, as suggested by its robustness, requires careful consideration of potential harms. Misestimations of depth in autonomous vehicles or healthcare systems could lead to accidents or misdiagnoses, respectively.

4. **Data Augmentation Limitations**: The effectiveness of the model is partly attributed to specific data augmentation techniques. However, these techniques may not cover all possible variations in real-world data, which could lead to overfitting or reduced performance in practical applications.

5. **Computational Efficiency**: The model's decoder is designed to be compact and efficient, with fewer parameters compared to other decoders. While this is beneficial for deployment on devices with limited computational resources, it may also limit the model's capacity to capture more complex features or scale to higher-resolution inputs without modifications.

6. **Ethical and Legal Considerations**: Deployment of the model in real-world applications must adhere to ethical standards and legal regulations, especially regarding privacy and data protection. The use of depth estimation in surveillance or data-sensitive environments could raise concerns that need to be addressed.

7. **Misunderstandings of Model Capabilities**: There is a risk that users may overestimate the model's capabilities based on its performance on benchmark datasets. Clear communication regarding the model's limitations and appropriate use cases is necessary to prevent misuse or overreliance on the model's output.

8. **Dependency on Quality and Diversity of Training Data**: The model's performance is highly dependent on the quality and diversity of the training data. If the training data lacks representation of certain demographics or environments, the model may exhibit biases or reduced performance for those underrepresented groups or scenarios.

In conclusion, while the model vinvino02/glpn-nyu shows promising results in depth estimation, it is important to consider these technical and sociotechnical limitations when deploying the model in real-world applications. Further research and testing are required to address these issues and ensure the model's safe and ethical use.

### Recommendations

Based on the provided references, here are some recommendations with respect to the foreseeable issues about the model vinvino02/glpn-nyu:

1. **Robustness to Image Corruption**: As mentioned in reference 3, the model has been tested for robustness against natural image corruptions, which is crucial for real-world applications. It is recommended to continue evaluating and improving the model's robustness to various types of image corruptions that were not covered in the initial tests. This could include more extensive testing on corrupted datasets or real-world scenarios to ensure the model's reliability.

2. **Generalization Ability**: Reference 6 and 8 highlight the model's generalization capabilities, particularly under cross-dataset validation. It is recommended to further test the model on a wider range of datasets, especially those that are significantly different from NYU Depth V2 and SUN RGB-D, to ensure that the model can maintain its performance across diverse environments and conditions.

3. **Data Augmentation Method**: The vertical CutDepth data augmentation method has shown to improve performance as per reference 5. It is recommended to explore additional data augmentation techniques that could further enhance the model's accuracy and generalization. This could involve experimenting with different hyperparameters or introducing new augmentation strategies.

4. **Model Compactness**: Reference 2 emphasizes the compactness of the proposed decoder, which is a significant advantage in terms of computational efficiency. It is recommended to maintain this focus on efficiency in future iterations of the model, potentially exploring ways to further reduce the parameter count without sacrificing performance.

5. **Ethical and Societal Considerations**: As a sociotechnic, it is important to consider the ethical implications of deploying the model in real-world applications. This includes ensuring that the model does not reinforce biases present in the training data, is used in a manner that respects privacy, and that its limitations are clearly communicated to users. [More Information Needed] on the specific ethical considerations that were taken into account during the development of the model.

6. **Transparency and Accessibility**: To foster trust and collaboration, it is recommended to make the code and pre-trained models readily available, as mentioned in reference 8. Additionally, providing thorough documentation and usage guidelines will help users understand how to effectively implement the model in their own projects.

7. **Continuous Evaluation and Improvement**: As with any deep learning model, it is recommended to continuously monitor the performance of vinvino02/glpn-nyu and update it as new data becomes available or as new techniques are developed in the field of depth estimation.

In summary, while the model vinvino02/glpn-nyu has demonstrated state-of-the-art performance and robustness, it is important to continue testing its generalization capabilities, explore further data augmentation methods, maintain model compactness, and address ethical and societal considerations in its deployment. Transparency in sharing the model's code and methodology will also be crucial for fostering trust and collaboration within the community.

## Training Details

### Training Data

The training data for the model vinvino02/glpn-nyu consists of images from the NYU Depth V2 dataset, which is specifically used for evaluating pre-trained models. The dataset includes a variety of indoor scenes, and for our model, only the official test set of 5050 images is utilized. Images are resized to the largest multiple of 32 below the original size for processing, and a range of data augmentation techniques are applied, including vertical CutDepth, to enhance the depth estimation performance. [More Information Needed] on data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

For the preprocessing of the data used in the model vinvino02/glpn-nyu, we perform several steps to ensure the input data is suitable for training and inference. The preprocessing steps include:

1. Data Augmentation: We apply various data augmentation techniques to the input images to improve the robustness and generalization of the model. The augmentation strategies include:
   - Horizontal flips with a 50% probability.
   - Random adjustments to brightness, contrast, gamma, hue, saturation, and value within specified ranges (±0.2 for brightness and contrast, ±20 for gamma and hue, ±30 for saturation, and ±20 for value).
   - Vertical CutDepth with a probability of 0.75 and a 25% possibility of being applied.

2. Resizing: The references do not explicitly mention the resizing of the images. However, it is common practice to resize images to a fixed size that is compatible with the input layer of the neural network. [More Information Needed] on the exact resizing dimensions used for the model.

3. Normalization: While not explicitly stated in the references, normalization is a standard preprocessing step in deep learning where pixel values are typically scaled to a range that the model expects, often [0, 1] or [-1, 1]. [More Information Needed] on the specific normalization technique used for the model.

4. Tokenization: Since this model is dealing with depth estimation from images, tokenization typically associated with text data is not applicable here. Therefore, there is no tokenization process involved in the preprocessing of the data for this model.

5. Loss Calculation Preprocessing: For the calculation of the training loss, we use the scale-invariant log scale loss. This requires computing the logarithm of the predicted output and the ground truth depth map values. The loss is computed as per the given equation, which involves the difference in the log values of the corresponding pixels in the predicted and ground truth depth maps.

6. Selective Feature Fusion (SFF): Although not a preprocessing step for the input data, the SFF module is used during the training to selectively focus on salient regions of the features. This is an architectural detail that affects how the model processes the data internally.

For the exact preprocessing code and parameters, such as image resizing dimensions or normalization values, [More Information Needed] as they are not provided in the references.

#### Training Hyperparameters

The training hyperparameters for the model vinvino02/glpn-nyu are as follows:

- Framework: PyTorch
- Learning Rate Strategy: One-cycle learning rate strategy
- Initial Learning Rate: 3e-5
- Maximum Learning Rate: 1e-4
- Learning Rate Schedule: Poly LR schedule with a factor of 0.9
- Total Number of Epochs: 25
- Batch Size: 12
- Optimizer: Adam
- Pre-trained Weights: MiT-b4 [Xie et al., 2021]
- Maximum Depth for Training: 10.0 (as specified in the command for NYU Depth V2 dataset)
- Loss Function: Scale-invariant log scale loss [Eigen et al., 2014]
- Data Augmentation: Vertical CutDepth with 50% probability, horizontal flips, random brightness (±0.2), contrast (±0.2), gamma (±20), hue (±20), saturation (±30), and value (±20)
- Probability for Vertical CutDepth: p = 0.75 with 25% possibility

Please note that the above details are specific to the NYU Depth V2 dataset as per the provided references.

#### Speeds, Sizes, Times

The model card description for vinvino02/glpn-nyu could include the following details based on the provided references:

```
# Model Card for vinvino02/glpn-nyu

## Model Description
vinvino02/glpn-nyu is a state-of-the-art deep learning model for depth estimation on the NYU Depth V2 dataset. It leverages a transformer encoder and a compact decoder to estimate accurate depth maps efficiently. The model has been trained using a scale-invariant log scale loss and enhanced with depth-specific data augmentation methods, including the proposed vertical CutDepth.

## Training and Evaluation Data
The model was trained and evaluated on the official test set of the NYU Depth V2 dataset, which consists of 5050 images. The images were resized to the largest multiple of 32 below the original size for processing and then resized back to the original dimensions for depth map prediction.

## Training Procedure
- Framework: PyTorch
- Learning Rate Strategy: One-cycle policy with a poly LR schedule
- Initial Learning Rate: 3e-5
- Maximum Learning Rate: 1e-4
- Total Epochs: 25
- Batch Size: 12
- Pre-trained Weights: MiT-b4
- Data Augmentation: Vertical CutDepth, horizontal flips, random brightness, contrast, gamma, hue, saturation, and value adjustments.

## Hyperparameters
- Maximum Depth for Training and Evaluation: 10.0 (as specified for the NYU Depth V2 dataset)
- Command to Train on NYU Depth V2: `$ python ./code/train.py --dataset nyudepthv2 --data_path ./datasets/ --max_depth 10.0 --max_depth_eval 10.0`

## Model Size and Performance
- Parameters: 0.66M
- The model outperforms other networks with larger decoders, such as MLPdecoder (3.19M parameters), BTS (5.79M parameters), and DPT (14.15M parameters), indicating an efficient design of the restoring path for the encoder.

## Loss Function
- Scale-invariant log scale loss as described by Eigen et al., 2014.

## Results
The model demonstrates state-of-the-art performance on the NYU Depth V2 dataset and shows better generalization ability and robustness compared to other models. The effectiveness of the vertical CutDepth data augmentation method has been validated through an ablation study.

## Additional Information
- Checkpoint Sizes: [More Information Needed]
- Throughput: [More Information Needed]
- Start or End Time of Training: [More Information Needed]

The code and additional details on the model's performance will be made available soon.
```

Please note that some specific details such as checkpoint sizes, throughput, and exact start or end times of training are not provided in the references and therefore are marked as "[More Information Needed]".

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model vinvino02/glpn-nyu evaluates on the following benchmarks or datasets:

1. NYU Depth V2 dataset: This is used for evaluating pre-trained models, specifically using the official test set of 5050 images for quantitative and qualitative evaluation, as well as for conducting ablation studies to demonstrate the effectiveness of the model's contributions.

2. KITTI dataset: The model is trained on approximately 23K images from the KITTI dataset and tested on 697 images. The KITTI dataset contains stereo camera images and corresponding 3D LiDAR scans of various driving scenes. For performance comparison, the model uses a crop defined by Garg and a maximum evaluation depth of 80m.

3. SUN RGB-D dataset: Although not explicitly mentioned in the provided references, the mention of "several experiments on the NYU Depth V2 and SUN RGB-D datasets" suggests that the SUN RGB-D dataset is also used for validation.

4. iBims-1 dataset: This indoor dataset is mentioned as part of additional results provided, indicating that the model has also been evaluated on it.

5. Cross-dataset validation: The model has been tested for generalization ability across different datasets, which implies that additional datasets may have been used for evaluation, although they are not explicitly named in the provided references.

#### Factors

The model vinvino02/glpn-nyu is designed for depth estimation, and its performance has been validated on the NYU Depth V2 and SUN RGB-D datasets. The following characteristics are foreseeable in influencing how the model behaves:

1. **Domain and Context**: The model has been trained and evaluated on indoor scene datasets (NYU Depth V2 and SUN RGB-D). Therefore, it is expected to perform well in similar indoor environments. However, its performance in outdoor scenes or environments that significantly differ from the training data is not guaranteed and may require additional validation.

2. **Robustness to Image Corruption**: The model has demonstrated robustness against 16 different types of image corruption, with each corruption applied at five different intensities. This suggests that the model should maintain performance in real-world scenarios where images may be degraded due to various factors such as noise, blur, or compression artifacts. This robustness makes it more suitable for safety-critical applications where reliability is crucial.

3. **Image Size and Preprocessing**: The model requires images to be resized to the largest multiple of 32 below the original image size before depth estimation. This preprocessing step is necessary for the model to function correctly, and the final depth map is resized back to the original image dimensions. The resizing process could influence the model's performance, especially if the original image size is not close to a multiple of 32.

4. **Data Augmentation and Generalization**: The model employs a specific data augmentation technique, vertical CutDepth, which has been shown to be effective in the ablation study. The model's ability to generalize well under cross-dataset validation suggests that it can handle variations in data not seen during training. However, the extent of this generalization to completely different datasets or real-world scenarios is not fully known.

5. **Population Subgroups**: The references do not provide specific information on the model's performance across different population subgroups. Since the datasets used are focused on indoor scenes, the model's interaction with varying human subjects, if any, is not discussed. [More Information Needed] to determine if there are disparities in performance when considering factors such as age, gender, or ethnicity of individuals present in the scenes.

6. **Computational Efficiency**: The model incorporates a selective feature fusion (SFF) module that allows it to focus on salient regions with a low computational burden. This suggests that the model is designed to be computationally efficient, which could be beneficial for deployment in real-time applications or on devices with limited processing power.

In summary, the model vinvino02/glpn-nyu is expected to perform well in indoor environments, particularly when dealing with image corruptions. It has been designed with computational efficiency in mind and has demonstrated the ability to generalize across similar datasets. However, performance disparities across different population subgroups or in vastly different domains from the training data are not addressed in the provided references, and further evaluation would be needed to uncover any potential issues in these areas.

#### Metrics

The evaluation metrics used for the model vinvino02/glpn-nyu are not explicitly listed in the provided references. However, since the model is evaluated on the NYU Depth V2 dataset and there is a mention of state-of-the-art performance in "most of the evaluation metrics," we can infer that standard depth estimation metrics are likely used. These typically include:

- Absolute Relative Difference (Abs Rel)
- Squared Relative Difference (Sq Rel)
- Root Mean Squared Error (RMSE)
- Root Mean Squared Error in log scale (RMSE log)
- Accuracy with different threshold values (e.g., δ1, δ2, δ3 where δ is the threshold for relative error)

Additionally, the model uses scale-invariant log scale loss for training, which suggests that the evaluation might also consider this loss function as a metric for performance comparison.

To confirm the exact metrics used for evaluation, more information from the actual results or methodology sections would be needed. Without explicit mention in the provided references, we cannot definitively state which metrics are used. Therefore, the answer is [More Information Needed].

### Results

Based on the provided references, here is the evaluation results description for the model `vinvino02/glpn-nyu`:

---

## Evaluation Results of vinvino02/glpn-nyu

Our model `vinvino02/glpn-nyu` has been rigorously evaluated on the NYU Depth V2 dataset, demonstrating state-of-the-art performance in depth estimation tasks. The model's architecture and the novel data augmentation method, including the vertical CutDepth, have significantly contributed to its high performance.

### Factors:
- **Dataset**: NYU Depth V2 official test set, consisting of 5050 images.
- **Image Preprocessing**: Images were resized to the largest multiple of 32 below the original size before depth map prediction, and the predicted depth maps were resized back to the original image size for evaluation.
- **Data Augmentation**: Enhanced depth-specific data augmentation method, including a novel technique called vertical CutDepth.
- **Cross-dataset Validation**: The model has shown robust generalization capabilities when validated across different datasets.

### Metrics:
- **Scale-invariant Log Scale Loss**: The model was trained using a scale-invariant log scale loss, which is crucial for depth estimation tasks.
- **Quantitative Metrics**: The model outperforms existing methods and recent state-of-the-art models like Adabins and DPT on most of the standard depth estimation metrics. However, specific metric values such as RMSE, Abs Rel, Sq Rel, etc., are not provided in the references and would require [More Information Needed].
- **Qualitative Evaluation**: The model has undergone qualitative evaluation, although specific examples or visual results are not included in the references provided.

### Additional Notes:
- The model has been validated through extensive quantitative and qualitative experiments.
- An ablation study confirms the effectiveness of the proposed architecture and data augmentation methods.
- The model demonstrates state-of-the-art performance on the NYU Depth V2 dataset as per the comparison presented in Table 1 of the references.

For detailed performance metrics and visual results, users are encouraged to refer to the supplementary material or the full paper associated with the model.

--- 

Please note that specific numerical evaluation results are not provided in the references, and thus [More Information Needed] for those details.

#### Summary

The model vinvino02/glpn-nyu has been rigorously evaluated on the NYU Depth V2 dataset, demonstrating state-of-the-art performance in depth estimation. The model outperforms existing methods, including recent state-of-the-art models like Adabins and DPT, which is attributed to its novel architecture and an enhanced depth-specific data augmentation method. The model's compact decoder, in conjunction with a transformer encoder, contributes significantly to its ability to estimate accurate depth maps efficiently.

Quantitative and qualitative evaluations, along with an ablation study, have been conducted to validate the effectiveness of each contribution within the model. The results indicate that the model is not only accurate in depth estimation but also robust to various illumination conditions, as evidenced by the visualized results.

Furthermore, the model shows promising generalization capabilities, as tested on the SUN RGB-D dataset, indicating its robustness and adaptability to different indoor environments. The model's performance is also resilient against image corruption, showcasing its reliability in practical applications.

The evaluation process for the NYU Depth V2 dataset involves using a specific Python script to compare predicted depth maps with ground truth, with a maximum depth evaluation threshold set to 10.0 meters. The results are saved in a designated results directory.

In summary, the vinvino02/glpn-nyu model is a highly effective and efficient solution for depth estimation, with proven state-of-the-art performance on the NYU Depth V2 dataset and strong generalization across different datasets. The code for the model is expected to be made available soon for further research and development in the field.

## Model Examination

### Model Card for vinvino02/glpn-nyu

#### Model Description
The `vinvino02/glpn-nyu` model is a state-of-the-art deep learning architecture designed for depth estimation tasks, specifically trained and validated on the NYU Depth V2 dataset. Our model introduces a novel decoder design that is both compact and efficient, with only 0.66M parameters, significantly fewer than other contemporary models such as MLPdecoder, BTS, and DPT. Despite its compactness, our network outperforms these heavier models, demonstrating the effectiveness of our design.

#### Model Architecture
Our model features an encoder with a carefully designed restoring path that allows for high performance with a minimal number of parameters. The decoder integrates multi-scale features and is robust against various types of image corruptions, making it suitable for safety-critical applications.

#### Data Augmentation and Robustness
We have conducted an ablation study on the data augmentation method used, particularly the vertical CutDepth technique. The results show that our model benefits from this augmentation, enhancing its robustness to natural image corruptions. This robustness is crucial for real-world applications where images are often imperfect.

#### Experiments and Validation
Extensive quantitative and qualitative experiments have been performed to validate the model's performance. Our approach has been compared with different decoder designs and against existing methods on the NYU Depth V2 and SUN RGB-D datasets. The model has demonstrated state-of-the-art performance and strong generalization capabilities across different datasets.

#### Explainability/Interpretability
[More Information Needed]

#### Additional Information
The model has been tested for robustness following the study on the robustness of CNNs by Hendrycks and Dietterich (2018). It has shown superior performance against image corruption, further proving its suitability for real-world scenarios. The code and supplementary materials, including additional results and comparisons, will be made available to facilitate further research and development in the field of depth estimation.

#### Conclusion
The `vinvino02/glpn-nyu` model is a highly efficient and robust solution for depth estimation tasks, offering state-of-the-art performance with a fraction of the parameters required by other models. Its design and data augmentation methods have been thoroughly validated, and it shows promising generalization and robustness characteristics.

## Environmental Impact

- **Hardware Type:** [More Information Needed]
- **Software Type:** The model vinvino02/glpn-nyu is trained on the PyTorch framework.
- **Hours used:** The amount of time used to train the model vinvino02/glpn-nyu is not explicitly stated in the provided references. Therefore, the answer is "[More Information Needed]".
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model vinvino02/glpn-nyu features a novel architecture designed for monocular depth estimation. The architecture is characterized by a global-local path strategy that aims to predict a depth map from a given RGB image. The model's encoder employs a hierarchical transformer to capture rich global information from the input image, which is initially processed into a sequence of patches and then passed through multiple self-attention and MLP-Conv-MLP layers with residual skips. This design helps in reducing computational costs while maintaining the ability to learn global dependencies.

For the decoding phase, the model utilizes a lightweight and effective decoder that restores the bottleneck feature to the original image size with much fewer convolution and bilinear upsampling layers than conventional methods. The channel dimension of the bottleneck feature is first reduced with a 1x1 convolution to decrease computational complexity. Then, consecutive bilinear upsampling is used to enlarge the feature to the desired output size. The final output is obtained by passing the upscaled features through two convolution layers and a sigmoid function, which predicts the depth map that is subsequently scaled in meters.

Additionally, the model incorporates a selective feature fusion (SFF) module through skip connections to exploit local structures with fine details. The SFF module allows the model to selectively focus on salient regions by estimating an attention map for both features, enhancing the model's performance with minimal additional computational burden.

The objective of the vinvino02/glpn-nyu model is to achieve state-of-the-art performance in depth estimation on the challenging NYU Depth V2 dataset. The model has been validated through extensive experiments, demonstrating its effectiveness, generalization ability, and robustness compared to other models. It also benefits from an improved depth-specific data augmentation method that further boosts its performance.

### Compute Infrastructure

The provided references do not include specific details about the compute infrastructure used for training the model vinvino02/glpn-nyu. Information such as the type of GPUs, the number of GPUs, the memory capacity, or the specific hardware configuration is not mentioned in the given text. Therefore, to answer the question regarding the compute infrastructure for the model vinvino02/glpn-nyu, I must say:

[More Information Needed]

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

