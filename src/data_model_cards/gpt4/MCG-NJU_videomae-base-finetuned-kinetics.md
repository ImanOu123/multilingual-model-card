# Model Card for MCG-NJU/videomae-base-finetuned-kinetics

The model MCG-NJU/videomae-base-finetuned-kinetics is a data-efficient video transformer pre-trained using a self-supervised learning approach with a high masking ratio, demonstrating strong performance and transferability on various video understanding tasks, including action recognition and detection, without the need for extra large-scale datasets.

## Model Details

### Model Description

Model Name: MCG-NJU/videomae-base-finetuned-kinetics

### Model Architecture:
The MCG-NJU/videomae-base-finetuned-kinetics model utilizes an asymmetric encoder-decoder architecture tailored for video self-supervised pre-training. The encoder and decoder are specifically designed to capture high-level spatio-temporal information in videos. The model is based on a 16-frame vanilla ViT-Base (Vision Transformer) and incorporates joint space-time attention mechanisms to enhance its understanding of video data. The decoder is discarded during the fine-tuning phase.

### Training Procedures:
- **Pre-training**: The VideoMAE model is pre-trained on the Kinetics-400 dataset for 800 epochs with a high masking ratio of 75%. During pre-training, 16 frames are sampled with a temporal stride of 4, and the batch size and base learning rate are set to 192 and 3e-4, respectively. The pre-training schedule is designed to encourage the model to learn useful spatiotemporal structures from the video data.
- **Fine-tuning**: After pre-training, the model undergoes fine-tuning on the Kinetics-400 dataset for 100 epochs using a batch size of 128. The base learning rate for fine-tuning is set to 5e-4, with a layer decay of 0.7 and a drop path rate of 0.2. Repeated augmentation techniques are employed during fine-tuning to improve the model's generalization capabilities. Dense sampling following the Slowfast protocol is used during this phase.

### Parameters:
- **Masking Ratio**: 75% during pre-training.
- **Batch Size**: 192 for pre-training, 128 for fine-tuning.
- **Base Learning Rate**: 3e-4 for pre-training, 5e-4 for fine-tuning.
- **Layer Decay**: 0.7 during fine-tuning.
- **Drop Path Rate**: 0.2 during fine-tuning.
- **Sampling**: 16 frames with a temporal stride of 4 for pre-training, dense sampling for fine-tuning.

### Evaluation:
For evaluation, the model follows the inference protocol of 5 clips × 3 crops.

### Important Disclaimers:
- The model's performance may vary when transferred to other video datasets due to potential domain shifts between the pre-training and target datasets.
- The high masking ratio is a core design feature of VideoMAE, and its impact has been investigated on the Kinetics-400 dataset.
- The longer pre-training schedule has been shown to provide slight gains, as evidenced by experiments on the Kinetics-400 and Something-Something V2 datasets.
- The effectiveness of temporal modeling may be less pronounced in datasets like Kinetics, where videos are mostly stationary and scene-related.

For more detailed instructions on fine-tuning and pre-training, please refer to the provided [FINETUNE.md](FINETUNE.md) and [PRETRAIN.md](PRETRAIN.md) documents.

[More Information Needed] for any additional specifics not covered by the provided references.

- **Developed by:** Zhan Tong; Yibing Song; Jue Wang; Limin Wang
- **Funded by:** [More Information Needed]
- **Shared by:** The contributors who made the model MCG-NJU/videomae-base-finetuned-kinetics available online as a GitHub repo include Ziteng Gao, Lei Chen, Chongjian Ge, Zhiyu Zhao, and Zhan Tong. Additionally, the project acknowledges the contributions of the codebases MAE-pytorch, BEiT, SlowFast, and pytorch-image-models, with thanks to the contributors of these repositories.
- **Model type:** The model MCG-NJU/videomae-base-finetuned-kinetics is a self-supervised video pre-training model using a masked autoencoder (MAE) approach with a plain Vision Transformer (ViT) backbone, trained on video modality with an extremely high masking ratio for efficient learning of spatiotemporal structures.
- **Language(s):** The model MCG-NJU/videomae-base-finetuned-kinetics does not process natural human language; instead, it is designed for action recognition and detection in video data.
- **License:** The model MCG-NJU/videomae-base-finetuned-kinetics is released under the CC-BY-NC 4.0 license. The license details can be found in the [LICENSE](https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE) file. Portions of the project that use other components, such as [SlowFast](https://github.com/facebookresearch/SlowFast) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), are licensed under the Apache 2.0 license, and [BEiT](https://github.com/microsoft/unilm/tree/master/beit) is licensed under the MIT license.
- **Finetuned from model:** The model MCG-NJU/videomae-base-finetuned-kinetics is fine-tuned from the base model "VideoMAE pre-trained ViT-B". Unfortunately, a direct link to the base model is not provided in the references above, so I cannot give you the exact link. However, you can typically find the pre-trained models in the model zoo documentation or repository mentioned in reference 6. Therefore, for the exact link, you would need to refer to the [MODEL_ZOO.md](MODEL_ZOO.md) file provided by the authors.
### Model Sources

- **Repository:** https://github.com/MCG-NJU/VideoMAE
- **Paper:** https://arxiv.org/pdf/2203.12602.pdf
- **Demo:** [More Information Needed]
## Uses

### Direct Use

The model MCG-NJU/videomae-base-finetuned-kinetics is designed to be used for video classification tasks, specifically pre-trained and fine-tuned on the Kinetics-400 dataset. Since the model has already been fine-tuned, it can be directly used for inference on video data without the need for additional fine-tuning, post-processing, or integration into a pipeline.

To use the model for inference, you would typically load the pre-trained model and pass a video through it to obtain the classification results. The model expects the input video to be pre-processed according to the specifications it was trained with, which includes sampling frames and possibly applying certain transformations to match the input format the model expects.

However, without specific code snippets provided in the references, I cannot give you an exact code example. If you want to use the model for inference directly from Huggingface, you would typically follow these steps:

1. Install the necessary libraries and dependencies as per [INSTALL.md](INSTALL.md).
2. Prepare your video data according to the specifications in [DATASET.md](DATASET.md).
3. Load the model from Huggingface.
4. Pre-process the video into the format expected by the model (e.g., frame sampling, resizing, normalization).
5. Pass the processed video through the model to obtain predictions.

Since the exact code snippet is not provided in the references, I would say [More Information Needed] for the code part. However, the general process described above is how you would use the model without additional fine-tuning or post-processing.

### Downstream Use

The model MCG-NJU/videomae-base-finetuned-kinetics is a deep learning model that has been pre-trained on the Kinetics-400 dataset and fine-tuned for video understanding tasks. It is based on the VideoMAE architecture, which is an extension of the Masked Autoencoder (MAE) approach for video data. This model can be used for various video analysis tasks such as action recognition, video classification, and other tasks that can benefit from understanding the content of video frames.

When fine-tuned for a specific task, the model can be adapted to the nuances of that task's dataset. For example, if you have a dataset of sports videos and you want to classify different types of sports, you can fine-tune this model on your dataset to learn the specific features of each sport.

To plug this model into a larger ecosystem or app, you would typically use it as a feature extractor or as an end-to-end model for video classification. The extracted features could be used for downstream tasks such as recommendation systems, content moderation, or interactive applications that require understanding of video content.

For fine-tuning or using the model, you would follow the instructions provided in the [FINETUNE.md](FINETUNE.md) file. Unfortunately, without direct access to the content of that file, I cannot provide a specific code snippet. However, the general process would involve loading the pre-trained model, preparing your dataset, and then training the model on your data with the appropriate hyperparameters as suggested in the documentation.

If you need to install the necessary dependencies or set up the environment, you would refer to the [INSTALL.md](INSTALL.md) file. Similarly, if you are interested in the pre-training process, you would look into the [PRETRAIN.md](PRETRAIN.md) file for instructions.

Since the exact code snippet for fine-tuning or integration is not provided in the references, I must say [More Information Needed] for the code part. However, the general usage pattern would involve leveraging Huggingface's `transformers` library to load the model and fine-tune it on your specific dataset.

### Out-of-Scope Use

The MCG-NJU/videomae-base-finetuned-kinetics model, as a powerful tool for video analysis, has the potential to be misused in ways that could have negative societal impacts. Users should be aware of the following guidelines to prevent misuse of the model:

1. Unauthorized Surveillance: Users should not use the model for unauthorized surveillance or monitoring activities. This includes any form of spying or eavesdropping on individuals without their consent, which is a violation of privacy rights and may be illegal in many jurisdictions.

2. Energy Consumption and Environmental Impact: While the pre-training phase of VideoMAE is energy-intensive and may lead to significant carbon emissions, users should not repeatedly pre-train the model for similar tasks where fine-tuning the existing pre-trained model would suffice. This practice helps to minimize the environmental impact.

3. Licensing and Ethical Use: The datasets used for training the model, such as Kinetics-400, have specific licenses (CC BY-NC 4.0) that restrict their use to non-commercial purposes. Users must adhere to these licenses and ensure that the model is not used for commercial exploitation without proper authorization.

4. Representation and Bias: Users should be cautious of potential biases in the model, as it may not capture motion information from very small objects effectively. This limitation should be considered when deploying the model in critical applications where such details are crucial.

5. Respect for Academic Use: The datasets employed are intended for academic purposes, and users should respect this context. The model should not be used in ways that contravene the spirit of academic research and collaboration.

6. Avoiding Harmful Applications: Users should not apply the model in contexts that could lead to harm, discrimination, or injustice. This includes any use that could contribute to human rights violations or exacerbate social inequalities.

In summary, users of the MCG-NJU/videomae-base-finetuned-kinetics model should ensure that their applications are ethical, legal, respect privacy, minimize environmental impact, adhere to licensing agreements, and do not perpetuate bias or discrimination. Any use case that falls outside these guidelines should be carefully reviewed and potentially avoided.

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the model MCG-NJU/videomae-base-finetuned-kinetics can be categorized into technical limitations and sociotechnical concerns:

Technical Limitations:
1. **Inadequate Capture of Small Motion**: The model may struggle to capture motion information from very small objects, as tokens containing small motion might be masked due to the high masking ratio, making it difficult for the model to reconstruct the masked small motion pattern (Reference 2).
2. **Energy Consumption**: The pre-training phase of VideoMAE is energy-consuming, which may lead to a significant carbon footprint. Although the model only needs to be pre-trained once, the initial environmental impact is a concern (Reference 3).
3. **Domain Shift**: When transferring pre-trained VideoMAE models to other video datasets, there is a performance drop compared to models directly pre-trained on the target datasets. This suggests that domain shift is an important issue to consider (Reference 10).
4. **Limited Temporal Modeling**: On datasets like Kinetics, which are mostly stationary and scene-related, the benefits of temporal modeling are not as pronounced, indicating a limitation in the model's ability to generalize across different types of video content (Reference 9).

Sociotechnical Concerns:
1. **Potential Misuse**: There is a risk of the model or its outputs being used for unauthorized surveillance or other unethical purposes, which raises privacy and civil liberties concerns (Reference 6).
2. **Data Licensing**: The datasets used for training and validation have custom licenses or are under CC BY-NC 4.0, which may restrict the commercial use of the model and its outputs (Reference 7).
3. **Negative Societal Impacts**: The high energy consumption of the pre-training phase could contribute to climate change, which is a significant societal concern (Reference 3).
4. **Lack of Multimodal Learning**: Currently, VideoMAE only leverages the RGB video stream without using additional audio or text streams, which could limit the model's understanding of context and reduce its effectiveness in certain applications (Reference 4).

Future work could address some of these limitations by incorporating larger and more diverse datasets, multimodal inputs, and optimizing the model to reduce its environmental impact. Additionally, ethical considerations and responsible use guidelines should be established to mitigate potential misuse of the technology.

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model MCG-NJU/videomae-base-finetuned-kinetics:

1. **Domain Shift**: The model may experience a performance drop when transferred to datasets that are significantly different from the Kinetics dataset it was trained on. To mitigate this, it is recommended to consider domain adaptation techniques when applying the model to new datasets or to fine-tune the model on a subset of the target dataset to better capture the domain-specific features.

2. **Data Quality Over Quantity**: The findings suggest that data quality is more important than data quantity in self-supervised video pre-training (SSVP) when there is a domain shift. Therefore, curating high-quality datasets that are more representative of the target domain should be prioritized over simply increasing the size of the training set.

3. **Temporal Modeling**: Since Kinetics videos are mostly stationary and scene-related, the effect of temporal modeling is not as pronounced. For datasets with more dynamic content, it may be beneficial to explore alternative temporal modeling techniques that can better capture motion and changes over time.

4. **Masking Ratio**: The model uses an extremely high masking ratio during pre-training. While this has shown to be effective, it is important to investigate the impact of different masking ratios on various datasets, as the optimal ratio may vary depending on the characteristics of the data.

5. **Sampling Methods**: The model uses dense sampling for Kinetics-400 and different sampling methods for other datasets. It is recommended to evaluate the impact of different sampling strategies on the model's performance for various types of video content and consider adapting the sampling method to the specific characteristics of the target dataset.

6. **Model Scaling**: Scaling up the pre-training configurations with larger video datasets or more powerful backbones has shown to improve performance. For applications requiring higher accuracy, consider using larger datasets for pre-training or employing more powerful model architectures.

7. **Inference Protocol**: The model uses a specific inference protocol (5 clips × 3 crops for Kinetics-400). It is recommended to ensure that the inference protocol is suitable for the target application and to consider adjusting it if necessary to optimize performance.

8. **Fine-Tuning**: Additional fine-tuning on the target dataset with labels can significantly improve transfer learning performance. It is recommended to fine-tune the model on the specific dataset it will be used for, especially if the dataset differs substantially from Kinetics.

In summary, when publishing the model to Huggingface, it is important to clearly communicate these recommendations and potential issues to users, so they can effectively apply and adapt the model to their specific use cases.

## Training Details

### Training Data

The training data for the model MCG-NJU/videomae-base-finetuned-kinetics consists of approximately 240k training videos from the Kinetics-400 dataset, which features 10-second clips across 400 different classes. The model was pre-trained for 800 epochs with a default masking ratio, and fine-tuning was performed using dense sampling following the Slowfast protocol. [More Information Needed] on data pre-processing and additional filtering.

### Training Procedure

#### Preprocessing

The preprocessing steps for the data used in the model MCG-NJU/videomae-base-finetuned-kinetics include the following:

1. Temporal Downsampling: As per the reference, we employ a strided temporal sampling strategy to reduce temporal redundancy. Specifically, we randomly sample a clip of `t` consecutive frames from the original video and then compress this clip to `T` frames using temporal sampling. The stride `τ` is set to 4 for the Kinetics dataset, as mentioned in reference 2.

2. Spatial Tokenization: The frames are tokenized using cube embedding to obtain video tokens. This is part of the customized design of our VideoMAE, which takes downsampled frames as inputs (reference 1).

3. Tube Masking: We apply a temporal tube masking mechanism to enforce a mask that expands over the entire temporal axis. This means that different frames share the same masking map. The mask is sampled from a Bernoulli distribution with a high masking ratio, which is typically between 90% to 95% for VideoMAE, as indicated in references 3 and 4.

4. Frame Resolution: Each frame in the compressed clip contains `H × W × 3` pixels, but the exact resolution is not specified in the provided references. [More Information Needed] for the specific resizing or resolution details.

5. Preprocessing for Fine-tuning: During fine-tuning on Kinetics-400, dense sampling is performed following the Slowfast protocol, and repeated augmentation is adopted. The model is trained with a batch size of 128 for 100 epochs, and the base learning rate, layer decay, and drop path are set to 5e-4, 0.7, and 0.2, respectively (reference 9).

6. Inference Protocol: For evaluation, all models share the same inference protocol, which is 5 clips × 3 crops for Kinetics-400 (reference 11).

The preprocessing steps are crucial for the model's performance as they directly impact the quality and efficiency of the training process. The high masking ratio and the temporal tube masking are particularly important for the model to learn to reconstruct the video data effectively, forcing it to understand high-level semantics and temporal dynamics within the video content.

#### Training Hyperparameters

The model MCG-NJU/videomae-base-finetuned-kinetics was pre-trained on the Kinetics-400 dataset for 800 epochs as per our default setting. During the fine-tuning phase, we adopted dense sampling following the Slowfast methodology. For evaluation, we used the protocol of 5 clips × 3 crops.

The fine-tuning hyperparameters for the Kinetics-400 dataset are as follows:
- We followed the default settings as shown in Table 9 and Table 12 of our documentation. [More Information Needed] for the specific values in these tables.
- For supervised training from scratch, the model was trained for 200 epochs, following the recipe in the referenced work [22].
- Repeated augmentation was used during the training, as mentioned in [32].

Unfortunately, the exact batch size, learning rate, layer decay, and drop path values used during the fine-tuning on Kinetics-400 are not specified in the provided references. Therefore, [More Information Needed] for these specific hyperparameters.

For the evaluation phase, we maintained a consistent approach across all models by using the 5 clips × 3 crops inference protocol.

It's important to note that the performance of VideoMAE can be further improved by fine-tuning on larger datasets or using more powerful backbones, as indicated by the improvements seen with ViT-L and ViT-H VideoMAE models pre-trained on Kinetics-700.

#### Speeds, Sizes, Times

The model MCG-NJU/videomae-base-finetuned-kinetics is a Video Masked Autoencoder (VideoMAE) that has been fine-tuned on the Kinetics-400 dataset. Below are the details regarding the model's training and evaluation:

- **Pre-training Details**: The model was pre-trained with a masking ratio of 75% for 800 epochs, as mentioned in reference 2. The batch size during pre-training was set to 192, and the base learning rate was 3e-4. The model samples 16 frames with a temporal stride of 4.

- **Fine-tuning Details**: For fine-tuning on Kinetics-400, the model was trained for 100 epochs with a batch size of 128. The base learning rate was set to 5e-4, with a layer decay of 0.7 and a drop path rate of 0.2. Dense sampling following the Slowfast methodology was used during fine-tuning, as per reference 2.

- **Evaluation Protocol**: The model adopts an inference protocol of 5 clips × 3 crops for evaluation on Kinetics-400, as stated in references 2 and 11.

- **Architectural Details**: The model uses an asymmetric encoder-decoder architecture with the encoder being a 16-frame vanilla ViT-Base. The decoder is discarded during the fine-tuning phase. Joint space-time attention is used to capture spatio-temporal information in the remaining tokens, as described in reference 5.

- **Throughput**: [More Information Needed]

- **Start or End Time of Training**: [More Information Needed]

- **Checkpoint Sizes**: [More Information Needed]

For more specific details such as throughput, start or end time of training, and checkpoint sizes, additional information would be required that is not provided in the references. These details are typically logged during the training process or can be found in the model's repository or associated documentation.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model MCG-NJU/videomae-base-finetuned-kinetics evaluates on the following benchmarks or datasets:

1. Kinetics-400: Contains around 240k training videos and 20k validation videos from 400 classes.
2. Something-Something V2: A large-scale video dataset with around 169k training videos and 20k validation videos across 174 motion-centric action classes.
3. UCF101: A smaller dataset with approximately 9.5k training videos and 3.5k validation videos.
4. HMDB51: Another small dataset with around 3.5k training videos and 1.5k validation videos.
5. AVA: A dataset for spatiotemporal localization of human actions with 211k training and 57k validation video segments.

#### Factors

The performance and behavior of the model MCG-NJU/videomae-base-finetuned-kinetics are influenced by several characteristics, as derived from the provided references:

1. **Domain and Context**: The model has been pre-trained and fine-tuned on the Kinetics-400 dataset, which consists of videos that are mostly stationary and scene-related. This suggests that the model may be better suited for recognizing actions in similar contexts where the background scenes are prominent and the actions do not involve significant camera movement. However, the model may not perform as well on datasets with a different domain, such as Something-Something V2, which contains motion-centric action classes and may require more temporal modeling.

2. **Data Quality vs. Quantity**: The model's performance indicates that data quality is more important than data quantity in self-supervised video pre-training (SSVP). This is evidenced by the model achieving better accuracy with only 42k pre-training videos compared to Kinetics pre-trained models with 240k videos. Therefore, the model's behavior is likely to be influenced by the quality of the data it is exposed to, and it may perform better with high-quality, relevant datasets even if they are smaller in size.

3. **Domain Shift**: There is an indication that domain shift between pre-training and target datasets could be an important issue. When transferring the pre-trained VideoMAE models to other video datasets, the results are slightly worse than models directly pre-trained on those target datasets. This suggests that the model may exhibit disparities in performance when applied to datasets with different characteristics from the Kinetics-400 dataset.

4. **Masking Ratio**: The model employs an extremely high masking ratio during pre-training (75%). This design choice is intended to enforce the network to capture more useful spatiotemporal structures. However, it is not clear how this high masking ratio might affect performance across different types of video content or population subgroups.

5. **Transfer Learning Performance**: The model shows an increase in transfer learning performance when fine-tuned on Kinetics-400 with labels. This implies that the model can benefit from additional supervised fine-tuning on specific datasets, which could be a factor in how it behaves when applied to new domains or contexts.

6. **Population Subgroups**: The references do not provide specific information on the model's performance across different population subgroups. Therefore, to understand disparities in performance across factors such as age, gender, or ethnicity, [More Information Needed] is required. Evaluation should be disaggregated across these factors to uncover any potential biases or limitations in the model's applicability to diverse populations.

In summary, the model's behavior is influenced by the domain and context of the training data, the quality of the data, potential domain shifts when applied to new datasets, and the high masking ratio used during pre-training. The impact on different population subgroups is not addressed in the provided references, and further evaluation would be necessary to assess this aspect.

#### Metrics

For the evaluation of the model MCG-NJU/videomae-base-finetuned-kinetics, the following metrics will be used:

1. Mean Average Precision (mAP) under IoU threshold of 0.5 for action detection, as mentioned in the context of transferring the learned VideoMAE on Kinetics-400 to the downstream action detection dataset AVA (Reference 1).

2. Accuracy, which is implied by the comparison of the number of pre-training videos and the resulting performance, indicating that accuracy is a metric of interest, especially when discussing the domain shift and the importance of data quality over quantity (Reference 2).

The references do not explicitly mention other specific metrics for evaluating tradeoffs between different errors. Therefore, based on the provided information, mAP and accuracy are the primary metrics used for evaluation. If there are other metrics used to evaluate tradeoffs between different types of errors, such as precision, recall, or F1 score, that information is not provided in the references given.

### Results

The evaluation results of the model MCG-NJU/videomae-base-finetuned-kinetics are as follows:

Factors:
1. Pre-training Dataset: The model is pre-trained on Kinetics-400 and Kinetics-700 datasets.
2. Pre-training Epochs: The model is pre-trained for 800 epochs on Kinetics-400.
3. Fine-tuning Dataset: The model is fine-tuned on Kinetics-400.
4. Masking Ratio: A masking ratio of 75% is used during pre-training.
5. Frame Sampling: 16 frames with a temporal stride of 4 are sampled.
6. Batch Size and Learning Rate: During pre-training, the batch size is 192 and the base learning rate is 3e-4. For fine-tuning, the batch size is 128 and the base learning rate is 5e-4.
7. Fine-tuning Epochs: The model is fine-tuned for 100 epochs.
8. Additional Configurations: Layer decay is set to 0.7 and drop path to 0.2 during fine-tuning.

Metrics:
1. Transfer Learning Performance: When the pre-trained ViT-B is fine-tuned on Kinetics-400 with labels, there is an increase of about 5 mAP (from 26.7 to 31.8).
2. Data Efficiency: VideoMAE shows to be a data-efficient learner, effectively training on limited video data without ImageNet pre-training and significantly outperforming training from scratch, MoCo v3 pre-training, and previous best performances on small-scale datasets like UCF101 and HMDB51.
3. Comparison with State-of-the-Art: On UCF101 and HMDB51, VideoMAE outperforms previous state-of-the-art methods without extra data.
4. Inference Protocol: For evaluation, a protocol of 5 clips × 3 crops is used.

The model demonstrates significant gains in most categories on SSV2, indicating its ability to capture more spatiotemporal structure representations. However, it performs slightly worse on some categories, particularly where motion information from very small objects is involved. The model's ability to capture the deformation of objects and movement from the squeeze of the hand is noted, which is not possible with image pre-training alone.

For more detailed analysis and results, including figures and tables, [More Information Needed].

#### Summary

The model MCG-NJU/videomae-base-finetuned-kinetics has been evaluated on several common video datasets, including Kinetics-400, Something-Something V2, UCF101, HMDB51, and AVA. The evaluation results highlight the model's ability to capture spatiotemporal structure representations effectively, outperforming ImageMAE and ImageNet-21k supervised pre-trained models in most categories on the Something-Something V2 dataset.

The model demonstrates significant data efficiency, as it can be effectively trained on limited video data, such as 9.5k clips from UCF101 and 3.5k clips from HMDB51, without relying on ImageNet pre-training. It significantly surpasses training from scratch, MoCo v3 pre-training, and previous best performances on these datasets.

However, the model does have limitations. For instance, it may struggle to capture motion information from very small objects, possibly due to the high masking ratio used during pre-training, which could lead to masking out tokens containing small motion patterns.

When the pre-trained ViT-B is fine-tuned on Kinetics-400 with labels, there is an observed increase in transfer learning performance by about 5 mAP. Scaling up the pre-training configurations with larger video datasets or more powerful backbones further improves performance. For example, a ViT-L VideoMAE pre-trained on Kinetics-700 achieves 39.3 mAP.

The pre-training of VideoMAE involves a masking ratio of 75% for 3200 epochs, with a batch size of 192 and a base learning rate of 3e-4. During fine-tuning, the model is trained with repeated augmentation and a batch size of 128 for 100 epochs, with a base learning rate of 5e-4, layer decay of 0.7, and drop path of 0.2.

In summary, the MCG-NJU/videomae-base-finetuned-kinetics model shows promising results in terms of data efficiency and performance on various video datasets, with potential areas for improvement in capturing motion from small objects.

## Model Examination

In the experimental section dedicated to explainability and interpretability of the model MCG-NJU/videomae-base-finetuned-kinetics, we would discuss the following points based on the provided references:

1. **Understanding Limitations in Small Object Motion Capture**: As noted in the first reference, our VideoMAE model may struggle with capturing motion information from very small objects. This could be due to the high masking ratio used during pre-training, which might result in the masking of all tokens that contain small motion patterns. This limitation is important to acknowledge as it can affect the model's performance in scenarios where fine-grained motion details are crucial.

2. **Spatiotemporal Structure Representation**: According to the second reference, our VideoMAE has shown significant gains in most categories on the Something-Something V2 dataset, indicating its ability to capture more spatiotemporal structure representations compared to ImageMAE and ImageNet-21k supervised pre-trained models. However, there are some categories where VideoMAE slightly underperforms, which warrants further investigation.

3. **Reconstruction Capability**: The third reference highlights the model's reconstruction capability, even with a high masking ratio. The reconstructed results from the validation set suggest that VideoMAE can learn representative features that capture the holistic spatiotemporal structure in videos, as mentioned in the fourth reference.

4. **Ablation Studies and Sampling Techniques**: The sixth reference details the ablation studies conducted with the default backbone and the sampling techniques used for fine-tuning on different datasets. This information is crucial for understanding the design decisions and their impact on the model's performance.

5. **Transfer Learning and Generalization**: The seventh reference discusses the generalization ability of VideoMAE, where it outperforms models pre-trained with MoCo v3. This indicates that VideoMAE learns more transferable representations, which is a positive aspect for its application to various video understanding tasks.

6. **Pre-training and Fine-tuning Protocols**: The eighth reference provides insights into the default settings for pre-training and fine-tuning on Kinetics-400, including the number of epochs and the sampling strategy used during these phases. This information is essential for replicating the training process or understanding the model's learning trajectory.

In summary, the experimental section on explainability and interpretability would focus on the model's limitations, its ability to capture spatiotemporal structures, reconstruction capabilities, design choices, and its generalization performance. Further detailed analysis and exploration of these aspects are suggested for future work to enhance our understanding of the VideoMAE model's behavior and improve its interpretability.

## Environmental Impact

- **Hardware Type:** [More Information Needed]
- **Software Type:** The model MCG-NJU/videomae-base-finetuned-kinetics is trained on the Kinetics-400 dataset.
- **Hours used:** The model card description for the model MCG-NJU/videomae-base-finetuned-kinetics should include the following information based on the references provided:

```
The MCG-NJU/videomae-base-finetuned-kinetics model underwent an extensive training regimen. Initially, the VideoMAE architecture was pre-trained for 800 epochs on the Kinetics-400 dataset, as mentioned in reference 4. This pre-training phase is crucial for the model to learn rich representations from the video data.

Following the pre-training, the model was fine-tuned on the same Kinetics-400 dataset. The fine-tuning process involved dense sampling as per the Slowfast methodology, and the model was trained for 200 epochs, as indicated in reference 4. During this phase, repeated augmentation techniques were employed to enhance the model's generalization capabilities.

For evaluation purposes, the model adheres to a standardized inference protocol of 5 clips × 3 crops, ensuring consistency and reliability in performance assessment across different datasets.

It is important to note that the specific amount of time taken to train the model is not directly provided in the references. The duration of training would depend on various factors, including the computational resources used (such as the type and number of GPUs), the implementation details, and any optimizations that may have been applied during the training process.

For more detailed information on the training duration, one would need to refer to the actual training logs or additional documentation that provides insights into the computational efficiency and time metrics for the training process.
```

Since the exact amount of time used to train the model is not specified in the provided references, the answer to the question is "[More Information Needed]".
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model MCG-NJU/videomae-base-finetuned-kinetics utilizes an asymmetric encoder-decoder architecture for video self-supervised pre-training, with the decoder being discarded during the fine-tuning phase. The architecture is based on a 16-frame vanilla Vision Transformer (ViT) Base model. The encoder and decoder's specific architectural design is detailed in the referenced Table 8, which is not provided here, so [More Information Needed] for the exact specifications.

The objective of the model during pre-training is to reconstruct the pixels of masked tokens, following a simple masking and reconstruction strategy inspired by ImageMAE but tailored for self-supervised video pre-training (SSVP). The model employs a high masking ratio of 75% to leverage the temporally redundant video content, allowing for effective learning even with a significant portion of the input data being masked.

For fine-tuning on the Kinetics-400 dataset, the model is trained with repeated augmentation and a batch size of 128 for 100 epochs. The base learning rate is set to 5e-4, with a layer decay of 0.7 and a drop path rate of 0.2. The fine-tuning process does not involve the decoder, focusing on the encoder to learn effective video representations.

In summary, the MCG-NJU/videomae-base-finetuned-kinetics model is a video masked autoencoder with a plain ViT backbone, designed for self-supervised pre-training and fine-tuned for video understanding tasks on the Kinetics-400 dataset.

### Compute Infrastructure

The provided references do not contain explicit information about the compute infrastructure used for the model MCG-NJU/videomae-base-finetuned-kinetics. Details such as the type of GPUs, the number of GPUs, the amount of RAM, or the specific hardware configurations are not mentioned in the excerpts provided. Therefore, to answer the question regarding the compute infrastructure used for the model, I would have to say:

[More Information Needed]

## Citation

```
@misc{zhan-videomae,
    author = {Zhan Tong and
              Yibing Song and
              Jue Wang and
              Limin Wang},
    title  = {VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
    url    = {https://arxiv.org/pdf/2203.12602.pdf}
}
```

