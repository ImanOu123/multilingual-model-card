# Model Card for ZinengTang/tvlt-base

The ZinengTang/tvlt-base model is a Textless Vision-Language Transformer (TVLT) that learns vision-and-language representations directly from raw video frames and audio spectrograms, without relying on text-based inputs or modules like tokenization or automatic speech recognition (ASR). TVLT is trained using masked autoencoding and contrastive modeling to align video and audio modalities.

## Model Details

### Model Description

Model Card for ZinengTang/tvlt-base:

Model Details:
- Architecture: TVLT (Textless Vision-Language Transformer), a minimal end-to-end vision-and-language transformer model. It consists of a 12-layer encoder (hidden size 768) and an 8-layer decoder (hidden size 512). (Reference 1, 2)
- Input Embeddings: The input embeddings are the sum of (1) modality embedding, (2) temporal/spatial embedding for video, (3) temporal/frequency embedding for audio, and (4) vision/audio patch embedding. (Reference 3)
- Audio Embeddings: Audio waveforms are converted to 128-dimensional log Mel-spectrograms, divided into patches, and projected to 768-dimensional patch embeddings. Experiments were done with 16x16 and 2x128 patch sizes. (Reference 4, 5)

Training:
- Pretrained Objectives: (1) vision-audio matching, (2) masked autoencoding (Reference 6)
- Pretrained Datasets: HowTo100M and YTtemporal videos (Reference 10)
- [More Information Needed] on training hyperparameters, compute, etc.

Evaluation:
- The pretrained model is finetuned and evaluated on downstream vision-and-language datasets. (Reference 7)
- [More Information Needed] on specific evaluation datasets and metrics.

Intended Use:
- The model can be used to directly process video (with audio) inputs for tasks like audio-image/video retrieval, audio-VQA, TTS-based VQA, or to extract visual-acoustic features for speech translation, multimodal content understanding, etc. (Reference 10)

Limitations and Bias:
- [More Information Needed]

Environmental Impact: 
- To reduce environmental damage, the code and pretrained checkpoint will be publicly released. (Reference 11)

Important Disclaimers:
- The TVLT framework is language-agnostic and can be adapted to non-English datasets without architectural changes. (Reference 11)
- Pretraining TVLT on larger-scale video datasets is recommended to unlock its full potential for more generic Vision-Language representation. (Reference 10)

Contact: 
- [More Information Needed]

- **Developed by:** Zineng Tang; Jaemin Cho; Yixin Nie; Mohit Bansal
- **Funded by:** According to the references provided, the TVLT project was supported by the following funding sources:

- ARO Award W911NF2110220
- DARPA KAIROS Grant FA8750-19-2-1004
- ONR Grant N000141812871
- NSF-AI Engage Institute DRL-211263

The references also state: "The views, opinions, and/or findings contained in this article are those of the authors and not of the funding agency."
- **Shared by:** Based on the provided references, the main contributors who made the model ZinengTang/tvlt-base available online are:

Zineng Tang, Jaemin Cho, Yixin Nie, and Mohit Bansal

The model weights are hosted on the Huggingface Hub. The codebase is based on the open-source ViLT repository on GitHub.

For additional details about code and GitHub repo, [More Information Needed].
- **Model type:** The ZinengTang/tvlt-base model is a multimodal vision-and-language transformer that is pretrained on video and audio data using self-supervised learning objectives of vision-audio matching and masked autoencoding.
- **Language(s):** The model ZinengTang/tvlt-base does not rely on written language or explicit modeling of text input, and instead learns visual-linguistic representations directly from visual and acoustic input at the perception level.
- **License:** Based on the references provided, the model ZinengTang/tvlt-base uses standard licenses from the community for the datasets, code, and models used in the project. However, the specific name and link to the license being used for this particular model is not directly mentioned.

[More Information Needed]

To properly answer this question, more details would be needed on the exact license that was chosen for the ZinengTang/tvlt-base model when preparing to publish it on Hugging Face. The references indicate an intent to release the code and models publicly using standard community licenses, but do not specify which license was selected for this model.
- **Finetuned from model:** Based on the provided references, the model ZinengTang/tvlt-base is not explicitly mentioned as being fine-tuned from another model. The references introduce the TVLT (Textless Vision-Language Transformer) architecture, but do not specify a particular base model that ZinengTang/tvlt-base is fine-tuned from.

[More Information Needed] on whether ZinengTang/tvlt-base is fine-tuned from another model, and if so, the name and link to that base model.
### Model Sources

- **Repository:** https://github.com/zinengtang/TVLT
- **Paper:** https://arxiv.org/pdf/2209.14156.pdf
- **Demo:** Based on the provided references, the demo link for the ZinengTang/tvlt-base model is:

[Emotional Analysis on Video and Audio](Demo_Emotional_Analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zinengtang/TVLT/blob/main/Demo_Emotional_Analysis.ipynb)

This demo notebook shows how to perform emotional analysis using the TVLT model on video and audio inputs.
## Uses

### Direct Use

The ZinengTang/tvlt-base model is a pre-trained TVLT model that can be used for vision-and-language tasks without requiring fine-tuning, post-processing, or plugging into a pipeline. The model takes video and audio as input and outputs aligned representations.

To use the pre-trained model directly, you can load the model weights hosted on the Huggingface Hub using the following code snippet:

```
load_hub_path="TVLT.ckpt"
```

This command will automatically download the pre-trained model for use in training scripts.

[More Information Needed] on the specific API calls or functions to use the loaded model for inference or evaluation on downstream tasks without fine-tuning.

Note that the pre-trained ZinengTang/tvlt-base model was trained on mostly English datasets (HowTo100M and YTTemporal180M), so it may not perform as well on non-English tasks without additional pre-training.

### Downstream Use

The ZinengTang/tvlt-base model can be used for various video-based and image-based vision-and-language tasks when fine-tuned or plugged into a larger ecosystem or app. Some potential use cases include:

1. Audio-to-Video Retrieval: The model can be fine-tuned on datasets like MSR-VTT, Youcook2, and CrossTask for audio-to-video retrieval tasks (Reference 1).

2. Multimodal Sentiment Analysis: The model can be fine-tuned for multimodal sentiment analysis on videos (Reference 2). Here's a code snippet for fine-tuning the model on the MOSEI dataset:

```bash
bash scripts/finetune_mosei.sh
```
(Reference 5)

3. Visual Question Answering (VQA): The model can be used for image-based VQA tasks by obtaining audio queries via text-to-speech (TTS) synthesis (Reference 2).

4. Extracting Visual-Acoustic Features: The pre-trained model can be used to extract visual-acoustic features for tasks like speech translation and multimodal content understanding (Reference 6).

5. Masked Autoencoding on Video Frames and Audio Spectrogram: The model can be used for masked autoencoding tasks on video frames and audio spectrograms. A demo notebook is available on Colab (Reference 7).

6. Emotional Analysis on Video and Audio: The model can be used for emotional analysis tasks on videos and audio. A demo notebook is available on Colab (Reference 8).

To unlock the full potential of the ZinengTang/tvlt-base model, it is recommended to pre-train it on large-scale video datasets for more generic vision-language representation (Reference 6).

### Out-of-Scope Use

Based on the provided references about the ZinengTang/tvlt-base model, here are some potential misuses and what users ought not do with the model:

1. The model is primarily trained on English datasets (HowTo100M and YTTemporal180M), so it may not perform well on non-English tasks without additional pretraining. Users should not assume the model will work effectively for languages other than English without further adaptation.

2. [More Information Needed] regarding specific harmful or unethical uses of the model, such as generating misleading or biased content.

3. While the model aims to reduce computational redundancy and environmental damage compared to other vision-and-language models, it still relies on large-scale pretraining. Users should be mindful of the computational resources and energy consumption required when using or fine-tuning the model.

4. [More Information Needed] about potential privacy concerns related to the training data or the model's ability to capture sensitive information from input data.

5. Users should adhere to the licenses provided for the datasets, code, and models used in the project. Misusing the model in violation of these licenses is not permitted.

6. [More Information Needed] regarding any specific limitations or biases of the model that users should be aware of to prevent misuse or misinterpretation of results.

In general, users should use the ZinengTang/tvlt-base model responsibly, be transparent about its limitations, and refrain from applying it to tasks or domains where it may produce harmful or misleading outputs without proper evaluation and adaptation.

### Bias, Risks, and Limitations

Based on the provided references, here are some known or foreseeable issues with the ZinengTang/tvlt-base model:

Limitations:
- The model is pretrained on mostly English datasets (HowTo100M and YTTemporal180M), so performance may be limited on non-English tasks without additional pretraining. [Reference 4]
- There is still a performance gap compared to text-based models using higher quality ASR or ground truth transcripts as input. Larger-scale pretraining on raw video signals may help close this gap. [Reference 6]

Potential Negative Impacts:
- Pretraining the model requires substantial computational resources, which can contribute to environmental damage. Publicly releasing the code and pretrained checkpoint can help mitigate this. [Reference 1]
- [More Information Needed] on other potential negative societal impacts.

Technical Limitations: 
- [More Information Needed] on specific technical limitations beyond the language and performance aspects noted above.

Sociotechnical Limitations:
- [More Information Needed] on potential sociotechnical limitations or issues that could arise from the model's use and interaction with society.

The references highlight the model's efficiency gains by eliminating the need for external ASR modules [Reference 1] and potential to inspire further research into simple, efficient vision-and-language frameworks [Reference 2]. However, more information would be needed to fully assess limitations and long-term societal implications.

### Recommendations

Based on the provided references about the TVLT model (ZinengTang/tvlt-base), here are some recommendations regarding foreseeable issues:

1. Language bias: The model was pretrained on datasets (HowTo100M and YTTemporal180M) that mostly contain English content. Therefore, the model's performance on non-English tasks might be limited without additional pretraining on more diverse language data. (Reference 4)

2. Computational resources: While TVLT is more efficient than models with explicit text-based modules, it still relies on pretraining to achieve decent performance. Further research on energy-efficient training methods could help reduce the environmental impact and computational requirements. (Reference 2, 3)

3. Representation quality: Although TVLT shows promising results, there is still a performance gap compared to text-based models using high-quality ASR or ground truth transcripts. Larger-scale pretraining on raw video signals could potentially improve the model's representation quality. (Reference 7)

4. Broader applicability: To unlock the full potential of TVLT, it is recommended to pretrain the model on large-scale, diverse video datasets. This would enable the model to be used for a wider range of tasks, such as audio-image/video retrieval, audio-VQA, TTS-based VQA, speech translation, and multimodal content understanding. (Reference 6, 9)

5. [More Information Needed] on the specific pretraining datasets used, their licenses, and any potential biases or limitations they may introduce to the model.

6. [More Information Needed] on the model's performance across different demographics, accents, and languages to assess its fairness and inclusivity.

## Training Details

### Training Data

The model ZinengTang/tvlt-base is pretrained on a combination of the HowTo100M dataset (0.92M videos) and a 20% subset of the YTTemporal180M dataset called YTT-S (0.93M videos), resulting in a total pretraining dataset size of 1.85M videos. Both datasets mostly contain English content, as HowTo100M videos are obtained from English queries and YTTemporal180M filtered out videos with non-English ASR results.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about the preprocessing for the data of the model ZinengTang/tvlt-base:

Video Preprocessing:
- Randomly sample a 15s video clip from each video to form a setting similar to the HowTo100M dataset (Reference 2)
- Extract an image frame from the middle timestep of each video segment (Reference 4)
- [More Information Needed] on the specific resizing or normalization applied to the video frames

Audio Preprocessing:
- Convert the 1D waveform of the raw audio signal to 128-dimensional log Mel-spectrogram having a dimension of T × 128 (time axis × frequency axis) (Reference 5)
- Treat the audio spectrogram as an image and divide it into patches (Reference 5)
- Apply a linear projection layer on each patch to obtain a 768-dimensional patch embedding (Reference 5)

Text Preprocessing:
- The model does not involve explicit modeling of text input, such as automatic speech recognition (ASR) or tokenization (Reference 8)
- [More Information Needed] on any potential text preprocessing if used

Embedding:
- The input embeddings of TVLT are the sum of (1) modality embedding, (2) temporal/spatial embedding for video, (3) temporal/frequency embedding for audio, and (4) vision/audio patch embedding (Reference 3)
- Temporal embedding is only added for video (Reference 3)

Dataset:
- The model is pretrained on HowTo100M dataset containing 136M video clips from 1.22M YouTube videos (Reference 1)
- 0.92M videos from HowTo100M were used for pretraining (Reference 1)
- The model can be further improved with larger-scale pretraining, e.g., on the full YTTemporal180M dataset (Reference 7)

#### Training Hyperparameters

Here are the training hyperparameters for the ZinengTang/tvlt-base model based on the provided references:

Pretraining:
- Optimizer: Adam 
- Learning rate: 1e-5
- Batch size: 4096
- Decay rate: 0.001 with cosine schedule
- Number of training steps: 200k
- Initialization: Weights initialized from masked autoencoder transformer pretrained on ImageNet
- Pretraining objectives: λ VAM = 1.0, λ MAE = 0.3
- Video frames: 8 uniformly sampled frames per clip
- Training time: 2 weeks on 4 NVIDIA GPUs
- Pretraining datasets: HowTo100M (used 0.92M videos)

Finetuning:
- Learning rate: 1e-5 
- Batch size: 256
- Decay rate: 0.001 with cosine schedule
- Video frames: 8 uniformly sampled frames per clip
- Hardware: 2 NVIDIA RTX A6000 GPUs

[More Information Needed] on the specific downstream tasks the model was finetuned on and their hyperparameters.

#### Speeds, Sizes, Times

Here are the details about the model ZinengTang/tvlt-base based on the provided references:

Throughput and Latency:
- TVLT is more than 27x and 28x faster than text-based TVLT for inference with video input lengths of 10s and 20s, respectively. (Reference 2)
- ASR dominates the inference time for text-based models. Although ASR helps reduce input length in transformers, it is still the bottleneck. (Reference 2, 3)
- [More Information Needed] on specific throughput numbers.

Model Size: 
- TVLT has only 1/3 of the parameters compared to the text-based counterpart. (Reference 2)
- [More Information Needed] on the exact model size.

Training Details:
- TVLT is pretrained on video datasets and then finetuned on downstream vision-and-language datasets for evaluation. (Reference 6) 
- [More Information Needed] on the specific datasets, training duration, hyperparameters etc.

Release Details:
- The code and pretrained checkpoint will be publicly released to reduce environmental damage. (Reference 9)
- [More Information Needed] on the exact release date.

Performance:
- TVLT outperforms other text-to-video retrieval models like HERO and DeCEMBERT, showing promising possibilities of speech-based video retrieval. (Reference 1)
- TVLT achieves competitive results with AVLnet and MILAN on audio-to-image retrieval (Places-400k dataset). (Reference 5)
- TVLT achieves comparable performance to the text-based counterpart on various downstream tasks. (Reference 8)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the ZinengTang/tvlt-base model evaluates on the following benchmarks and datasets:

Video-based tasks:
1. Video retrieval:
   - MSR-VTT [82]
   - Youcook2 [91] 
   - CrossTask [93]
2. Multimodal sentiment analysis:
   - [More Information Needed]

Image-based tasks:
1. Image retrieval:
   - [84] (dataset name not provided)
2. Visual question answering (VQA):
   - [4] (likely referring to VQAv2)
   - [21] (dataset name not provided)

The model is first pretrained on the combination of HowTo100M and YTT-S datasets, with a total size of 1.85M videos. It is then finetuned and evaluated on the downstream tasks mentioned above.

#### Factors

Based on the provided references, here are some foreseeable characteristics that may influence how the ZinengTang/tvlt-base model behaves:

Domain and Context:
- The model is pretrained on the HowTo100M and YTTemporal180M datasets, which contain mostly English instructional videos, lifestyle vlogs, and auto-suggested YouTube videos on topics like science and home improvement (References 4, 5, 6). This suggests the model may perform better on tasks related to these domains and contexts.
- The model is language-agnostic and can be adapted to non-English datasets without architectural changes (Reference 1). However, since the pretraining datasets are mostly in English, the model might not perform as well on non-English tasks without additional pretraining (Reference 2).

Population Subgroups:
- [More Information Needed] The provided references do not contain specific information about the model's performance across different population subgroups.

Evaluation and Disaggregation:
- [More Information Needed] The references do not provide details on how the model's evaluation is disaggregated across different factors to uncover performance disparities.

Other Factors:
- The model's performance may be influenced by the audio masking strategy, encoder/decoder architectures, and pretraining objectives used during training (Reference 8).
- The model uses a joint encoder for visual and audio modalities, which may affect its performance compared to modality-specific encoders (Reference 7).
- The model's reliance on pretraining to achieve decent performance on visual-linguistic tasks may have environmental implications, although the model's architecture helps reduce pretraining computation compared to vision-and-language models with explicit text-based modules (Reference 3).

#### Metrics

Based on the references provided, the following metrics are likely used to evaluate the ZinengTang/tvlt-base model, considering tradeoffs between different errors:

For audio-to-video retrieval tasks on MSR-VTT, Youcook2, and CrossTask datasets, standard retrieval metrics like recall@k would be used to measure performance and compare against baselines like AVLnet.

For video-based multimodal sentiment analysis on the CMU-MOSEI dataset, accuracy and F1 scores are likely the key metrics to assess performance on the sentiment classification task.

For image-based retrieval on the Places-400k dataset, recall@k metrics would again be employed to evaluate the model in comparison to prior work like MILAN.

For visual question answering on datasets like VQAv2, standard VQA accuracy metrics are expected to be reported.

[More Information Needed] on the specific tradeoffs and error analysis conducted between these different tasks and metrics when developing and evaluating the tvlt-base model. The references do not go into detail on prioritizing different types of errors.

In summary, the key evaluation revolves around retrieval metrics for the video/image-to-text tasks, and classification metrics like accuracy and F1 for the sentiment analysis and VQA tasks. But more details would be needed to comment on the tradeoffs and error analysis between the different downstream tasks the model was evaluated on.

### Results

Based on the provided references, here are the evaluation results for the ZinengTang/tvlt-base model:

Performance:
- Audio-to-Video Retrieval: TVLT outperforms the text-based counterpart when pretrained on HowTo100M or YTT-S datasets (Reference 1, 3).
- Multimodal Sentiment Analysis (CMU-MOSEI): TVLT outperforms its text variant when pretrained on YTT-S (Reference 1).
- Audio-to-Image Retrieval (Places-400k): TVLT achieves competitive results with AVLnet and MILAN (Reference 7).
- Visual Question Answering: TVLT slightly underperforms the text-based counterpart but remains competitive (Reference 2, 5).

Efficiency:
- Inference Latency: TVLT is 27x faster during inference compared to text-based counterparts due to the removal of ASR from the processing pipeline (Reference 2).

Environmental Impact:
- The code and pretrained checkpoint will be publicly released to reduce environmental damage (Reference 6).

[More Information Needed] for specific metrics and scores, as they are not directly provided in the given references.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the ZinengTang/tvlt-base model:

- TVLT outperforms its text-based counterpart in audio-to-video retrieval tasks when pretrained on HowTo100M or YTT-S datasets. It also outperforms the text variant on CMU-MOSEI sentiment analysis when pretrained on YTT-S. (Reference 1)

- Although TVLT slightly underperforms the text-based counterpart on audio-to-image retrieval and visual question answering, it achieves comparable results while being 27x faster during inference due to the removal of ASR from the processing pipeline. (Reference 2)

- TVLT outperforms other models that take raw visual and audio signals as input, such as AVLnet on three audio-to-video retrieval tasks (MSR-VTT, Youcook2, CrossTask) and Multilogue-Net on multimodal tasks. (Reference 10)

- [More Information Needed] on specific performance metrics and scores for the mentioned tasks and datasets.

- The TVLT framework is language-agnostic and can be adapted to non-English datasets without architectural changes. It eliminates the need for external ASR modules, reducing the computation of the typical vision-and-language pipeline. (Reference 8)

## Model Examination

Here is my attempt at an experimental section on explainability/interpretability for the ZinengTang/tvlt-base model:

The TVLT framework used in the ZinengTang/tvlt-base model is a language-agnostic method that learns cross-modal representations directly from visual and acoustic inputs, without relying on text or modality-specific encoders [6, 7]. This makes the model more compact and efficient compared to typical vision-and-language models that use external ASR modules [1, 6].

[More Information Needed] on specific techniques used to analyze or visualize what the TVLT model has learned and how it makes predictions. Potential areas to explore interpretability include:

- Analyzing the learned visual and acoustic representations and alignments 
- Visualizing neuron activations on different inputs
- Applying attribution methods to identify important visual/acoustic features
- Probing the model's reasoning capabilities through carefully designed inputs

Publishing the code and pretrained checkpoint [1] will enable the community to further study the model's behaviors and develop explainability methods. Overall, the model's simplicity and efficiency provide a promising foundation for future interpretability work in the textless vision-and-language domain. However, more focused experiments are needed to shed light on this particular model's inner workings and decision making process.

## Environmental Impact

- **Hardware Type:** Based on the references provided, the TVLT model was trained on the following hardware:

For pretraining:
- 4 NVIDIA RTX A6000 GPUs (each 49GB memory) for 2 weeks

For finetuning on downstream tasks:
- 2 NVIDIA RTX A6000 GPUs
- **Software Type:** Based on the provided references, the model ZinengTang/tvlt-base was trained using the following software:

* PyTorch: Versions 1.10.0 and 1.12.1 have been tested
* Torchvision: Versions 0.11.1 and 0.12.1 have been tested
* Torchaudio: Versions 0.10.0 and 0.13.1 have been tested

The exact versions used for training are not specified. Other PyTorch versions may work if compatible with the CUDA and cuDNN versions.

Additional dependencies can be installed via:
```
pip install -r requirements.txt
```

However, the full requirements.txt file is not provided in the references. [More Information Needed] on the complete list of software dependencies and versions used for training the model.
- **Hours used:** According to the references provided, the TVLT model was pretrained for 200k steps, which took 2 weeks with 4 NVIDIA GPUs (Reference 1). However, the specific GPU model used for pretraining is not mentioned.

Therefore, the answer to the question "Provide the amount of time used to train the model ZinengTang/tvlt-base" is:

The TVLT model was pretrained for 200k steps, which took 2 weeks using 4 NVIDIA GPUs. [More Information Needed] on the specific GPU model used for pretraining.
- **Cloud Provider:** The model ZinengTang/tvlt-base is trained on 4 NVIDIA GPUs, as mentioned in the References:

"Pretraining takes 2 weeks with 4 NVIDIA"

However, the specific cloud provider is not mentioned in the given references. [More Information Needed] on the exact cloud provider used for training this model.
- **Carbon Emitted:** [More Information Needed]

Based on the provided references, there is no specific information about the amount of carbon emitted during the training of the ZinengTang/tvlt-base model. The references mention that pretraining took 2 weeks with 4 NVIDIA GPUs, but do not provide details on the specific GPU model or the energy consumption. To accurately estimate the carbon emissions, more details would be needed about the hardware, energy usage, and the carbon intensity of the electricity grid where the training was performed.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The main architecture of TVLT is a transformer consisting of a 12-layer encoder (hidden size 768) and an 8-layer decoder (hidden size 512). The decoder is shallow and only serves for the masked autoencoding objective. After pretraining, only the encoder representation is used for finetuning on downstream tasks.

The input embeddings of TVLT are the sum of (1) modality embedding, (2) temporal/spatial embedding for video, (3) temporal/frequency embedding for audio, and (4) vision/audio patch embedding. Temporal embedding is only added for video.

For audio embeddings, the 1D waveform is converted to a 128-dimensional log Mel-spectrogram. The spectrogram is treated as an image, divided into patches, and a linear projection layer is applied on each patch to obtain a 768-dimensional patch embedding.

Model Objective:
TVLT is pretrained with two objectives: masked autoencoding and contrastive modeling to align video and audio. TVLT makes no assumptions about the existence of written language and does not involve explicit modeling of text input.

[More Information Needed] on the specific pretraining datasets and hyperparameters used for the ZinengTang/tvlt-base model.

### Compute Infrastructure

Based on the provided references, here is the compute infrastructure information for the ZinengTang/tvlt-base model:

Pretraining of TVLT and its text-based counterpart was done for 200k steps using:
- Adam optimizer with learning rate 1e-5
- Batch size 4096
- Decay rate 0.001 with cosine schedule
- Weights initialized from a masked autoencoder transformer pretrained on ImageNet
- Pretraining objectives: λ VAM = 1.0 and λ MAE = 0.3
- 8 frames uniformly sampled from each video clip
- Pretraining took 2 weeks on 4 NVIDIA [More Information Needed] GPUs

The main TVLT architecture consists of:
- A 12-layer transformer encoder (hidden size 768) 
- An 8-layer transformer decoder (hidden size 512)
- The shallow decoder is only used for the masked autoencoding objective during pretraining
- Only the encoder representation is used for finetuning on downstream tasks after pretraining

[More Information Needed] on the specific GPU models used and other compute infrastructure details.

## Citation

```
@misc{zineng-tvlt,
    author = {Zineng Tang and
              Jaemin Cho and
              Yixin Nie and
              Mohit Bansal},
    title  = {TVLT: Textless Vision-Language Transformer},
    url    = {https://arxiv.org/pdf/2209.14156.pdf}
}
```

