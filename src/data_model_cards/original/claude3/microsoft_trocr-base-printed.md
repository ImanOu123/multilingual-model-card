# Model Card for microsoft/trocr-base-printed

The microsoft/trocr-base-printed model is an end-to-end Transformer-based OCR model for printed text recognition. It leverages pre-trained image and text Transformers to achieve state-of-the-art results on printed text recognition tasks without relying on complex pre/post-processing steps.

## Model Details

### Model Description

Model Architecture:
TrOCR is built with the Transformer encoder-decoder architecture. The encoder is an image Transformer that extracts visual features from image patches. The decoder is a text Transformer for language modeling that generates the wordpiece sequence guided by the visual features and previous predictions.

Training Procedures:
1. Pre-training Phase:
   - Stage 1: Pre-train TrOCR on a large-scale synthetic dataset of hundreds of millions of printed textline images.
   - Stage 2: Further pre-train separate models on task-specific datasets (printed and handwritten) containing millions of textline images each, initialized by the Stage 1 model.
2. Fine-tuning: Pre-trained TrOCR models are fine-tuned on downstream text recognition tasks, except for scene text recognition experiments.

Parameters:
[More Information Needed]

Important Disclaimers:
- The outputs of TrOCR models are based on Byte Pair Encoding (BPE) and SentencePiece, and do not rely on any task-related vocabularies.
- TrOCR achieves state-of-the-art results on printed, handwritten, and scene text image datasets without complex pre/post-processing steps.
- TrOCR can be easily extended for multilingual text recognition by leveraging multilingual pre-trained models in the decoder-side and expanding the dictionary.

For more information and access to the TrOCR models and code, please visit https://aka.ms/trocr.

- **Developed by:** Minghao Li; Tengchao Lv; Jingye Chen; Lei Cui; Yijuan Lu; Dinei Florencio; Cha Zhang; Zhoujun Li; Furu Wei
- **Funded by:** Based on the provided references, it appears that the TrOCR model, including microsoft/trocr-base-printed, is a project from Microsoft. The model and code are publicly available, and the contact persons listed for communications related to TrOCR are Lei Cui and Furu Wei, both with Microsoft email addresses.

However, there is no explicit mention in the given references of what specific people or organizations funded the development of the microsoft/trocr-base-printed model.

[More Information Needed]
- **Shared by:** Based on the provided references, the main contributors who made the TrOCR model available are:

Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, and Furu Wei

For help or issues using TrOCR, the references indicate to contact Lei Cui (lecu@microsoft.com) and Furu Wei (fuwei@microsoft.com).

The exact GitHub repository for the microsoft/trocr-base-printed model is [More Information Needed], as it is not explicitly provided in the references. The references mention a general link https://aka.ms/trocr for the TrOCR models and code being publicly available, but do not specify the exact repo path for this particular model.
- **Model type:** The microsoft/trocr-base-printed model is a Transformer-based OCR model that is pre-trained on a large-scale synthetic printed textline image dataset and fine-tuned on the SROIE dataset for printed text recognition, leveraging both computer vision and natural language processing modalities in an end-to-end manner.
- **Language(s):** The microsoft/trocr-base-printed model processes English text for printed text recognition in images.
- **License:** This project is licensed under the license found in the LICENSE file in the root directory of this source tree. Portions of the source code are based on the [fairseq](https://github.com/pytorch/fairseq) project. [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)
- **Finetuned from model:** Based on the provided references, the microsoft/trocr-base-printed model is initialized and fine-tuned from pre-trained models in two stages:

1. Encoder Initialization: The encoder is initialized using models like DeiT (Touvron et al. 2021) and BEiT (Bao, Dong, and Wei 2021). (Reference 4)

2. Decoder Initialization: The decoder is initialized using RoBERTa (Liu et al. 2019) models and MiniLM (Wang et al. 2020b) models. (Reference 8)

Specifically, the model is first pre-trained on a large-scale synthetic printed textline image dataset. Then in the second stage, it is further pre-trained on a smaller printed text dataset before being fine-tuned for downstream tasks. (Reference 6-7)

However, the exact model names and links for the base models used for initialization are not provided in the given references. [More Information Needed]
### Model Sources

- **Repository:** https://github.com/microsoft/unilm/tree/master/trocr
- **Paper:** https://arxiv.org/pdf/2109.10282.pdf
- **Demo:** [More Information Needed]

The provided references do not contain a direct link to a demo of the model microsoft/trocr-base-printed. More information would be needed to provide a demo link for this specific model in the model card description.
## Uses

### Direct Use

The microsoft/trocr-base-printed model can be used for text recognition tasks without requiring fine-tuning, post-processing, or plugging into a pipeline. The model takes textline images as input and directly outputs the predicted text.

To use the model for inference:

```python
Please see detials in [pic_inference.py](https://github.com/microsoft/unilm/blob/master/trocr/pic_inference.py).
```

The model outputs are based on Byte Pair Encoding (BPE) and SentencePiece, and do not rely on any task-related vocabularies. During inference, the decoder starts from the "[BOS]" token to predict the output iteratively while continuously taking the newly generated output as the next input, until the "[EOS]" token is predicted, indicating the end of the sentence.

[More Information Needed] on the specific API usage and required input image format.

### Downstream Use

The microsoft/trocr-base-printed model can be fine-tuned for printed text recognition tasks. It is pre-trained on large-scale synthetic data and can be further fine-tuned on human-labeled datasets like SROIE for specific printed text recognition applications.

When plugged into a larger ecosystem or app, the model can be used as an OCR component to extract text from printed images. It takes in textline images as input, extracts visual features using the encoder, and generates the recognized text using the decoder in an autoregressive manner.

[More Information Needed] for a specific code snippet, as the references do not provide a direct code block for using the model in an application. However, the model is publicly available and can be integrated into OCR pipelines for printed text recognition.

The model achieves state-of-the-art performance on printed text benchmarks without requiring complex pre/post-processing steps, making it suitable for various text recognition scenarios in real-world applications.

For any further information or updates regarding the microsoft/trocr-base-printed model, please feel free to reach out to the project organizer.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the microsoft/trocr-base-printed model could foreseeably be misused and what users ought not do:

The model should not be used to recognize and extract text from images without permission, as this could violate copyright or privacy rights. Users should ensure they have the necessary rights and consents before applying the model to images.

The model should not be relied upon for recognizing text in high-stakes domains like healthcare or finance without extensive testing and human oversight. Incorrect text extraction in these areas could lead to serious consequences.

Users should be aware the model may exhibit biases based on the data it was trained on. It should not be deployed in ways that could discriminate against certain demographics or perpetuate harmful stereotypes.

The model should not be used as the sole authoritative source for text recognition. Its outputs should be treated as predictions to be verified by humans, not as ground truth.

[More Information Needed] on the specific training data, intended use cases, and performance characteristics of microsoft/trocr-base-printed to fully assess potential misuse and limitations. The provided references focus more on technical model details rather than usage guidelines and societal implications. Consulting with the model developers and other relevant experts would help fill in these gaps to responsibly document and publish the model.

### Bias, Risks, and Limitations

Based on the provided references about the TrOCR model (microsoft/trocr-base-printed), here are some potential issues and limitations:

Technical limitations:
- The model relies on resizing input images to a fixed 384x384 size and splitting them into 16x16 patches. This may limit performance on text images of very different sizes or aspect ratios. [More Information Needed] on how well it handles varied image sizes.
- [More Information Needed] on computational requirements and inference speed compared to other OCR approaches. The references don't provide details on efficiency.

Foreseeable misunderstandings and harms:
- The model is trained on printed text recognition. Applying it to handwritten text or more complex scenes may lead to degraded performance and unreliable results. Users should be clearly informed of the intended use case.
- [More Information Needed] on how the model handles text in different languages and scripts. Multilingual support is mentioned but details are lacking.
- As with any AI system, overreliance on the model's outputs without human verification could lead to propagating errors in downstream applications. Human oversight is advised.

Sociotechnical considerations:
- The training data sources and any potential biases therein are [More Information Needed]. Biases in training data could lead to biased or discriminatory model behavior.
- [More Information Needed] on use of user data and privacy implications. How are user-provided images handled? Is any data retained?
- Intended use cases and potential misuse scenarios should be clearly outlined to users, along with guidance on responsible deployment.

In summary, while TrOCR shows promising technical capabilities, more information is needed in several key areas to robustly assess limitations and sociotechnical implications. Responsible disclosure and oversight are advised.

### Recommendations

Based on the provided references about the TrOCR model (microsoft/trocr-base-printed), here are some recommendations regarding foreseeable issues:

1. Bias and Fairness: [More Information Needed] The references do not provide details on what types of text images the model was trained and evaluated on. It's important to assess if the training data adequately represents diverse languages, scripts, fonts, handwriting styles, etc. to avoid biased performance. 

2. Robustness and Failure Modes: [More Information Needed] More analysis is needed on how the model performs on challenging text images - e.g. with background noise, distortions, low resolution, occlusion, etc. Understanding failure modes is important for real-world deployment.

3. Computational Efficiency: The references mention using image patches of size 16x16 as input to the image Transformer encoder (ref 7). This may be computationally expensive, especially for high-resolution images. Efficiency optimizations and impact on latency should be studied.

4. Responsible Usage: The ability to accurately recognize text in images could potentially be misused for unintended purposes like surveillance or privacy violation. Establishing usage guidelines and considering appropriate restrictions may be prudent.

5. Explainability and Interpretability: [More Information Needed] To build trust with users and aid debugging, it would be valuable to explore techniques to explain the model's predictions and failure modes in an interpretable manner.

In summary, key areas to focus on are analyzing potential biases, robustness, computational efficiency, responsible usage policies, and model interpretability. More targeted testing and documentation in these areas would help identify and mitigate foreseeable issues before releasing the model.

## Training Details

### Training Data

The microsoft/trocr-base-printed model was pre-trained on a large-scale synthetic dataset of hundreds of millions of printed textline images, and further fine-tuned on two relatively small datasets corresponding to printed and receipt text, totaling 3.3M textlines. The training data was augmented using six kinds of image transformations to enhance variety.

[More Information Needed] for links to documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about the preprocessing for the data of the model microsoft/trocr-base-printed:

Tokenization:
The outputs of the TrOCR models are based on Byte Pair Encoding (BPE) and SentencePiece for tokenization, and do not rely on any task-related vocabularies. (Reference 6)

Resizing/Rewriting:
The printed dataset consists of 3.3M textline images, which are obtained through the following steps (Reference 2):
1. Collect around 53K real-world receipt images 
2. Recognize the text on the receipts using commercial OCR engines
3. Crop the textlines based on their coordinates from the OCR results
4. Rectify the cropped textlines into normalized images
5. Use TRDG to synthesize an additional 1M printed textline images with two receipt fonts and built-in printed fonts
6. Apply six kinds of image transformations plus keeping the original images for data augmentation (Reference from Krishnan and Jawahar 2016)

[More Information Needed] on the specific resizing dimensions or normalization techniques used.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the microsoft/trocr-base-printed model:

- Batch size: 2,048
- Learning rate: 5e-5
- Image resolution: 384×384
- Patch size: 16×16
- Encoder: BEiT BASE with 12 layers, 768 hidden size, and 12 attention heads
- Decoder: RoBERTa LARGE, using only the last 12 layers
- Beam size: 10
- Tokenizer: BPE and SentencePiece tokenizer from Fairseq

[More Information Needed] on the exact number of training steps/epochs, optimizer, learning rate schedule, and other hyperparameters not explicitly mentioned in the provided references.

The model was trained on 32 V100 GPUs with 32GB memory for pre-training and 8 V100 GPUs for fine-tuning. The total number of parameters for the TrOCR BASE model is approximately 334M.

#### Speeds, Sizes, Times

Here are the details about the model microsoft/trocr-base-printed based on the provided references:

Model Architecture:
- Uses a Transformer-based encoder-decoder architecture
- Encoder: BEiT base model 
- Decoder: Transformer decoder initialized with RoBERTa base
- Outperforms CNN-based feature extractors and RNN-based language modeling without complex pre/post-processing

Training:
- Pre-trained on large-scale synthetic data
- Fine-tuned on the SROIE dataset
- Trained using 32 V100 GPUs with 32GB memory for pre-training
- Fine-tuned using 8 V100 GPUs
[More Information Needed] on exact training time

Model Checkpoint:
- Checkpoint file: trocr-base-printed.pt
- Download link: https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-printed.pt
[More Information Needed] on checkpoint size

Performance:
- Achieves state-of-the-art performance on the SROIE leaderboard for printed text recognition
[More Information Needed] on exact throughput metrics

The TrOCR model code and checkpoints are publicly available at https://aka.ms/trocr.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the TrOCR models were evaluated on the following benchmarks and datasets for printed text recognition:

1. IAM Handwriting Database: The TrOCR models were evaluated on this dataset and compared against existing methods. The TrOCR LARGE model achieved better results than methods using CTC decoders. (Reference 1)

2. SROIE (Scanned Receipts OCR and Information Extraction) dataset (Task 2): This dataset focuses on text recognition in receipt images. The evaluation was performed on cropped images of textlines obtained from the ground truth bounding boxes. (Reference 7)

[More Information Needed] for other specific benchmarks or datasets used to evaluate the microsoft/trocr-base-printed model, as the provided references do not mention them explicitly.

#### Factors

Based on the provided references, here are some foreseeable characteristics that may influence how the microsoft/trocr-base-printed model behaves:

1. Domain and context: The model is designed for printed text recognition (References 1, 4). It can handle scanned documents, photos of documents, scene photos, or subtitle text superimposed on images (Reference 4). The model's performance may vary depending on the specific domain and context of the input images.

2. Text characteristics: The model's performance may be influenced by various text characteristics such as font, size, spacing, and layout. [More Information Needed] on how the model handles different text characteristics and if there are any disparities in performance.

3. Image quality: The quality of input images, including resolution, contrast, and noise, can impact the model's text recognition accuracy. [More Information Needed] on how the model performs across different levels of image quality and if there are any performance disparities.

4. Language and script: The model can be extended for multilingual text recognition by leveraging multilingual pre-trained models and expanding the dictionary (Reference 1). However, [More Information Needed] on the specific languages and scripts supported by the microsoft/trocr-base-printed model and if there are any performance disparities across different languages or scripts.

5. Population subgroups: [More Information Needed] on how the model performs across different population subgroups, such as demographics, geographic regions, or socioeconomic factors. Ideally, evaluation should be disaggregated to uncover any potential disparities in performance.

To comprehensively assess the model's behavior and identify any disparities, further evaluation and disaggregation of results across various factors would be necessary. The provided references do not contain sufficient information to fully address all aspects of the question.

#### Metrics

Based on the references provided, the evaluation metrics for the microsoft/trocr-base-printed model would likely be:

For printed text datasets like IAM:
- Case-sensitive Character Error Rate (CER)

For scene text datasets: 
- Word Accuracy, using a 36-character lowercase alphanumeric charset for fair comparison

[More Information Needed] on the specific tradeoffs considered between different error types for this particular model. The references discuss metrics used for OCR models in general, but do not provide model-specific information about error tradeoffs for microsoft/trocr-base-printed.

### Results

Here are the evaluation results for the microsoft/trocr-base-printed model based on the provided references:

Model Architecture:
- Encoder: BEiT BASE 
- Decoder: RoBERTa LARGE
- Total parameters: 334M

The TrOCR BASE model consists of the BEiT BASE encoder and the RoBERTa LARGE decoder.

Performance:
The TrOCR models outperform existing SOTA models on printed text recognition tasks. Specifically:

- TrOCR models achieve state-of-the-art results on the leaderboard of the SROIE dataset compared to CNN-based methods for visual feature extraction and RNN-based methods for language modeling.

- Transformer-based TrOCR models are competitive with CNN-based networks for visual feature extraction and RNN-based networks for language modeling on printed text recognition without complex pre/post-processing.

Inference Speed:
[More Information Needed]

The references do not provide specific inference speed metrics for the microsoft/trocr-base-printed model. More information would be needed to report inference speed.

Data and Fine-tuning:
- The TrOCR model can be pre-trained on large-scale synthetic data and fine-tuned on human-labeled datasets for downstream text recognition tasks.

- Fine-tuning is performed for printed text recognition tasks.

- Outputs are based on Byte Pair Encoding (BPE) and SentencePiece tokenization, and do not rely on task-specific vocabularies.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the microsoft/trocr-base-printed model:

The TrOCR base models, which include microsoft/trocr-base-printed, achieve competitive performance on printed text recognition tasks compared to existing state-of-the-art models. Key points:

- TrOCR models outperform existing SOTA models with pure Transformer structures on the IAM Handwriting Database (Reference 5)
- TrOCR base models have fast inference speed, though not as fast as the small model (Reference 3)
- Combining TrOCR with pre-trained vision models like BEiT further improves text recognition performance (Reference 2)

The TrOCR architecture uses a standard Transformer encoder-decoder setup without relying on convolutions or complex pre/post-processing. This makes the model simple yet effective.

[More Information Needed] on the specific performance metrics (e.g. character error rate) for the microsoft/trocr-base-printed model on benchmark datasets. The references discuss TrOCR models in general but do not provide detailed results for this particular base printed model.

## Model Examination

Here is a possible answer for the "Explainability/Interpretability" section of the model card for microsoft/trocr-base-printed:

The TrOCR model uses a Transformer architecture that takes advantage of pre-trained image and text Transformer models. This allows the model to learn from large-scale unlabeled data for both image understanding and language modeling, without requiring an external language model. 

The model does not rely on any convolutional networks or introduce image-specific inductive biases, which makes the architecture simpler and easier to interpret compared to previous approaches like CRNN that used convolutional layers for feature extraction.

[More Information Needed] on specific explainability/interpretability techniques that have been applied to analyze the TrOCR model's predictions and internal representations. Future work could explore methods to visualize attention weights, generate explanations for predictions, and probe what the model has learned. Comparing the interpretability of TrOCR to other text recognition models would also provide useful insights.

The model outputs are based on Byte Pair Encoding (BPE) and SentencePiece tokenization rather than a fixed character vocabulary. [More Information Needed] on how the choice of tokenization affects the interpretability of the model's text generation process.

## Environmental Impact

- **Hardware Type:** The model microsoft/trocr-base-printed is trained on 8 V100 GPUs for fine-tuning, as mentioned in the references:

"We use 32 V100 GPUs with the memory of 32GBs for pre-training and 8 V100 GPUs for fine-tuning."
- **Software Type:** The TrOCR models, including microsoft/trocr-base-printed, are built upon the Fairseq toolkit according to Reference 2:

"The TrOCR models are built upon the Fairseq (Ott et al. 2019) which is a popular sequence modeling toolkit."

So the software type that the model microsoft/trocr-base-printed is trained on is Fairseq.
- **Hours used:** Based on the provided references, there is no specific information about the amount of time used to train the microsoft/trocr-base-printed model. The references mention using 32 V100 GPUs for pre-training and 8 V100 GPUs for fine-tuning, but do not specify the duration of training.

[More Information Needed] on the exact training time for the microsoft/trocr-base-printed model.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the microsoft/trocr-base-printed model. They only state that "We use 32 V100 GPUs with the memory of 32GBs for pre-training and 8 V100" GPUs, but do not specify the cloud provider those GPUs are from.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the microsoft/trocr-base-printed model. To answer this question, more specific details would be needed, such as the computing resources used for training, the duration of the training process, and the energy consumption of the hardware. Without these details, it is not possible to estimate the carbon emissions associated with training this particular model.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
TrOCR is built with the Transformer encoder-decoder architecture. The encoder is an image Transformer that extracts visual features from the input image, which is resized to 384x384 and split into a sequence of 16x16 patches. The decoder is a standard text Transformer for generating the wordpiece sequence, guided by the visual features and previous predictions.

The encoder is initialized with pre-trained models like DeiT (Touvron et al. 2021) and BEiT (Bao, Dong, and Wei 2021). [More Information Needed] on the specific pre-trained model used for microsoft/trocr-base-printed.

The decoder leverages pre-trained language models. [More Information Needed] on the specific pre-trained language model used in the decoder of microsoft/trocr-base-printed.

Objective:
TrOCR is an end-to-end text recognition approach that aims to recognize and transcribe text from images. It leverages pre-trained image Transformer and text Transformer models to jointly learn visual understanding and wordpiece-level text generation.

The model can be pre-trained on large-scale synthetic data and fine-tuned on human-labeled datasets for specific text recognition tasks such as printed, handwritten, and scene text recognition.

TrOCR aims to achieve state-of-the-art performance on various OCR benchmark datasets without relying on complex pre/post-processing steps. It can also be easily extended to multilingual text recognition by using multilingual pre-trained models in the decoder and expanding the dictionary.

### Compute Infrastructure

Based on the provided references, here is the compute infrastructure information I could find about the microsoft/trocr-base-printed model:

We use 32 V100 GPUs with the memory of 32GBs for pre-training and 8 V100 GPUs for fine-tuning. For all the models, the batch size is set to 2,048 and the learning rate is 5e-5.

The TrOCR-Base-SROIE model file can be downloaded from:
[trocr-base-printed.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-printed.pt)

[More Information Needed] on the exact training time, number of parameters, and other compute infrastructure details for the microsoft/trocr-base-printed model specifically.

## Citation

```
@misc{minghao-trocr,
    author = {Minghao Li and
              Tengchao Lv and
              Jingye Chen and
              Lei Cui and
              Yijuan Lu and
              Dinei Florencio and
              Cha Zhang and
              Zhoujun Li and
              Furu Wei},
    title  = {TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models},
    url    = {https://arxiv.org/pdf/2109.10282.pdf}
}
```

