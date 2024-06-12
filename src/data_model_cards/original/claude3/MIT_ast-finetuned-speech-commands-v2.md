# Model Card for MIT/ast-finetuned-speech-commands-v2

The MIT/ast-finetuned-speech-commands-v2 model is an Audio Spectrogram Transformer (AST) model fine-tuned on the Speech Commands V2 dataset, achieving state-of-the-art accuracy of 98.1%. AST is a convolution-free, purely attention-based model that can be applied to various audio classification tasks with variable-length inputs.

## Model Details

### Model Description

Here are the basic details about the MIT/ast-finetuned-speech-commands-v2 model:

Model Architecture:
- Audio Spectrogram Transformer (AST), a convolution-free, purely attention-based model 
- The 2D audio spectrogram is split into a sequence of 16x16 patches with overlap, linearly projected to 1-D patch embeddings
- Each patch embedding is added with a learnable positional embedding 
- An additional classification token is prepended to the sequence
- The output embedding is input to a Transformer, and the output of the classification token is used for classification with a linear layer

Training Procedures: 
- Trained on the Speech Commands V2 dataset
- [More Information Needed] on specific training procedures used

Parameters:
- Model size is base384
- audioset_pretrain=True, using AudioSet and ImageNet pretrained weights
- fstride=tstride=10

Important Disclaimers:
[More Information Needed]

Please refer to the paper "AST: Audio Spectrogram Transformer" (https://arxiv.org/abs/2104.01778) for more details. The model achieves 98.1% accuracy on the Speech Commands V2 benchmark.

Let me know if you need any other information to complete the model card. I'm happy to help track down additional details.

- **Developed by:** Yuan Gong; Yu-An Chung; James Glass
- **Funded by:** Based on the provided references, there is no explicit information about the funding sources for the MIT/ast-finetuned-speech-commands-v2 model. The references mainly discuss the technical details and performance of the Audio Spectrogram Transformer (AST) model, but do not mention the specific organizations or people that funded the project.

[More Information Needed]
- **Shared by:** According to the references provided, the contributors who made the model MIT/ast-finetuned-speech-commands-v2 available online are:

Yuan Gong, Yu-An Chung, and James Glass

The work is partly supported by Signify.
- **Model type:** The MIT/ast-finetuned-speech-commands-v2 model is a purely attention-based audio classification model that was pretrained on AudioSet and finetuned on the Speech Commands V2 dataset, achieving 98.1% accuracy on the evaluation set.
- **Language(s):** The MIT/ast-finetuned-speech-commands-v2 model processes audio data from the Speech Commands V2 dataset, which contains spoken English words and background noise.
- **License:** [More Information Needed]

The provided references do not contain any information about the license being used for the model MIT/ast-finetuned-speech-commands-v2. More details would be needed from the model developers to determine the specific license that applies to this model.
- **Finetuned from model:** The model MIT/ast-finetuned-speech-commands-v2 is likely fine-tuned from the AudioSet pretrained AST model mentioned in Reference 7:

[Full AudioSet, 10 tstride, 10 fstride, with Weight Averaging (0.459 mAP)](https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1)

Reference 6 also mentions that setting `audioset_pretrain=True` when creating the AST model will automatically download the AudioSet pretrained model for fine-tuning on a new task like Speech Commands.

However, to confirm this is the exact base model used, [More Information Needed] from the model authors or documentation.
### Model Sources

- **Repository:** https://github.com/YuanGongND/ast
- **Paper:** https://arxiv.org/pdf/2104.01778.pdf
- **Demo:** [More Information Needed]

The references provided do not contain a direct link to a demo for the specific model "MIT/ast-finetuned-speech-commands-v2". While the references mention a demo for an audio large language model called LTU, and a Google Colab script for AST inference, there is no information about a demo specifically for the "MIT/ast-finetuned-speech-commands-v2" model.
## Uses

### Direct Use

The MIT/ast-finetuned-speech-commands-v2 model can be used for inference without fine-tuning, post-processing, or plugging into a pipeline. Here are the key points:

1. The model is pretrained on AudioSet and fine-tuned on the Speech Commands V2 dataset, achieving 98.12% accuracy on the evaluation set.

2. It supports variable length input, so you can directly use it for your audio classification task, as long as your audio data is prepared in 16kHz.

3. To use the pretrained model, you need to normalize the input audio to roughly 0 mean and 0.5 std. You can use the AudioSet normalization: 
```
input_spec = (input_spec + 4.26) / (4.57 * 2)
```

4. We provide a one-click, self-contained Google Colab script for AST inference and attention visualization. You can test the model with your own audio by clicking the "Open in Colab" badge:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuanGongND/ast/blob/master/colab/AST_Inference_Demo.ipynb)

5. [More Information Needed] on the specific code snippet for using the model without fine-tuning, post-processing or plugging into a pipeline.

In summary, the MIT/ast-finetuned-speech-commands-v2 model can be easily used for audio classification inference with the proper input normalization. The provided Colab demo makes it straightforward to test the model on your own audio data.

### Downstream Use

The MIT/ast-finetuned-speech-commands-v2 model can be used for speech command recognition tasks when fine-tuned. It achieves state-of-the-art accuracy of 98.1% on the Speech Commands V2 dataset.

To use the fine-tuned model in your own training pipeline, you can take the `ast/src/models/ast_models.py` file, set `audioset_pretrain=True`, and integrate it with your pipeline. Make sure to normalize the audio input to roughly 0 mean and 0.5 std, similar to what the pretrained model expects. You can use the AudioSet normalization: 

```python
input_spec = (input_spec + 4.26) / (4.57 * 2)
```

Note that AST requires a smaller learning rate (10x smaller than CNN models) and converges faster, so tune the learning rate and scheduler appropriately for your task.

The model supports variable length inputs, so it can be applied to different speech tasks without architecture changes. In an application, you would feed 16kHz audio (that the model was trained on) and get the predicted speech command as output.

[More Information Needed] for providing a complete code snippet, as the references don't include one. Overall, the AST model provides a powerful, purely attention-based solution for speech command recognition that can be dropped into many pipelines and applications.

### Out-of-Scope Use

Based on the provided references, here are some potential ways the MIT/ast-finetuned-speech-commands-v2 model could be misused and what users should not do:

The model should not be used for tasks beyond simple speech command classification, as it was specifically trained on the Speech Commands V2 dataset for that purpose. Applying it to significantly different audio classification tasks may lead to unreliable or biased results.

Users should not assume the model will perform optimally on speech data of lengths differing from what it was trained on (around 1 second clips). Feeding much longer or shorter audio could degrade accuracy.

The model should not be deployed in high-stakes real-world scenarios like voice-controlled medical devices or emergency response systems without thorough testing and validation, as errors could have severe consequences. 

Users ought not extrapolate the model's strong benchmark performance to claim human-level accuracy in understanding speech commands, as the model's robustness to accents, noise, and adversarial attacks is [More Information Needed].

To mitigate potential misuse, the model card should clearly state the intended use case of simple speech command recognition, emphasize that predictions are probabilistic and can be erroneous, and caution against application in sensitive domains without extensive testing on representative real-world data.

[More Information Needed] on details of the model architecture, training data, and evaluation of fairness and robustness that would allow deeper analysis of misuse potential and recommendations for mitigation.

### Bias, Risks, and Limitations

Based on the provided references about the MIT/ast-finetuned-speech-commands-v2 model, here are some potential issues and limitations:

Foreseeable harms and misunderstandings:
- The model is trained on the Speech Commands dataset, so it may not generalize well to speech in other domains or with different accents/languages. Applying it outside its intended use case could lead to poor performance or misclassifications.
- [More Information Needed] on potential biases in the training data that could lead to unfair or discriminatory model outputs.

Technical limitations: 
- The model expects 16kHz audio input that is roughly normalized to 0 mean and 0.5 std (ref 4). Using audio that deviates significantly from this could degrade performance.
- A fixed AST architecture is used which may not be optimal for all speech classification tasks (ref 10).
- [More Information Needed] on compute requirements and latency which could limit real-time applications.

Sociotechnical limitations:
- As a speech classifier, the model has no contextual understanding. Outputs should be carefully interpreted based on the specific application. 
- [More Information Needed] on how the model handles sensitive content and potential privacy concerns with classifying human speech.

To summarize, key limitations appear to be the narrow training domain, fixed input requirements, and lack of contextual awareness. More information is needed to fully characterize potential biases, resource requirements, and sociotechnical implications. Responsible usage requires understanding these constraints and carefully validating the model's fitness for a given real-world application.

### Recommendations

Based on the provided references about the AST model and its finetuning on the Speech Commands V2 dataset, here are some recommendations regarding foreseeable issues:

1. Input normalization: The pretrained AST model expects the input audio to be roughly normalized to 0 mean and 0.5 std. When using the model, ensure the input data is properly normalized to avoid performance degradation. [Reference 7]

2. Audio sampling rate: The pretrained model and training pipeline use 16kHz audio. Prepare your data in 16kHz to be compatible with the pretrained model. [Reference 5, 6]

3. Model architecture: While the AST model can handle variable-length inputs, the architecture used for Speech Commands classification (AST-S) may not be optimal for other tasks or audio lengths. Consider tuning the architecture if applying the model to significantly different tasks or audio lengths. [Reference 10]

4. Generalization to non-speech audio: The model was finetuned on Speech Commands V2, which consists of speech data. Performance on non-speech audio tasks may vary, and additional finetuning or data augmentation techniques might be necessary. [More Information Needed]

5. Fairness and bias: [More Information Needed] Analyze the model's performance across different demographics, accents, and languages to ensure fairness and mitigate potential biases.

6. Privacy concerns: [More Information Needed] Consider the privacy implications of deploying a speech recognition model, such as the potential for misuse or unintended identification of individuals.

7. Robustness to noise and distortions: [More Information Needed] Evaluate the model's performance under various noise conditions and audio distortions to understand its limitations and potential failure modes.

## Training Details

### Training Data

The MIT/ast-finetuned-speech-commands-v2 model was trained on the Speech Commands V2 dataset, which contains spoken words for speech command classification. The dataset was automatically downloaded and the model was trained, validated and evaluated using the recipe in `ast/egs/speechcommands/run_sc.sh`.

### Training Procedure

#### Preprocessing

For the MIT/ast-finetuned-speech-commands-v2 model, the preprocessing steps include:

Data augmentation techniques such as:
- mixup with mixup ratio=0.5
- spectrogram masking with max time mask length of 192 frames and max frequency mask length of 48 bins

[More Information Needed] on specific details about tokenization and resizing/rewriting of the input data for this model.

The model was trained on the Speechcommands V2 dataset. The provided recipe script `ast/egs/speechcommands/run_sc.sh` automatically downloads the dataset and runs the training, validation and evaluation.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the model MIT/ast-finetuned-speech-commands-v2:

- Pretrained model: AudioSet pretrained AST model (0.459 mAP)
- Input time dimension: 100 frames
- Time stride: 10
- Frequency stride: 10
- No weight averaging used
- Achieved 98.12% accuracy on the Speech Commands V2-35 evaluation set

[More Information Needed] for the following hyperparameters:
- Number of training epochs
- Learning rate schedule
- Batch size
- Optimizer
- Loss function
- Data augmentation techniques used, if any

The model expects 16kHz audio input since the pretrained model was trained on 16kHz audio.

#### Speeds, Sizes, Times

Here are the details about the MIT/ast-finetuned-speech-commands-v2 model based on the provided references:

Model Performance:
- The model achieves 98.12% accuracy on the Speech Commands V2-35 evaluation set.

Model Checkpoint:
- The model checkpoint can be downloaded from this link: https://www.dropbox.com/s/q0tbqpwv44pquwy/speechcommands_10_10_0.9812.pth?dl=1
- [More Information Needed] on the exact checkpoint size.

Training Details:
- The model uses a 10 time step stride and 10 frequency stride.
- Weight averaging was not used during training.
- The model architecture is a purely attention-based AST model without convolutions.
- The AST model was initialized with AudioSet pretrained weights for better performance.
- [More Information Needed] on the exact training start and end times, as well as throughput metrics.

In summary, the MIT/ast-finetuned-speech-commands-v2 model achieves state-of-the-art 98.12% accuracy on the Speech Commands V2-35 dataset using a purely attention-based architecture. The model checkpoint is publicly available, but more details are needed on the checkpoint size, training times, and throughput.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the MIT/ast-finetuned-speech-commands-v2 model evaluates on the Speech Commands V2 dataset. Specifically, reference 10 states:

"The Speechcommands recipe is in `ast/egs/speechcommands/run_sc.sh`, the script will automatically download the Speechcommands V2 dataset, train an AST model on the training set, validate it on the validation set, and evaluate it on the test set."

The model achieves 98.12% accuracy on the Speech Commands V2 evaluation set, as mentioned in reference 11:

"We attached our log file in `ast/egs/speechcommends/test-speechcommands-f10-t10-p-b128-lr2.5e-4-0.5-false`, the model achieves `98.12%` accuracy."

#### Factors

Based on the provided references, here are some key characteristics that may influence the behavior of the MIT/ast-finetuned-speech-commands-v2 model:

Domain and Context:
- The model was trained and evaluated on the Speech Commands V2 dataset, which contains spoken word commands (References 4, 6, 11). Performance on other types of audio data like music, environmental sounds, or conversational speech is unknown and would require further evaluation.

Population Subgroups:
- [More Information Needed] The references do not provide details on performance across different speaker demographics (e.g. age, gender, accent). Ideally, the model should be evaluated on a diverse range of speakers to uncover any performance disparities.

Model Characteristics:
- AST is a purely attention-based model that can handle variable length audio inputs without architecture changes (References 1, 2, 10).
- It achieves state-of-the-art accuracy of 98.1% on the Speech Commands V2 dataset (Reference 1).
- The same architecture is used across different tasks and audio lengths, indicating potential as a generic audio classifier (References 6, 11).

To comprehensively characterize model behavior, additional information and disaggregated evaluation would be needed on aspects like:
- Performance on more diverse audio domains beyond speech commands
- Breakdown of results across speaker demographics 
- Failure modes and error analysis on misclassified examples
- Robustness to factors like background noise, recording quality, etc.

#### Metrics

Based on the references provided, the key evaluation metric used for the Speech Commands V2 dataset is accuracy on the test set. Specifically, reference 9 mentions a model achieving 98.12% accuracy on the evaluation set of Speech Commands V2.

The references do not provide details on analyzing tradeoffs between different types of errors for the Speech Commands models. To fully answer the question about error tradeoffs, [More Information Needed] on metrics like precision, recall, or a confusion matrix that would allow analyzing the types of errors made.

In summary, accuracy appears to be the primary metric, but more details would be needed to analyze tradeoffs between different errors for the MIT/ast-finetuned-speech-commands-v2 model.

### Results

Here are the evaluation results for the MIT/ast-finetuned-speech-commands-v2 model based on the provided references:

Model Performance:
- The AST-S model achieves 98.11±0.05% accuracy on the Speech Commands V2 test set, outperforming the previous state-of-the-art model (Ref 3)
- A specific model checkpoint achieves 98.12% accuracy on the Speech Commands V2 evaluation set (Ref 6, 7)

Training Configuration:
- Batch size: 128 (Ref 3)
- Optimizer: Adam with initial learning rate of 2.5e-4, decreased by a factor of 0.85 every epoch after the 5th epoch (Ref 3)
- Number of training epochs: Up to 20, with best model selected using the validation set (Ref 3)
- Data augmentation: Frequency and time masking, random noise, and mixup augmentation (Ref 4)
- Patch split: 10x10 (10 stride in time, 10 stride in frequency) (Ref 6)

[More Information Needed] for other specific training hyperparameters or configurations.

The model is a convolution-free, purely attention-based model (AST) that supports variable length input (Ref 8). It achieves state-of-the-art performance on the Speech Commands V2 dataset.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the model MIT/ast-finetuned-speech-commands-v2:

The AST-S model, which is an Audio Spectrogram Transformer model pretrained on ImageNet and then finetuned on the Speech Commands V2 dataset, achieves an accuracy of 98.11±0.05% on the test set. This outperforms the previous state-of-the-art CNN model in [9] which achieved 97.7% accuracy.

The model was trained with frequency and time masking, random noise, and mixup augmentation, a batch size of 128, and the Adam optimizer with an initial learning rate of 2.5e-4 decreased by a factor of 0.85 every epoch after the 5th epoch, for up to 20 epochs total. The best model was selected using the validation set.

[More Information Needed] on the exact training hyperparameters and data augmentation used for the MIT/ast-finetuned-speech-commands-v2 model specifically. The references provide some training details but it's unclear if they apply to this exact model.

In summary, the AST model architecture, when finetuned on the Speech Commands V2 dataset, achieves state-of-the-art accuracy of 98.11% on the test set, demonstrating the effectiveness of this convolution-free attention-based approach for speech command recognition.

## Model Examination

Here is my attempt at answering the question about the MIT/ast-finetuned-speech-commands-v2 model, based on the provided references:

This model is a finetuned version of the Audio Spectrogram Transformer (AST) architecture, trained on the Speech Commands V2 dataset. The AST model is a purely attention-based model for audio classification that supports variable length input. 

Some key points about the interpretability and explainability of this model:

- The AST architecture is simpler and has fewer parameters compared to CNN-attention hybrid models, which may make it easier to interpret and explain the model's behavior. However, [More Information Needed] on specific techniques used to analyze the interpretability of this finetuned model.

- AST models can be applied to various audio classification tasks without changing the architecture, just by varying the input length. The MIT/ast-finetuned-speech-commands-v2 model takes 1 second audio clips as input, tuned for the Speech Commands task. [More Information Needed] on how the input length and task impact interpretability.

- [More Information Needed] on any visualization or explanation methods (e.g. attention weight analysis, input perturbation studies, etc.) applied to better understand and explain the decisions made by this specific finetuned AST model.

In summary, the simple, flexible AST architecture used in this model may have advantages for interpretability compared to other audio models, but more specific studies and techniques are needed to provide a fuller picture of the explainability of the MIT/ast-finetuned-speech-commands-v2 model. The developers aim to expand this section as more interpretability work is done.

## Environmental Impact

- **Hardware Type:** The model MIT/ast-finetuned-speech-commands-v2 was trained on 4 GTX TITAN GPUs with 12GB memory, according to the information provided in Reference 6:

"The recipe was tested on 4 GTX TITAN GPUs with 12GB memory."
- **Software Type:** Based on the provided references, the model MIT/ast-finetuned-speech-commands-v2 is trained on the Speech Commands V2 dataset. Specifically, reference 4 states:

"We evaluate AST on various audio classification benchmarks, where it achieves new state-of-the-art results of 0.485 mAP on AudioSet, 95.6% accuracy on ESC-50, and 98.1% accuracy on Speech Commands V2."

The references do not provide information about the specific software used for training the model. Therefore, for the software type, the answer is [More Information Needed].
- **Hours used:** Based on the provided references, the amount of time used to train the model MIT/ast-finetuned-speech-commands-v2 is not explicitly mentioned. The references discuss training details such as number of epochs, learning rate schedule, and batch size, but do not specify the total training time.

[More Information Needed] on the exact amount of time used to train the MIT/ast-finetuned-speech-commands-v2 model.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the MIT/ast-finetuned-speech-commands-v2 model. The references only state that the Speechcommands recipe was tested on 4 GTX TITAN GPUs with 12GB memory, but do not specify if this was done on a cloud platform or local hardware.
- **Carbon Emitted:** [More Information Needed]

Based on the provided references, there is no information about the amount of carbon emitted during the training of the model MIT/ast-finetuned-speech-commands-v2. The references discuss model accuracy, datasets, hyperparameters, and training settings, but do not mention anything related to carbon emissions or environmental impact of the model training process.

To provide the carbon emissions for this specific model, more details would be needed such as the compute infrastructure used, total training time, energy consumption, and carbon intensity of the electricity grid. Without this information, it is not possible to accurately estimate the carbon footprint of training the MIT/ast-finetuned-speech-commands-v2 model.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The Audio Spectrogram Transformer (AST) is a convolution-free, purely attention-based model for audio classification. It splits the 2D audio spectrogram into a sequence of 16x16 patches with overlap, linearly projects them to a sequence of 1-D patch embeddings, and adds learnable positional embeddings. An additional classification token is prepended to the sequence. The output embedding is input to a Transformer, and the output of the classification token is used for classification with a linear layer.

Code:
```
ast_mdl = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=True)
test_output = ast_mdl(test_input)
```

Objective:
AST-finetuned-speech-commands-v2 is finetuned on the Speech Commands V2 dataset for speech command classification. AST achieves state-of-the-art accuracy of 98.1% on this task.

[More Information Needed] on specific training details and hyperparameters used for finetuning on Speech Commands V2.

### Compute Infrastructure

The compute infrastructure used for training the MIT/ast-finetuned-speech-commands-v2 model was:

4 GTX TITAN GPUs with 12GB memory each

[More Information Needed] on other compute infrastructure details such as CPU, RAM, storage, etc. The provided references do not contain this information.

## Citation

```
@misc{yuan-ast,
    author = {Yuan Gong and
              Yu-An Chung and
              James Glass},
    title  = {AST: Audio Spectrogram Transformer},
    url    = {https://arxiv.org/pdf/2104.01778.pdf}
}
```

