# Model Card for microsoft/speecht5_vc

The microsoft/speecht5_vc model is a voice conversion model based on the SpeechT5 framework. It converts a speaker-dependent source speech waveform into a different one while preserving the linguistic information, and achieves significant improvements over state-of-the-art baselines like the Voice Transformer Network (VTN).

## Model Details

### Model Description

Model Description for microsoft/speecht5_vc:

Model Architecture:
- Encoder-decoder Transformer model with modal-specific pre/post-nets (Fig 2a in Reference 1)
- Speech encoder pre-net: convolutional feature extractor from wav2vec 2.0 to downsample raw waveform (Reference 3)
- Speech decoder pre-net: 3 fully-connected layers with ReLU, concatenated with speaker embedding (Reference 3) 
- Speech decoder post-net: linear layer to predict log Mel-filterbank, followed by 1D conv layers and stop token prediction (Reference 5)

Training Procedure: 
- Pre-trained on 960 hours of LibriSpeech audio data (Reference 7)
- Optimized with Adam, learning rate 2e-4, 500k steps on 32 V100 GPUs (Reference 7)
- Trained with bidirectional masked prediction and sequence-to-sequence generation tasks (Reference 6)

Important Parameters:
- Input: raw speech waveform 
- Output: log Mel-filterbank and stop token
- [More Information Needed] on model size 

Disclaimers:
- [More Information Needed]

For more details, please refer to the full paper. The code and model are available at https://github.com/microsoft/SpeechT5.

- **Developed by:** Junyi Ao; Rui Wang; Long Zhou; Chengyi Wang; Shuo Ren; Yu Wu; Shujie Liu; Tom Ko; Qing Li; Yu Zhang; Zhihua Wei; Yao Qian; Jinyu Li; Furu Wei; Peng Cheng Laboratory
- **Funded by:** Based on the provided references, the following people and organizations are mentioned in relation to the SpeechT5 project and the microsoft/speecht5_vc model:

Peng Cheng Laboratory

However, there is no explicit mention of the specific funding sources for this project. To confidently list the funders, [More Information Needed].
- **Shared by:** Based on the references provided, the contributors who made the model microsoft/speecht5_vc available online as a GitHub repo are:

Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, and Furu Wei from Microsoft and Peng Cheng Laboratory.

The references mention that for help or issues using SpeechT5 models, a GitHub issue should be submitted. This implies the models are available on GitHub. The paper authors are listed, indicating they contributed to making the models available.

However, more specific information would be needed to determine the exact process and roles of the contributors in releasing the model on GitHub.
- **Model type:** The microsoft/speecht5_vc model is a self-supervised pre-trained encoder-decoder model that uses joint speech and text pre-training for spoken language processing tasks such as voice conversion (VC), leveraging unlabeled speech and text data.
- **Language(s):** The microsoft/speecht5_vc model processes speech data for the task of voice conversion (VC).
- **License:** [More Information Needed]

The references provided do not contain any information about the specific license being used for the model microsoft/speecht5_vc. More details would be needed from the model developers or documentation to determine the applicable license.
- **Finetuned from model:** Based on the provided references, the model microsoft/speecht5_vc is likely fine-tuned from the SpeechT5 Base model that was pre-trained on the 960 hours LibriSpeech dataset and the LibriSpeech LM Dataset.

The pre-trained SpeechT5 Base model can be found at:
- HuggingFace: https://huggingface.co/ajyy/SpeechT5/resolve/main/speecht5_base.pt
- Google Drive: https://drive.google.com/file/d/1Sq00uZ1pw6Z4OUaqhOWzQEJxIVWgAO5U/view?usp=sharing

However, more specific information is needed to confirm if microsoft/speecht5_vc was indeed fine-tuned from this exact pre-trained model.
### Model Sources

- **Repository:** https://github.com/microsoft/SpeechT5/
- **Paper:** https://arxiv.org/pdf/2110.07205.pdf
- **Demo:** [More Information Needed]

Based on the provided references, there is no specific information about a demo link for the model microsoft/speecht5_vc. The references mention various SpeechT5 models and their release dates, but do not include any demo links. More information would be needed from the SpeechT5 documentation or GitHub repository to determine if a demo exists for this particular model.
## Uses

### Direct Use

Based on the provided references, there is no direct information on how the microsoft/speecht5_vc model can be used without fine-tuning, post-processing, or plugging into a pipeline. The references mainly discuss the pre-training and fine-tuning process of the SpeechT5 model on various spoken language processing tasks, including voice conversion (VC).

The references do not provide a specific code snippet demonstrating the usage of the microsoft/speecht5_vc model without additional steps.

[More Information Needed] on the specific usage details and code snippet for utilizing the microsoft/speecht5_vc model directly without fine-tuning, post-processing, or integration into a pipeline.

### Downstream Use

The microsoft/speecht5_vc model can be fine-tuned for voice conversion (VC) tasks. As mentioned in the references, SpeechT5 performs better than the state-of-the-art voice Transformer network on the VC task after fine-tuning.

To fine-tune the model for voice conversion, you would need to:

1. Prepare a dataset for voice conversion with paired speech samples from source and target speakers.
2. Load the pre-trained microsoft/speecht5_vc model.
3. Fine-tune the model using the voice conversion dataset with the appropriate loss function, optimizer, and hyperparameters as mentioned in the references.

[More Information Needed] for the specific code snippet to fine-tune the model, as no direct code block is provided in the references.

Once fine-tuned, the microsoft/speecht5_vc model can be integrated into a larger ecosystem or application that requires voice conversion functionality, such as:

- Voice-based assistants or chatbots to generate responses in different voices
- Dubbing or translation systems to convert the voice of an actor or speaker to another language or accent
- Voice privacy tools to anonymize or mask the identity of a speaker

[More Information Needed] for the specific integration details and code snippets, as they would depend on the specific ecosystem or application.

### Out-of-Scope Use

Based on the provided references about the SpeechT5 model and its voice conversion (VC) capabilities, here are some potential misuses and what users should not do with the microsoft/speecht5_vc model:

The model should not be used to convert voices without the explicit consent of the speakers involved. Voice conversion could potentially be misused for impersonation or creating misleading audio content.

Users should not attempt to use the model to convert voices to mimic real individuals, especially public figures or celebrities, as this could enable the creation of fake audio content that spreads misinformation or causes reputational harm.

The model should not be employed for any kind of voice spoofing attack to bypass voice authentication systems.

[More Information Needed] on specific technical details of how the model architecture could be misused or exploited.

In general, the microsoft/speecht5_vc model should only be used for legitimate research, creative, and educational purposes with proper disclosure that the audio has been generated by voice conversion. Users have an ethical obligation not to utilize the powerful VC capabilities for deception, harassment, or infringement of an individual's rights and privacy related to their voice and speech.

### Bias, Risks, and Limitations

Based on the provided references about the SpeechT5 model, here are some potential issues and limitations for the microsoft/speecht5_vc voice conversion model:

Foreseeable harms and misunderstandings:
- The voice conversion capability could potentially be misused to generate fake or misleading audio content that imitates real people's voices without consent. This has risks of fraud, identity theft, or reputational damage.
- Voice conversion models may perpetuate biases present in the training data, such as underrepresentation or stereotyping of certain accents, demographics, or speaking styles.

Technical limitations: 
- Voice conversion quality is likely not perfect and artifacts may be present, especially for speakers or styles not well-represented in the training data. More information is needed on conversion naturalness and speaker similarity.
- [More Information Needed] on the specific voice conversion use case and performance of microsoft/speecht5_vc, as the references focus more broadly on the SpeechT5 framework.

Sociotechnical limitations:
- Broad accessibility of voice conversion technology may require thoughtful release strategies, documentation, and safeguards to mitigate potential misuse. [More Information Needed] on the specific model release plans.
- As with other speech synthesis models, the development of voice conversion systems would benefit from socially-aware data collection and increased inclusion of diverse voices. [More Information Needed] on the specific data sources and practices used.

In summary, while voice conversion with models like microsoft/speecht5_vc enables compelling creative use cases, it also poses non-trivial risks that require proactive cross-disciplinary collaboration to address. More specific information on the model's performance, training data, and release plans would help further characterize limitations and inform risk mitigation strategies.

### Recommendations

Based on the provided references about the SpeechT5 model and its application to voice conversion (VC) in the microsoft/speecht5_vc model, here are some recommendations regarding foreseeable issues:

1. Data privacy and consent: Ensure that the speech data used for pre-training and fine-tuning the model has been obtained with proper consent and adheres to data privacy regulations. [More Information Needed] on the specific data sources and privacy measures taken.

2. Potential misuse: Voice conversion technology could be misused for creating deepfakes or impersonating individuals without their consent. Establish clear guidelines and restrictions on the use of the model to prevent misuse.

3. Bias and fairness: Evaluate the model's performance across different demographics, accents, and languages to ensure it does not exhibit biases or perform poorly for certain groups. [More Information Needed] on the diversity of the training data and any bias mitigation techniques applied.

4. Transparency and accountability: Clearly document the model's capabilities, limitations, and intended use cases in the model card. Provide information on the training data, model architecture, and any known issues or biases.

5. Ongoing monitoring and maintenance: Continuously monitor the model's performance and impact in real-world applications. Establish channels for users to report issues or concerns, and be prepared to update or refine the model as needed.

6. Ethical considerations: Engage with ethicists, legal experts, and stakeholders to discuss the broader societal implications of voice conversion technology and ensure its development and deployment align with ethical principles.

[More Information Needed] on the specific steps taken to address these issues in the development and release of the microsoft/speecht5_vc model. Providing detailed information on data privacy, bias mitigation, intended use cases, and ethical considerations in the model card will help users make informed decisions about using the model responsibly.

## Training Details

### Training Data

The training data of the model microsoft/speecht5_vc is the CMU Arctic dataset, which contains speech recordings of four speakers (two female and two male) reading the same 1,132 phonetically balanced English utterances. For each speaker, the first 932 sentences are used for training, the last 100 for testing, and the remaining 100 for validation.

[More Information Needed] for documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the data of the model microsoft/speecht5_vc:

Tokenization:
[More Information Needed] - The references do not provide specific details about tokenization for the voice conversion task.

Speech Input Preprocessing:
The speech input is converted to 80-dimensional log Mel-filterbank features, as mentioned in reference 7:
"where x f n denotes n-th an 80-dimensional log Melfilterbank from X f ."

Text Input Preprocessing: 
The text input is corrupted using a mask-based noising function to generate the inputX t = (x t 1 , ...,x t M ), as stated in reference 7:
"SpeechT5 is trained to reconstruct the model output Y t = (y t 1 , ..., y t N t ) to the original text X t , using the corrupted textX t = (x t 1 , ...,x t M ) as the input generated with a mask-based noising function."

However, more specific details about the text preprocessing are not provided in the given references.

Resizing/Rewriting:
[More Information Needed] - The references do not mention any specific resizing or rewriting techniques applied to the input data for the voice conversion task.

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the microsoft/speecht5_vc model:

- Fine-tuned the pre-trained model on the 460 hours LibriTTS clean sets
- Used the L1 loss, L_s_bce loss, and attention loss for fine-tuning
- Trained the model on 8 V100 GPUs with a batch size of 16000 tokens per GPU
- Used the Adam optimizer with a learning rate based on the inverse square root with a maximum learning rate of 10^-4 within 100k steps
- Applied 10k warm-up steps

[More Information Needed] on the exact number of training epochs/steps, learning rate schedule details, and other hyperparameters like weight decay, gradient clipping, etc.

#### Speeds, Sizes, Times

Here are the details about the model microsoft/speecht5_vc based on the provided references:

Training Data:
- 960 hours of LibriSpeech audio for speech pre-training

Training Configuration:
- Batch size: around 90s samples per GPU for speech and 12k tokens per GPU for text 
- Update frequency: 2
- Total training steps: 500k
- Optimizer: Adam with learning rate warmup to 2×10^-4 in the first 8% of updates, then linear decay
- Training hardware: 32 V100 GPUs

Model Architecture:
- Encoder-decoder backbone: 12 Transformer encoder blocks, 6 Transformer decoder blocks
- Model dimension: 768
- Inner dimension (FFN): 3,072 
- Number of attention heads: 12
- Speech-encoder pre-net: 7 blocks of temporal convolutions with 512 channels

Fine-tuning:
- Objective: CTC loss (weight 0.5) + cross-entropy loss (weight 0.5)
- Batch size: up to 256k audio samples per GPU
- Learning rate schedule: warmup for the first 10% steps, constant for the next 40% steps, linear decay for the rest
- Training hardware: 8 V100 GPUs

Evaluation Results:
- Outperforms the state-of-the-art voice Transformer network (VTN) variants on voice conversion in terms of Mel-cepstral distortion (MCD)

[More Information Needed] for the following:
- Throughput 
- Training start/end time
- Checkpoint sizes

The code and model are released at https://github.com/microsoft/SpeechT5.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the SpeechT5 model for voice conversion (microsoft/speecht5_vc) evaluates on the following benchmarks:

1. Voice conversion from speaker bdl to slt and clb to slt, as used in the Voice Transformer Network (VTN) (Reference 8).

The model outperforms the state-of-the-art VTN variants in terms of Mel-Cepstral Distortion (MCD), including:
- VTN fine-tuned from ASR or TTS
- Many-to-many VTN

2. Subjective evaluation metrics (Reference 1):
- Naturalness: 2.91 MOS
- CMOS: +0.29 gain with respect to the baseline model

[More Information Needed] on the specific datasets used for evaluation.

#### Factors

Based on the provided references about the SpeechT5 model and its voice conversion (VC) capabilities in the microsoft/speecht5_vc variant, here are some key characteristics that may influence the model's behavior:

Domain and Context:
- The model was trained on spoken language processing tasks like ASR, TTS, ST, VC, SE and SID. Performance on voice conversion tasks outside this domain is unknown. [More Information Needed] on broader applicability.

- Voice conversion quality was evaluated converting between specific speakers (bdl to slt, clb to slt). Generalization to other speaker pairs or many-to-many conversion is unclear without further testing. [More Information Needed]

Population Subgroups:
- No information was provided on the demographics of speakers used for training data. [More Information Needed] to determine if the model performs equitably across genders, ages, accents, languages and other speaker attributes.

- Disaggregated evaluation metrics were not provided to uncover potential performance disparities. Ideally, voice conversion quality should be broken down by speaker demographics. [More Information Needed]

In summary, more details are required on the training data composition and fine-grained performance analysis to fully characterize foreseeable factors influencing the model's real-world behavior and uncover any concerning biases or disparities. Transparency on these points in the model card would help users make informed decisions about appropriate use cases and limitations.

#### Metrics

Based on the references provided, the key evaluation metrics for the microsoft/speecht5_vc voice conversion model are:

1. MCD (Mel-Cepstral Distortion): A lower MCD indicates better voice conversion performance. The SpeechT5 model outperforms the strong baseline and state-of-the-art Voice Transformer Network (VTN) variants in terms of MCD.

2. WER (Word Error Rate): The converted speech is evaluated using a public ASR model (HuBERT LARGE). The WER of the test set with this ASR model is comparable to that of VTN.

The references mention a tradeoff between MCD and WER, suggesting that optimizing for one metric may impact the other. No specific details are provided on how to balance these metrics.

[More Information Needed] on other potential metrics or tradeoffs considered during the model development and evaluation process.

### Results

Evaluation Results:

Factors:
- Task: Voice Conversion (VC) in a many-to-many setting on the CMU Arctic dataset
- Source and target speakers: clb (female), bdl (male), slt (female), and rms (male)
- Dataset split: 932 utterances for training, 100 for validation, 100 for test per speaker
- Waveform synthesis: Parallel WaveGAN vocoder used

Metrics:
- Mel-Cepstral Distortion (MCD): A smaller MCD indicates better performance. SpeechT5 achieves a significant gain compared to the strong baseline model and outperforms state-of-the-art Voice Transformer Network (VTN) variants.
- Word Error Rate (WER): Evaluated using the public ASR model HuBERT LARGE. The WER of the SpeechT5 model on the test set is comparable to that of VTN.

[More Information Needed] for the exact MCD and WER values.

Fine-tuning Details:
[More Information Needed]

For help or issues using SpeechT5 models, please submit a GitHub issue or contact Long Zhou (lozhou@microsoft.com).

#### Summary

Based on the provided references, here is a summary of the evaluation results for the microsoft/speecht5_vc model:

The proposed SpeechT5 model, which microsoft/speecht5_vc is based on, achieves significant improvements compared to strong baseline models and state-of-the-art Voice Transformer Network (VTN) variants on the Voice Conversion (VC) task. Specifically:

- For converting speaker bdl to slt and clb to slt, SpeechT5 outperforms the baseline model in terms of Mel-Cepstral Distortion (MCD). 

- SpeechT5 also achieves better MCD results than VTN models fine-tuned from ASR or TTS, as well as the many-to-many VTN model.

- In terms of Word Error Rate (WER), SpeechT5 shows improvements over the baseline for bdl to slt conversion. [More Information Needed] for clb to slt WER comparison.

The references demonstrate the effectiveness and superiority of the SpeechT5 pre-training method for the VC task. However, more specific information about the microsoft/speecht5_vc model's performance metrics would be needed to provide additional details in the summary.

## Model Examination

Based on the provided references, here is what I can say about the explainability and interpretability of the microsoft/speecht5_vc model:

The microsoft/speecht5_vc model is a voice conversion model developed using the SpeechT5 framework. SpeechT5 treats various spoken language tasks, including voice conversion, as a speech-to-speech or speech-to-text problem.

The references show that the SpeechT5 voice conversion model significantly outperforms strong baseline models and state-of-the-art Voice Transformer Network (VTN) variants in terms of Mel-Cepstral Distortion (MCD). Specifically, it achieves better voice conversion quality than VTN models fine-tuned from ASR or TTS and many-to-many VTN models.

However, the provided references do not contain specific details on the explainability or interpretability techniques used for the microsoft/speecht5_vc model. [More Information Needed] on how the model's decisions and outputs can be understood or interpreted by humans. Details on any visualization tools, attribution methods, or other explainability approaches applied to this model are not available in the given context.

To improve this section, the model developers would need to provide more information on any efforts undertaken to make the model's behavior more transparent and understandable to users. This could include techniques to attribute importance to input features, visualize learned representations, or generate human-interpretable explanations of the model's predictions.

## Environmental Impact

- **Hardware Type:** The model microsoft/speecht5_vc is trained on V100 GPUs, as mentioned in several places in the provided references:

1. "We pre-train the proposed SpeechT5 model on 32 V100 GPUs with a batch size of around 90s samples per GPU for speech and 12k tokens per GPU for text and set the update frequency to 2 for 500k steps."

2. "We train on 8 V100 GPUs with a batch size of up to 256k audio samples per GPU."

3. "The model is trained on 8 V100 GPUs by the Adam optimizer with a batch size of 16000 tokens per GPU."

7. "We train on 8 V100 GPUs in a speakerindependent manner by using the training data of the LibriTTS."
- **Software Type:** Based on the provided references, the model microsoft/speecht5_vc is trained using the SpeechT5 framework, which consists of a Transformer encoder-decoder model as the backbone network. The references mention fine-tuning the model using CTC loss and cross-entropy loss on 8 V100 GPUs.

However, the specific software type or deep learning framework (such as PyTorch or TensorFlow) used for training the model is not explicitly mentioned in the given references. Therefore, for the software type, the answer would be:

[More Information Needed]
- **Hours used:** Based on the provided references, the SpeechT5 model for voice conversion (microsoft/speecht5_vc) was trained for 120k steps with a batch size of 45,000 tokens per GPU on 8 V100 GPUs (Reference 6). The learning rate was 0.0004, with a 10k step warm-up and inverse square root decay for the remaining steps.

However, the exact amount of time (in hours or days) is not explicitly mentioned in the given references. To provide a more precise answer, [More Information Needed] on the total training time.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider that the model microsoft/speecht5_vc was trained on. More information would be needed to determine this.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain specific information about the amount of carbon emitted during the training of the microsoft/speecht5_vc model. To accurately report the carbon emissions, more details would be needed, such as the total compute time, hardware used, and energy sources. Without this information, it is not possible to provide a reliable estimate of the carbon footprint for this specific model's training process.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The SpeechT5 model consists of an encoder-decoder Transformer backbone and six modal-specific pre/post-nets. The pre-nets convert input speech or text to a unified hidden representation space, which is then fed into the shared encoder-decoder for sequence-to-sequence conversion. The post-nets generate the output in the speech or text modality based on the decoder output.

For voice conversion (VC), the speech-encoder pre-net is the convolutional feature extractor from wav2vec 2.0, which downsamples raw waveforms. The speech-decoder pre-net is composed of fully connected layers with ReLU activation, fed with log Mel-filterbank features. Speaker embeddings from x-vectors are concatenated to support multi-speaker VC. The speech-decoder post-net consists of a linear layer to predict log Mel-filterbanks, followed by convolutional layers for refinement, and a linear module to predict the stop token.

Objective:
The objective of the microsoft/speecht5_vc model is to perform voice conversion, converting speech from one speaker to another. It is fine-tuned from the pre-trained SpeechT5 model, which is jointly pre-trained on large-scale speech and text data using a unified-modal framework. The model aims to outperform state-of-the-art baselines in terms of metrics like Mel-Cepstral Distortion (MCD).

[More Information Needed] on specific training details, datasets used, and evaluation results for the microsoft/speecht5_vc model.

### Compute Infrastructure

Based on the references provided, here is the compute infrastructure information I could find about the microsoft/speecht5_vc model:

The model was trained on 8 NVIDIA V100 GPUs with a batch size of 16000 tokens per GPU. The Adam optimizer was used with a learning rate based on the inverse square root, with a maximum learning rate of 10^-4 within 100k steps and 10k warm-up steps.

[More Information Needed] on the exact model architecture, total training time, dataset sizes used, and other infrastructure details.

## Citation

```
@misc{junyi-speecht,
    author = {Junyi Ao and
              Rui Wang and
              Long Zhou and
              Chengyi Wang and
              Shuo Ren and
              Yu Wu and
              Shujie Liu and
              Tom Ko and
              Qing Li and
              Yu Zhang and
              Zhihua Wei and
              Yao Qian and
              Jinyu Li and
              Furu Wei and
              Peng Cheng Laboratory},
    title  = {SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing},
    url    = {https://arxiv.org/pdf/2110.07205.pdf}
}
```

