# Model Card for openai/whisper-base.en

openai/whisper-base.en is an English-only speech recognition model based on the Whisper architecture. It is a general-purpose model trained on a large dataset of diverse audio and can be used for English speech recognition tasks.

## Model Details

### Model Description

Model Details for openai/whisper-base.en:

Model Architecture:
- Encoder-decoder Transformer architecture
- Audio is resampled to 16kHz, converted to 80-channel log-mel spectrogram
- Encoder processes input with 2 conv layers, sinusoidal position embeddings, and transformer blocks
- Decoder uses learned position embeddings and tied input-output token representations
- Encoder and decoder have same width and number of transformer blocks

Training Procedures: 
- Trained on 680,000 hours of diverse labeled audio data
- Multitask training on speech recognition, translation, language ID, voice activity detection
- Tasks jointly represented as token sequences for decoder to predict
- Uses special tokens as task specifiers
- Raw text transcripts used without significant standardization
- [More Information Needed] on other training details like optimizer, learning rate, etc.

Model Parameters:
- [More Information Needed]

Important Disclaimers:
- Model may generate plausible but incorrect speaker names in transcripts
- [More Information Needed] on other potential biases, limitations or risks

The Whisper-base.en model is a general-purpose English speech recognition model that is part of the larger multilingual Whisper model family. It demonstrates strong zero-shot generalization abilities that improve with model size. However, more specific details on model parameters, training hyperparameters, and comprehensive analysis of limitations are needed to complete the model card. Please reach out if you require any additional information to be included.

- **Developed by:** Alec Radford; Jong Wook Kim; Tao Xu; Greg Brockman; Christine Mcleavey; Ilya Sutskever
- **Funded by:** Based on the provided references, the model openai/whisper-base.en was developed by OpenAI. The references mention the following people and teams at OpenAI:

- Nick Ryder, Will Zhuk, and Andrew Carr for the conversation that inspired the project
- The Acceleration and Supercomputing teams at OpenAI for their work on software and hardware infrastructure 
- Pamela Mishkin for advising the project from a policy perspective
- Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever

However, there is no explicit mention of the specific funding sources or organizations for this project. More information would be needed to definitively list the funders.

[More Information Needed]
- **Shared by:** Based on the provided references, the main contributors who made the model openai/whisper-base.en available online as a GitHub repo are:

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever from OpenAI.

The references also mention thanking the Acceleration and Supercomputing teams at OpenAI for their work on software and hardware infrastructure used in the project, as well as Pamela Mishkin for advising from a policy perspective. However, it's unclear if they were directly involved in making the model available on GitHub.

[More Information Needed] on additional contributors, if any, who were specifically involved in publishing the GitHub repo.
- **Model type:** The openai/whisper-base.en model is a supervised encoder-decoder Transformer model trained on a large dataset of diverse audio for multilingual speech recognition, speech translation, and language identification.
- **Language(s):** The openai/whisper-base.en model is an English-only speech recognition model that performs better than the multilingual base model for English speech transcription tasks.
- **License:** Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.
- **Finetuned from model:** Based on the provided references, the model openai/whisper-base.en does not appear to be fine-tuned from another model. The references mention training a suite of models of various sizes from scratch to study the scaling properties of Whisper (Reference 3). There is no indication that the base model was fine-tuned from a pre-existing model.

[More Information Needed] on whether openai/whisper-base.en was fine-tuned from another model. The provided references do not contain this information.
### Model Sources

- **Repository:** https://github.com/openai/whisper
- **Paper:** https://arxiv.org/pdf/2212.04356.pdf
- **Demo:** Here is the model card description for openai/whisper-base.en based on the provided references:

Model name: openai/whisper-base.en

Link to demo: [More Information Needed]

The references do not provide a direct link to a demo of the openai/whisper-base.en model. The Colab example notebook linked is for demonstrating usage on the LibriSpeech dataset in general, but is not a hosted demo of this specific model. More information would be needed to include a link to an interactive online demo of openai/whisper-base.en.

Please let me know if you need any other details filled out for the model card. As the project organizer, I'm happy to track down additional information as needed or put you in touch with the relevant team members who can provide more specifics. We're excited to publish this model to the community on Hugging Face.
## Uses

### Direct Use

The openai/whisper-base.en model can be used for English speech recognition without any fine-tuning, post-processing or complex pipeline. It can be used directly in Python like this:

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3") 
print(result["text"])
```

This code snippet loads the base model, transcribes the audio from the "audio.mp3" file, and prints out the recognized text.

The model can also be used from the command line to transcribe English audio files:

whisper audio.flac audio.mp3 audio.wav --model medium

This command will transcribe the specified audio files using the medium sized model. The default small model also works well for English transcription.

[More Information Needed] on using the model for other languages, translation, or language identification. The model card focuses on its strong English speech recognition capabilities that work out-of-the-box.

### Downstream Use

The openai/whisper-base.en model can be fine-tuned or integrated into a larger speech processing pipeline in several ways:

1. It can perform end-to-end speech recognition, transcribing audio into text. This can be done in Python:

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3") 
print(result["text"])
```

2. The model provides lower-level access through `whisper.detect_language()` and `whisper.decode()` methods for more granular control.

3. Whisper aims to be a single model handling the entire pipeline, reducing the complexity of separate components like voice activity detection, speaker diarization, etc. 

4. [More Information Needed]

5. [More Information Needed]  

6. Audio can be loaded and preprocessed using utility functions:

```python
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)
```

7. The model is trained to directly predict raw transcripts without significant standardization, simplifying the pipeline.

8. [More Information Needed]

9. Fine-tuning on high-quality supervised data and reinforcement learning could help reduce errors like incorrect speaker names or hallucinated text.

In summary, openai/whisper-base.en provides a powerful and flexible foundation for speech recognition tasks that can be used directly or integrated into larger systems with some additional fine-tuning and optimization.

### Out-of-Scope Use

Based on the provided references about the Whisper model, here are some potential ways the model openai/whisper-base.en may foreseeably be misused and what users ought not do with it:

- The model may hallucinate or generate transcripts entirely unrelated to the actual audio input in some cases, especially on the first/last few words (ref 1). Users should not rely on it for applications requiring high accuracy on every single word.

- The model's performance degrades significantly with noisy audio (ref 4, 5). Users should not expect it to work reliably in very noisy environments without further fine-tuning. 

- The model has a tendency to make incorrect guesses about speaker names when they are not inferable from the audio (ref 8). Users should not assume the model can accurately identify speakers just from voice.

- [More Information Needed] on other specific ways the model may be intentionally misused, such as for generating deceptive or inappropriate content. The references focus more on robustness limitations than deliberate misuse.

In general, users should be aware of the model's limitations, not rely on it for high-stakes applications without further testing and adaptation, and use it responsibly in accordance with ethical AI principles. More targeted testing would help identify additional foreseeable misuses to call out.

### Bias, Risks, and Limitations

Based on the provided references about the Whisper speech recognition model, here are some known or foreseeable issues, limitations, and potential harms:

Technical Limitations:
1. Performance on lower-resource languages is still quite poor, likely due to the English-centric training data. Targeted efforts to increase data for rarer languages could significantly improve average speech recognition performance. (Reference 6)

2. The model struggles with certain stubborn error types in long-form transcription, such as getting stuck in repeat loops, not transcribing the first or last few words, or hallucinating unrelated transcripts. Fine-tuning on high-quality supervised data or using reinforcement learning could help reduce these errors. (Reference 3)

3. The model has a tendency to transcribe plausible but incorrect speaker names, as this information is rarely inferable from the audio alone. (Reference 7)

Sociotechnical Issues and Potential Harms:
1. [More Information Needed] The references do not directly discuss sociotechnical issues or potential harms.

Misunderstandings and Unknowns:
1. It's unclear to what degree the benefits of Whisper stem from training its encoder, decoder, or both. Further studies could help clarify this by ablating design components or comparing with existing speech recognition encoders. (Reference 1)

2. The robustness properties of Whisper have been the focus, so the potential improvements from fine-tuning on high-quality supervised data for specific domains are unknown and require further study. (Reference 5)

To summarize, the main known issues are technical limitations related to performance on low-resource languages, certain stubborn error types, and incorrect speaker name transcription. More information is needed to assess potential sociotechnical issues and harms. Key areas for further study include understanding the relative contributions of model components and the effects of fine-tuning.

### Recommendations

Based on the provided references about the Whisper model, here are some recommendations regarding foreseeable issues with the openai/whisper-base.en model:

1. The model's performance on lower-resource languages could be improved by increasing the amount of training data for those languages, as the current pre-training dataset is heavily biased towards English. [More Information Needed] on specific plans to source and incorporate more diverse language data.

2. The model may struggle with certain types of errors in long-form transcription, such as getting stuck in repeat loops, failing to transcribe the beginning or end of audio segments, or hallucinating unrelated transcripts. Fine-tuning on high-quality supervised datasets or using reinforcement learning could help reduce these errors. [More Information Needed] on plans to address these specific error types.

3. The model has a tendency to make plausible but incorrect guesses for speaker names, as this information is rarely inferable from the audio alone. [More Information Needed] on potential mitigation strategies for this issue.

4. The model's performance varies widely depending on the language, with notably lower accuracy on many non-English languages. Refer to the WER/CER breakdown by language in the paper's Appendix D for more details. [More Information Needed] on plans to improve performance on the lowest-scoring languages.

5. [More Information Needed] on any fine-tuning efforts or direct comparisons with other models, which could provide valuable insights into the model's strengths and weaknesses.

In summary, key areas to monitor and potentially address include language bias in training data, specific error types in long-form transcription, incorrect speaker labeling, and performance disparities across different languages. More information is needed on concrete plans to tackle these issues and further validate the model's performance.

## Training Details

### Training Data

The training data for openai/whisper-base.en consists of a large dataset of diverse audio transcripts sourced primarily from English-centric parts of the internet, with minimal data pre-processing and filtering of machine-generated transcripts. [More Information Needed] on the specifics of data pre-processing and additional filtering steps.

### Training Procedure

#### Preprocessing

For the openai/whisper-base.en model, the preprocessing steps include:

1. Text preprocessing:
   - Minimal standardization of transcripts, relying on the model to learn the mapping between utterances and their transcribed form. [Reference 1]
   - Text normalization for English texts, including handling punctuation, formatting, and stylistic aspects. [Reference 3]
   - Removing machine-generated transcripts using heuristics to avoid learning "transcript-ese". [References 4, 5]

2. Audio preprocessing:
   - Breaking audio files into 30-second segments paired with the corresponding transcript subset. [Reference 6]
   - Training on all audio segments, including those without speech, for voice activity detection. [Reference 6]
   - Converting audio to log mel spectrogram representation:
     ```python
     mel = whisper.log_mel_spectrogram(audio).to(model.device)
     ```
     [Reference 7]

3. Language detection:
   - Using an audio language detector to ensure the spoken language matches the transcript language. [Reference 2]
   - Including English transcripts even if the audio language doesn't match. [Reference 2]

[More Information Needed] on specific details of tokenization and resizing/rewriting.

The model can be used for transcription within Python:

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])
```

The `transcribe()` method processes the audio with a sliding 30-second window and performs autoregressive sequence-to-sequence predictions on each window. [Reference 8]

Lower-level access to the model is provided through `whisper.detect_language()` and `whisper.decode()` functions. [Reference 8]

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the openai/whisper-base.en model:

- Optimizer: AdamW
- Gradient clipping: Gradient norm clipping was used
- Learning rate schedule: Linear learning rate decay to zero after a warmup over the first 2048 updates
- Batch size: 256 segments
- Training duration: Trained for 220 updates, which is between two and three passes over the dataset
- Precision: Trained with FP16 and dynamic loss scaling
- Other techniques used: Data parallelism across accelerators, activation checkpointing

[More Information Needed] on the exact learning rate, weight decay, warmup steps, and other hyperparameters not directly mentioned in the references.

#### Speeds, Sizes, Times

Based on the provided references, here are the details I can provide about the openai/whisper-base.en model:

Model Architecture:
- Encoder-decoder Transformer architecture
- Audio is re-sampled to 16,000 Hz
- 80-channel log-magnitude Mel spectrogram representation is computed on 25-millisecond windows with a stride of 10 milliseconds

Training:
- Trained with AdamW optimizer and gradient norm clipping
- Linear learning rate decay to zero after a warmup over the first 2048 updates
- Batch size of 256 segments was used
- Trained for 2^20 updates, which is between two and three passes over the dataset
- No data augmentation or regularization used during training

Performance:
- The base.en model performs well for English-only applications
- Difference in performance compared to multilingual base model becomes less significant for larger model sizes

[More Information Needed] for the following:
- Throughput
- Start or end time of training
- Checkpoint sizes

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the openai/whisper-base.en model was evaluated on the following benchmarks and datasets:

Short-form English-only datasets:
- LibriSpeech test-clean and test-other splits
- TED-LIUM 3 test split
- Common Voice 5.1 English subset
- Artie bias corpus (subset of Common Voice)
- CallHome and Switchboard corpora
- WSJ corpus
- CORAAL interviews

Other English datasets:
- CHiME-6 binaural recordings
- AMI-IHM and AMI-SDM1 corpora
- Rev16 (subset of 16 files from Rev.AI's Podcast Transcription Benchmark)

Multilingual benchmarks:
- Multilingual LibriSpeech (MLS)
- VoxPopuli

The model's performance was also broken down by language using WER/CER metrics on the Common Voice 15 and Fleurs datasets. Additional detailed results can be found in Appendix D.1, D.2, and D.4 of the referenced paper.

[More Information Needed] on the specific training details and hyperparameters used for the openai/whisper-base.en model.

#### Factors

Based on the provided references about the Whisper speech recognition model, here are some key characteristics that could influence the behavior and performance of the openai/whisper-base.en model:

Domain and Context:
- Performance degrades with increasing levels of background noise, especially compared to models trained primarily on clean datasets like LibriSpeech (Reference 5)
- Robustness likely stems in part from having a strong decoder that is an audio conditional language model (Reference 2)
- Many remaining errors in long-form transcription are stubborn and non-human/perceptual in nature, like getting stuck in loops or hallucinating unrelated text (References 1, 3)

Population Subgroups:
- Performance on a given language is strongly predicted by the amount of training data for that language (Reference 4) 
- Current pre-training data is very English-heavy, so the model likely performs much worse on lower-resource languages that have less than 1000 hours of training data (References 4, 6)
- [More Information Needed] on detailed performance breakdowns and disparities across different languages, accents, demographics

Evaluation:
- Using a text normalizer in evaluation to reduce penalties for innocuous, non-semantic differences in transcripts (Reference 8)
- Some evaluation datasets may not accurately reflect robustness if commercial models were trained on them (Reference 10)
- Performance varies widely by language - a breakdown is provided for the large models on Common Voice and Fleurs datasets (Reference 11)
- [More Information Needed] on disaggregated evaluations across key subgroups to uncover disparities

In summary, key factors that likely influence openai/whisper-base.en's behavior are the noise level and length of the input audio, the amount of training data for the particular language, and quirks of the evaluation datasets and metrics used. More disaggregated evaluations are needed to fully characterize performance disparities across subgroups.

#### Metrics

Based on the references provided, the key metrics and considerations for evaluating the openai/whisper-base.en model are:

- Word Error Rate (WER) is the primary metric typically used in speech recognition research to compare systems. However, WER has limitations as it penalizes all differences between model output and reference transcripts, including innocuous formatting and style differences. This is particularly challenging for zero-shot models like Whisper that don't observe dataset-specific transcript formats. (References 1-2)

- To address WER limitations, the team developed a text normalizer to standardize transcripts before WER calculation and minimize penalization of non-semantic differences. This normalizer was iteratively refined through manual inspection. The code for the normalizer is being released to enable fair comparisons. (References 2-3, 5) 

- Character Error Rate (CER) is used as an additional metric for some languages. (Reference 9)

- To assess room for improvement, human performance levels were estimated by obtaining professional transcripts for a subset of the Kincaid46 dataset. Distributions of WER from Whisper, commercial services, and the best open-source model were compared. (References 10-11)

In summary, the primary metrics are WER and CER, with text normalization applied to transcripts to enable fairer comparisons in light of Whisper's zero-shot nature. Human performance levels help indicate headroom for further model improvements. [More Information Needed] on the specific WER/CER results for openai/whisper-base.en.

### Results

Evaluation Results of openai/whisper-base.en:

Performance Metrics:
- Word Error Rate (WER) on English speech recognition: The model achieves close to human-level performance on English speech recognition tasks, with diminishing returns on performance improvements as model size increases beyond the base size. (Reference 1)
- [More Information Needed] on exact WER values for openai/whisper-base.en.
- Performance on multilingual speech recognition, speech translation, and language identification continues to improve with larger model sizes beyond the base model. (Reference 1)
- Compared to a supervised model with similar performance on LibriSpeech test-clean, the zero-shot Whisper base model achieves an average 55.2% relative error reduction when evaluated on other speech recognition datasets. (Reference 8)

Evaluation Datasets:
- LibriSpeech (Reference 8)
- Common Voice 15 (Reference 9)
- Fleurs (Reference 9)
- [More Information Needed] on additional evaluation datasets and results.

Factors Impacting Performance:
- Dataset size: Performance improves with increasing dataset size, but shows diminishing returns beyond the 680,000 hours used for training the Whisper models. (References 3, 4, 5)
- Language: Performance varies widely depending on the language. WER and CER breakdowns by language are available for some datasets in the paper. (Reference 9)
- [More Information Needed] on other factors impacting performance.

Comparisons:
- Outperforms open-source models like NVIDIA STT Conformer-CTC Large. (Reference 10)
- [More Information Needed] on detailed comparisons to the 4 commercial ASR services mentioned. (Reference 10)

Methodology Notes:
- Extensive text normalization is performed before WER calculation to minimize penalization of zero-shot models like Whisper. (Reference 6, 7)
- Human performance estimates were obtained on a subset of the Kincaid46 dataset to assess room for improvement. (Reference 11)

[More Information Needed] on additional evaluation details for the openai/whisper-base.en model specifically. The references provide general insights about the Whisper model family, but more details on the base.en model's evaluation are needed to comprehensively answer the question.

#### Summary

Based on the provided references about the Whisper model, here is a summary of the evaluation results for the openai/whisper-base.en model:

- Performance continues to increase with model size across multilingual speech recognition, speech translation, and language identification tasks, with some diminishing returns for English speech recognition possibly due to approaching human-level performance. (Reference 1)

- Whisper models are evaluated in a zero-shot manner on a wide set of existing speech processing datasets to test generalization across domains, tasks, and languages. (Reference 2) 

- Whisper's performance varies widely by language, with performance on a language being well predicted by the amount of training data available for that language. (Reference 8, 9)

- On low-data multilingual speech recognition benchmarks like Multilingual LibriSpeech (MLS) and VoxPopuli, Whisper's performance is reported but these benchmarks provide limited coverage of Whisper's multilingual capabilities which include training data for 75 languages. (Reference 10)

- Compared to other open-source and commercial ASR models, Whisper performs better on most datasets, especially those with uncommon words. Some commercial models may have been trained on the evaluation datasets though. (Reference 6, 7)

[More Information Needed] on the specific performance metrics for the openai/whisper-base.en model size. The references discuss Whisper models in general but do not provide results broken down by the specific base.en model.

## Model Examination

Based on the provided references about the Whisper model, here is what I can say about the explainability and interpretability of the openai/whisper-base.en model that our team has worked on:

The Whisper model's robustness seems to be partially due to its strong decoder, which is an audio conditional language model. However, it's currently unclear to what degree the benefits come from training the encoder, decoder, or both. Studying this further by ablating design components or using Whisper's decoder with other encoders could provide more interpretability insights. [More Information Needed]

Whisper still makes some stubborn errors in long-form transcription that seem decidedly non-human/perceptual, like getting stuck in loops, skipping words, or hallucinating unrelated transcripts. Fine-tuning on high-quality datasets or using reinforcement learning to optimize decoding could help explain and reduce these failure modes. [More Information Needed]

Whisper's performance varies significantly by language, with much lower accuracy for lower-resource languages. This is likely due to the English-centric pre-training data. Increasing training data for other languages is a clear path to improving explainability of the model's language-specific capabilities. [More Information Needed]

In summary, while the Whisper model shows impressive robustness and performance, there are still open questions around the relative contributions of different model components, stubborn error modes, and language-specific capabilities that require further research to fully explain and interpret the model's behaviors. Our team will continue to study these areas to improve the model's interpretability.

## Environmental Impact

- **Hardware Type:** [More Information Needed]

The provided references do not specify the hardware type that the model openai/whisper-base.en was trained on. The references mention using data parallelism across accelerators and FP16 precision, but do not explicitly state whether GPUs, TPUs, or other hardware was used for training this specific model.
- **Software Type:** Based on the provided references, the model openai/whisper-base.en is trained using:

- Data parallelism across accelerators using FP16 with dynamic loss scaling and activation checkpointing
- AdamW optimizer
- Gradient norm clipping
- Linear learning rate decay to zero after a warmup over the first 2048 updates
- Batch size of 256 segments
- Trained for 2^20 updates (between two and three passes over the dataset)

The references do not explicitly mention the specific software or deep learning framework used for training the model. [More Information Needed] on the exact software stack and libraries used in the training process.
- **Hours used:** Based on the information provided in the references, the Whisper models were trained for 2^20 updates, which is between two and three passes over the dataset (Reference 2). However, the exact amount of time used to train the specific model openai/whisper-base.en is not explicitly mentioned.

[More Information Needed] on the precise training time for the openai/whisper-base.en model.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the openai/whisper-base.en model. More information would be needed from the model developers or documentation to determine which cloud provider, if any, was utilized during training.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the openai/whisper-base.en model. To answer this question, more specific information would be needed, such as the compute resources used for training, the duration of the training process, and the energy sources powering the training infrastructure.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The openai/whisper-base.en model uses an encoder-decoder Transformer architecture. Key components include:

- Audio is resampled to 16kHz and converted to an 80-channel log-mel spectrogram using a 25ms window and 10ms stride
- Input is normalized to [-1, 1] with ~0 mean across the pretraining data
- Encoder has a small 2-layer CNN stem with stride 2 in the second layer, followed by sinusoidal positional embeddings and Transformer blocks
- Encoder and decoder have the same width and number of blocks
- Decoder uses learned positional embeddings and tied input-output token embeddings
- Pre-activation residual blocks are used in the Transformer
- Final layer norm is applied to the encoder output

Model Objective:
The openai/whisper-base.en model is trained on a multitask mixture of speech processing tasks, including:

- Multilingual speech recognition 
- Speech translation
- Spoken language identification
- Voice activity detection

These tasks are jointly represented as a sequence of tokens to be predicted by the decoder. Special tokens are used to specify the task or classification targets. The model is trained on a large 680,000 hour dataset of transcribed audio from the internet with the goal of learning general speech representations that can be used for zero-shot transfer to downstream tasks without finetuning.

[More Information Needed] on the exact pretraining data composition, training hyperparameters, and performance benchmarks for this specific base.en model variant.

### Compute Infrastructure

Based on the provided references, here is the available information about the compute infrastructure used for training the openai/whisper-base.en model:

The models were trained using:
- Data parallelism across accelerators
- FP16 precision with dynamic loss scaling
- Activation checkpointing

The optimizer and training hyperparameters:
- AdamW optimizer
- Gradient norm clipping
- Linear learning rate decay to zero after a warmup over the first 2048 updates
- Batch size of 256 segments
- Trained for 2^20 updates (between two and three passes over the dataset)

[More Information Needed] on the specific hardware (GPU/TPU models and counts) used for training.

The references do not include code blocks directly relevant to the compute infrastructure.

## Citation

```
@misc{alec-robust,
    author = {Alec Radford and
              Jong Wook Kim and
              Tao Xu and
              Greg Brockman and
              Christine Mcleavey and
              Ilya Sutskever},
    title  = {Robust Speech Recognition via Large-Scale Weak Supervision},
    url    = {https://arxiv.org/pdf/2212.04356.pdf}
}
```

