# Model Card for openai/whisper-base

openai/whisper-base is a general-purpose speech recognition model that is trained on a large dataset of diverse audio. It is a multitasking model capable of multilingual speech recognition, speech translation, and language identification.

## Model Details

### Model Description

Model Details for openai/whisper-base:

Model Architecture:
- Encoder-decoder Transformer architecture
- Audio is re-sampled to 16,000 Hz
- 80-channel log-magnitude Mel spectrogram representation is computed on 25-millisecond windows with a stride of 10
- Pre-activation residual blocks are used in the Transformer
- Final layer normalization is applied to the encoder output
- Decoder uses learned position embeddings and tied input-output token representations
- Encoder and decoder have the same width and number of transformer blocks

Training Procedures:
- Trained on a large dataset of 680,000 hours of labeled audio
- No data augmentation or regularization is used, relying on dataset diversity for generalization and robustness
- Trained for 2^20 updates, which is between two and three passes over the dataset
- Early stopping based on validation loss is used

Parameters:
[More Information Needed]

Important Disclaimers:
- Whisper models may have a tendency to transcribe plausible but incorrect guesses for speaker names, as this information is rarely inferable from the most recent 30 seconds of audio alone

Additional Notes:
- Whisper is a general-purpose speech recognition model trained on a diverse audio dataset
- It is a multitasking model capable of multilingual speech recognition, speech translation, and language identification

For more details on the model architecture, please refer to Figure 1 in the provided references.

- **Developed by:** Alec Radford; Jong Wook Kim; Tao Xu; Greg Brockman; Christine Mcleavey; Ilya Sutskever
- **Funded by:** Based on the provided references, the model openai/whisper-base was developed by OpenAI. The references mention the Acceleration and Supercomputing teams at OpenAI and their critical work on software and hardware infrastructure used in this project. However, there is no explicit information about the funding sources for this specific model.

[More Information Needed]
- **Shared by:** Based on the provided references, the main contributors who made the model openai/whisper-base available online as a GitHub repo are:

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever from OpenAI.

The references also mention thanking the Acceleration and Supercomputing teams at OpenAI for their work on the software and hardware infrastructure used in the project, as well as Pamela Mishkin for advising from a policy perspective. However, it is not explicitly stated if they directly contributed to making the model available on GitHub.

[More Information Needed] on additional contributors, if any, who were directly involved in publishing the model to the GitHub repository.
- **Model type:** The openai/whisper-base model is a supervised encoder-decoder Transformer model trained on a large dataset of diverse audio for speech recognition, speech translation, and language identification.
- **Language(s):** The openai/whisper-base model can perform multilingual speech recognition, speech translation, and language identification across 75 languages.
- **License:** Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.
- **Finetuned from model:** Based on the provided references, the Whisper model does not appear to be fine-tuned from another model. The references describe Whisper as "a general-purpose speech recognition model" that is "trained on a large dataset of diverse audio" and leverages "web-scale text from the internet for training." There is no mention of Whisper being fine-tuned from a pre-existing base model.

[More Information Needed] to definitively state whether or not Whisper is fine-tuned from another model. The provided references do not contain enough information to make that determination.
### Model Sources

- **Repository:** https://github.com/openai/whisper
- **Paper:** https://arxiv.org/pdf/2212.04356.pdf
- **Demo:** [More Information Needed]

The references provided do not contain a direct link to a demo of the openai/whisper-base model. More information would be needed to definitively answer this question.
## Uses

### Direct Use

The openai/whisper-base model can be used for speech recognition without any fine-tuning, post-processing or complex pipeline. It is designed as a general-purpose model that works well across many domains, tasks and languages out of the box.

To use the model, you can simply load it and call the transcribe() method with an audio file:

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3") 
print(result["text"])
```

This will read the audio file, process it using a sliding window, and output the transcribed text. 

The model can also detect the spoken language automatically. To get more low-level control, you can use the detect_language() and decode() methods instead of transcribe().

Additionally, you can run the model from the command line to transcribe audio files:

whisper audio.flac audio.mp3 audio.wav --model base

To transcribe non-English speech, add the --language flag. And to translate the speech to English, add --task translate.

So in summary, the openai/whisper-base model is very flexible and easy to use for speech recognition tasks without requiring dataset-specific fine-tuning or complex post-processing steps. The sensible defaults make it suitable for many applications out-of-the-box.

### Downstream Use

The openai/whisper-base model can be fine-tuned or integrated into a larger system in several ways:

1. Fine-tuning for improved performance on specific tasks like long-form transcription, by training on a high-quality supervised dataset and/or using reinforcement learning. This could help reduce errors like repeat loops, missing first/last words, or hallucination. [More Information Needed] for code snippet.

2. Integrating into a speech processing pipeline as a single model that performs the entire pipeline, rather than just the core speech recognition. This simplifies the system architecture. The model uses a simple token format to specify tasks like transcription, translation, voice activity detection, alignment, and language identification. [More Information Needed] for code snippet.

3. Using the model's lower-level `whisper.detect_language()` and `whisper.decode()` methods to incorporate it into custom applications, in addition to the higher-level `transcribe()` method. Here's an example of using the model directly in Python:

```python
import whisper

model = whisper.load_model("base") 
result = model.transcribe("audio.mp3")
print(result["text"])
```

4. Leveraging the model's multitasking abilities (multilingual speech recognition, speech translation, language identification) to enable rich functionality in applications. [More Information Needed] for code snippet.

In summary, the openai/whisper-base model provides flexibility to improve performance via fine-tuning, simplify speech processing pipelines, access low-level model functionality for custom use cases, and utilize its multitasking capabilities. More specific code snippets would depend on the particular integration or fine-tuning approach.

### Out-of-Scope Use

Based on the provided references about the Whisper model, here are some potential ways the model could be misused and what users should avoid doing:

- Users should not rely on Whisper for transcribing sensitive information like names or personal details, as the model has a tendency to make plausible but incorrect guesses for names of speakers based on its training data. [Reference 7]

- Whisper should not be used in high-stakes domains requiring near-perfect accuracy, as it still makes some stubborn errors in long-form transcription like getting stuck in loops or hallucinating unrelated text. Further work is needed to reduce these failure modes. [References 2, 4] 

- Users should be cautious about applying Whisper to low-resource languages, as the model's performance is still quite poor for many languages that had less training data. Increasing language-specific training data is an important direction for improving the model. [Reference 3]

- As a general-purpose model, Whisper should not be used without further testing and potential fine-tuning for specialized domains where high-quality supervised data exists, as this could yield better performance than the base model. [References 5, 8]

[More Information Needed] on other specific ways the model could be misapplied or abused given its current capabilities and limitations. A deeper analysis of potential societal impacts and failure modes would be beneficial to include.

### Bias, Risks, and Limitations

Based on the provided references about the Whisper model, here are some known or foreseeable issues, limitations, and potential harms:

Technical Limitations:
1. Poor performance on low-resource languages due to English-centric training data (Reference 1)
2. Remaining errors in long-form transcription, such as getting stuck in repeat loops, missing first/last words, or hallucinating unrelated transcripts (Reference 3)
3. Diminishing returns in English speech recognition performance as model size increases, possibly due to approaching human-level saturation (Reference 7)

Sociotechnical Limitations:
1. Tendency to transcribe plausible but incorrect speaker names, as this information is rarely inferable from the audio alone (Reference 6)
2. [More Information Needed] on potential biases or fairness issues arising from the English-centric training data

Potential Misunderstandings:
1. Users may overestimate the model's ability to accurately transcribe low-resource languages or infer speaker identity
2. [More Information Needed] on how users might misinterpret or misuse the model's outputs

Foreseeable Harms:
1. [More Information Needed] on potential privacy risks from transcribing sensitive audio data
2. [More Information Needed] on how transcription errors or hallucinations could lead to harmful consequences in high-stakes applications

To address these issues, the references suggest:
1. Increasing training data for low-resource languages (Reference 1) 
2. Studying the impact of encoder/decoder design choices and fine-tuning (References 2, 5)
3. Improving decoding strategies to reduce errors (Reference 4)
4. Releasing text normalization code for better comparison and out-of-distribution testing (Reference 8)

However, more information is needed to fully assess potential biases, misunderstandings, privacy risks, and high-stakes harms. Ongoing testing, monitoring, and stakeholder engagement will be important for responsible development and deployment.

### Recommendations

Based on the provided references about the Whisper speech recognition model, here are some recommendations regarding foreseeable issues with the openai/whisper-base model:

1. Performance on low-resource languages could be improved by increasing the amount of training data for those languages, as the current dataset is English-heavy. More diverse data collection is needed.

2. [More Information Needed] on the specific contributions of the encoder vs. decoder to Whisper's robustness. Ablation studies or experiments with other encoders could provide insights.

3. Decoding errors like looping, skipping words, or hallucination are still issues, especially on long-form transcription. Fine-tuning on high-quality datasets or using reinforcement learning techniques may help reduce these.

4. [More Information Needed] on whether incorporating unsupervised pre-training or self-teaching methods used in other state-of-the-art models could further improve Whisper's performance.

5. Fine-tuning Whisper on specific domains where supervised data is available is likely to boost results and enable direct comparisons to other models. More studies are needed on fine-tuning.

6. The model has a tendency to make incorrect guesses about speaker names, as this information is rarely inferable from the audio alone. [More Information Needed] on mitigation strategies.

7. While performance continues to scale with model size in most tasks, diminishing returns are seen for English ASR, possibly due to saturation as human-level performance is approached. Analyzing scaling trends can inform efficient model sizing.

8. Whisper aims to be a fully-featured recognition system, but some components like voice activity detection, speaker diarization and inverse text normalization may still need to be handled separately. [More Information Needed] on Whisper's capabilities in these areas and opportunities for a more integrated system.

In summary, key issues include low-resource language performance, decoding errors, analyzing the impact of different architectural components and training techniques, studying fine-tuning behavior, mitigating incorrect speaker labeling, model scaling efficiency, and expanding Whisper's capabilities as an integrated speech recognition system. Collecting more diverse data, conducting ablation studies, and exploring techniques used in other state-of-the-art models could help address these issues.

## Training Details

### Training Data

The training data for the openai/whisper-base model consists of a large dataset of 680,000 hours of diverse and multilingual audio data collected from the internet. The data underwent minimal pre-processing and filtering to remove low-quality and machine-generated transcripts, and to ensure the spoken language matches the transcript language.

[More Information Needed] for links to documentation related to data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

Here are the details about preprocessing for the openai/whisper-base model, based on the provided references:

Tokenization:
[More Information Needed]

Audio Preprocessing:
- All audio is re-sampled to 16,000 Hz
- An 80-channel log-magnitude Mel spectrogram representation is computed on 25-millisecond windows with a stride of 10 milliseconds

Text Preprocessing:
- Whisper models are trained to predict the raw text of transcripts without any significant standardization, relying on the expressiveness of sequence-to-sequence models to learn the mapping between utterances and their transcribed form (Reference 1)
- For English texts, the following normalization steps are performed to standardize the text into a consistent form (Reference 4):
  [More Information Needed]
- The text normalizer is released to allow for easy comparison and to help others study the performance of speech recognition systems in out-of-distribution settings (Reference 10)

Data Filtering:
- An audio language detector, fine-tuned on VoxLingua107, is used to ensure that the spoken language matches the language of the transcript according to CLD2. If the languages do not match, the (audio, transcript) pair is not included as a speech recognition training example, with an exception for English transcripts (Reference 2)
- Audio files are broken into 30-second segments paired with the corresponding transcript subset (Reference 3)
- Several automated filtering techniques were developed to address subpar transcripts in the raw dataset (Reference 6)

#### Training Hyperparameters

Based on the provided references, here are the training hyperparameters for the openai/whisper-base model:

- Data parallelism across accelerators using FP16 with dynamic loss scaling and activation checkpointing was used for training (Reference 1)
- Optimizer: AdamW (Reference 1) 
- Gradient norm clipping was used (Reference 1)
- Learning rate schedule: Linear learning rate decay to zero after a warmup over the first 2048 updates (Reference 1)
- Batch size: 256 segments (Reference 1)
- Number of training updates: 2^20 updates, which is between two and three passes over the dataset (Reference 1)
- No data augmentation or regularization was used, relying on the diversity of the large dataset to encourage generalization and robustness (Reference 2)

[More Information Needed] on the exact values used for:
- Learning rate 
- Weight decay
- Warmup steps
- Gradient clipping threshold
- Activation checkpointing configuration

The model was briefly fine-tuned on a subset of transcripts that do not include speaker annotations to remove the behavior of incorrectly guessing speaker names (Reference 5).

#### Speeds, Sizes, Times

Based on the provided references, here are the details I can provide about the openai/whisper-base model:

Model size: The base model is part of a suite of models of various sizes trained to study the scaling properties of Whisper. However, the exact number of parameters for the base model is [More Information Needed].

Training:
- Models were trained with AdamW optimizer and gradient norm clipping.
- A linear learning rate decay to zero was used after a warmup over the first 2048 updates. 
- Batch size of 256 segments was used.
- Models were trained for 2^20 updates, which is between two and three passes over the dataset.
- No data augmentation or regularization was used during training.

[More Information Needed] on specific throughput, start/end times, and checkpoint sizes for the base model.

The model architecture is an encoder-decoder Transformer. Audio is resampled to 16kHz and converted to an 80-channel log-mel spectrogram using 25ms windows and 10ms stride.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the openai/whisper-base model was evaluated on the following benchmarks and datasets:

Short-form English-only datasets:
- LibriSpeech (test-clean and test-other splits)
- TED-LIUM 3 (test split)
- Common Voice 5.1 (English subset)
- Artie bias corpus (subset of Common Voice)
- CallHome and Switchboard (from LDC2002S09 and LDC2002T43)
- WSJ (from LDC93S6B and LDC94S13B, preprocessed using s5 recipe)
- CORAAL (231 interviews, preprocessed using FairSpeech project script)
- Rev16 (subset of 16 files from Rev.AI's Podcast Transcription Benchmark)

Other datasets:
- CHiME-6 (preprocessed from CHiME-5 using s5 track1 recipe stage 0)
- AMI-IHM and AMI-SDM1 (preprocessed using s5b recipe stages 0 and 2)
- Common Voice 15 (used for language-specific performance breakdown)
- Fleurs (used for language-specific performance breakdown)

The model was also compared against open-source models like NVIDIA STT Conformer-CTC Large and 4 commercial ASR services on various datasets. Detailed results can be found in the paper.

[More Information Needed] on the specific datasets used for the commercial ASR comparison and the "Meanwhile" dataset mentioned.

#### Factors

Based on the provided references about the openai/whisper-base model, here are some key foreseeable characteristics that may influence the model's behavior and performance:

1. Language and amount of training data: The model's performance on a given language is strongly correlated with the amount of training data available for that language. The current pre-training dataset is English-heavy, so the model likely performs better on English compared to lower-resource languages (Reference 2).

2. Domain and context: The model may struggle with domain-specific terminology or contexts that are underrepresented in the training data. Fine-tuning on high-quality supervised datasets from specific domains could help improve performance (Reference 1).

3. Audio quality and noise: The model's performance may degrade when dealing with noisy audio or audio from challenging environments like pubs or restaurants. Testing the model's robustness to white noise and pub noise additions could reveal disparities in performance (Reference 6).

4. Speaker demographics: [More Information Needed] The references do not directly address performance disparities across speaker demographics like age, gender, or accent.

5. Transcript formatting: The model's zero-shot performance may be impacted by variations in transcript formatting across datasets, as it does not observe dataset-specific formatting examples during training (Reference 8).

6. Evaluation metrics: Standard evaluation metrics like WER may not always correlate well with human judgment, especially for zero-shot models. Developing better evaluation metrics is an active research area (Reference 8).

To uncover potential performance disparities, the model should be evaluated on a diverse range of datasets covering different languages, domains, audio qualities, and speaker demographics. Disaggregated evaluation results would provide a clearer picture of the model's strengths and weaknesses.

#### Metrics

Based on the references provided, the key metrics and considerations for evaluating the openai/whisper-base model are:

1. Word Error Rate (WER) is the primary metric, but it has limitations as it penalizes innocuous differences in transcript style. To mitigate this, extensive text standardization is done before calculating WER, especially for zero-shot models like Whisper. (References 1-3)

2. Character Error Rate (CER) is used for some languages, particularly in the Common Voice and Fleurs datasets. (Reference 9)

3. BLEU scores are also calculated in some cases, as mentioned in Reference 9, but details are in the appendix of the full paper. [More Information Needed]

4. To assess how close Whisper's performance is to human level, professional human transcripts were obtained for a subset of the Kincaid46 dataset and compared to Whisper and other models. (Reference 10)

5. Whisper's performance is compared to open-source models like NVIDIA STT Conformer-CTC Large as well as commercial ASR services. (Reference 11)

In summary, WER is the primary metric, supplemented by CER for certain languages, with efforts made to standardize the text to make the metrics more meaningful. Human-level performance and comparisons to other open-source and commercial models provide additional context for Whisper's performance.

### Results

Evaluation Results of openai/whisper-base:

Performance (Accuracy):
- English speech recognition performance is close to human-level and shows diminishing returns with increasing model size. (Reference 1)
- Performance continues to increase with model size for multilingual speech recognition, speech translation, and language identification. (Reference 1)
- Achieves an average 55.2% relative error reduction compared to a supervised LibriSpeech model with similar performance when evaluated on other speech recognition datasets. (Reference 7)
- [More Information Needed] for exact accuracy metrics.

Performance (WER/CER):
- Performance varies widely depending on the language. (Reference 9)
- [More Information Needed] for exact WER/CER metrics for openai/whisper-base. References only provide metrics for large-v3 and large-v2 models.

Performance (Language Identification):
- Underperforms supervised SOTA on the Fleurs dataset by 13.6%. (Reference 10)
- Achieves 80.3% accuracy on the 82 overlapping languages in the Fleurs dataset. (Reference 10)

Robustness:
- Designed to be a single robust model that works reliably across domains, tasks, and languages without dataset-specific fine-tuning. (Reference 2)

Scalability:
- Trained on one of the largest ever datasets in supervised speech recognition with 680,000 hours of labeled audio. (Reference 4)
- Performance scales with dataset size, but shows diminishing returns from 54,000 to 680,000 hours. (Reference 3, 5)

[More Information Needed] for Efficiency, Fairness, and Interpretability.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the openai/whisper-base model:

- Whisper's performance varies widely depending on the language. WER (word error rate) and CER (character error rate) metrics are used to evaluate performance on datasets like Common Voice 15 and Fleurs. Full results can be found in the paper's Appendix D.1, D.2 and D.4.

- With the exception of English speech recognition, Whisper's performance continues to increase with model size across multilingual speech recognition, speech translation, and language identification. English performance may be saturating as it approaches human-level. 

- Whisper performs better than compared models on most datasets, especially those with uncommon words, even though some commercial systems may have been trained on the public datasets.

- To quantify how close Whisper is to human-level performance, professional human transcripts were obtained for 25 recordings from the Kincaid46 dataset. [More Information Needed] on the results of this comparison.

- Common errors include truncation, hallucination, and struggling with the first/last few words. Fine-tuning on high-quality data or using reinforcement learning could help reduce these.

- [More Information Needed] for code examples, as no code blocks were provided in the references.

In summary, Whisper shows strong zero-shot generalization across languages, with performance scaling with model size, but still has room for improvement especially on low-resource languages and in reducing specific error types.

## Model Examination

Based on the provided references about the Whisper model, here is an experimental section on explainability and interpretability for the openai/whisper-base model:

The openai/whisper-base model is a robust speech recognition system that aims to generalize well across domains, tasks, and languages without the need for dataset-specific fine-tuning. To better understand the model's behavior and decision-making process, several areas could be explored:

1. Ablation studies: Investigating the contributions of the encoder and decoder components to the model's robustness by training variations such as a decoder-less CTC model or combining the Whisper encoder with existing speech recognition encoders like wav2vec 2.0. [More Information Needed]

2. Error analysis: Examining the types of errors made by the model, such as perception-related errors, repetition loops, hallucinations, and incorrect speaker name predictions. This analysis could help identify areas for improvement in the model's decoding strategies and training data. [More Information Needed]

3. Language-specific performance: Analyzing the relationship between the amount of training data for each language and the model's performance in that language. This could provide insights into the model's generalization capabilities and highlight the need for more diverse training data. [More Information Needed]

4. Fine-tuning impact: Studying the effects of fine-tuning the model on high-quality supervised datasets for specific domains or tasks. This could help understand the model's adaptability and potential for further improvement in specialized applications. [More Information Needed]

5. Interpretability techniques: Applying interpretability methods such as attention visualization, saliency maps, or feature attribution to understand which parts of the input audio the model focuses on when making predictions. [More Information Needed]

By exploring these areas, we aim to gain a deeper understanding of the openai/whisper-base model's inner workings, identify its strengths and weaknesses, and discover potential avenues for improvement in terms of robustness, generalization, and interpretability.

## Environmental Impact

- **Hardware Type:** Based on the provided references, the model openai/whisper-base was trained using:

Data parallelism across accelerators using FP16 with dynamic loss scaling and activation checkpointing.

However, the specific hardware type (e.g., GPU, TPU) is not explicitly mentioned. Therefore, for the hardware type used for training openai/whisper-base:

[More Information Needed]
- **Software Type:** The model openai/whisper-base is trained using Python, as evidenced by the code snippets in the references:

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])
```

and

```python
import whisper

model = whisper.load_model("base")
```

These code examples demonstrate loading and using the Whisper model in Python.
- **Hours used:** Based on the information provided in the references, the Whisper models were trained for 2^20 updates, which is between two and three passes over the dataset (Reference 2). However, the specific training time for the openai/whisper-base model is not explicitly mentioned.

[More Information Needed] on the exact training time for the openai/whisper-base model.
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider used for training the openai/whisper-base model. More information would be needed to determine which cloud provider, if any, was utilized during the model's training process.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the openai/whisper-base model. To answer this question, more specific details would be needed, such as the hardware used for training, the energy consumption of the training process, and the carbon intensity of the electricity used.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
- Encoder-decoder Transformer architecture
- Audio is re-sampled to 16,000 Hz
- 80-channel log-magnitude Mel spectrogram representation is computed on 25-millisecond windows with a stride of 10 milliseconds
- Input is globally scaled to be between -1 and 1 with approximately zero mean across the pre-training dataset
- Encoder processes input with a small stem consisting of two convolution layers (filter width of 3, GELU activation, second layer has stride of two)
- Sinusoidal position embeddings are added to the output of the stem
- Encoder Transformer blocks are applied, using pre-activation residual blocks
- Final layer normalization is applied to the encoder output
- Decoder uses learned position embeddings and tied input-output token representations
- Encoder and decoder have the same width and number of transformer blocks

Model Objective:
The Whisper model is trained on various speech processing tasks, including:
- Multilingual speech recognition
- Speech translation
- Spoken language identification
- Voice activity detection

These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a single model to replace many stages of a traditional speech-processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.

The model is trained on a large dataset of diverse audio and aims to be a general-purpose speech recognition model that can perform multiple tasks without the need for fine-tuning.

### Compute Infrastructure

Based on the provided references, here is the information about the compute infrastructure used for training the openai/whisper-base model:

The models were trained using:
- Data parallelism across accelerators
- FP16 precision with dynamic loss scaling
- Activation checkpointing

The model was trained for 2^20 updates, which is between two and three passes over the dataset of 680,000 hours.

Batch size: 256

Optimizer:
- AdamW optimizer
- Gradient norm clipping
- Linear learning rate decay to zero after a warmup over the first 2048 updates

[More Information Needed] on the exact compute infrastructure details such as number and type of accelerators used, training time, etc.

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

