# Model Card for openai/whisper-base

The model openai/whisper-base is a general-purpose, multilingual speech recognition model that is robust and capable of performing tasks such as speech recognition, translation, and language identification without the need for dataset-specific fine-tuning, trained on a diverse set of 680,000 hours of audio data. It utilizes a minimalist approach to data pre-processing and is designed to work "out of the box" for a broad range of environments.

## Model Details

### Model Description

Model Name: openai/whisper-base

### Model Architecture
The openai/whisper-base model is built on an encoder-decoder Transformer architecture, as described by Vaswani et al. (2017). This architecture has been chosen for its proven ability to scale reliably. The model is designed to handle various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are represented as a sequence of tokens predicted by the decoder, allowing the model to replace multiple stages of a traditional speech-processing pipeline.

### Training Procedures
The model was trained using data parallelism across accelerators, employing FP16 precision with dynamic loss scaling and activation checkpointing to manage memory efficiently. The AdamW optimizer was used along with gradient norm clipping and a linear learning rate decay schedule that started after a warmup over the first 2048 updates. The batch size was set to 256, and the model was trained for approximately 2-3 passes over the dataset, amounting to 2^20 updates.

### Parameters
The base model is part of a suite of models of various sizes, which were trained to study the scaling properties of Whisper. The specific hyperparameters for the base model, such as the number of layers, attention heads, and other architectural details, are not explicitly mentioned in the provided references. [More Information Needed] for the exact number of parameters in the openai/whisper-base model.

### Important Disclaimers
1. The model has a tendency to transcribe plausible but often incorrect guesses for the names of speakers, which is a result of the pre-training dataset's characteristics.
2. The model's performance varies widely depending on the language, with different WERs (word error rates) or CERs (character error rates) for each language.
3. For long-form audio transcription, the model relies on accurate prediction of timestamp tokens, and inaccuracies in one window may affect subsequent windows. Heuristics have been developed to mitigate failure cases in long-form transcription.
4. The robustness of the model is partially attributed to its strong decoder, which functions as an audio conditional language model. However, it is unclear how much of the benefit comes from the encoder, the decoder, or both.
5. Overfitting is not considered a large concern due to the limited number of training epochs and the diversity of the dataset, which is expected to encourage generalization and robustness. No data augmentation or regularization was used.
6. The model is a general-purpose speech recognition model trained on a large and diverse dataset, capable of multitasking across different speech processing tasks.

For further details and specific metrics, interested parties are encouraged to refer to the original paper and its appendices.

Contact Person for Model Card Updates: [Your Name/Contact Information]

- **Developed by:** Alec Radford; Jong Wook Kim; Tao Xu; Greg Brockman; Christine Mcleavey; Ilya Sutskever
- **Funded by:** The references provided do not explicitly mention the people or organizations that funded the project for the model openai/whisper-base. However, since the project is attributed to OpenAI, it is reasonable to infer that OpenAI is the primary organization behind the funding and development of the Whisper model. Specific funding sources or external organizations are not listed in the provided references.

For a definitive list of funders, one would typically need to look at OpenAI's funding sources or any press releases or acknowledgments made by OpenAI regarding this specific project. Since this information is not included in the provided references, the answer is:

[More Information Needed]
- **Shared by:** The contributors that made the model openai/whisper-base available online as a GitHub repo include Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever. Additionally, the team would like to thank Nick Ryder, Will Zhuk, and Andrew Carr for the conversation that inspired the project, the Acceleration and Supercomputing teams at OpenAI for their work on software and hardware infrastructure, and Pamela Mishkin for advising the project from a policy perspective.
- **Model type:** The openai/whisper-base model is a general-purpose, multitasking speech recognition Transformer model trained on diverse audio data using large-scale supervised pre-training, capable of multilingual recognition, speech translation, and language identification.
- **Language(s):** The model openai/whisper-base is a multilingual speech recognition model capable of processing and transcribing audio in multiple languages, with varying performance across different languages as detailed in the provided references.
- **License:** The model openai/whisper-base is released under the MIT License. You can find the details of the license here: [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE).
- **Finetuned from model:** The model openai/whisper-base is not explicitly mentioned as being fine-tuned from another model in the provided references. However, reference 4 mentions a "VoxLingua107 model" from which language targets are sourced. This does not necessarily imply that openai/whisper-base is fine-tuned from VoxLingua107, but rather that it utilizes language tokens sourced from it. Therefore, based on the given references, there is no direct evidence that openai/whisper-base is fine-tuned from another model. 

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/openai/whisper
- **Paper:** https://arxiv.org/pdf/2212.04356.pdf
- **Demo:** The demo for the model openai/whisper-base can be found in the Colab example provided in the references. Here is the link to the demo:

[Colab example](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)
## Uses

### Direct Use

The openai/whisper-base model is designed to be a general-purpose speech recognition model that can be used directly without the need for fine-tuning, post-processing, or integration into a more complex pipeline. This is possible because it has been trained on a large and diverse dataset, enabling it to handle a variety of speech recognition tasks effectively.

To use the model for transcribing audio, you can simply load the model and call the `transcribe()` method with the path to your audio file. Here's a code snippet demonstrating how to do this:

```python
import whisper

# Load the base model
model = whisper.load_model("base")

# Transcribe an audio file
result = model.transcribe("audio.mp3")

# Print the transcribed text
print(result["text"])
```

This code snippet will transcribe the speech from "audio.mp3" without any additional steps required. The `transcribe()` method handles the audio processing and transcription internally, using a sliding 30-second window to perform autoregressive sequence-to-sequence predictions.

Additionally, the model can perform language identification and translation tasks without further adjustments. For example, to transcribe and translate Japanese speech into English, you would use the following command in the terminal (not in Python):

```
whisper japanese.wav --language Japanese --task translate
```

This command uses the Whisper model to first identify the language of the speech in "japanese.wav" and then translate the speech into English. The model's multitasking capabilities allow it to perform these functions seamlessly, leveraging its training on multilingual data.

### Downstream Use

The openai/whisper-base model is designed to be a robust speech processing system that can be used "out of the box" for a variety of speech-related tasks without the need for dataset-specific fine-tuning. However, for specialized tasks or to improve performance on specific types of data, fine-tuning the model can be beneficial.

When fine-tuning openai/whisper-base for a specific task, you would typically start with the pre-trained model and continue the training process on a dataset that is representative of the task or domain you are interested in. This could involve adjusting the model's parameters to better capture the nuances of the target domain's speech patterns, vocabulary, and acoustic characteristics.

For integration into a larger ecosystem or app, the model can be used as a service that provides speech recognition capabilities. This could be part of a voice-controlled application, a transcription service, or any system that requires converting speech to text. The model's ability to transcribe and translate speech in multiple languages makes it particularly versatile for global applications.

Here is an example of how you might use the openai/whisper-base model in Python to transcribe audio:

```python
import whisper

# Load the pre-trained base model
model = whisper.load_model("base")

# Transcribe an audio file
result = model.transcribe("audio.mp3")

# Output the transcription
print(result["text"])
```

If you were to fine-tune the model, the code snippet would be similar, but you would load your fine-tuned model instead of the base model. The fine-tuning process itself would require additional steps, including preparing a dataset, setting up a training loop, and possibly adjusting hyperparameters, which is not directly covered in the provided references. Therefore, for the fine-tuning process itself, [More Information Needed].

### Out-of-Scope Use

As a sociotechnic on the deep learning model development team, it is my responsibility to consider the potential misuses of the openai/whisper-base model and to advise users on what they should not do with the model. Here are some foreseeable misuses and associated guidance:

1. **Privacy Violations**: Users should not use the Whisper model to transcribe audio recordings without the consent of the individuals being recorded. This includes private conversations, confidential information, or any audio obtained in a manner that violates privacy laws or expectations.

2. **Misrepresentation and Deception**: The model should not be used to generate transcripts that are then misrepresented as verbatim or official records when they have not been verified for accuracy. Given the model's potential for errors, such as hallucination or misalignment, users should be cautious about presenting the output as entirely factual without human review.

3. **Bias Amplification**: Users should avoid using the model in contexts where biased transcriptions could lead to harm or unfair treatment. Since the model's training data is English-heavy, it may not perform equally well on lower-resource languages, potentially leading to misrepresentation or misunderstanding of speakers of those languages.

4. **Intellectual Property Infringement**: The model should not be used to transcribe copyrighted material without permission from the copyright holder, as this could constitute a violation of intellectual property laws.

5. **Manipulation and Misinformation**: Users should not use the model to transcribe audio for the purpose of creating manipulated or misleading content, such as deepfakes or selectively edited recordings that could be used to spread misinformation or harm reputations.

6. **Security Risks**: Users should be cautious about using the model to transcribe sensitive or secure communications, as the model's outputs could potentially be intercepted or stored in a manner that compromises security.

7. **Unethical Research or Surveillance**: The model should not be used as a tool for unauthorized surveillance or research that does not comply with ethical standards, including informed consent and the protection of participants' rights.

In summary, users of the openai/whisper-base model should respect privacy, avoid misrepresentation, be aware of potential biases, respect intellectual property rights, refrain from creating misinformation, protect sensitive information, and adhere to ethical standards in research and surveillance. It is crucial that users of the model consider the societal implications of their applications and use the technology responsibly.

### Bias, Risks, and Limitations

The known and foreseeable issues stemming from the model openai/whisper-base include both technical and sociotechnical limitations:

1. **Perception-Related Errors**: Larger models have reduced perception-related errors, such as confusing similar-sounding words. However, there are still errors in long-form transcription that are non-human and non-perceptual in nature. These include seq2seq model failures, language model limitations, and text-audio alignment issues, leading to problems like repeat loops, incomplete transcriptions, and complete hallucinations where the transcript is unrelated to the audio. [Reference 1, 3]

2. **Language Bias and Data Scarcity**: The model's performance on non-English languages is suboptimal due to the English-heavy pre-training dataset. This bias stems from the data collection pipeline, which sourced primarily from English-centric parts of the internet. As a result, many languages are underrepresented, and performance is directly correlated with the amount of training data available for each language. [Reference 2]

3. **Fine-Tuning and Domain Adaptation**: The model has been studied in a zero-shot setting, focusing on robustness and general reliability. However, for domains with high-quality supervised speech data, fine-tuning could improve performance. This also allows for direct comparisons with other models and could address some of the generalization issues. [Reference 4]

4. **Generalization and Robustness**: Despite high performance on in-distribution (IID) test sets, machine learning models, including Whisper, can make mistakes when evaluated in slightly different settings. This highlights the lack of generalization between datasets and the need for robustness to distribution shifts and other perturbations. [Reference 5, 7]

5. **Model Components and Architecture**: It is unclear how much of Whisper's benefits come from its encoder, decoder, or both. Ablation studies or experiments with different combinations of encoders and decoders could provide insights into the contributions of each component to the overall performance. [Reference 6, 11]

6. **Multitask Training and Special Tokens**: The model is trained on various speech processing tasks using a multitask training format with special tokens. While this allows a single model to replace many stages of a traditional speech-processing pipeline, it may also introduce complexity and potential errors related to task specification and token prediction. [Reference 8]

7. **Speaker Name Prediction**: The model has a tendency to transcribe plausible but incorrect guesses for the names of speakers, which is a result of the pre-training dataset including speaker names in transcripts. This could lead to misinformation and confusion when the model is used in real-world applications. [Reference 9]

8. **Model Size and Diminishing Returns**: While performance generally increases with model size, there are diminishing returns, especially for English speech recognition, which could be due to saturation effects from approaching human-level performance. This suggests that simply scaling up the model may not be the most efficient way to improve performance further. [Reference 10]

In summary, the openai/whisper-base model has made significant strides in speech recognition but still faces challenges related to error types, language bias, fine-tuning, generalization, model architecture, task specificity, and scaling. Addressing these issues will require a combination of technical improvements, more diverse and representative training data, and careful consideration of the sociotechnical impact of the model's use in various domains and languages.

### Recommendations

Based on the references provided, here are the recommendations with respect to the foreseeable issues about the model openai/whisper-base:

1. **Improved Decoding Strategies**: To address the non-human/perceptual errors such as repeat loops and hallucinations, it is recommended to explore fine-tuning the Whisper models on high-quality supervised datasets and/or using reinforcement learning to optimize decoding performance. This could help in reducing the stubborn errors that are not related to perception.

2. **Increase Training Data for Lower-Resource Languages**: The performance of Whisper on many languages is currently suboptimal due to a lack of training data. To improve speech recognition performance across a wider range of languages, it is recommended to expand the pre-training dataset with more diverse language data, particularly for those languages that are underrepresented.

3. **Handling Long-Form Audio**: For transcribing long-form audio, it is important to ensure accurate prediction of timestamp tokens. The development of heuristics to avoid failure cases in long-form transcription is recommended. This includes using beam search with multiple beams and log probability scoring, as well as setting thresholds for voice activity detection and constraining the initial timestamp token to improve reliability.

4. **Text Normalizer**: The release of the code for the text normalizer is a positive step that allows for comparison and helps in studying the performance of speech recognition systems in out-of-distribution settings. It is recommended to encourage the community to use and contribute to the text normalizer to further improve its effectiveness.

5. **General-Purpose Robustness**: Whisper aims to be a robust speech processing system that generalizes well across domains, tasks, and languages without the need for dataset-specific fine-tuning. It is recommended to continue evaluating Whisper on a wide set of existing speech processing datasets to ensure that it maintains high-quality results and generalizes effectively.

6. **Ethical and Societal Considerations**: [More Information Needed] While the technical recommendations are clear, there is a need for further information regarding the ethical and societal implications of deploying Whisper. This includes considerations around privacy, consent, and the potential for misuse of speech recognition technology. It is recommended that the team includes ethicists, lawyers, and rights advocates in the conversation to address these concerns and ensure responsible use of the technology.

In summary, the recommendations include improving decoding strategies, increasing training data for lower-resource languages, refining long-form audio transcription methods, leveraging community contributions to the text normalizer, ensuring general-purpose robustness, and addressing ethical and societal considerations.

## Training Details

### Training Data

The training data for the model openai/whisper-base consists of a large and diverse dataset sourced primarily from English-centric parts of the internet, with a focus on raw text of transcripts without significant standardization. This approach allows the model to learn the mapping between spoken utterances and their transcribed forms directly. For languages other than English, the dataset includes audio-transcript pairs where the spoken language matches the language of the transcript, with an exception made for English transcripts. [More Information Needed] on data pre-processing and additional filtering.

### Training Procedure

#### Preprocessing

The preprocessing for the data of the model openai/whisper-base involves several steps to prepare the audio data for training and to ensure that the text data is in a standardized form that the model can learn from effectively. Here are the details based on the provided references:

1. **Text Normalization**: The model uses a comprehensive text normalization process to standardize English texts into a form that penalizes word errors due to mistranscription rather than formatting or punctuation differences. This normalization is designed to work with any UTF-8 string and is more intricate than standard ASCII character normalization (Reference 1).

2. **Data Pre-processing**: The approach to data pre-processing is minimalist, with the model trained to predict the raw text of transcripts without significant standardization. This leverages the expressiveness of sequence-to-sequence models to learn the mapping between spoken utterances and their transcribed text, simplifying the speech recognition pipeline (Reference 2).

3. **Audio Language Detection**: An audio language detector is used to ensure that the spoken language matches the language of the transcript. This detector was fine-tuned on VoxLingua107. If the languages do not match, the (audio, transcript) pair is not included in the training dataset, except when the transcript language is English (Reference 3).

4. **Model Architecture**: The model uses an encoder-decoder Transformer architecture, which has been shown to scale reliably. All audio is resampled to 16,000 Hz, and an 80-channel log-magnitude Mel spectrogram is computed on 25-millisecond windows with a stride of 10 milliseconds (Reference 6).

5. **Audio Segmentation**: Audio files are broken into 30-second segments, each paired with the corresponding transcript that occurs within that time frame. The model is trained on all audio, including segments with no speech, which are used for training voice activity detection (Reference 7).

6. **Voice Activity Detection**: The model uses a no-speech probability threshold and an average log-probability threshold to improve the reliability of voice activity detection. Additionally, the initial timestamp token is constrained to be between 0.0 and 1.0 second to avoid ignoring the first few words in the input (Reference 10).

7. **Transcription Method**: The `transcribe()` method in the Python API reads the entire audio file and processes it with a sliding 30-second window, performing autoregressive sequence-to-sequence predictions on each window (Reference 11).

The preprocessing steps are designed to handle the complexities of speech recognition while minimizing the need for extensive data cleaning or standardization, allowing the model to learn directly from the raw text of transcripts. This approach helps the model to generalize better to real-world speech variations.

#### Training Hyperparameters

The training hyperparameters for the model openai/whisper-base are as follows:

- **Model Size**: Various sizes were trained, but for the base model, specific dimensions are not provided in the reference text. [More Information Needed]
- **Parallelism**: Data parallelism across accelerators was used.
- **Precision**: FP16 with dynamic loss scaling.
- **Activation Checkpointing**: Implemented as per Griewank & Walther, 2000; Chen et al., 2016.
- **Optimizer**: AdamW, as per Loshchilov & Hutter, 2017.
- **Gradient Norm Clipping**: As per Pascanu et al., 2013.
- **Learning Rate Schedule**: Linear decay to zero after a warmup over the first 2048 updates.
- **Batch Size**: 256 segments.
- **Training Updates**: The models were trained for 2^20 updates, which corresponds to two to three passes over the dataset.
- **Regularization**: No data augmentation or explicit regularization was used; the model relies on the diversity of the large dataset for generalization.
- **Epochs**: The base model was trained for a few epochs, as overfitting was not a large concern due to the large dataset size.
- **Additional Techniques**: For the V2 Large model, SpecAugment and Stochastic Depth were added, but it is not clear if these were used for the base model. [More Information Needed]
- **Voice Activity Detection**: A no-speech probability threshold of 0.6 combined with an average log-probability threshold of -1 was used to improve reliability.
- **Timestamp Token Constraint**: The initial timestamp token was constrained to be between 0.0 and 1.0 second.

For full training hyperparameters, the reference suggests checking Appendix F, which is not provided in the reference text. Therefore, for some specific details like the exact learning rate, weight decay, or the dimensions of the base model, [More Information Needed].

#### Speeds, Sizes, Times

The model card for `openai/whisper-base` should include the following details based on the provided references:

- **Throughput**: [More Information Needed]
- **Start or End Time**: The models, including the base model, were trained for 2^20 updates (Reference 6). However, the exact start or end time of the training process for the `whisper-base` model is not provided in the references.
- **Checkpoint Sizes**: While the references mention the use of activation checkpointing (Reference 3), the specific checkpoint sizes for the `whisper-base` model are not detailed in the provided text.

For more detailed information on these specific aspects, one would need to refer to the full training logs or additional documentation that is not included in the provided references.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model openai/whisper-base evaluates on the following benchmarks or datasets:

- LibriSpeech: Specifically, the test-clean and test-other splits from the LibriSpeech ASR corpus.
- TED-LIUM 3: The test split of TED-LIUM Release 3, using the segmented manual transcripts included in the release.
- Common Voice 5.1: The English subset of the Common Voice Corpus 5.1.
- Artie bias corpus: A subset of the Common Voice dataset.
- CallHome and Switchboard: Two corpora from LDC2002S09 and LDC2002T43.
- WSJ: Using LDC93S6B and LDC94S13B datasets with the s5 recipe for preprocessing.
- CORAAL: The 231 interviews from CORAAL, with preprocessing script from the FairSpeech project.
- VoxPopuli: ASR data in 16 languages, including English, collected using the get asr data.py script from the official repository.
- Common Voice 9: The Common Voice Corpus 9.
- CoVOST 2: The X into English data collected using the official repository.
- Multilingual LibriSpeech (MLS): The test splits from each language in the MLS corpus.
- Fleurs: Audio files and transcripts collected using the HuggingFace datasets implementation, used as a translation dataset.
- Rev16: A subset of 16 files from the 30 podcast episodes in Rev.AI's Podcast Transcription Benchmark, specifically the files available in the speech-datasets repository as of their 202206 version.

Additionally, the model's performance on multilingual speech recognition is reported on two low-data benchmarks:

- Multilingual LibriSpeech (MLS)
- VoxPopuli

The model's performance is also evaluated using WERs (word error rates) or CER (character error rates) on the Common Voice 15 and Fleurs datasets, with additional metrics available in the referenced paper's appendices.

#### Factors

The performance of the openai/whisper-base model is influenced by several characteristics that can be identified from the provided references:

1. **Language and Data Availability**: As indicated in reference 1, the amount of training data for a language is a strong predictor of the model's performance in that language. The model's training dataset is currently skewed towards English, resulting in poorer performance on lower-resource languages. Therefore, the model's behavior will vary significantly across languages, with better performance in English and potentially lower performance in languages that are underrepresented in the training data.

2. **Decoding Strategies**: Reference 2 and 4 highlight that the model's decoding strategies affect its performance, especially in long-form transcription. Issues such as repeat loops, missing transcriptions of the beginning or end of audio segments, and complete hallucinations can occur. These errors are more pronounced in non-human/perceptual aspects of language processing and may be mitigated by improved decoding strategies or fine-tuning on high-quality supervised datasets.

3. **Model Size and Generalization**: Reference 7 suggests that the model's size influences its zero-shot generalization capabilities. Larger models tend to perform better across various tasks, including multilingual speech recognition and speech translation. However, there are diminishing returns for English speech recognition, possibly due to approaching human-level performance.

4. **Robustness and Decoder Strength**: Reference 8 discusses the robustness of the Whisper model, which may be partially attributed to its strong decoder. The model's behavior could be affected by the training of its encoder, decoder, or both. The interaction between these components and their individual contributions to performance is an area for further study.

5. **Evaluation Metrics and Standardization**: Reference 5 points out the challenges in developing evaluation metrics that correlate well with human judgment. The model uses extensive standardization of text before calculating word error rates (WER) to minimize penalization for acceptable variations. This suggests that the model's performance evaluation may not fully capture user satisfaction in real-world applications.

6. **Population Subgroups**: Although not explicitly mentioned in the references, it is reasonable to infer that the model's performance may vary across different population subgroups, such as speakers with accents or dialects that are underrepresented in the training data. [More Information Needed] to provide a detailed analysis of disparities in performance across these subgroups.

In summary, the openai/whisper-base model's behavior is influenced by the language and amount of training data, decoding strategies, model size, robustness and decoder strength, and the choice of evaluation metrics. Performance disparities may exist across languages, domains, contexts, and population subgroups, and further disaggregated evaluation is needed to uncover and address these disparities.

#### Metrics

For the evaluation of the openai/whisper-base model, we will primarily use the Word Error Rate (WER) metric, as it is the standard for speech recognition systems (Reference 2). However, we are aware of the limitations of WER, such as penalizing innocuous differences in transcript style that do not affect the intelligibility or correctness of the transcripts from a human perspective (Reference 1 and 2).

To address these limitations and better correlate with human judgment, we have implemented extensive standardization of text before the WER calculation to minimize penalization of differences that are not errors in the true sense (Reference 1). This approach helps to mitigate the impact of transcript style variations on the WER metric.

Additionally, we have considered the irreducible error present in each dataset due to factors like ambiguous speech or labeling errors. To gauge how close the Whisper model's performance is to human performance, we have compared it against transcripts produced by professional transcribers using a subset of recordings from the Kincaid46 dataset (Reference 3).

Furthermore, we have acknowledged the different error types that larger models like Whisper face, such as repeat loops and issues with text-audio alignment, which are not typical human errors. These are taken into account when considering the model's decoding strategies and overall performance (Reference 4).

Lastly, we recognize that Whisper's performance varies by language, and we use both WER and Character Error Rate (CER) for languages with different scripts, as shown in the performance breakdown by language using datasets like Common Voice 15 and Fleurs (Reference 6). We also consider the amount of training data available for each language, as this is a strong predictor of performance, and aim to increase data for lower-resource languages (Reference 5).

In summary, while WER is the primary metric for evaluating the openai/whisper-base model, we supplement it with text standardization, comparisons to human transcribers, and additional metrics like CER where appropriate, to provide a more nuanced and accurate assessment of the model's performance across different languages and datasets.

### Results

The evaluation results of the model `openai/whisper-base` are based on several factors and metrics as follows:

1. **Zero-Shot Generalization**: The `openai/whisper-base` model is a zero-shot model, meaning it is evaluated without fine-tuning on specific datasets. This approach is designed to test the model's ability to generalize across various speech recognition tasks.

2. **Word Error Rate (WER)**: The primary metric used for evaluating the `openai/whisper-base` model is the Word Error Rate. WER is a common metric in speech recognition that measures the difference between the model's output and the reference transcript.

3. **Text Normalization**: To address the issue of WER penalizing minor formatting differences that would not affect human judgment, extensive standardization of text before the WER calculation is performed. The team has released the code for the text normalizer to facilitate fair comparisons and aid in the study of speech recognition performance in out-of-distribution settings.

4. **Performance by Language**: The performance of the Whisper models, including `openai/whisper-base`, varies by language. The evaluation includes metrics such as WER and Character Error Rate (CER) on datasets like Common Voice 15 and Fleurs. However, specific performance breakdowns for the `openai/whisper-base` model by language are not provided in the references and would require [More Information Needed].

5. **Comparison with Other Models**: The `openai/whisper-base` model's performance is compared with both open-source models and commercial Automatic Speech Recognition (ASR) services. The comparison includes an analysis of the distribution of word error rates from Whisper and other ASR services. However, specific figures or detailed results for the `openai/whisper-base` model are not provided in the references, so [More Information Needed] for exact numbers.

6. **Model Size and Performance**: The references suggest that there is a study on the zero-shot generalization of Whisper models as a function of model size, with larger models generally performing better across various tasks. However, there is a mention of diminishing returns for English speech recognition, possibly due to approaching human-level performance. Specific results related to the `openai/whisper-base` model size and performance are not detailed, so [More Information Needed].

7. **Relative Error Reduction**: When compared to a supervised LibriSpeech model with similar performance on the LibriSpeech test-clean, the zero-shot Whisper model achieves an average relative error reduction of 55.2% when evaluated on other speech recognition datasets. This suggests that the `openai/whisper-base` model has strong generalization capabilities, although specific numbers for this model are not provided.

In summary, the `openai/whisper-base` model is evaluated on its ability to generalize in a zero-shot setting using WER as the primary metric, with text normalization to ensure fair comparison. Its performance varies by language and is compared with other models and commercial services. However, specific evaluation results for the `openai/whisper-base` model are not detailed in the provided references, and therefore [More Information Needed] for precise figures and comparisons.

#### Summary

The evaluation results for the model `openai/whisper-base` indicate that it performs comparably to a supervised LibriSpeech model on the LibriSpeech test-clean dataset, with a significant average relative error reduction of 55.2% across various other speech recognition datasets in a zero-shot setting. This demonstrates the model's robustness and its ability to generalize well across different domains, tasks, and languages without the need for dataset-specific fine-tuning.

Whisper models, including `openai/whisper-base`, are designed to approach the ideal of uniform performance across all datasets, leveraging a broad and diverse distribution of audio for training. The model's multilingual capabilities are highlighted by its training on speech recognition data in 75 languages, although its performance varies widely depending on the language, as shown by word error rates (WERs) and character error rates (CERs) on the Common Voice 15 and Fleurs datasets.

The model's zero-shot generalization ability was also studied as a function of model size, with the `openai/whisper-base` model likely following the trend where performance increases with model size across various tasks, except for English speech recognition where diminishing returns are observed due to approaching human-level performance.

Lastly, the evaluation process for Whisper models, including `openai/whisper-base`, involves extensive standardization of text before WER calculation to minimize penalization for transcript format differences, acknowledging the challenge of developing evaluation metrics that correlate well with human judgment in speech recognition.

[More Information Needed] for specific metrics and results related to the `openai/whisper-base` model, as the references provided do not include direct figures or detailed outcomes for this specific model variant.

## Model Examination

Explainability/Interpretability of openai/whisper-base:

The openai/whisper-base model is a robust speech recognition system that has been trained on a diverse set of audio data. Its architecture is designed to handle a wide range of speech recognition tasks, including transcription and language identification. The model's robustness can be attributed to its strong decoder, which functions as an audio conditional language model, as mentioned in reference 2. This decoder plays a crucial role in the model's ability to discern and transcribe spoken words accurately.

One of the key challenges in the interpretability of the Whisper model is understanding the contributions of its encoder and decoder components to its overall performance. As suggested in reference 2, further studies could involve ablating different design components or combining the Whisper decoder with other speech recognition encoders to assess their individual impacts.

The model has been observed to reduce perception-related errors, such as confusing similar-sounding words, as it scales up. However, it still faces challenges with non-human/perceptual errors, such as repeat loops and text-audio alignment issues, as noted in reference 3. These errors are indicative of the limitations of current seq2seq and language models, which could be areas for future research in explainability.

Moreover, the model has a tendency to make plausible but incorrect guesses for speaker names, as highlighted in reference 6. This behavior is influenced by the pre-training dataset, which often includes speaker names in the transcripts. Understanding and mitigating this tendency could improve the model's interpretability in real-world applications.

Finally, the release of the code for the text normalizer, as mentioned in reference 7, is a step towards improving the comparability and study of speech recognition systems' performance, particularly in out-of-distribution settings. This tool can aid in the interpretability of the model by providing a standardized method for text normalization.

In summary, while the openai/whisper-base model demonstrates strong performance in speech recognition, there are several areas where its explainability could be further developed. These include dissecting the contributions of its encoder and decoder, addressing non-perceptual error patterns, and understanding the influence of training data on its predictive behaviors. Further research and development are needed to enhance the model's interpretability, which could involve fine-tuning on high-quality datasets, reinforcement learning optimization, and improved decoding strategies as suggested in reference 1.

## Environmental Impact

- **Hardware Type:** [More Information Needed]
- **Software Type:** The model openai/whisper-base is trained on Python 3.9.9 and PyTorch 1.10.1.
- **Hours used:** The model openai/whisper-base was trained for 2^20 updates, which is between two and three passes over the dataset.
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture for `openai/whisper-base` is a Transformer sequence-to-sequence model, which is well-suited for large-scale supervised pre-training for speech recognition tasks. The architecture is based on an encoder-decoder Transformer framework, as validated by Vaswani et al. (2017), to scale reliably. The encoder and decoder have the same width and number of transformer blocks, with the transformer utilizing pre-activation residual blocks and a final layer normalization applied to the encoder output. The decoder employs learned position embeddings and tied input-output token representations.

The objective of the `openai/whisper-base` model is to serve as a general-purpose speech recognition model that is robust and versatile. It is trained on a large and diverse dataset, enabling it to perform multiple tasks such as multilingual speech recognition, speech translation, and language identification. The model is designed to work "out of the box" in a broad range of environments without the need for supervised fine-tuning of the decoder for each deployment distribution. The multitask training format uses special tokens that act as task specifiers or classification labels, allowing the model to replace many stages of a traditional speech-processing pipeline.

### Compute Infrastructure

The compute infrastructure used for training the model openai/whisper-base involved data parallelism across accelerators and utilized FP16 precision with dynamic loss scaling and activation checkpointing for efficiency. The models were trained using the AdamW optimizer with gradient norm clipping and a linear learning rate decay schedule after a warmup period. A batch size of 256 segments was used during training. The specific hardware details, such as the type and number of accelerators, are not provided in the references, so [More Information Needed] for those particulars.

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

