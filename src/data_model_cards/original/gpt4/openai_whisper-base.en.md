# Model Card for openai/whisper-base.en

The model openai/whisper-base.en is a general-purpose, multilingual speech recognition model that is capable of performing tasks such as speech transcription, translation, and language identification without the need for fine-tuning, trained on a diverse dataset of 680,000 hours of audio. It utilizes a Transformer sequence-to-sequence architecture and is designed to approach human-level accuracy and robustness in speech processing.

## Model Details

### Model Description

Model Name: openai/whisper-base.en

### Model Architecture
The openai/whisper-base.en model is built upon an encoder-decoder Transformer architecture, which has been well-validated for its scalability and reliability. The model is designed for various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are represented as a sequence of tokens predicted by the decoder, allowing the model to streamline the traditional speech-processing pipeline with a multitask approach.

### Training Procedures
- The model is trained on a large dataset with diverse audio samples.
- A minimalist approach to data pre-processing was adopted, with the model trained to predict raw text transcripts without significant standardization.
- The training utilized data parallelism across accelerators, employing FP16 precision with dynamic loss scaling and activation checkpointing.
- AdamW optimizer was used with gradient norm clipping and a linear learning rate decay schedule.
- A batch size of 256 segments was used, and the model was trained for 2^20 updates, which corresponds to two to three passes over the dataset.
- No significant data augmentation or regularization techniques were used, relying instead on the diversity of the dataset to promote generalization and robustness.

### Parameters
- The model uses an 80-channel log-magnitude Mel spectrogram representation of audio, re-sampled to 16,000 Hz.
- The spectrogram is computed on 25-millisecond windows with a stride of 10 milliseconds.
- The model was trained with a suite of different sizes to study scaling properties, but specific parameter counts for the base model are not provided in the references.

### Important Disclaimers
- The model has a tendency to transcribe plausible but often incorrect guesses for the names of speakers due to biases in the pre-training dataset.
- Performance varies widely depending on the language, with word error rates (WERs) and character error rates (CERs) differing across languages.
- The model's robustness may be partially attributed to its strong decoder, which functions as an audio conditional language model, but the exact contributions of the encoder and decoder to the overall performance are not fully understood.
- Long-form audio transcription relies on accurate prediction of timestamp tokens, and errors in one window may affect subsequent windows. Heuristics have been developed to mitigate failure cases in long-form transcription.

For further details on the model's architecture, training procedures, parameters, and performance, interested parties are encouraged to refer to the original research paper and supplementary materials.

- **Developed by:** Alec Radford; Jong Wook Kim; Tao Xu; Greg Brockman; Christine Mcleavey; Ilya Sutskever
- **Funded by:** The references provided do not explicitly mention the people or organizations that funded the project for the model openai/whisper-base.en. However, since the project is associated with OpenAI, it is reasonable to infer that OpenAI is the primary organization behind the development and funding of the Whisper project. Specific funding sources or additional sponsors are not listed in the provided references.

For a definitive list of funders, more information would be needed.
- **Shared by:** The contributors that made the model openai/whisper-base.en available online as a GitHub repo include Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever. Additionally, the project acknowledges the contributions of Nick Ryder, Will Zhuk, Andrew Carr, the Acceleration and Supercomputing teams at OpenAI, Pamela Mishkin, and the developers of various software packages such as Numpy, SciPy, ftfy, PyTorch, pandas, and scikit-learn.
- **Model type:** The model openai/whisper-base.en is a general-purpose, autoregressive sequence-to-sequence speech recognition model trained with data parallelism and multitasking capabilities for multilingual recognition, translation, and language identification, using a minimalist approach to data pre-processing without significant standardization.
- **Language(s):** The model openai/whisper-base.en is designed for English-only speech recognition applications.
- **License:** The model openai/whisper-base.en is released under the MIT License. You can find the details of the license here: [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE).
- **Finetuned from model:** [More Information Needed]
### Model Sources

- **Repository:** https://github.com/openai/whisper
- **Paper:** https://arxiv.org/pdf/2212.04356.pdf
- **Demo:** [Colab example](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)
## Uses

### Direct Use

The model `openai/whisper-base.en` is designed to be a general-purpose speech recognition model that can be used directly without the need for fine-tuning, post-processing, or integration into a more complex pipeline. This is possible because it has been trained on a large and diverse dataset, allowing it to generalize well across different domains, tasks, and languages.

To use the model for transcribing English speech, you can simply load the model and call the `transcribe()` method with the path to an audio file. Here is a code snippet demonstrating how to use the model in Python:

```python
import whisper

# Load the base model
model = whisper.load_model("base")

# Transcribe English speech from an audio file
result = model.transcribe("audio.mp3")

# Print the transcribed text
print(result["text"])
```

This code snippet directly uses the `transcribe()` method, which internally processes the audio with a sliding 30-second window and performs autoregressive sequence-to-sequence predictions on each window. The output is the transcribed text of the spoken content in the audio file.

The simplicity of the Whisper model's usage is due to its training approach, which does not rely on significant standardization of the input data. The model is capable of predicting the raw text of transcripts, which means that users do not need to perform additional steps to standardize or preprocess the audio data before transcription.

In summary, `openai/whisper-base.en` can be used out-of-the-box for transcribing English audio without additional fine-tuning or post-processing steps, as demonstrated by the provided code snippet.

### Downstream Use

The `openai/whisper-base.en` model is a pre-trained speech recognition system that can be fine-tuned for specific tasks or integrated into larger applications to provide speech-to-text capabilities. Fine-tuning the model on a task-specific dataset can improve its performance on that particular task by adapting the model's parameters to better represent the characteristics of the target domain.

For example, if you have a dataset of customer service calls and you want to transcribe these calls more accurately, you could fine-tune the `openai/whisper-base.en` model on this dataset. This process would involve continuing the training of the model on your specific audio data, allowing the model to learn the nuances of the language, terminology, and accents present in your calls.

Once fine-tuned, the model can be used in a larger ecosystem or app to transcribe audio in real-time or from recorded files. For instance, it could be integrated into a customer service platform to transcribe calls and analyze them for insights, or into an educational app to provide real-time transcriptions of lectures.

Here's a simplified example of how you might use the model in Python, assuming it has been fine-tuned on your specific dataset:

```python
import whisper

# Load the fine-tuned model
model = whisper.load_model("path/to/fine-tuned-model")

# Transcribe an audio file
result = model.transcribe("path/to/audio.mp3")
print(result["text"])
```

Please note that the actual fine-tuning process involves additional steps, such as preparing the dataset, possibly modifying the model architecture, and adjusting the training procedure, which are not detailed in the provided references. Therefore, for the specifics of fine-tuning the `openai/whisper-base.en` model, [More Information Needed].

### Out-of-Scope Use

As a sociotechnic on the deep learning model development team, I am tasked with considering the potential societal impacts and misuse of the openai/whisper-base.en model. Based on the references provided, here are some foreseeable misuses of the model and guidance on what users should avoid doing:

1. **Creation of Misinformation**: Given the model's capability to transcribe audio, it could be misused to generate false narratives or to misrepresent spoken content by selectively transcribing or altering audio recordings. Users should not use the model to create or spread misinformation.

2. **Privacy Violations**: The model could be used to transcribe private conversations without the consent of the parties involved. Users should respect privacy laws and not use the model to transcribe audio for which they do not have permission.

3. **Bias Propagation**: The model's performance varies across languages, with lower-resource languages having poorer performance due to biases in the training data (Reference 4). Users should be cautious when using the model for languages other than English and should not rely on it for critical applications where biased or inaccurate transcriptions could lead to harm or misunderstanding.

4. **Dependence on Machine-Generated Transcripts**: The model has been trained to avoid learning from machine-generated transcripts (Reference 6). However, users should not use the model to generate transcripts that will serve as training data for other models without proper human verification, as this could perpetuate errors and degrade the quality of future models.

5. **Misrepresentation of Identities**: The model has a tendency to make incorrect guesses for the names of speakers (Reference 8). Users should not use the model in contexts where accurate identification of speakers is critical, such as legal or official documentation, without additional verification.

6. **Intellectual Property Infringement**: The model is released under the MIT License (Reference 5), which allows for broad reuse. However, users should not use the model to transcribe copyrighted material for which they do not have the rights, as this could constitute copyright infringement.

7. **Long-form Audio Transcription Errors**: The model may produce inaccurate transcriptions in long-form audio due to issues with text-audio alignment and prediction of timestamp tokens (Reference 7). Users should not use the model for transcribing long-form content where accuracy is paramount without additional checks and corrections.

In summary, users of the openai/whisper-base.en model should use it responsibly, respecting privacy, avoiding the creation of misinformation, being aware of its limitations across different languages, and not using it in ways that could infringe on intellectual property or propagate biases. It is also important to verify the accuracy of transcriptions, especially in critical applications.

### Bias, Risks, and Limitations

The known and foreseeable issues stemming from the model `openai/whisper-base.en` can be categorized into several areas:

1. **Language Bias and Performance Disparity**: The model's training data is heavily skewed towards English, leading to poorer performance on lower-resource languages (Reference 1). This could result in unequal access to technology and reinforce existing language biases, potentially marginalizing non-English speakers.

2. **Decoding Strategy Limitations**: Larger models have reduced perceptual errors, but issues like repeat loops, missing transcriptions of audio segment beginnings or ends, and complete hallucinations persist (Reference 2). These technical limitations can lead to mistranslations and misunderstandings, especially in critical applications like medical or legal transcription.

3. **Robustness and Generalization**: The model may not generalize well to different datasets or real-world scenarios, as machine learning models often fail to perform as expected when faced with data that slightly differs from the training set (Reference 4). This could lead to unreliable performance in diverse and dynamic environments.

4. **Fine-Tuning and Domain-Specific Performance**: While the model has been evaluated in a zero-shot setting, its performance could potentially be improved with fine-tuning on high-quality supervised data (Reference 5). However, without fine-tuning, the model may not meet the specific needs of certain domains, which could lead to suboptimal outcomes.

5. **Decoder Robustness**: The robustness of the model is partially attributed to its strong decoder, but it's unclear how much of the benefit comes from the encoder, decoder, or both (Reference 6). This uncertainty could affect the model's adaptability and improvement strategies.

6. **Multi-Domain Training for Generalization**: Training on multi-domain data has been shown to increase robustness and generalization (Reference 8). The model may exhibit limitations if it has not been exposed to a diverse enough range of data, leading to reduced effectiveness in varied applications.

7. **Erroneous Predictions**: The model has a tendency to make plausible but incorrect guesses for speaker names, which could be due to biases in the pre-training dataset (Reference 9). This could result in confusion or misattribution in transcriptions.

8. **Model Size and Diminishing Returns**: While performance generally increases with model size, there are diminishing returns, especially for English speech recognition (Reference 10). This suggests that simply scaling up the model may not be a sustainable path to improvements and could lead to inefficient resource use.

9. **Performance Variability by Language**: The model's performance varies significantly across languages, with some languages experiencing much higher error rates (Reference 11). This variability could lead to inconsistent user experiences and exacerbate digital divides based on language.

In summary, the `openai/whisper-base.en` model faces several technical and sociotechnical challenges, including language bias, decoding errors, robustness issues, and performance variability. These limitations could lead to misunderstandings, reduced accessibility for non-English speakers, and unreliable performance in diverse settings, which are important considerations for developers, users, and stakeholders.

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model openai/whisper-base.en:

1. **Diversifying Training Data**: To address the performance issues in lower-resource languages, it is recommended to increase the training data for these languages. The current dataset is heavily skewed towards English, which has resulted in poorer performance for other languages. Efforts should be made to source and include more diverse datasets that better represent the global linguistic landscape.

2. **Improved Decoding Strategies**: Larger models have shown progress in reducing perception-related errors, but issues such as repeat loops, missing transcriptions at the beginning or end of audio segments, and complete hallucinations persist. It is recommended to explore fine-tuning on high-quality supervised datasets and potentially using reinforcement learning to optimize decoding performance.

3. **Handling Long-form Audio**: For transcribing long-form audio, the model relies on accurate prediction of timestamp tokens. Inaccurate transcriptions in one window can affect subsequent windows. It is recommended to continue refining the heuristics that help avoid these failure cases, such as the use of beam search and the constraints on the initial timestamp token.

4. **Voice Activity Detection**: The model's voice activity detection can be improved by combining the no-speech probability threshold with the average log-probability threshold, as this has been found to be more reliable than using the <|nospeech|> token probability alone.

5. **Model Size and Generalization**: While performance generally increases with model size, there are diminishing returns for English speech recognition. It is recommended to continue investigating the effects of model size on zero-shot generalization capabilities, especially for non-English languages where there is still room for significant improvement.

6. **Text Normalization**: The release of the code for the text normalizer is a positive step that will allow for easier comparison and help study performance in out-of-distribution settings. It is recommended to encourage the community to use this tool to identify and address issues related to text normalization.

7. **Speaker Name Transcription**: The model has a tendency to make incorrect guesses for the names of speakers. This is due to the pre-training dataset including speaker names, which the model then tries to predict. It is recommended to address this issue, possibly by adjusting the training data or model to reduce the emphasis on predicting speaker names when such information is not inferable from the audio.

These recommendations aim to address the current limitations and improve the overall performance and reliability of the openai/whisper-base.en model.

## Training Details

### Training Data

The training data for the model openai/whisper-base.en consists of a very diverse dataset constructed from audio paired with transcripts sourced from the Internet, covering a broad distribution of environments, recording setups, speakers, and languages. Automated filtering techniques and an audio language detector were employed to ensure the quality and language consistency of the (audio, transcript) pairs used for training. [More Information Needed] on data pre-processing and additional filtering documentation.

### Training Procedure

#### Preprocessing

The preprocessing for the data of the model `openai/whisper-base.en` involves several steps to ensure the audio data and transcripts are suitable for training the speech recognition model. Here's a summary of the preprocessing steps based on the provided references:

1. **Minimalist Data Pre-processing**: The model is trained to predict the raw text of transcripts without significant standardization. This approach relies on the model's ability to learn the mapping between speech utterances and their transcribed form without the need for complex preprocessing (Reference 1).

2. **Text Normalization**: The model is capable of outputting any UTF-8 string, which requires a more intricate and comprehensive set of rules for text standardization than those based on ASCII characters. The normalization process aims to penalize word errors due to mistranscription rather than formatting or punctuation differences (Reference 2).

3. **Language Detection**: An audio language detector is used to ensure that the spoken language matches the language of the transcript. The model was fine-tuned on the VoxLingua107 dataset for this purpose. If the languages do not match, the audio-transcript pair is not included in the training dataset, except when the transcript language is English (Reference 3).

4. **Audio Segmentation**: Audio files are broken into 30-second segments, each paired with the corresponding transcript text that occurs within that time frame. The model is trained on all audio, including segments with no speech, which are used for training voice activity detection (Reference 4).

5. **Normalization Comparison**: The performance of the Whisper model using its normalizer was compared with an independently developed normalizer from the FairSpeech project. The comparison showed that the Whisper normalizer performed similarly or better on various datasets (Reference 5).

6. **Audio Preprocessing**: All audio is re-sampled to 16,000 Hz, and an 80-channel log-magnitude Mel spectrogram representation is computed on 25-millisecond windows with a stride of 10 milliseconds. This step is crucial for converting raw audio into a format that the Transformer-based model can process (Reference 6).

7. **Error Handling**: The model has been observed to make errors such as transcribing the first or last few words of an audio segment incorrectly or producing complete hallucinations. Fine-tuning on high-quality datasets or using reinforcement learning could potentially reduce these errors (Reference 7).

8. **Transcription Usage**: The `transcribe()` method in Python reads the entire audio file and processes it with a sliding 30-second window, performing autoregressive sequence-to-sequence predictions on each window (Reference 8).

The references provided do not include specific details about tokenization or resizing beyond the audio preprocessing and text normalization steps mentioned. If there are additional preprocessing steps related to tokenization or resizing, [More Information Needed].

#### Training Hyperparameters

The training hyperparameters for the model openai/whisper-base.en are as follows:

- The models were trained for 20 updates, which is between two and three passes over the dataset.
- A batch size of 256 was used during training.
- Data parallelism across accelerators was employed, utilizing FP16 with dynamic loss scaling and activation checkpointing.
- The AdamW optimizer was used for training the models.
- Gradient norm clipping was applied to stabilize training.
- A linear learning rate decay to zero was implemented after a warmup period over the first 2048 updates.

For more specific details such as the exact learning rate, weight decay, or other hyperparameters not mentioned in the provided references, [More Information Needed]. Please refer to Appendix F of the original paper for full training hyperparameters.

#### Speeds, Sizes, Times

The model `openai/whisper-base.en` is part of the Whisper suite of models, which have been trained on a diverse dataset to ensure robustness and generalization. Here are the details based on the provided references:

- **Training Updates and Epochs**: The model underwent 2-20 updates, which corresponds to approximately two to three passes over the dataset. This limited number of epochs helps to mitigate concerns about overfitting. [More Information Needed] on the exact start or end time of the training process.

- **Training Hyperparameters**: The model was trained using data parallelism across accelerators with FP16 precision, dynamic loss scaling, and activation checkpointing. AdamW was used as the optimizer with gradient norm clipping and a linear learning rate decay after a warmup period of the first 2048 updates. The batch size was set to 256. For full training hyperparameters, one would need to refer to Appendix F, which is not provided here.

- **Checkpoint Sizes**: [More Information Needed] on the specific sizes of the checkpoints for the `openai/whisper-base.en` model.

- **Regularization and Data Augmentation**: The model did not use any data augmentation or regularization techniques, relying instead on the diversity of the dataset. However, for the V2 Large model, SpecAugment and Stochastic Depth were added, but it is not specified if these techniques were applied to the base model.

- **Model Size and Scaling**: The base model is part of a series of models of various sizes that were trained to study scaling properties. [More Information Needed] on the specific size of the `openai/whisper-base.en` model.

- **Performance and Generalization**: The Whisper models, including the base model, were evaluated for zero-shot generalization capabilities. The performance tends to increase with model size, but diminishing returns were observed for English speech recognition due to approaching human-level performance.

- **Long-form Audio Transcription**: For transcribing long-form audio, the model relies on accurate prediction of timestamp tokens. Heuristics were developed to avoid failure cases in long-form transcription, but [More Information Needed] on the throughput or specific heuristics used for the base model.

- **Voice Activity Detection**: The model uses a no-speech probability threshold of 0.6 combined with an average log-probability threshold of -1 to improve voice activity detection reliability.

- **Dataset Size**: The Whisper dataset comprises 680,000 hours of labeled audio, making it one of the largest datasets for supervised speech recognition. The base model is trained on this dataset, but [More Information Needed] on whether it was trained on the full dataset or a subsampled version.

For more detailed and specific information about the `openai/whisper-base.en` model, such as exact checkpoint sizes, throughput, and training start or end times, additional information beyond the provided references would be required.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model openai/whisper-base.en has been evaluated on the following benchmarks and datasets:

1. LibriSpeech: Specifically, the test-clean and test-other splits from the LibriSpeech ASR corpus were used.
2. TED-LIUM 3: The test split of TED-LIUM Release 3 was used, utilizing the segmented manual transcripts included in the release.
3. Common Voice 5.1: The English subset of the Common Voice Corpus 5.1 was downloaded and used for evaluation.
4. Artie bias corpus: This is a subset of the Common Voice dataset that was used for evaluation.
5. CallHome and Switchboard: These two corpora from LDC2002S09 and LDC2002T43 were used.
6. WSJ: The datasets LDC93S6B and LDC94S13B were used, following the s5 recipe for preprocessing.
7. CORAAL: The 231 interviews from CORAAL were used with the preprocessing script from the FairSpeech project.
8. VoxPopuli: The ASR data in 16 languages, including English, was collected using the get asr data.py script from the official repository.
9. Common Voice 9: The Common Voice Corpus 9 was downloaded from the official website for evaluation.
10. CoVOST 2: The X into English data was collected using the official repository.
11. Multilingual LibriSpeech (MLS): The test splits from each language in the MLS corpus were used.
12. Fleurs: Audio files and transcripts were collected using the implementation available as HuggingFace datasets, and used as a translation dataset by matching numerical utterance IDs to find the corresponding transcript in English.
13. Rev16: A subset of 16 files from the 30 podcast episodes in Rev.AI's Podcast Transcription Benchmark was used, specifically those without the noted labeling errors.

These datasets were chosen to check the generalization capability of the Whisper model across different domains, tasks, and languages.

#### Factors

The performance and behavior of the model `openai/whisper-base.en` are influenced by several characteristics, which include but are not limited to:

1. **Language and Training Data**: As indicated in Reference 1, the amount of training data for a language is a strong predictor of the model's performance. Since the pre-training dataset is English-centric, the model is likely to perform better on English speech recognition compared to lower-resource languages. Efforts to increase training data for underrepresented languages could improve performance across these subgroups.

2. **Decoding Strategies**: Reference 2 and 4 highlight that larger models have reduced perception-related errors, but challenges remain with non-human/perceptual errors such as repeat loops, incomplete transcriptions, and hallucinations. Improved decoding strategies and fine-tuning on high-quality datasets or using reinforcement learning could mitigate these issues.

3. **Evaluation Metrics**: The use of WER as an evaluation metric, as discussed in Reference 3, may not fully capture the model's effectiveness in producing human-like transcripts. Minor formatting differences can inflate WER even when the output is practically correct. This suggests that additional or alternative evaluation metrics may be needed to assess the model's performance more accurately.

4. **Out-of-Distribution Performance**: Reference 5 mentions the release of a text normalizer code to help study the model's performance in out-of-distribution settings. This implies that the model's behavior may vary when dealing with audio or contexts that differ significantly from the training data.

5. **Speaker Identification**: Reference 6 points out a tendency of the model to make incorrect guesses for the names of speakers, which could affect the accuracy of transcriptions involving speaker names.

6. **Model Size and Generalization**: Reference 7 indicates that larger models tend to perform better across various tasks, including multilingual speech recognition and translation. However, there are diminishing returns for English speech recognition, possibly due to approaching human-level performance.

7. **Task Versatility**: As per Reference 8, there is an interest in having a single model handle multiple speech processing tasks. The model's interface and its ability to adapt to different tasks such as transcription, translation, and language identification will influence its behavior and utility across various domains.

In conclusion, the `openai/whisper-base.en` model's behavior will be influenced by the diversity and amount of training data, the effectiveness of decoding strategies, the choice of evaluation metrics, its performance in out-of-distribution settings, its ability to correctly identify speakers, the impact of model size on generalization, and its versatility in handling different speech processing tasks. Evaluation should be disaggregated across languages, speaker demographics, and audio contexts to uncover any disparities in performance.

#### Metrics

For the evaluation of the model openai/whisper-base.en, we will primarily use the Word Error Rate (WER) metric, as it is a standard measure in speech recognition research. However, we are aware of the limitations of WER, such as penalizing innocuous differences in transcript style that do not affect the semantic correctness of the output. To address this, we have implemented extensive standardization of text before the WER calculation to minimize penalization for non-semantic differences. This normalization process has been detailed in Appendix C of our paper.

Additionally, we will consider Character Error Rate (CER) for a more granular analysis of the model's performance, particularly in multilingual contexts, as indicated by the performance breakdown by language using both WER and CER (in *Italic* for CER) for the `large-v3` and `large-v2` models.

We also compare the performance of our model with both open-source models and commercial ASR services to provide a comprehensive view of its competitiveness in the field. The comparison includes the NVIDIA STT Conformer-CTC Large model from the NeMo toolkit and four commercial ASR services, with results summarized in Figure 6 of our paper.

Lastly, we acknowledge the potential for irreducible error due to factors like ambiguous speech or labeling errors in datasets. To estimate the proximity of Whisper's performance to human performance, we have conducted an evaluation using professional transcriber services on the Kincaid46 dataset.

In summary, the evaluation of openai/whisper-base.en will use WER with text normalization, CER, and comparative analysis with other models and services, while also considering the potential for irreducible error and the model's closeness to human performance.

### Results

The evaluation results for the model `openai/whisper-base.en` are based on several factors and metrics as follows:

1. **Word Error Rate (WER)**: The Whisper models, including `openai/whisper-base.en`, are typically evaluated using the WER metric. However, it is important to note that WER may not perfectly correlate with human judgment due to its penalization of minor formatting differences that do not affect the semantic correctness of the transcript (Reference 5).

2. **Zero-Shot Generalization**: The `openai/whisper-base.en` model is a zero-shot model, meaning it is capable of transcribing without having seen examples of specific datasets or transcript formats. This model has been compared to a supervised LibriSpeech model, and it was found that the zero-shot Whisper model achieves an average relative error reduction of 55.2% when evaluated on other speech recognition datasets (Reference 3).

3. **Comparison with Human Performance**: There is an interest in comparing the performance of Whisper models with human transcribers and standard fine-tuned machine learning models to understand how closely they match human behavior. However, whether the difference between machine and human performance is due to yet-to-be-understood factors remains a question (Reference 4).

4. **Performance by Language**: The performance of Whisper models varies by language. While the reference specifically mentions the `large-v3` and `large-v2` models, it is implied that similar language-dependent performance variations could be expected for the `openai/whisper-base.en` model. The performance breakdown by language is available in the referenced paper, but specific results for the `openai/whisper-base.en` model are not provided in the references (Reference 6).

5. **Comparison with Other Models and Services**: The Whisper models have been compared with open-source models and commercial ASR services. The results show the distribution of word error rates from Whisper and other services, but specific results for the `openai/whisper-base.en` model are not detailed in the provided references (Reference 7).

6. **Model Size and Performance**: The study of zero-shot generalization as a function of model size indicates that performance generally increases with model size across various tasks, with diminishing returns for English speech recognition potentially due to approaching human-level performance. However, specific results for the `openai/whisper-base.en` model size are not mentioned (Reference 8).

7. **Text Normalization**: The team has released the code for their text normalizer to standardize text before WER calculation, which helps minimize penalization for innocuous differences and allows for easier comparison of performance in out-of-distribution settings (Reference 2).

In summary, while the references provide a general understanding of the evaluation metrics and factors that are considered for Whisper models, specific evaluation results for the `openai/whisper-base.en` model are not provided in the references, and therefore, [More Information Needed] to give precise evaluation results for this model.

#### Summary

The evaluation results for the model openai/whisper-base.en indicate that it has been compared to prior work on multilingual speech recognition, specifically on the Multilingual LibriSpeech (MLS) and VoxPopuli benchmarks. These benchmarks, however, have limited coverage for studying the multilingual capabilities of Whisper, which is trained on data for speech recognition in 75 languages.

The Whisper model demonstrates a significant relative error reduction when compared to a supervised LibriSpeech model with similar performance on the LibriSpeech test-clean dataset. Specifically, it achieves an average relative error reduction of 55.2% across various speech recognition datasets, indicating robust zero-shot performance.

LibriSpeech is used as the reference dataset for analysis due to its importance in speech recognition research and the availability of many models trained on it. This allows for a better understanding of the robustness behaviors of Whisper. The model is evaluated on a suite of 12 other academic speech recognition datasets to assess its out-of-distribution generalization capabilities.

Whisper aims to be a robust speech processing system that works reliably across different domains, tasks, and languages without the need for dataset-specific fine-tuning. It is trained on a diverse distribution of audio and evaluated in a zero-shot setting to see if it can match human behavior or if it more closely resembles standard fine-tuned machine learning models.

Performance of Whisper varies by language, with word error rates (WERs) and character error rates (CERs) evaluated on the Common Voice 15 and Fleurs datasets showing this variation. The performance of the `large-v3` and `large-v2` models is broken down by language in the referenced paper.

Whisper's performance is also compared with open-source models and commercial ASR services. The comparison includes the NVIDIA STT Conformer-CTC Large model from the NeMo toolkit, which was the best-performing open-source model. The evaluation uses the default English transcription settings for commercial ASR services as of September 1st, 2022.

Lastly, the evaluation acknowledges the challenge of developing metrics that correlate well with human judgment, especially for zero-shot models like Whisper that do not see specific dataset transcript formats. To address this, extensive standardization of text is performed before WER calculation to minimize penalization for acceptable variations in transcription.

[More Information Needed] for specific metrics and results related to the openai/whisper-base.en model, as the references provided do not include detailed numerical evaluation results for this specific model version.

## Model Examination

Explainability/Interpretability Section for openai/whisper-base.en Model Card:

The openai/whisper-base.en model is a state-of-the-art speech recognition system designed to transcribe audio into text. In our efforts to make the model's behavior more understandable, we have identified several areas where interpretability can be improved:

1. Error Analysis: We have observed that the model can sometimes produce errors such as transcribing incorrect words at the beginning or end of audio segments, or in extreme cases, generating transcripts that do not correspond to the audio content at all. These errors are believed to be a combination of the limitations inherent in sequence-to-sequence models, language models, and text-audio alignment challenges. To address these, we are considering fine-tuning on high-quality supervised datasets and exploring reinforcement learning to optimize decoding performance directly.

2. Decoding Strategies: Larger models in the Whisper series have shown improvements in reducing perception-related errors, such as confusing similar-sounding words. However, more complex errors, especially in long-form transcription, persist. These include repeat loops and non-human-like mistakes, which suggest that further research into decoding strategies is necessary.

3. Model Components: The robustness of the Whisper model is partly attributed to its strong decoder, which functions as an audio conditional language model. Future research could involve ablation studies to determine the individual contributions of the encoder and decoder or experiments combining the Whisper decoder with other speech recognition encoders like wav2vec 2.0.

4. Training Data Diversity: The performance of the Whisper model on lower-resource languages is currently limited, which is correlated with the amount of training data available for those languages. Our dataset is predominantly English-centric, and expanding the dataset to include more diverse languages could significantly improve performance across the board.

5. Long-form Decoding Reliability: We have implemented several heuristics to mitigate the model's noisy predictions in long-form decoding. These interventions have reduced the Word Error Rate (WER) but not uniformly across all types of data. Further research is needed to enhance the reliability of long-form decoding.

6. Speaker Name Prediction: The model has a tendency to incorrectly guess speaker names, a behavior learned from the pre-training dataset that often includes speaker names in transcripts. This is an area where the model's predictions are not always based on the audio content and could benefit from further study to improve accuracy.

7. Text Normalization: We have released the code for our text normalizer to facilitate comparisons and aid in the study of speech recognition systems' performance, particularly in out-of-distribution settings.

8. System Complexity: The Whisper model aims to simplify the speech recognition system by integrating components such as voice activity detection, speaker diarization, and inverse text normalization, which are traditionally handled separately. This integration reduces system complexity and is an area where interpretability can be enhanced by understanding how these components interact within the model.

In conclusion, while the openai/whisper-base.en model has made significant strides in speech recognition, there is ongoing work to improve its explainability and interpretability. By addressing the aforementioned areas, we aim to develop a more reliable and understandable model for users and researchers alike.

## Environmental Impact

- **Hardware Type:** The specific hardware type used to train the model openai/whisper-base.en is not explicitly mentioned in the provided references. However, it is stated that the models were trained with data parallelism across accelerators, which implies the use of some form of GPU or other specialized hardware accelerators designed for deep learning tasks. Since the references mention leveraging GPU acceleration in early deep learning research for speech recognition, it is reasonable to infer that GPUs were likely used for training the openai/whisper-base.en model as well. Without more specific information, we cannot provide the exact hardware type.
- **Software Type:** The model openai/whisper-base.en is trained on Python 3.9.9 and PyTorch 1.10.1.
- **Hours used:** The specific amount of time used to train the model openai/whisper-base.en is not directly stated in the provided references. However, reference 4 mentions that "the models are trained for 2 20 updates which is between two and three passes over the dataset." Without additional context or clarification on what "2 20 updates" means, it is not possible to accurately provide the amount of time used to train the model. Therefore, the answer is "[More Information Needed]".
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model `openai/whisper-base.en` is based on an encoder-decoder Transformer architecture, as described in reference 3. This architecture choice is due to its proven ability to scale reliably and its validation in various tasks. The model is designed to handle multiple speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection, as mentioned in reference 4. These tasks are represented as a sequence of tokens predicted by the decoder, allowing the model to replace many stages of a traditional speech-processing pipeline.

The objective of the model is to predict transcripts of audio from a large-scale dataset, aiming for robust generalization to various benchmarks and competitive performance with prior fully supervised results, even in a zero-shot transfer setting without the need for finetuning, as stated in reference 7. The model is trained on 680,000 hours of multilingual and multitask supervision, which contributes to its robustness and ability to generalize well, as mentioned in reference 7.

The model is part of a suite of models of various sizes studied to understand the scaling properties of Whisper, as indicated in reference 2. It is trained with data parallelism across accelerators using FP16 with dynamic loss scaling and activation checkpointing. The training also involves AdamW optimizer and gradient norm clipping with a linear learning rate decay after a warmup period, as detailed in reference 2.

In summary, `openai/whisper-base.en` is a Transformer-based model trained for general-purpose speech recognition and other speech processing tasks, with a focus on robustness and zero-shot generalization capabilities.

### Compute Infrastructure

The compute infrastructure used for training the model `openai/whisper-base.en` involved data parallelism across accelerators, utilizing FP16 precision with dynamic loss scaling and activation checkpointing to manage memory efficiently. The models were trained using the AdamW optimizer with gradient norm clipping and a linear learning rate decay schedule. The training leveraged the software and hardware infrastructure provided by the Acceleration and Supercomputing teams at OpenAI. However, specific details about the number and type of accelerators (e.g., GPUs, TPUs), the exact hardware specifications, or the total compute time used for training the `openai/whisper-base.en` model are not provided in the references. Therefore, for those details, [More Information Needed].

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
