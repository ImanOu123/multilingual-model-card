# Model Card for google/flan-t5-xxl

google/flan-t5-xxl is an 11 billion parameter language model that has been instruction-finetuned using the Flan (Finetuning language models) procedure on a large collection of datasets. Flan-T5-XXL achieves strong few-shot performance compared to even much larger models, but should not be used directly in applications without first assessing safety and fairness concerns.

## Model Details

### Model Description

Model Architecture:
- Dense encoder-decoder model
- 11 billion parameters

Training Procedures:
- Instruction finetuning procedure called Flan (Finetuning language models) was used
- Finetuned on a collection of 1,836 tasks from mixtures like Muffin, T0-SF, NIV2, and CoT
- Used a constant learning rate schedule
- Finetuned using the JAX-based T5X framework
- [More Information Needed] on specific hyperparameters like learning rate, batch size, dropout, and number of finetuning steps

Important Disclaimers:
- Flan-T5-XXL has not been tested in real world applications
- The model was finetuned on a large text corpus that was not filtered for explicit content or assessed for biases
- The model may be vulnerable to generating inappropriate content or replicating inherent biases from the training data

- **Developed by:** Hyung Won; Chung * Le Hou; Shayne Longpre; Barret Zoph; Yi Tay; William Fedus; Yunxuan Li; Xuezhi Wang; Mostafa Dehghani; Siddhartha Brahma; Albert Webson Shixiang; Shane Gu; Zhuyun Dai; Mirac Suzgun; Xinyun Chen; Aakanksha Chowdhery; Alex Castro-Ros; Marie Pellat; Kevin Robinson; Dasha Valter; Sharan Narang; Gaurav Mishra; Adams Yu; Vincent Zhao; Yanping Huang; Andrew Dai; Hongkun Yu; Slav Petrov; Ed H Chi; Jeff Dean; Jacob Devlin; Adam Roberts; Denny Zhou Quoc; V Le; Jason Wei;  Google
- **Funded by:** Based on the provided references, the model google/flan-t5-xxl was developed by researchers at Google, including Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson Shixiang, Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou Quoc, V Le, and Jason Wei.

The references do not explicitly mention the funding organizations for this project. [More Information Needed] on the specific funding sources.
- **Shared by:** Based on the references provided, the main contributors who made the model google/flan-t5-xxl available online are:

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson Shixiang, Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou Quoc, V Le, Jason Wei from Google.

The references do not provide specific information on who created the GitHub repo or uploaded the model. [More Information Needed]
- **Model type:** The google/flan-t5-xxl model is an encoder-decoder language model that was initialized from a pretrained T5 model and then instruction fine-tuned on a collection of datasets spanning a variety of tasks, in order to improve its zero-shot and few-shot performance on unseen tasks.
- **Language(s):** The model google/flan-t5-xxl takes text as input and processes natural human language.
- **License:** [More Information Needed]

The provided references do not contain any information about the specific license being used for the google/flan-t5-xxl model. More details would be needed from the model developers or documentation to determine the applicable license.
- **Finetuned from model:** The model google/flan-t5-xxl is fine-tuned from the PaLM model, as mentioned in the references:

"The model is initialized from PaLM . We also have a variation initialized from U-PaLM (Tay et al., 2022b)."

However, no direct link to the PaLM model is provided in the given references. 

[More Information Needed] for the link to the base PaLM model.
### Model Sources

- **Repository:** https://github.com/google-research/t5x
- **Paper:** https://arxiv.org/pdf/2210.11416.pdf
- **Demo:** [More Information Needed]

Based on the provided references, there is no information about a demo link for the google/flan-t5-xxl model. The references discuss various aspects of the Flan-T5 models, such as their evaluation on benchmarks, ethical considerations, and prompts used, but do not mention a specific demo link for the XXL variant.
## Uses

### Direct Use

The model google/flan-t5-xxl can be used for zero-shot inference on various NLP tasks such as reasoning and question answering without requiring any fine-tuning, post-processing or plugging into a pipeline. 

To run inference, you can use the `t5x/infer.py` script as shown in the code snippet below:

```sh
INFER_OUTPUT_DIR="..."  # directory to write infer output
T5X_DIR="..."  # directory where the t5x is cloned, e.g., ${HOME}"/t5x".
TFDS_DATA_DIR="..."
CHECKPOINT_PATH="..."

python3 ${T5X_DIR}/t5x/infer.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_infer.gin" \
  --gin.CHECKPOINT_PATH=\"${CHECKPOINT_PATH}\" \
  --gin.INFER_OUTPUT_DIR=\"${INFER_OUTPUT_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
```

The same `seqio.Task` used for training can be leveraged for inference as well, but the targets features are only used for logging predictions and not for the actual inference.

[More Information Needed] on the specific `seqio.Task` to use for the NLP task of interest with google/flan-t5-xxl.

### Downstream Use

The google/flan-t5-xxl model can be used for various natural language processing tasks when fine-tuned on specific datasets or instructions. Some key use cases include:

- Zero-shot and few-shot learning on NLP tasks like reasoning and question answering, by providing task instructions or a few examples (based on Reference 8)
- Translating between languages, when fine-tuned on machine translation datasets like wmt_t2t_ende_v003 (based on Reference 7). Example code for fine-tuning:

[More Information Needed]

- Improving performance on a wide range of NLP benchmarks compared to base T5 models, thanks to instruction fine-tuning (based on Reference 9)

In general, instruction fine-tuning allows the model to better generalize to new tasks from natural language descriptions (Reference 10). The model can be plugged into NLP applications or research projects that require strong zero-shot or few-shot performance on language understanding and generation.

To maximize the model's capabilities, fine-tuning on a large collection of diverse NLP tasks phrased as instructions is recommended, potentially including chain-of-thought data for reasoning tasks (Reference 11). Scaling up the number of tasks, model size, and using chain-of-thought fine-tuning can further boost performance (Reference 11).

### Out-of-Scope Use

Based on the provided references, the google/flan-t5-xxl model may foreseeably be misused in the following ways:

1. Generating abusive, toxic, or harmful language. The model should not be used for generating such content. (Reference 1, 4, 6)

2. Replicating or amplifying inherent biases present in the underlying training data. The model's outputs may contain representational biases or specific forms of gender bias. (Reference 4, 6)

3. Deploying the model directly in real-world applications without proper assessment of safety, fairness, and potential harms specific to the application context. (Reference 2, 5, 6)

Users ought not to:

1. Apply the model for any unacceptable use cases, such as generating abusive speech. (Reference 1)

2. Use the model directly in any application without a prior assessment of safety and fairness concerns specific to the application. (Reference 5)

3. Assume that the instruction-finetuned model alone is sufficient for safe deployment. Additional measures should be taken to assess risks and potential harms in the application context before deployment. (Reference 2, 3)

[More Information Needed] regarding specific examples of potential misuse or harm in real-world applications, as the model has not been tested in such contexts. (Reference 6)

### Bias, Risks, and Limitations

Based on the provided references, here are the known or foreseeable issues stemming from the google/flan-t5-xxl model:

1. Potential for generating toxic, abusive, or inappropriate content: The model was trained on a large corpus of text data that was not filtered for explicit content, making it potentially vulnerable to generating harmful content (References 5, 8).

2. Replicating inherent biases in the underlying training data: As the training data was not assessed for existing biases, the model may reproduce these biases in its outputs (References 5, 8).

3. Risks and potential harms in specific application contexts: While instruction finetuning improves many capabilities, downstream developers should still consider the full range of potential risks and anticipate application-specific risks before deployment (References 2, 3, 6, 7).

4. Lack of real-world testing: The model has not been tested in real-world applications, which may lead to unforeseen issues or limitations (References 5, 8).

5. [More Information Needed] on the model's performance in complex reasoning, planning, and explanation tasks, as the provided references do not specifically mention google/flan-t5-xxl in this context (Reference 9 mentions Flan-PaLM, but not google/flan-t5-xxl).

To mitigate these risks, it is crucial to conduct a thorough assessment of safety and fairness concerns specific to the intended application before deploying the google/flan-t5-xxl model (References 6, 7).

### Recommendations

Based on the provided references about the google/flan-t5-xxl model, here are some recommendations regarding foreseeable issues:

1. The model should not be used for generating abusive speech or other unacceptable content. Appropriate safety measures and filters should be put in place before deployment.

2. Downstream developers need to carefully assess the full range of potential risks and harms specific to their application context before using the model. The model alone cannot be fully evaluated for all possible issues.

3. The model may be vulnerable to generating inappropriate content or replicating inherent biases from the large text corpus it was trained on, which was not filtered. Thorough testing for toxic language, representational bias, gender bias, and other harms should be conducted.

4. The model has not been tested in real-world applications. Extensive real-world testing and monitoring for emerging issues is recommended before deployment. 

5. While instruction fine-tuning improves many capabilities, it may lead to new or different forms of bias and harm in certain use cases. These risks need to be anticipated and mitigated.

6. [More Information Needed] on specific design decisions around prompting and sampling to mitigate potential harms.

In summary, google/flan-t5-xxl is a powerful model but comes with significant risks that require proactive assessment, testing, and mitigation measures before it can be responsibly deployed. Real-world piloting and ongoing monitoring for emerging issues is strongly advised.

## Training Details

### Training Data

The training data of the model google/flan-t5-xxl is a combination of tasks from FLAN, T0, Natural Instructions, along with some dialog, program synthesis, and chain-of-thought reasoning tasks, as described in Figure 2 of the referenced paper. The specific datasets used are publicly available and pointers and citations are provided in Table 24 of the paper.

### Training Procedure

#### Preprocessing

For the preprocessing of the data used to train the google/flan-t5-xxl model:

Tokenization:
The model uses the SentencePiece vocabulary 'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model' with 100 extra IDs, as shown in this code snippet from the references:

vocabulary = seqio.SentencePieceVocabulary(
    'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model', extra_ids=100)
output_features = {
    'inputs': seqio.Feature(vocabulary=vocabulary),
    'targets': seqio.Feature(vocabulary=vocabulary)
}

Resizing/Rewriting: 
[More Information Needed]

Other preprocessing details:
- The model was finetuned on a combination of 1,836 tasks from the Muffin, T0-SF, NIV2, and CoT mixtures.
- Instructional templates were used for each task in the Muffin, T0-SF and NIV2 mixtures. For the CoT mixture, around 10 instruction templates were manually written for each of the 9 datasets.
- To create few-shot templates, various exemplar delimiters like "Q:"/"A:" were written and randomly applied at the example level.

[More Information Needed] on other specific preprocessing steps like normalization, filtering, etc.

#### Training Hyperparameters

Here are the training hyperparameters for the google/flan-t5-xxl model based on the provided references:

Model size:
- Flan-T5-XXL has 11 billion weights. (Reference 7)

Important hyperparameters:
- Learning rate, batch size and dropout were the most important hyperparameters for instruction finetuning. (Reference 2)

Optimizer:
- Adafactor optimizer was used. (Reference 3)

Packing:
- Packing was used to combine multiple training examples into a single sequence, separating inputs from targets using an end-of-sequence token. Masking was applied to prevent the tokens from attending to others across the packed example boundary. (Reference 3)

Compute used for finetuning:
- [More Information Needed]

Specific values for learning rate, batch size, dropout, and number of finetuning steps:
- [More Information Needed] (Reference 4 mentions these values are provided in Appendix E, but Appendix E is not included in the given references)

Training script:
```sh
[More Information Needed]
```
(Reference 5 mentions using `t5x/train.py` script to run a training job, but the full command is not provided)

#### Speeds, Sizes, Times

Based on the provided references, here are the details I can provide about the google/flan-t5-xxl model:

Model size:
- Flan-T5-XXL has 11 billion parameters. (Reference 3)

Checkpoint format:
- The T5 model checkpoints, which Flan-T5 is based on, have been released in a native T5X format for maximal efficiency. (Reference 5)

Safety considerations:
- Flan-T5 should not be used directly in any application without first assessing safety and fairness concerns specific to that application, as language models like Flan-T5 could potentially be used for harmful language generation. (Reference 6)

Performance notes:
- Flan-T5-XXL's responses seem to be more influenced by the toxicity level of the input prompts compared to Flan-PaLM models, sometimes even producing more toxic responses than human baselines. This needs further investigation. (Reference 8)

[More Information Needed]:
- Throughput details
- Training start and end times
- Exact checkpoint sizes

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the google/flan-t5-xxl model was evaluated on the following benchmarks and datasets:

1. MMLU (Hendrycks et al., 2020) - includes exam questions from 57 tasks such as mathematics, history, law, and medicine. Evaluated using both direct prompting and chain-of-thought (CoT) prompting.

2. BBH (BIG-Bench Hard) - [More Information Needed]. Evaluated using both direct prompting and chain-of-thought (CoT) prompting.

3. TyDiQA - a question-answering benchmark across 8 typologically diverse languages. Evaluated using direct prompting exact-match score.

4. MGSM (Shi et al., 2022) - a multilingual benchmark of math word problems from Cobbe et al. (2021) manually translated into 10 languages. [More Information Needed on evaluation method]

5. Some Responsible AI benchmarks - [More Information Needed]

The model's performance was also summarized using a "normalized average" metric, which is the macro-average over six normalized scores: MMLU-Direct, MMLU-CoT, BBH-Direct, BBH-CoT, TyDiQA-Direct, and MGSM-CoT.

#### Factors

Based on the provided references, here are some key characteristics that may influence the behavior of the google/flan-t5-xxl model:

1. Domain and context of application: The model's performance and potential risks/harms can vary depending on the specific downstream use case it is applied to. Developers should carefully consider risks in their particular application context before deployment. [References 1-3]

2. Sensitive/unacceptable use cases: The model should not be used for generating abusive speech or other unacceptable content, as it may be vulnerable to replicating inappropriate content or biases from its training data. [References 4, 9]

3. Gender bias: Evaluation on gender bias benchmarks shows the model performs better on stereotypical examples compared to "gotcha" examples that go against stereotypes. There are also disparities in performance when the correct output uses "she" vs "he" pronouns. This indicates a need for disaggregated evaluation across gender. [References 5-7]

4. Dehumanization: Some potential dehumanization harms were noticed in error analysis that are not captured by current benchmarks. [Reference 6]

5. Population subgroups: [More Information Needed] The references do not provide details on specific population subgroups that may be disparately impacted.

In general, disaggregated evaluation across various factors is recommended to uncover potential performance disparities and harms. Real-world testing and risk assessment specific to the application context is also advised before deploying the model. [References 3, 9, 11]

#### Metrics

Based on the references provided, the following metrics will be used to evaluate the google/flan-t5-xxl model, considering tradeoffs between different types of errors:

1. Performance on held-out tasks not included in the finetuning data, focusing on world knowledge and reasoning tasks. This includes evaluating the model on a range of benchmarks, including multilingual ones (Reference 1).

2. A single "normalized average" metric, which is the macro-average over six normalized scores: MMLU-Direct, MMLU-CoT, BBH-Direct, BBH-CoT, TyDiQA-Direct, and MGSM-CoT (Reference 2).

3. Direct prompting and chain-of-thought (CoT) prompting for MMLU and BBH benchmarks, and direct prompting exact-match score for TyDiQA (Reference 4).

4. [More Information Needed] on specific metrics used for Responsible AI evaluations mentioned in Reference 5.

5. [More Information Needed] on metrics used to assess potential harms to end users, including toxic language harms, representational bias, and specific forms of gender bias (Reference 6).

6. Manual evaluation investigating the effect of instruction finetuning on the ability for models to give open-ended responses to challenging inputs, using an evaluation set of 190 examples (Reference 10).

### Results

Based on the provided references, here are the evaluation results for the google/flan-t5-xxl model:

Evaluation Benchmarks:
- MMLU (exam questions from 57 tasks)
- BBH (BIG-Bench Hard) 
- TyDiQA (question-answering across 8 diverse languages)
- MGSM (multilingual math word problems in 10 languages)
- Responsible AI benchmarks

Evaluation Methods and Metrics:
- Direct prompting (model directly predicts answer)
- Chain-of-thought (CoT) prompting (model provides reasoning before final answer)
- Exact-match score for TyDiQA direct prompting
- Normalized average metric (macro-average over normalized scores for MMLU-Direct, MMLU-CoT, BBH-Direct, BBH-CoT, TyDiQA-Direct, MGSM-CoT)

Key Results:
- Flan-T5-XL (3B parameters) achieved 52.4% on MMLU, surpassing GPT-3 175B (43.9%)
- Flan-PaLM outperformed PaLM on several Responsible AI benchmarks measuring toxic language harms
- Instruction finetuning improved open-ended responses to challenging inputs in a manual evaluation of 190 examples
- [More Information Needed] for specific scores on BBH, TyDiQA, MGSM and the normalized average metric

Real-World Testing:
- Flan-T5 has not been tested in real-world applications

Ethical Considerations & Risks:
- Flan-T5 was finetuned on unfiltered data and may be vulnerable to generating inappropriate content or replicating data biases

[More Information Needed] for code examples, as no direct code blocks were provided in the references.

#### Summary

Based on the provided references, here is a summary of the evaluation results for the google/flan-t5-xxl model:

- Flan-T5-XL, despite being only 3B parameters, achieved a MMLU score of 52.4%, surpassing GPT-3 175B's score of 43.9%. This shows the effectiveness of instruction finetuning.

- In a manual evaluation on an evaluation set of 190 examples, Flan-T5-XXL prefixed responses with articles (e.g. "the engineer") in 96% of cases, especially in zero-shot settings. After adapting the exact string matching scoring to ignore articles, Flan-PaLM's performance exceeded PaLM at every scale, approaching human baseline performance even with the smallest 8B model and in zero-shot settings. 

- [More Information Needed] on Flan-T5-XXL's specific performance on other benchmarks like TyDiQA, MGSM, BBH, and Responsible AI evaluations, as the references focus more on Flan-PaLM.

In summary, the references highlight the strong zero-shot and few-shot capabilities of Flan-T5-XXL compared to much larger models, demonstrating the power of instruction finetuning. However, more specific results on Flan-T5-XXL across the full range of benchmarks would be needed to give a complete picture of its performance.

## Model Examination

[More Information Needed]

The provided references do not contain specific information about explainability or interpretability work done on the google/flan-t5-xxl model. The references discuss the model's training procedure, potential risks and ethical considerations, and the general goal of developing models that can generalize to unseen tasks. However, there are no details about efforts to make the model's predictions or inner workings more explainable or interpretable. More information would be needed from the model developers or additional research to adequately address this experimental section in the model card.

## Environmental Impact

- **Hardware Type:** Based on the provided references, the model google/flan-t5-xxl was likely trained on TPUs (Tensor Processing Units). Specifically, it mentions:

"We use the JAX-based T5X framework"

and

"T5X can be run with [XManager](https://github.com/deepmind/xmanager) on
[Vertex AI](https://cloud.google.com/vertex-ai). Vertex AI is a platform for
training that creates TPU instances and runs code on the TPUs."

So while it doesn't explicitly state the hardware used for google/flan-t5-xxl, given that it uses the T5X framework which is designed to run on TPUs via Vertex AI, it's reasonable to infer that TPUs were likely the hardware used for training this model.
- **Software Type:** Based on the provided references, the model google/flan-t5-xxl is likely trained using the T5X framework, which is a JAX and Flax based implementation of the original T5 codebase. The references mention that T5X is a "new and improved implementation" of T5 using JAX and Flax.

However, to definitively state the exact software used to train the specific model google/flan-t5-xxl, [More Information Needed]. The references discuss T5 and T5X in general terms, but do not specify the software used for the flan-t5-xxl variant.
- **Hours used:** Based on the provided references, the exact amount of time used to train the Flan-T5-XXL model is not explicitly stated. The references mention training details for some Flan models, but not specifically for the XXL variant.

[More Information Needed] on the training time for the Flan-T5-XXL model.
- **Cloud Provider:** Based on the provided references, the model google/flan-t5-xxl was likely trained on Google Cloud using TPUs. Specifically, the references mention:

1. A quick start guide for training models with TPUs on Google Cloud.
2. Using XManager on Vertex AI, a Google Cloud platform for training models on TPUs.

However, the references do not explicitly state that the google/flan-t5-xxl model was trained using this setup. To confirm this, [More Information Needed] directly related to the training of the google/flan-t5-xxl model.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the google/flan-t5-xxl model. To answer this question, more specific details would be needed regarding the computational resources used, energy consumption, and carbon footprint of the training process.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
- Dense encoder-decoder model
- Based on pretrained T5-XXL (11B parameters) and fine-tuned with instructions for better zero-shot and few-shot performance

Training Objective:
The model was finetuned using the Flan (Finetuning language models) procedure on a collection of datasets phrased as instructions. The objective was to improve the model's ability to follow instructions and generalize to unseen tasks.

[More Information Needed] on the exact training hyperparameters, number of training steps, and hardware used.

### Compute Infrastructure

Based on the provided references, here is the information about the compute infrastructure used for the google/flan-t5-xxl model:

The model was trained using the JAX-based T5X framework on TPU chips (approximately 512 v4 TPU chips for 37 hours for instruction finetuning Flan-PaLM 540B, which uses 0.2% of the pre-training compute). 

T5X can also be run on GPUs in single-node or multi-node configurations with a SLURM+pyxis cluster. NVIDIA has released an updated version of the T5X repository with H100 FP8 support and broad GPU performance improvements in the NVIDIA Rosetta repository.

[More Information Needed] on the exact compute infrastructure used for pre-training and finetuning the google/flan-t5-xxl model specifically. The references provide general information about T5X compute but do not give specifics for this particular model.

## Citation

```
@misc{hyung-scaling,
    author = {Hyung Won and
              Chung * Le Hou and
              Shayne Longpre and
              Barret Zoph and
              Yi Tay and
              William Fedus and
              Yunxuan Li and
              Xuezhi Wang and
              Mostafa Dehghani and
              Siddhartha Brahma and
              Albert Webson Shixiang and
              Shane Gu and
              Zhuyun Dai and
              Mirac Suzgun and
              Xinyun Chen and
              Aakanksha Chowdhery and
              Alex Castro-Ros and
              Marie Pellat and
              Kevin Robinson and
              Dasha Valter and
              Sharan Narang and
              Gaurav Mishra and
              Adams Yu and
              Vincent Zhao and
              Yanping Huang and
              Andrew Dai and
              Hongkun Yu and
              Slav Petrov and
              Ed H Chi and
              Jeff Dean and
              Jacob Devlin and
              Adam Roberts and
              Denny Zhou Quoc and
              V Le and
              Jason Wei and
              Google},
    title  = {Scaling Instruction-Finetuned Language Models},
    url    = {https://arxiv.org/pdf/2210.11416.pdf}
}
```

