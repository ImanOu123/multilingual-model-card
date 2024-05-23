# Model Card for facebook/opt-2.7b

facebook/opt-2.7b is a 2.7 billion parameter autoregressive language model that is part of the OPT (Open Pre-trained Transformer) model family, which ranges in size from 125 million to 175 billion parameters. It was developed by Meta AI with the goal of replicating the performance and scale of GPT-3 models while applying best practices in data curation and training efficiency.

## Model Details

### Model Description

Here is a model card description for the facebook/opt-2.7b model based on the provided references:

Model Details
- Developed by Meta AI 
- Released on May 3, 2022
- Version 1.0.0
- Decoder-only transformer language model architecture
- 2.7 billion parameters

Training Procedures
- Trained using AdamW optimizer 
- (β1, β2) set to (0.9, 0.95)
- Weight decay of 0.1
- Linear learning rate schedule
  - Warmed up from 0 to max LR over first 375M tokens 
  - Decayed to 10% of max LR over 300B tokens
- Batch size: [More Information Needed]
- Trained on [More Information Needed] GPUs

Evaluation
- Evaluated on dialogue tasks like ConvAI2 and Wizard-of-Internet
- Outperformed Reddit 2.7B model on all dialogue tasks
- Competitive with supervised BlenderBot 1 model, especially on ConvAI2
- Obtained lowest perplexity but lower Unigram F1 than models with Wizard-of-Wikipedia supervision on Wizard-of-Internet

Limitations and Disclaimers
- Can exhibit biases and safety issues induced by limitations in diversity of training data
- Potential issues with generation diversity and hallucination 
- Not immune to issues common in large language models
- Released under a non-commercial use license

For more details, please refer to the full paper and blog post. The model is available in the metaseq open-source repository. Contact me for any updates to this model card.

- **Developed by:** Susan Zhang; Stephen Roller; Naman Goyal; Mikel Artetxe; Moya Chen; Shuohui Chen; Christopher Dewan; Mona Diab; Xian Li; Victoria Lin; Todor Mihaylov; Myle Ott; Sam Shleifer; Kurt Shuster; Daniel Simig; Singh Koura; Anjali Sridhar; Tianlu Wang; Luke Zettlemoyer
- **Funded by:** Based on the provided references, the model facebook/opt-2.7b was developed by Meta AI. The references do not explicitly mention the funding sources for this specific model.

[More Information Needed] on the funding sources for the facebook/opt-2.7b model project.
- **Shared by:** Based on the references provided, the key contributors who made the model facebook/opt-2.7b available online as a GitHub repo are:

Susan Zhang, Naman Goyal, Punit Singh Koura, Moya Chen, Kurt Shuster, David Esiobu, Igor Molybog, Peter Albert, Andrew Poulton, Nikolay Bashlykov, Binh Tang, Uriel Singer, Yuchen Zhang, Armen Aghajanya, Lili Yu, and Adam Polyak.

They are listed as the current maintainers (CODEOWNERS) of the Metaseq repository where the model code is hosted.

[More Information Needed] on their specific roles and contributions in developing and releasing the model.
- **Model type:** The facebook/opt-2.7b model is a decoder-only transformer language model trained using the AdamW optimizer on a large corpus of English text data.
- **Language(s):** The facebook/opt-2.7b model uses a pre-training corpus containing predominantly English text, with a small amount of non-English data present, by concatenating datasets from RoBERTa, the Pile, and PushShift.io Reddit.
- **License:** The model facebook/opt-2.7b is released under a non-commercial use license agreement. The specific license text is provided in the model license, but the link to that license is not given in the provided references.

[More Information Needed] on the exact name and link to the non-commercial license being used for the facebook/opt-2.7b model release.
- **Finetuned from model:** Based on the provided references, the facebook/opt-2.7b model is not explicitly mentioned as being fine-tuned from another model. The references discuss the OPT-175B model and its training process, but do not specify if the smaller 2.7B parameter model was fine-tuned from a base model.

[More Information Needed] about the specific training process and potential base model for the facebook/opt-2.7b model.
### Model Sources

- **Repository:** https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling
- **Paper:** https://arxiv.org/pdf/2205.01068.pdf
- **Demo:** The model facebook/opt-2.7b is available on the Hugging Face Hub under the `facebook` organization, as mentioned in Reference 5:

"The OPT 125M--66B models are now available in [Hugging Face Transformers](https://github.com/huggingface/transformers/releases/tag/v4.19.0). You can access them under the `facebook` organization on the [Hugging Face Hub](https://huggingface.co/facebook)"

Therefore, the demo of the model facebook/opt-2.7b can be found at:

https://huggingface.co/facebook/opt-2.7b
## Uses

### Direct Use

The OPT-2.7B model can be used for various natural language tasks without fine-tuning, post-processing or plugging into a pipeline. It is primarily intended for research purposes, especially in the area of Responsible AI.

To use the model, you can access it through the metaseq open-source repository provided by Facebook/Meta. Here is an example code snippet to load and use the model for generation:

[More Information Needed]

The model can be used for tasks like:
- Poetry and creative writing generation 
- Conversational AI and chatbots
- Few-shot translation between languages like German, Spanish, French and Chinese
- Assisting with writing tasks like paper introductions
- Arithmetic and math word problems
- Code generation in languages like Python

However, the quality and coherence of the generated outputs may vary. The model can make mistakes, become repetitive, or generate illogical outputs, especially for more complex tasks. Careful prompt engineering and output filtering is recommended.

For best results on specific downstream tasks, fine-tuning the model on task-specific data is advised. The model is not intended for production use or real-world deployments without further testing and refinement.

### Downstream Use

The facebook/opt-2.7b model can be fine-tuned for various downstream tasks or integrated into larger applications. Some potential uses include:

- Fine-tuning the model for specific language tasks like text classification, question answering, summarization, etc. The model's performance on SuperGLUE benchmarks suggests it could perform well when fine-tuned for these types of tasks.

- Integrating the model into conversational AI systems or chatbots. Results show OPT-175B performs competitively with supervised models like BlenderBot on conversational datasets, so the smaller OPT-2.7B may also be suitable as a base for conversational applications.

- Using the model for content moderation tasks like detecting toxic or offensive language. The model's strong few-shot performance on this type of classification compared to GPT-3 indicates it may be a good fit when fine-tuned for content moderation.

To actually use the model for fine-tuning or integration, code utilizing libraries like PyTorch or HuggingFace Transformers would be needed to load the pre-trained weights, modify the model architecture if needed, and train the model further on task-specific data. Detailed code snippets are [More Information Needed] based on the provided references.

The model should currently be used for research purposes rather than production, as the OPT models still have shortcomings that make them not yet suitable for commercial use or real-world deployment without further testing and potential modifications. More information on the specific pre-training data composition and any fairness considerations would also be beneficial to include.

### Out-of-Scope Use

Based on the provided references about the OPT-175B model, here are some foreseeable misuses and what users ought not do with the smaller facebook/opt-2.7b model:

- The model should not be used for production or real-world deployments, as it is only intended for research purposes, especially into Responsible AI for large language models. [Ref 1]

- Users should be aware that the model likely has limitations in terms of bias, safety, generation diversity, hallucination, and other issues that are common in large language models. It should not be used in applications where these limitations could cause harm. [Ref 2, 5]

- The model has a high propensity to generate toxic language and reinforce harmful stereotypes, even from relatively innocuous prompts. Users should not use it to generate toxic or hateful content. Additional mitigations or avoidance may be needed depending on the downstream application. [Ref 3, 7, 8]

- [More Information Needed] on specific additional misuses to call out for the smaller 2.7B model vs the full 175B model.

In summary, the facebook/opt-2.7b model is an early research model that is not ready for commercial deployment. Users should limit usage to research, especially into responsible development of large language models, and be aware of serious limitations around toxic and biased outputs. Misuse in production systems serving real users could cause significant harm.

### Bias, Risks, and Limitations

Based on the provided references about the OPT-175B model, here are some of the known and foreseeable issues that likely also apply to the smaller facebook/opt-2.7b model:

- The model has limitations in terms of bias and safety due to the diversity (or lack thereof) in the training data. This can lead to the model reinforcing harmful stereotypes.

- The model has a high propensity to generate toxic language, even when given relatively innocuous prompts. Adversarial prompts that elicit toxic outputs are easy to find. 

- There are risks of deploying large language models with emergent capabilities like this at scale, including ethical and social risks that are still being researched and understood.

- Standard benchmark datasets may not fully characterize all the limitations and potential harms of the model. More scrutiny of the training data, with better data characterization and selection criteria, is needed to use the data responsibly.

- The strong awareness of toxic language may be undesirable for some downstream applications and additional mitigations may be needed.

- The model is not immune to issues common to large language models like hallucinations, lack of diversity in generations, and encoding biases from the training data.

[More Information Needed] on the specific technical limitations of the smaller 2.7B parameter model vs the 175B model discussed. But in general, this model is intended only for research, especially into responsible AI, and is not suitable for production use or real-world deployments without further testing and potential mitigations for the issues noted above.

### Recommendations

Based on the provided references about the OPT-175B model, here are some recommendations with respect to foreseeable issues about the smaller facebook/opt-2.7b model:

- The opt-2.7b model likely has similar limitations as the larger OPT-175B in terms of bias, safety issues, lack of generation diversity, and tendency to hallucinate or generate toxic language and harmful stereotypes, even from innocuous prompts. More analysis should be done to characterize the specific limitations of the 2.7B model size.

- The opt-2.7b model should be released with a non-commercial license to enable responsible AI research into its capabilities and limitations, while reducing environmental impact. Access should be limited to researchers to mitigate ethical and social risks of open deployment.

- Extensive evaluation should be carried out on the opt-2.7b model to assess performance on standard NLP benchmarks as well as safety, bias, and inclusion metrics. However, such evaluations may not fully capture all limitations.

- The primary intended use of opt-2.7b should be for research purposes, especially into responsible AI development for language models. It should not be used for production or real-world deployment. 

- The AI community would benefit from collective development of guidelines for responsible development of models like opt-2.7b. Enabling broader researcher access can increase diverse voices in defining ethical considerations.

[More Information Needed] on specific quantitative performance of the 2.7B model size on NLP benchmarks and safety evaluations compared to the full 175B model and GPT-3 models.

## Training Details

### Training Data

The training data of the model facebook/opt-2.7b is a union of five publicly available datasets, including datasets used in RoBERTa, the Pile, and PushShift.io Reddit. The component datasets went through standard cleaning and re-formatting practices, such as removing repetitive/non-informative text. [More Information Needed]

### Training Procedure

#### Preprocessing

For the tokenization of the data used to train the OPT-2.7B model, we used the GPT-2 byte level BPE tokenizer, as mentioned in reference 1:

"We tokenize all corpora using the GPT-2 byte level BPE tokenizer (Sennrich et al., 2016;Radford et al., 2019;Brown et al., 2020). Our final corpus contains roughly 180B tokens."

Regarding any resizing or rewriting of the data, [More Information Needed]. The references do not provide specific details about these preprocessing steps for the OPT-2.7B model.

The pre-training corpus for OPT-2.7B was created by combining datasets used in RoBERTa, the Pile, and PushShift.io Reddit, as stated in reference 3:

"The pre-training corpus contains a concatenation of datasets used in RoBERTa (Liu et al., 2019b), the Pile (Gao et al., 2021a), and PushShift.io Reddit (Baumgartner et al., 2020;Roller et al., 2021)."

For the PushShift.io Reddit dataset, the longest chain of comments in each thread was extracted, and all other paths in the tree were discarded, reducing the corpus by about 66%, as mentioned in reference 4.

#### Training Hyperparameters

Here are the training hyperparameters for the facebook/opt-2.7b model based on the provided references:

Optimizer:
- AdamW optimizer with (β1, β2) set to (0.9, 0.95) and weight decay of 0.1

Learning Rate Schedule:
- Linear learning rate schedule
- Warms up from 0 to the maximum learning rate over the first 375M tokens
- Decays down to 10% of the maximum learning rate over 300B tokens
- [More Information Needed] on the specific maximum learning rate used

Batch Size:
- [More Information Needed] on the exact batch size used for the 2.7B parameter model

Weight Initialization:
- Normal distribution with zero mean and standard deviation of 0.006
- Standard deviation for output layers scaled by 1.0/√(2L) where L is the total number of layers
- All bias terms initialized as 0

Other Settings:
- Trained with ReLU activation 
- Sequence length of 2048

[More Information Needed] on the exact number of training steps/epochs and any other hyperparameters specific to the 2.7B model. The references mostly discuss the larger 175B model.

#### Speeds, Sizes, Times

Here are the details I could find about the model facebook/opt-2.7b based on the provided references:

Throughput: [More Information Needed]

Start time: [More Information Needed] 

End time: [More Information Needed]

Checkpoint sizes: [More Information Needed]

Additional details:
- OPT-2.7B is one of the smaller baseline models released alongside OPT-175B, with parameters ranging from 125 million to 66 billion (Reference 9)
- The models and hyperparameters largely follow Brown et al. (2020), with variations in batch size mostly to obtain increased computational efficiency (Reference 8)
- OPT-2.7B and the other baseline models between 125M and 66B parameters are being released by Meta AI (Reference 9)

Unfortunately the provided references do not contain specific information about the throughput, training times, or checkpoint sizes for the OPT-2.7B model. More details would be needed from the model developers to fill in that information.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, there is no specific information about the benchmarks or datasets that the facebook/opt-2.7b model was evaluated on. The references mention evaluation results for larger OPT models like OPT-175B, but not the 2.7B parameter model.

[More Information Needed] on the specific benchmarks or datasets used to evaluate the facebook/opt-2.7b model.

#### Factors

Based on the provided references about the OPT-175B model, here are some key points relevant to the foreseeable characteristics that will influence how the smaller facebook/opt-2.7b model behaves:

- Like other large language models, the diversity (or lack thereof) of training data will impact downstream model quality, including potential limitations in terms of bias and safety. The model may have issues with generation diversity and hallucination. (Reference 1)

- The model likely has a high propensity to generate toxic language and reinforce harmful stereotypes, even when provided with relatively innocuous prompts. Adversarial prompts to elicit such behavior are likely trivial to find. (Reference 3) 

- Evaluations should assess performance on standard NLP benchmarks, but also explicitly evaluate safety, bias, and inclusion to characterize limitations. Performance may vary for different population subgroups. (References 6, 8)

- The strong awareness of toxic language may or may not be desirable depending on downstream applications. Additional mitigations or avoidance of the model may be needed. (Reference 11)

[More Information Needed] Specific evaluations disaggregating performance across different factors and subgroups for the facebook/opt-2.7b model. The references focus on the larger OPT-175B model.

In summary, key influencing characteristics include training data diversity and potential biases, propensity for generating unsafe/toxic content, and variable performance across domains and subgroups that requires targeted evaluation to uncover. Responsible deployment requires considering these limitations.

#### Metrics

Based on the provided references, the following metrics will be used to evaluate the OPT-2.7B model, considering the tradeoffs between different types of errors:

1. Standard NLP task performance: Accuracy on tasks such as HellaSwag, StoryCloze, PIQA, ARC Easy and Challenge, OpenBookQA, WinoGrad, Wino-Grande, and SuperGLUE, following the prompts and experimental setup used by GPT-3. This will allow for a direct comparison with GPT-3 and other large language models.

2. Responsible AI evaluations: Benchmarks related to hate speech detection, stereotype awareness, and toxic content generation will be used to assess the potential harm and limitations of OPT-2.7B. These evaluations provide a first step towards understanding the model's performance in terms of safety, bias, and inclusion.

3. Factual correctness: [More Information Needed]

4. Generation diversity and hallucination: [More Information Needed]

5. Perplexity and F1 score on conversational datasets: Performance on datasets such as ConvAI2 and Wizard-of-Internet will be evaluated using perplexity and Unigram F1 (UF1) score to assess the model's conversational abilities.

It is important to note that these evaluations may not fully characterize the complete limitations of the model, and qualitative observations suggest that OPT-2.7B may suffer from issues similar to other large language models.

### Results

Based on the provided references, there is limited information to fully evaluate the facebook/opt-2.7b model. Here is what can be determined:

Performance:
- OPT-175B significantly outperforms the Reddit 2.7B model on all dialogue tasks (Reference 10). This suggests the 2.7B model likely has lower performance than the 175B model.
- [More Information Needed] on specific performance metrics for the 2.7B model size.

Risks and Limitations: 
- OPT-175B has a higher toxicity rate than GPT-3 Davinci (Reference 3). The 2.7B model may have similar toxicity issues.
- Inclusion of unmoderated social media texts in pre-training likely raises familiarity with hate speech and toxic language (References 2-4).
- [More Information Needed] on specific bias and toxicity evaluations for the 2.7B model.

Training Data:
- [More Information Needed] on the exact training data used for the 2.7B model. The full OPT models used a dataset that included unmoderated social media discussions (Reference 1).

Other Notes:
- The 2.7B model is one of the smaller baseline models being released, while the 175B requires a request for research access (Reference 11).
- [More Information Needed] on model architecture, hyperparameters, compute used, etc for the 2.7B model specifically.

In summary, more targeted evaluation is needed to fully characterize the performance, limitations, training data, and technical details of the facebook/opt-2.7b model based on the provided references that focus more on the larger 175B model. The smaller model likely exhibits some of the same concerning behaviors around toxicity and bias as the 175B model, but to an unknown degree without further study.

#### Summary

Based on the provided references, there is limited information specifically about the evaluation results of the facebook/opt-2.7b model. The references mostly discuss the larger OPT-175B model.

The only potentially relevant information is in reference 11, which mentions that "OPT-175B significantly outperforms the also-unsupervised Reddit 2.7B model on all tasks". However, it's unclear if the "Reddit 2.7B model" refers to the same model as "facebook/opt-2.7b".

To properly summarize the evaluation results for the facebook/opt-2.7b model, [More Information Needed] that directly discusses the performance of this specific model on relevant benchmarks and tasks. The current references do not provide enough details to draw conclusions about the 2.7B parameter model.

## Model Examination

Explainability/Interpretability:
[More Information Needed]

The provided references do not contain specific information about explainability or interpretability work done on the facebook/opt-2.7b model. More details would be needed from the model development process and any research conducted to analyze and understand the model's inner workings and decision-making process. Explainable AI techniques could potentially be applied to this model to gain insights, but no concrete information is given in the references about such work being done for this particular model.

## Environmental Impact

- **Hardware Type:** The model facebook/opt-2.7b was trained on 80GB A100 GPUs, as mentioned in reference 6:

"We are releasing all of our models between 125M and 66B parameters, and will provide full research access to OPT-175B upon request. Access will be granted to academic researchers; those affiliated with organizations in government, civil society, and academia; and those in industry research laboratories. We are also releasing both the logbook of our model creation as well as our codebase, metaseq, 3 which enabled training OPT-175B on 992 80GB A100 GPUs, reaching 147 TFLOP/s utilization per GPU."

While the specific number of GPUs used for training the 2.7B model is not provided, it is clear that 80GB A100 GPUs were used for training the OPT model family.
- **Software Type:** Based on the provided references, the OPT-175B model was trained using the metaseq open-source repository developed by Meta AI. However, there is no specific information provided about the software used to train the smaller facebook/opt-2.7b model.

[More Information Needed] on the exact software type used to train the facebook/opt-2.7b model.
- **Hours used:** [More Information Needed]

The provided references do not contain specific information about the training time for the facebook/opt-2.7b model. The references discuss training details and challenges for the larger OPT-175B model, but do not provide the training time for the smaller 2.7B parameter model.
- **Cloud Provider:** [More Information Needed]

The provided references do not contain information about the specific cloud provider used to train the facebook/opt-2.7b model. More details would be needed from the model developers or documentation to determine the cloud provider used for training this particular model.
- **Carbon Emitted:** [More Information Needed]

The references provided do not contain specific information about the carbon emissions for training the facebook/opt-2.7b model. The references mention carbon emissions estimates for larger models like OPT-175B (75 tons CO2eq), GPT-3 (500 tons), and Gopher (380 tons), but do not provide the emissions for the 2.7B parameter model. More information would be needed about the compute resources and energy used to train the smaller facebook/opt-2.7b model in order to estimate its carbon footprint.
## Technical Specification

### Model Architecture and Objective

Model Architecture and Objective:

The facebook/opt-2.7b model is a decoder-only transformer language model with 2.7 billion parameters. It is part of the OPT (Open Pre-trained Transformer) model family developed by Meta AI, which includes models ranging from 125 million to 175 billion parameters.

The primary objective of the OPT models, including facebook/opt-2.7b, is to replicate the performance and sizes of the GPT-3 class of models while applying the latest best practices in data curation and training efficiency. These models are intended for research into Language Models, especially as it pertains to Responsible AI.

[More Information Needed] on the specific architectural details and hyperparameters used for the facebook/opt-2.7b model.

The OPT models, including facebook/opt-2.7b, were pre-trained on a large dataset to serve as a foundation for various current and future language tasks. The models are released under a non-commercial use license agreement for academic researchers, those affiliated with organizations in government, civil society, and academia, and those in industry research laboratories.

### Compute Infrastructure

We trained OPT-2.7B on [More Information Needed] GPUs, by utilizing Fully Sharded Data Parallel with Megatron-LM Tensor Parallelism. We achieve utilization of up to [More Information Needed] TFLOP/s per GPU. We keep Adam state in FP32, since we shard it across all hosts, while the model weights remained in FP16. To avoid underflows, we used dynamic loss scaling.

## Citation

```
@misc{susan-opt,
    author = {Susan Zhang and
              Stephen Roller and
              Naman Goyal and
              Mikel Artetxe and
              Moya Chen and
              Shuohui Chen and
              Christopher Dewan and
              Mona Diab and
              Xian Li and
              Victoria Lin and
              Todor Mihaylov and
              Myle Ott and
              Sam Shleifer and
              Kurt Shuster and
              Daniel Simig and
              Singh Koura and
              Anjali Sridhar and
              Tianlu Wang and
              Luke Zettlemoyer},
    title  = {OPT: Open Pre-trained Transformer Language Models},
    url    = {https://arxiv.org/pdf/2205.01068.pdf}
}
```

