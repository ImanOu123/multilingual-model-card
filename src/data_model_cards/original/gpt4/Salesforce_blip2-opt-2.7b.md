# Model Card for Salesforce/blip2-opt-2.7b

Salesforce/blip2-opt-2.7b is a state-of-the-art vision-language pre-training model that leverages a two-stage pre-training strategy with frozen image encoders and large language models, demonstrating significant efficiency in trainable parameters and excelling in zero-shot vision-language tasks. It introduces a Querying Transformer (Q-Former) to bridge the modality gap and enable new zero-shot instructed image-to-text generation capabilities.

## Model Details

### Model Description

Model Name: Salesforce/blip2-opt-2.7b

Model Architecture:
The Salesforce/blip2-opt-2.7b model is built upon the BLIP-2 framework, which is designed for vision-language pre-training. The architecture includes a trainable module called Q-Former that bridges the gap between a frozen image encoder and a frozen large language model (LLM). The Q-Former consists of two transformer submodules: an image transformer for visual feature extraction and a text transformer that serves as both a text encoder and decoder. The model uses learnable query embeddings as input to the image transformer, which interact with frozen image features through self-attention and cross-attention layers. The Q-Former is initialized with pre-trained BERT base weights, and the cross-attention layers are randomly initialized. The model contains 188 million parameters, with the queries considered as model parameters.

Training Procedures:
The model undergoes a two-stage pre-training process. In the first stage, the model is pre-trained from scratch for 250,000 steps, and in the second stage, it is further pre-trained for 80,000 steps. The pre-training uses the AdamW optimizer with specific hyperparameters, including a cosine learning rate decay with a peak learning rate of 1e-4 and a linear warmup of 2,000 steps. The model is trained with images of size 224×224, augmented with random resized cropping and horizontal flipping. The pre-training scripts and config files are available for customization.

Parameters:
The model uses 32 queries with a dimension of 768, which is the same as the hidden dimension of the Q-Former. The output query representation Z has a size of 32 × 768. The pre-training hyperparameters are consistent across all models, and during pre-training, the frozen ViTs' and LLMs' parameters are converted into FP16 (except for FlanT5, which uses BFloat16) to make the process more computationally efficient.

Important Disclaimers:
The model leverages frozen pretrained image encoders and LLMs to achieve state-of-the-art performance on various vision-language tasks with a smaller number of trainable parameters during pre-training. It is important to note that the performance improvements and efficiency gains are in the context of the specific tasks and datasets evaluated. The model demonstrates emerging capabilities in zero-shot instructed image-to-text generation, which is a step towards building multimodal conversational AI agents. However, the performance on tasks not evaluated and in real-world applications may vary, and users should exercise caution and perform additional validation as needed.

[More Information Needed]: For any specific disclaimers related to ethical considerations, biases, or limitations inherent to the model's training data or architecture, additional information would be required.

- **Developed by:** Junnan Li; Dongxu Li; Silvio Savarese; Steven Hoi
- **Funded by:** The references provided do not explicitly mention the people or organizations that funded the project for the model Salesforce/blip2-opt-2.7b. Therefore, to answer the question regarding the funding sources for this project, I would have to say:

[More Information Needed]
- **Shared by:** The contributors who made the model Salesforce/blip2-opt-2.7b available online as a GitHub repo are Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi, as indicated in the provided reference citation:

```
@inproceedings{li2023blip2,
      title={{BLIP-2:} Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models}, 
      author={Junnan Li and Dongxu Li and Silvio Savarese and Steven Hoi},
      year={2023},
      booktitle={ICML},
}
```
- **Model type:** The Salesforce/blip2-opt-2.7b is a compute-efficient, vision-language pre-trained model that leverages frozen image encoders and language models for multimodal tasks, using methods like image-text contrastive learning and generative learning, and demonstrates strong performance in zero-shot image-to-text generation tasks.
- **Language(s):** The model Salesforce/blip2-opt-2.7b processes natural human language in the form of text prompts and instructions for tasks such as image-to-text generation, including visual knowledge reasoning and image captioning.
- **License:** [More Information Needed]
- **Finetuned from model:** The model Salesforce/blip2-opt-2.7b is fine-tuned from the pre-trained weights of BERT base. The link to the BERT base model is: [BERT base (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805).
### Model Sources

- **Repository:** https://github.com/salesforce/LAVIS/tree/main/projects/blip2
- **Paper:** https://arxiv.org/pdf/2301.12597.pdf
- **Demo:** The link to the demo of the model Salesforce/blip2-opt-2.7b is provided in the first reference. Here is the link to the Notebook Demo for instructed vision-to-language generation: [Notebook Demo](https://github.com/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb).
## Uses

### Direct Use

The Salesforce/blip2-opt-2.7b model can be used for zero-shot instructed image-to-text generation tasks without the need for fine-tuning, post-processing, or plugging into a pipeline. This means that the model can generate text based on an image and a text prompt directly after loading it, without any additional training or complex setup.

Here's a code snippet demonstrating how to use the model for a zero-shot visual question answering task, as referenced in the provided materials:

```python
import torch
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load the pre-trained model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")

# Load an image from a URL
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Define a question about the image
question = "how many dogs are in the picture?"

# Process the image and the question, and move the tensors to the GPU
inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

# Generate an answer to the question based on the image
outputs = model.generate(**inputs, max_length=16, num_beams=5, length_penalty=-1)

# Decode the generated text
answer = processor.decode(outputs[0], skip_special_tokens=True)

print(answer)
```

This code snippet uses the model to answer a question about an image in a zero-shot manner, meaning it does not require any additional training data or fine-tuning. The model takes an image and a question as inputs and generates an answer based on its pre-trained knowledge. The `generate` function is used with a beam search strategy to find the most likely answer, and the `length_penalty` parameter is set to encourage shorter, more concise answers.

### Downstream Use

The Salesforce/blip2-opt-2.7b model is a state-of-the-art deep learning model designed for vision-language tasks, such as image captioning and instructed image-to-text generation. When fine-tuned for a specific task, this model can generate descriptive text based on the visual content of an image. This capability can be integrated into various applications, such as content creation tools, accessibility apps for the visually impaired, or automated social media management systems.

For example, in an image captioning task, the model can be fine-tuned to generate captions for images in a dataset. The fine-tuning process involves using a dataset with images and corresponding text descriptions, and training the model to predict the descriptions given the images. The prompt "a photo of" is used as an initial input to the language model, and the model is trained with a language modeling loss while keeping the language model frozen and updating the parameters of the Q-Former and the image encoder.

In a larger ecosystem or app, the model can be used for zero-shot instructed image-to-text generation, where it can generate text based on instructions without the need for task-specific fine-tuning. This can be particularly useful for applications that require on-the-fly text generation based on images, such as generating product descriptions in e-commerce platforms or providing instant explanations for educational content.

Here is a code snippet demonstrating how to use the Salesforce/blip2-opt-2.7b model for generating a caption for an image, assuming the model has been fine-tuned for the image captioning task:

```python
from PIL import Image
import requests
from lavis.models import Blip2ForConditionalGeneration
from lavis.processors import Blip2Processor

# Load the processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")

# Load an image from a URL
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Generate a caption for the image
inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print(caption)
```

This code snippet loads a pre-trained model and processor, fetches an image from a URL, processes the image, and generates a caption. The generated caption is then printed out. Note that the actual use of the model may require additional steps, such as setting up the appropriate environment and ensuring that the model is fine-tuned for the desired task.

### Out-of-Scope Use

As the sociotechnic member of the team, I must highlight potential misuses of the Salesforce/blip2-opt-2.7b model and advise against certain applications to ensure responsible usage:

1. **Generating Offensive Content**: Given that BLIP-2 inherits risks from LLMs, including the potential output of offensive language, users should not employ the model in contexts where it might generate harmful or derogatory content. This is particularly important in public-facing applications or platforms with diverse user bases.

2. **Propagating Social Bias**: The model may inadvertently reflect or amplify social biases present in the training data. Users should avoid using the model in decision-making processes or scenarios where such biases could lead to unfair or discriminatory outcomes.

3. **Privacy Violations**: Since LLMs can leak private information, users must not use Salesforce/blip2-opt-2.7b to process images or text that contain sensitive personal data unless appropriate safeguards are in place to prevent privacy breaches.

4. **Misrepresentation of Facts**: As noted in the references, the model may not always have up-to-date information or could activate incorrect reasoning paths. Users should not rely on the model for critical applications where factual accuracy is paramount, such as in legal, medical, or safety-critical domains.

5. **Deepfakes and Misinformation**: The model's capability to generate text from images could be misused to create convincing but false narratives or deepfakes, contributing to misinformation. Users should not use the model to create or spread deceptive content.

6. **Unsupervised Use by Minors**: Given the potential for generating inappropriate content, the model should not be used unsupervised by minors or in educational settings without proper moderation.

7. **Automated Decision-Making**: Users should not use the model for automated decision-making without human oversight, especially in contexts where the decisions have significant consequences for individuals' lives or rights.

In conclusion, while Salesforce/blip2-opt-2.7b has many beneficial applications, it is crucial that users operate it within ethical boundaries and with an awareness of its limitations. Any application that risks harm, violates privacy, or propagates falsehoods should be strictly avoided. Users are encouraged to implement additional layers of content filtering and human oversight to mitigate these risks.

### Bias, Risks, and Limitations

The known and foreseeable issues with the Salesforce/blip2-opt-2.7b model can be categorized into technical limitations and sociotechnical concerns:

Technical Limitations:
1. Inaccurate Knowledge: The model may generate unsatisfactory results due to the LLM's inaccurate knowledge base, which can lead to incorrect information being presented in the image-to-text generation process.
2. Incorrect Reasoning Path: The model might activate incorrect reasoning paths during the generation process, leading to outputs that do not accurately reflect the content or context of the input images.
3. Outdated Information: Since the model uses frozen components, it may not have up-to-date information, which could result in the generation of outdated or irrelevant text in response to new image content.
4. Frozen Model Risks: The use of frozen models means that BLIP-2 inherits the risks associated with LLMs, such as the potential to output offensive language, propagate social biases, or leak private information.

Sociotechnical Concerns:
1. Propagation of Social Bias: The model may inadvertently perpetuate existing social biases present in the training data, which could lead to unfair or discriminatory outcomes.
2. Offensive Language: There is a risk that the model could generate offensive or inappropriate language, which could have negative social impacts and harm the reputation of entities using the model.
3. Privacy Leaks: The model could potentially leak private information if such data were present in the training set, raising concerns about user privacy and data protection.
4. Misunderstandings: Users may misunderstand the capabilities of the model, expecting it to have up-to-date knowledge or to understand context in ways that it currently does not, leading to overreliance or misuse of the technology.

Remediation approaches to address some of these issues include using instructions to guide the model's generation process and training on a filtered dataset with harmful content removed. However, these approaches may not fully mitigate all the technical and sociotechnical limitations of the model.

### Recommendations

The model Salesforce/blip2-opt-2.7b, as a state-of-the-art vision-language pre-training method, has demonstrated significant capabilities in various tasks. However, there are several foreseeable issues that need to be addressed:

1. **Inaccurate Knowledge and Outdated Information**: Since BLIP-2 can generate unsatisfactory results due to inaccurate knowledge from the LLM or outdated information, it is recommended to continuously update the model with the latest data and potentially retrain the LLM component to ensure the information it relies on remains current.

2. **Risk of Offensive Language and Social Bias**: The model inherits risks from LLMs, such as outputting offensive language or propagating social bias. To mitigate these risks, it is advisable to:
   - Implement content filters to remove harmful content from the training dataset.
   - Employ bias detection and mitigation techniques during both pre-training and fine-tuning stages.
   - Regularly evaluate the model outputs and update the training data or model parameters to reduce biases.

3. **Privacy Concerns**: There is a risk of leaking private information. To prevent this, privacy-preserving measures should be taken, such as:
   - Anonymizing the dataset to remove any personally identifiable information.
   - Using differential privacy techniques during training to ensure individual data points do not influence the model outputs significantly.

4. **Guidance for Model Usage**: Since the model can be guided by instructions, providing clear documentation and examples on how to use instructions effectively can help users achieve better results and avoid unintended consequences.

5. **Model Monitoring and Remediation**: Continuous monitoring of the model's performance and the impact of its outputs on society is crucial. Establishing a feedback loop where users can report issues and having a team ready to address these issues can help in timely remediation.

6. **Ethical and Legal Considerations**: Engage with ethicists, lawyers, and sociologists to review the model's applications and its broader societal impact. Ensure compliance with legal standards and ethical guidelines for AI.

7. **Resource Efficiency**: Leverage techniques like int8 quantization, as mentioned in the documentation, to reduce the model's memory footprint, making it more accessible and sustainable.

8. **Usage and Evaluation**: Encourage users to follow the provided scripts and documentation for dataset preparation, model evaluation, and fine-tuning to ensure the model is used as intended and its performance is accurately assessed.

By addressing these recommendations, the team can help ensure that Salesforce/blip2-opt-2.7b is used responsibly and effectively while minimizing potential negative impacts on society.

## Training Details

### Training Data

The training data for the model Salesforce/blip2-opt-2.7b consists of a large-scale pre-training dataset that includes 129M images in total, sourced from various datasets such as COCO, Visual Genome, CC3M, CC12M, SBU, and a significant portion from the LAION400M dataset. The data is processed using the CapFilt method to ensure quality and relevance for vision-language tasks. For more details on data pre-processing and additional filtering, refer to the [Dataset Download](https://opensource.salesforce.com/LAVIS//latest/getting_started.html#auto-downloading-and-loading-datasets) documentation.

### Training Procedure

#### Preprocessing

For the Salesforce/blip2-opt-2.7b model, the preprocessing of the data involves several steps to prepare the images and text for training. Here's a detailed description of the preprocessing steps based on the provided references:

1. **Image Preprocessing**:
   - Images are resized to a fixed resolution of 224×224 pixels.
   - Data augmentation techniques are applied, including random resized cropping and horizontal flipping, to improve the robustness and generalization of the model.

2. **Text Preprocessing**:
   - Text data is tokenized using a tokenizer compatible with the frozen large language models (LLMs) used in the BLIP-2 framework. The specific tokenizer details are not provided in the references, so [More Information Needed] for the exact tokenization process.
   - For the image captioning task, a prompt "a photo of" is used as an initial input to the LLM to guide the generation of captions.

3. **Feature Extraction**:
   - The frozen image encoder (e.g., ViT-L/14 or ViT-g/14) is used to extract visual features from the images. The last layer of the ViT is removed, and the output features from the second last layer are used.
   - The dimensionality of the output features from the image encoder is reduced by the Q-Former, which uses a set number of learnable query embeddings (32 queries, each with a dimension of 768) to interact with the image features and extract the most relevant visual information for the text.

4. **Modality Interaction**:
   - The Q-Former module allows for interaction between the visual and textual modalities. Depending on the pre-training task, different self-attention masks are applied to control the interaction between the queries and the text.

5. **Data Sources**:
   - The pre-training dataset includes a large collection of images from various sources such as COCO, Visual Genome, CC3M, CC12M, SBU, and LAION400M datasets. The CapFilt method is adopted for filtering the data, although specific preprocessing steps for this method are not detailed in the references provided.

6. **Hyperparameters and Training Scripts**:
   - The model is pre-trained and fine-tuned using specific scripts and config files that can be modified for customized training. The exact preprocessing steps within these scripts are not detailed in the references, so [More Information Needed] for any additional preprocessing steps encoded in the training scripts.

In summary, the preprocessing for the Salesforce/blip2-opt-2.7b model involves resizing and augmenting images, tokenizing text, extracting and reducing the dimensionality of visual features, and controlling modality interactions through attention mechanisms. For specific details on tokenization and any additional preprocessing steps within the training scripts, [More Information Needed].

#### Training Hyperparameters

The training hyperparameters for the model Salesforce/blip2-opt-2.7b are as follows:

- **Optimizer**: AdamW with β1 = 0.9, β2 = 0.98, and a weight decay of 0.05.
- **Learning Rate Schedule**: Cosine learning rate decay with a peak learning rate of 1e-4, a linear warmup of 2k steps, and a minimum learning rate of 5e-5 at the second stage.
- **Image Size**: 224×224, augmented with random resized cropping and horizontal flipping.
- **Pre-training Steps**: 250k steps in the first stage and 80k steps in the second stage.
- **Batch Size**: 2320/1680 for ViT-L/ViT-g in the first stage and 1920/1520 for OPT/FlanT5 in the second stage.
- **Precision**: Conversion of the frozen ViTs' and LLMs' parameters into FP16 during pre-training, except for FlanT5 where BFloat16 is used.
- **Pre-training Data**: A dataset comprising 129M images, including COCO, Visual Genome, CC3M, CC12M, SBU, and 115M images from the LAION400M dataset, with the CapFilt method applied.

For the specific model Salesforce/blip2-opt-2.7b, the processor and model instantiation, as well as an example of how to prepare inputs for inference, are provided in reference 6. However, this does not include additional hyperparameters specific to the training of Salesforce/blip2-opt-2.7b beyond those mentioned above. For further customization of training, the config files can be modified as indicated in reference 4.

If there are additional hyperparameters specific to Salesforce/blip2-opt-2.7b that are not covered in the provided references, then [More Information Needed].

#### Speeds, Sizes, Times

The model Salesforce/blip2-opt-2.7b underwent a two-stage pre-training process. In the first stage, the model was pre-trained for 250k steps with a batch size of 2320 for ViT-L and 1680 for ViT-g. In the second stage, the model was further pre-trained for 80k steps with a batch size of 1920 for OPT and 1520 for FlanT5. During pre-training, the parameters of the frozen vision transformers (ViTs) and large language models (LLMs) were converted to FP16, except for FlanT5, which used BFloat16. This approach did not result in any performance degradation compared to 32-bit models and made the pre-training more computationally friendly.

The pre-training utilized the AdamW optimizer with β1 = 0.9, β2 = 0.98, and a weight decay of 0.05. A cosine learning rate decay was used with a peak learning rate of 1e-4 and a linear warmup of 2k steps. The minimum learning rate at the second stage was set to 5e-5. Images of size 224×224 were used, augmented with random resized cropping and horizontal flipping.

The pre-training dataset was the same as that used by BLIP, consisting of 129M images from various sources, including COCO, Visual Genome, CC3M, CC12M, SBU, and 115M images from the LAION400M dataset. The CapFilt method was adopted for pre-training.

For the image encoder, two state-of-the-art pre-trained vision transformer models were explored: ViT-L/14 from CLIP and ViT-g/14 from EVA-CLIP. The last layer of the ViT was removed, and the output features from the second last layer were used. For the language model, the unsupervised-trained OPT model family and the instruction-trained FlanT5 model family were explored.

The Q-Former, which is part of the model, contains 188M parameters, with the pre-trained weights of BERT base and randomly initialized cross-attention layers.

Regarding the performance, BLIP-2, which includes Salesforce/blip2-opt-2.7b, has shown improved performance on various zero-shot vision-language tasks, establishing new state-of-the-art results.

For specific details such as throughput, start or end time, and checkpoint sizes, the provided references do not contain this information. Therefore, for these specific details, the answer is "[More Information Needed]".

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model Salesforce/blip2-opt-2.7b evaluates on the following benchmarks or datasets:

- COCO (Common Objects in Context) for both image-to-text retrieval and text-to-image retrieval tasks.
- Flickr30K dataset for image-to-text retrieval and text-to-image retrieval tasks.
- Zero-shot VQAv2 (Visual Question Answering version 2) for zero-shot visual question answering.
- NoCaps for zero-shot image captioning.

For further details on the evaluation setup and hyperparameters, the appendix of the official implementation paper can be referenced. Additionally, scripts for evaluating pretrained and finetuned models are available at the provided GitHub link.

#### Factors

The model Salesforce/blip2-opt-2.7b is a state-of-the-art vision-language model that has shown improved performance on zero-shot vision-language tasks as indicated in Table 1 of the provided references. However, there are several characteristics that can influence its behavior:

1. **Domain and Context**: The model's performance is influenced by the domain and context of the images and text it is processing. Since it has been trained on common vision-language datasets, its performance may be optimized for the types of images and contexts present in those datasets. If the model encounters images or text from domains that are underrepresented in the training data, its performance may degrade.

2. **Population Subgroups**: The model may exhibit disparities in performance across different population subgroups, especially if the training data contains biases. As mentioned in the references, the model inherits the risks of large language models (LLMs), such as propagating social bias. This could manifest in less accurate or potentially offensive outputs for certain demographics or cultural contexts that are not well-represented in the training data.

3. **Up-to-Date Information**: The model may not have up-to-date information about new image content, which could affect its performance in rapidly evolving domains or with recent events that were not part of the training data.

4. **Reasoning Path Activation**: The model's reasoning path activation can influence its image-to-text generation. If the incorrect reasoning path is activated, it could lead to unsatisfactory results.

5. **Inherited Risks from LLMs**: As the model uses frozen models from LLMs, it may output offensive language, leak private information, or propagate social bias, which are known risks associated with LLMs.

6. **Remediation Approaches**: The model's behavior can be influenced by remediation approaches such as using instructions to guide the model's generation or training on a filtered dataset with harmful content removed. These approaches can help mitigate some of the risks mentioned above.

Evaluation of the model should be disaggregated across these factors to uncover any disparities in performance. This would involve testing the model across a variety of domains, contexts, and population subgroups to ensure that it performs equitably and does not amplify existing biases. However, without specific disaggregated evaluation results, [More Information Needed] to make definitive statements about the model's performance across these different factors.

#### Metrics

For evaluating the Salesforce/blip2-opt-2.7b model, the metrics used will reflect the model's performance on various zero-shot vision-language tasks as indicated in the references. Specifically, the model's performance is measured using the following metrics:

1. **Zero-Shot VQA (Visual Question Answering)**: The model's ability to answer questions about images without any task-specific training is evaluated using the VQAv2 dataset. The metric used here is accuracy, which measures the percentage of questions that the model answers correctly.

2. **Zero-Shot Captioning**: The model's capability to generate descriptive captions for images without fine-tuning is assessed using the NoCaps dataset. The metric used for this task is the CIDEr score, which evaluates the consensus between the generated captions and a set of reference captions.

3. **General Vision-Language Tasks**: The model's overall performance on a range of vision-language tasks is evaluated, though the specific metrics for these tasks are not explicitly mentioned in the provided references. Common metrics for such tasks include BLEU, METEOR, and ROUGE, which are standard for evaluating the quality of generated text in comparison to reference text.

The references also highlight the importance of considering the tradeoffs between different errors. For instance, the model may generate unsatisfactory image-to-text results due to various reasons, including inaccurate knowledge or activating incorrect reasoning paths. While the specific metrics for evaluating these errors are not detailed in the provided references, it is crucial to monitor the model's outputs for such issues and potentially use qualitative analysis or user studies to assess the impact of these errors.

Moreover, the model inherits risks from the underlying large language models (LLMs), such as outputting offensive language, propagating social bias, or leaking private information. While these are not quantifiable with standard evaluation metrics, they are critical aspects to consider when deploying the model. Remediation approaches such as using instructions to guide the model's generation or training on a filtered dataset with harmful content removed can help mitigate these risks.

In summary, the Salesforce/blip2-opt-2.7b model will be evaluated using accuracy for zero-shot VQA, CIDEr score for zero-shot captioning, and potentially other standard text generation metrics for general vision-language tasks. Additionally, qualitative assessments and remediation strategies will be necessary to address the model's limitations and risks associated with LLMs.

### Results

The evaluation results of the model Salesforce/blip2-opt-2.7b are as follows:

- BLIP-2 demonstrates state-of-the-art performance on various zero-shot vision-language tasks, as shown in Table 1 of the provided references. It outperforms previous models with fewer trainable parameters during vision-language pre-training.
  
- For image-text retrieval tasks, the model was fine-tuned on the COCO dataset without the need for language model fine-tuning. It was evaluated on both COCO and Flickr30K datasets for image-to-text and text-to-image retrieval tasks. The fine-tuning involved the same objectives used in pre-training: Image-Text Contrastive (ITC), Image-Text Matching (ITM), and Image-Text Grounding (ITG) losses.

- During inference, the model uses a two-step process where it first selects 128 candidates based on image-text feature similarity and then re-ranks them based on pairwise ITM scores. The results, detailed in Table 5, show that BLIP-2 achieves significant improvement over existing methods on zero-shot image-text retrieval.

- The ITC and ITM losses are highlighted as essential for learning image-text similarity, and the ITG loss is shown to be beneficial for image-text retrieval by enforcing the queries to extract visual features relevant to the text, thus improving vision-language alignment. This is supported by the results in Table 6.

- BLIP-2, when equipped with powerful Large Language Models (LLMs) like OPT, has established new state-of-the-art results on zero-shot VQAv2 with a score of 65.0 compared to the previous best of 56.3 by Flamingo. It also sets a new record on zero-shot captioning on the NoCaps dataset with a CIDEr score of 121.6, surpassing the previous best of 113.2.

- However, it is noted that BLIP-2's image-to-text generation can sometimes yield unsatisfactory results due to various factors such as inaccurate knowledge from the LLM, activating incorrect reasoning paths, or lacking up-to-date information about new image content. Additionally, the model inherits risks from LLMs, including the potential for outputting offensive language, propagating social bias, or leaking private information.

For more detailed hyperparameters and specific numerical results, reference to the appendix and Table 5 and 6 would be necessary. [More Information Needed] regarding the exact figures for the metrics on the COCO and Flickr30K datasets, as they are not explicitly provided in the reference text.

#### Summary

The model Salesforce/blip2-opt-2.7b has demonstrated state-of-the-art performance on various zero-shot vision-language tasks. It outperforms previous models with a significant margin while utilizing fewer trainable parameters during vision-language pre-training. Specifically, BLIP-2 has achieved a new high score on zero-shot VQAv2 with a score of 65.0, surpassing the previous best score of 56.3 held by Flamingo. Additionally, it has set a new record on zero-shot captioning on the NoCaps dataset with a CIDEr score of 121.6, compared to the former best of 113.2. These results highlight the model's effectiveness in understanding and generating relevant responses to visual and textual inputs.

## Model Examination

### Model Card: Salesforce/blip2-opt-2.7b

#### Explainability/Interpretability

The Salesforce/blip2-opt-2.7b model is a state-of-the-art vision-language pre-training model that leverages a Querying Transformer (Q-Former) to bridge the gap between a frozen image encoder and a frozen Large Language Model (LLM). The model is designed to understand and generate text descriptions from images, and it has been fine-tuned for tasks such as image captioning.

Explainability and interpretability are critical aspects of deep learning models, especially in the context of vision-language tasks where the reasoning behind the generated text is not always transparent. For the Salesforce/blip2-opt-2.7b model, we have taken several steps to enhance the understanding of how the model processes and generates its outputs:

1. **Bottleneck Architecture**: The model uses a bottleneck architecture where the output query representation Z (32 × 768) is much smaller than the size of frozen image features. This design forces the queries to extract the most relevant visual information for the text, which can help in understanding the model's focus during the generation process.

2. **Two-Stage Pre-Training**: The Q-Former is pre-trained in two stages, which helps in isolating the vision-language representation learning from the vision-to-language generative learning. This separation allows for a clearer understanding of how the model learns to bridge the modality gap.

3. **Learnable Query Embeddings**: The model uses a set number of learnable query embeddings as input to the image transformer. These queries interact with each other and with frozen image features through self-attention and cross-attention layers. By examining these interactions, we can gain insights into the attention patterns and how the model prioritizes different aspects of the image.

4. **Controlled Interaction**: Depending on the pre-training task, different self-attention masks are applied to control the interaction between queries and text. This allows for a controlled study of how text influences the visual understanding of the model.

5. **Prompt-Based Fine-Tuning**: For image captioning, the model is fine-tuned with prompts like "a photo of," which guides the LLM to generate relevant captions. This prompt-based approach provides a way to control and interpret the model's output by analyzing how different prompts affect the generated text.

6. **Performance Overview**: Table 1 in the references provides an overview of the model's performance on various zero-shot vision-language tasks. By comparing the model's outputs with state-of-the-art models, we can better understand the improvements and limitations of our model.

While these features enhance the explainability of the Salesforce/blip2-opt-2.7b model, it is important to note that the model, like other LLMs, may still produce unsatisfactory results due to inaccurate knowledge, incorrect reasoning paths, or outdated information. Additionally, the model inherits risks such as outputting offensive language, propagating social bias, or leaking private information. Continuous efforts in model monitoring, evaluation, and updating are necessary to address these challenges and improve the model's interpretability.

[More Information Needed] on specific methods or tools used for explainability and interpretability beyond the architectural and training procedure descriptions provided.

## Environmental Impact

- **Hardware Type:** [More Information Needed]
- **Software Type:** The model Salesforce/blip2-opt-2.7b is trained on software that includes the `salesforce-lavis` library, which can be installed using pip as shown in reference 3. Additionally, the model is integrated into the Hugging Face Transformers library, as mentioned in reference 8. The Transformers library documentation can be found at the provided link, and the model leverages int8 quantization through the `bitsandbytes` library, also mentioned in reference 8.
- **Hours used:** The model Salesforce/blip2-opt-2.7b required less than 6 days for the first stage of training and less than 3 days for the second stage of training, as per reference 5.
- **Cloud Provider:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The Salesforce/blip2-opt-2.7b model leverages a novel architecture known as the Querying Transformer (Q-Former) to bridge the modality gap between visual and language representations. The Q-Former is a trainable module that connects a frozen image encoder with a frozen Large Language Model (LLM). It is designed to extract a fixed number of output features from the image encoder, which is independent of the input image resolution.

The Q-Former consists of two transformer submodules that share the same self-attention layers: an image transformer and a text transformer. The image transformer interacts with the frozen image encoder to extract visual features, while the text transformer can function as both a text encoder and a text decoder. The model uses a set of learnable query embeddings as input to the image transformer, which interact with each other through self-attention layers and with frozen image features through cross-attention layers. These queries can also interact with the text through the same self-attention layers, with different self-attention masks applied depending on the pre-training task to control the interaction between queries and text.

The Q-Former is initialized with the pre-trained weights of BERT base, and the cross-attention layers are randomly initialized. The queries are considered as model parameters, and the Q-Former contains a total of 188M parameters.

BLIP-2, the model in which the Q-Former is used, is pre-trained in two stages. The first stage focuses on vision-language representation learning with a frozen image encoder, and the second stage is dedicated to vision-to-language generative learning with a frozen LLM. BLIP-2 has demonstrated state-of-the-art performance on various vision-language tasks, including zero-shot VQAv2, and has shown emerging capabilities in zero-shot image-to-text generation.

The model architecture of Salesforce/blip2-opt-2.7b is specifically designed to efficiently pre-train on vision-language tasks by leveraging off-the-shelf frozen pre-trained image encoders and LLMs, achieving high performance with significantly fewer trainable parameters compared to existing methods.

### Compute Infrastructure

The compute infrastructure used for the Salesforce/blip2-opt-2.7b model leverages a pre-training process that is computationally friendly due to the use of frozen models. During pre-training, the parameters of the frozen Vision Transformers (ViTs) and Large Language Models (LLMs) are converted into FP16 precision format, except for FlanT5 where BFloat16 is used. This approach does not result in performance degradation compared to using 32-bit models.

The pre-training consists of two stages, with the first stage running for 250k steps and the second stage for 80k steps. The batch sizes used are 2320/1680 for ViT-L/ViT-g in the first stage and 1920/1520 for OPT/FlanT5 in the second stage. The AdamW optimizer is used with specific hyperparameters, and a cosine learning rate decay schedule is applied.

However, the specific hardware configuration (such as the type and number of GPUs, CPU specifications, memory, etc.) used for training the Salesforce/blip2-opt-2.7b model is not provided in the references. Therefore, to give a complete answer regarding the compute infrastructure, [More Information Needed] would be required.

## Citation

```
@misc{junnan-blip,
    author = {Junnan Li and
              Dongxu Li and
              Silvio Savarese and
              Steven Hoi},
    title  = {BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
    url    = {https://arxiv.org/pdf/2301.12597.pdf}
}
```

