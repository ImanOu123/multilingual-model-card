# Model Card for Salesforce/blip2-opt-2.7b

BLIP-2 is a generic and efficient vision-language pre-training method that leverages frozen pre-trained image encoders and large language models. It achieves state-of-the-art performance on various vision-language tasks with significantly fewer trainable parameters compared to existing methods, while also demonstrating emerging capabilities in zero-shot instructed image-to-text generation.

## Model Details

### Model Description

Model Card for Salesforce/blip2-opt-2.7b

Model Architecture:
- BLIP-2 is a vision-language pre-training method that bootstraps from frozen pre-trained unimodal models.
- It uses a Querying Transformer (Q-Former) to bridge the modality gap between vision and language.
- Q-Former is connected to a frozen image encoder (e.g., ViT-L/14) and a frozen language model (e.g., OPT-2.7B).
- The Q-Former uses 32 queries, each with a dimension of 768, to extract visual information relevant to the text.

Training Procedures:
- BLIP-2 is pre-trained in two stages:
  1. Vision-language representation learning stage with a frozen image encoder.
  2. Vision-to-language generative learning stage with a frozen language model.
- In the first stage, the model is trained from scratch using the following command:
  ```bash run_scripts/blip2/train/pretrain_stage1.sh```
- In the second stage, the model is further pre-trained using the following command:
  ```bash run_scripts/blip2/train/pretrain_stage2.sh```
- The model is fine-tuned for specific tasks, such as image captioning, using task-specific scripts (e.g., ```bash run_scripts/blip2/train/train_caption_coco.sh```).
- The pre-training uses AdamW optimizer with specific hyperparameters, a cosine learning rate decay, and data augmentation techniques.

Parameters:
- BLIP-2 has significantly fewer trainable parameters compared to existing methods.
- The exact number of parameters for the Salesforce/blip2-opt-2.7b variant is [More Information Needed].
- During pre-training, the frozen models' parameters are converted to FP16 or BFloat16 for computational efficiency.

Important Disclaimers:
[More Information Needed]

For more information or updates to this model card, please contact the project organizer at [More Information Needed].

- **Developed by:** Junnan Li; Dongxu Li; Silvio Savarese; Steven Hoi
- **Funded by:** Based on the provided references, the model Salesforce/blip2-opt-2.7b appears to be developed by researchers at Salesforce, including Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi (Reference 3). However, there is no explicit mention of the funding sources for this project.

[More Information Needed]
- **Shared by:** Based on the provided references, the following information can be gathered about the contributors who made the model Salesforce/blip2-opt-2.7b available online:

The model Salesforce/blip2-opt-2.7b is the official implementation of the BLIP-2 paper by Salesforce. The model has been integrated into the Hugging Face 🤗 Transformers library.

However, the references do not explicitly mention the names of the individual contributors who worked on developing and releasing this model. More specific information would be needed to determine the exact contributors involved in making the model available on GitHub.

[More Information Needed]
- **Model type:** BLIP-2 is a vision-language pre-training method that bootstraps from frozen pre-trained unimodal models, using a Querying Transformer (Q-Former) to bridge the modality gap through a two-stage pre-training process involving vision-language representation learning and vision-to-language generative learning.
- **Language(s):** The Salesforce/blip2-opt-2.7b model uses natural language instructions to control image-to-text generation, enabling a wide range of zero-shot capabilities such as visual knowledge reasoning, visual commonsense reasoning, visual conversation, and personalized image-to-text generation.
- **License:** [More Information Needed]

The provided references do not contain any information about the specific license being used for the Salesforce/blip2-opt-2.7b model. More details would be needed from the model creators or documentation to determine the license under which this model is being released.
- **Finetuned from model:** Based on the provided references, the Salesforce/blip2-opt-2.7b model is fine-tuned from the following base models:

1. A frozen pre-trained image encoder (specific model name not provided in the references)
2. A frozen pre-trained large language model (LLM) called OPT (specific model version and link not provided in the references)

The Q-Former component in BLIP-2 is pre-trained in two stages: first with the frozen image encoder for vision-language representation learning, and then with the frozen OPT LLM for vision-to-language generative learning.

[More Information Needed] on the specific model names and links for the frozen image encoder and OPT LLM used as the base models for fine-tuning Salesforce/blip2-opt-2.7b.
### Model Sources

- **Repository:** https://github.com/salesforce/LAVIS/tree/main/projects/blip2
- **Paper:** https://arxiv.org/pdf/2301.12597.pdf
- **Demo:** Here is the link to the demo notebook for the Salesforce/blip2-opt-2.7b model:

[Notebook Demo](https://github.com/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb) on instructed vision-to-language generation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb)

The notebook shows how to use the BLIP-2 model to perform zero-shot instructed image-to-text generation.
## Uses

### Direct Use

The Salesforce/blip2-opt-2.7b model can be used for zero-shot instructed image-to-text generation without requiring fine-tuning, post-processing, or plugging into a pipeline. Here's a code snippet demonstrating how to load an image and perform zero-shot image-to-text generation:

```python
import torch
from PIL import Image

image = Image.open("image.jpg")
```

After loading the image, you can directly use the BLIP-2 model to generate text based on the image.

BLIP-2 can also compute the image-text matching score using the same interface as BLIP. Refer to this notebook for an example:
https://github.com/salesforce/LAVIS/blob/3446bac20c5646d35ae383ebe6d13cec4f8b00cb/examples/blip2_image_text_matching.ipynb

Additionally, BLIP-2 supports the Unified Feature Extraction Interface of LAVIS. Check out this notebook for an example:
https://github.com/salesforce/LAVIS/blob/3446bac20c5646d35ae383ebe6d13cec4f8b00cb/examples/blip2_feature_extraction.ipynb

[More Information Needed] on the specific code for generating text from the image and using the model for feature extraction.

### Downstream Use

The Salesforce/blip2-opt-2.7b model can be fine-tuned for various vision-language tasks such as:

1. Image Captioning: The model can be fine-tuned on datasets like COCO to generate text descriptions for images. It achieves state-of-the-art performance on zero-shot captioning on the NoCaps dataset.

2. Visual Question Answering (VQA): By fine-tuning on annotated VQA data like VQAv2 and GQA, the model can answer questions about images. It achieves state-of-the-art results on these datasets.

3. Visual Reasoning and Conversation: The model can perform zero-shot image-to-text tasks like visual knowledge reasoning, visual commonsense reasoning, and personalized image-to-text generation.

When plugged into a larger ecosystem or app, the Salesforce/blip2-opt-2.7b model can enable various functionalities such as:

1. Automatic image captioning for accessibility or content organization.
2. Visual question answering for interactive user experiences or information retrieval.
3. Engaging in visual conversations or generating personalized responses based on images.

To evaluate the pretrained or fine-tuned models, you can run the following scripts:

[More Information Needed]

The model's strong zero-shot capabilities and generalization to out-of-domain images make it suitable for integration into diverse applications dealing with visual content understanding and generation.

For more information on dataset preparation and specific fine-tuning procedures, please refer to the provided documentation and resources.

### Out-of-Scope Use

Based on the provided references about the Salesforce/blip2-opt-2.7b model, here are some potential ways the model may be misused and what users ought not do:

The model could output inaccurate or outdated information when generating text from images, as it relies on the knowledge from the frozen language model which may not have the latest information. Users should not rely on the model's outputs as factual without verifying the information from other authoritative sources.

The model inherits risks from large language models such as generating offensive language, propagating social biases, or leaking private information. Users should not use the model to generate text that could be harmful or discriminatory. 

The model has capabilities for zero-shot image-to-text generation that can reason about images. However, users should not assume the model has human-level visual understanding or commonsense reasoning abilities. The model's outputs may not always be logically sound.

[More Information Needed] on other specific ways the model architecture or training data could lead to potential misuse or unintended negative consequences. The references do not provide enough technical details to comment further.

In general, users ought not deploy the model in high-stakes domains or applications that could cause serious harm if the model makes mistakes or generates problematic content. Careful testing, monitoring, and human oversight are needed to mitigate risks.

### Bias, Risks, and Limitations

Based on the provided references, here are some known or foreseeable issues stemming from the Salesforce/blip2-opt-2.7b model:

1. Unsatisfactory image-to-text generation results due to inaccurate knowledge from the LLM, activating incorrect reasoning paths, or not having up-to-date information about new image content (Reference 1).

2. Inheriting risks from LLMs, such as outputting offensive language, propagating social bias, or leaking private information, due to the use of frozen models (Reference 1).

3. [More Information Needed] on the specific remediation approaches mentioned, such as using instructions to guide the model's generation or training on a filtered dataset with harmful content removed (Reference 5).

4. [More Information Needed] on the potential limitations or biases introduced by the frozen image encoders used in the BLIP-2 architecture (Reference 2).

5. [More Information Needed] on the sociotechnical implications of BLIP-2 being considered a step towards building a multimodal conversational AI agent (Reference 2).

6. The model's performance on various zero-shot vision-language tasks, while improved compared to previous state-of-the-art models, may still have limitations or edge cases that require further investigation (Reference 8).

7. [More Information Needed] on the potential privacy concerns related to the use of in-batch negatives instead of the momentum queue in BLIP (Reference 9).

### Recommendations

Based on the provided references about the Salesforce/blip2-opt-2.7b model, here are some recommendations regarding foreseeable issues:

1. The model could produce unsatisfactory or inaccurate results for image-to-text generation tasks due to reasons like incorrect knowledge from the LLM, activating wrong reasoning paths, or lacking up-to-date information about new image content. To mitigate this, consider using instructions to guide the model's generation or training on filtered datasets with harmful content removed.

2. Since BLIP-2 uses frozen models, it inherits risks associated with LLMs such as outputting offensive language, propagating social biases, or leaking private information. It's important to be aware of and monitor these potential issues.

3. [More Information Needed] on the specific data and pre-training approaches used for the Salesforce/blip2-opt-2.7b model to assess potential biases or fairness concerns.

4. While the model achieves state-of-the-art performance on various vision-language tasks with fewer trainable parameters, [More Information Needed] on the computational resources and environmental impact of training and deploying the model at scale.

5. As the model demonstrates emerging zero-shot image-to-text generation capabilities, it's crucial to establish responsible usage guidelines and consider potential misuse cases. [More Information Needed] on any safety considerations specific to the Salesforce/blip2-opt-2.7b model.

Overall, ongoing monitoring, responsible deployment practices, and transparency about the model's capabilities and limitations are recommended to address foreseeable issues as the Salesforce/blip2-opt-2.7b model is published and used in real-world applications.

## Training Details

### Training Data

The training data of Salesforce/blip2-opt-2.7b consists of 129M images from datasets including COCO, Visual Genome, CC3M, CC12M, SBU, and LAION-400M. The CapFilt method was used to filter the image-text pairs for training ([More Information Needed]).

### Training Procedure

#### Preprocessing

Based on the provided references, here are the details about preprocessing for the data of the model Salesforce/blip2-opt-2.7b:

Image preprocessing:
- Images are resized to 224x224
- Data augmentation is applied, including:
  - Random resized cropping 
  - Horizontal flipping

Tokenization:
[More Information Needed]

Additional notes:
- The model uses a frozen image encoder (ViT-L/14 from CLIP or ViT-g/14 from EVA-CLIP). The last layer of the ViT is removed and the second last layer's output features are used.
- The model uses a frozen language model decoder (OPT).
- During pre-training, the frozen ViT and LLM parameters are converted to FP16 to save memory/compute.

#### Training Hyperparameters

Here are the detailed training hyperparameters for the Salesforce/blip2-opt-2.7b model:

Optimizer:
- AdamW optimizer 
- β1 = 0.9, β2 = 0.98
- Weight decay = 0.05

Learning Rate Schedule:
- Cosine learning rate decay
- Peak learning rate = 1e-4 
- Linear warmup for 2k steps
- Minimum learning rate at second stage = 5e-5

Image Augmentation:
- Image size: 224 x 224
- Random resized cropping
- Horizontal flipping

Pre-training:
- First stage: 250k steps 
- Second stage: 80k steps
- Batch size in first stage: 2320 for ViT-L, 1680 for ViT-g
- Batch size in second stage: 1920 for OPT, 1520 for FlanT5
- Frozen ViT and LLM parameters converted to FP16 during pre-training, except FlanT5 which uses BFloat16

[More Information Needed] on the exact pre-training scripts used.

Pre-training Data:
- Same dataset as BLIP with 129M images total
- Includes COCO, Visual Genome, CC3M, CC12M, SBU, and 115M images from LAION-400M
- CapFilt method used to [More Information Needed]

#### Speeds, Sizes, Times

Here are the details about the Salesforce/blip2-opt-2.7b model based on the provided references:

Throughput: [More Information Needed]

Start or end time: [More Information Needed] 

Checkpoint sizes: [More Information Needed]

The model achieves improved performance on various zero-shot vision-language tasks compared to previous state-of-the-art models, while requiring substantially fewer trainable parameters during vision-language pre-training (Reference 1).

To use the model:

```
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")
```

(Reference 5)

The model can be found on the Hugging Face hub under Salesforce/blip2-opt-2.7b (Reference 6).

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Based on the provided references, the Salesforce/blip2-opt-2.7b model evaluates on the following benchmarks and datasets:

1. Zero-shot vision-language tasks (Table 1)
2. Image captioning: 
   - Finetuned on COCO dataset
   - Evaluated on COCO test set and zero-shot transfer to NoCaps dataset
3. Visual Question Answering (VQA):
   - Finetuned on VQAv2 training and validation splits, and Visual Genome training samples
   - Achieves state-of-the-art results on VQAv2 and GQA datasets (Table 2)

The model demonstrates strong performance and generalization ability, setting new state-of-the-art on zero-shot captioning with a CIDEr score of 121.6 on NoCaps (previous best was 113.2). It also outperforms Flamingo on zero-shot VQAv2 with a score of 65.0 vs 56.3.

#### Factors

Based on the provided references about the Salesforce/blip2-opt-2.7b model, here are some foreseeable characteristics that may influence the model's behavior:

1. Domain and context: The model's performance may vary across different image domains and contexts. For example, the model may have better accuracy on common objects and scenes it was trained on, compared to rare or novel visual concepts. The model's knowledge cutoff date can also impact its ability to reason about new image content.

2. Population subgroups: As mentioned in reference 1, the model inherits risks of the underlying language model, which could propagate social biases. This suggests the model's outputs may exhibit disparities or biases when analyzing images depicting different demographic groups. Disaggregated evaluation across population subgroups would be needed to uncover such biases.

3. Question types: The model's performance on visual question answering may vary depending on the type and complexity of questions asked. [More Information Needed] on specific question types the model excels at or struggles with.

4. Offensive/harmful content: Reference 1 notes the model could output offensive language, likely in response to images containing offensive or harmful content. The prevalence of such content in the model's training data and the effectiveness of filtering techniques used would impact this behavior.

5. Private information leakage: [More Information Needed] on whether the model is susceptible to leaking private information present in training images.

In summary, key factors that require more disaggregated evaluation to uncover their influence on the model's behavior include: image domains, population subgroups, question types, offensive/harmful content, and private information. Targeted testing across these dimensions would help characterize the model's strengths, weaknesses, and fairness.

#### Metrics

Based on the provided references, the following metrics are used for evaluating the Salesforce/blip2-opt-2.7b model:

For image captioning:
- The model is finetuned on COCO dataset and evaluated on both COCO test set and zero-shot transfer to NoCaps dataset. The evaluation metrics for image captioning are not explicitly mentioned. [More Information Needed]

For visual question answering (VQA):
- The model is evaluated on the zero-shot VQA task.
- Beam search with a beam width of 5 is used during generation.
- Length-penalty is set to -1 to encourage shorter answers that align better with human annotation.
- The model achieves 65.0 accuracy on zero-shot VQAv2, outperforming Flamingo (56.3 accuracy).

The references mention that BLIP-2 achieves state-of-the-art performance on various vision-language tasks while having a small number of trainable parameters during pre-training. However, specific tradeoffs between different errors are not discussed in the provided information. [More Information Needed]

### Results

Evaluation results of the model Salesforce/blip2-opt-2.7b based on the Factors and Metrics:

Performance:
- Achieves state-of-the-art results on VQAv2 and GQA tasks (Reference 2)
- Demonstrates strong zero-shot performance on various vision-language tasks (Reference 1)
- Achieves state-of-the-art performance on image captioning with significant improvement on zero-shot transfer to NoCaps (Reference 6)

Efficiency:
- Requires substantially fewer trainable parameters during vision-language pre-training compared to previous state-of-the-art models (Reference 1)

Intended Uses:
- Enables a wide range of zero-shot image-to-text capabilities, including visual knowledge reasoning, visual commonsense reasoning, visual conversation, and personalized image-to-text generation (Reference 3)

Limitations and Bias:
- Image-to-text generation could have unsatisfactory results due to inaccurate knowledge from the LLM, activating incorrect reasoning paths, or not having up-to-date information about new image content (Reference 10)
- Inherits the risks of LLMs, such as outputting offensive language, propagating social bias, or leaking private information (Reference 10)

Training Data:
- VQA data includes training and validation splits from VQAv2, as well as training samples from Visual Genome (Reference 7)

Training Procedure:
- Finetunes the parameters of the Q-Former and the image encoder while keeping the LLM frozen (Reference 8)
- Finetunes with the open-ended answer generation loss (Reference 8)
- Conditions Q-Former on the question to extract more relevant image features (Reference 8)

Zero-shot Usage:
```
[More Information Needed]
```

Finetuning Usage:
```
[More Information Needed]
```

#### Summary

Based on the provided references, here is a summary of the evaluation results for the Salesforce/blip2-opt-2.7b model:

- BLIP-2 achieves state-of-the-art performance on various zero-shot vision-language tasks while requiring substantially fewer trainable parameters compared to previous models. (Reference 1)

- For the image captioning task, BLIP-2 achieves state-of-the-art performance with significant improvement on the NoCaps dataset, demonstrating strong generalization ability to out-of-domain images. (Reference 4)

- BLIP-2 achieves state-of-the-art results on the VQAv2 and GQA datasets for the visual question answering task. (Reference 6) 

- On the zero-shot VQA task, BLIP-2 outperforms Flamingo80B by 8.7% on VQAv2 with 54x fewer trainable parameters. (Reference 9)

- BLIP-2 establishes a new state-of-the-art on zero-shot captioning, achieving a CIDEr score of 121.6 on NoCaps compared to the previous best of 113.2. (Reference 11)

[More Information Needed] on the exact performance metrics for some of the mentioned tasks.

## Model Examination

Here is my attempt at an experimental section on explainability/interpretability for the Salesforce/blip2-opt-2.7b model:

The BLIP-2 model achieves strong zero-shot performance on various vision-language tasks by effectively combining a frozen image encoder with a frozen language model. This allows the model to understand images while preserving its capability to follow text prompts and instructions.

However, the model can still produce unsatisfactory or inaccurate image-to-text generation in some cases. This may be due to:
- The language model outputting incorrect knowledge 
- Activating an incorrect reasoning path for the given image
- Lacking up-to-date information about new visual concepts

Furthermore, by using frozen pretrained components, BLIP-2 inherits some risks of large language models, such as potentially generating offensive language, propagating social biases, or leaking private information.

[More Information Needed] on specific techniques to analyze what features or reasoning paths the BLIP-2 model uses to generate text for a given image. Visualizing the model's attention weights or most activated neurons could provide insights into its decision process.

Potential remediation approaches to improve the model's outputs include:
- Using instructions to better guide the model's generation 
- [More Information Needed] on other techniques

In summary, while BLIP-2 demonstrates impressive zero-shot capabilities, more work is needed to fully understand and explain its reasoning process from image to generated text. Improved interpretability could help identify failure modes and mitigate risks inherited from the underlying frozen models.

## Environmental Impact

- **Hardware Type:** Based on the provided references, the Salesforce/blip2-opt-2.7b model was trained using the following hardware:

During pre-training, the frozen ViT and LLM parameters were converted to FP16 precision, except for FlanT5 where BFloat16 was used. This allowed for more computationally friendly pre-training due to the use of frozen models.

However, the specific hardware type (e.g., GPU model) used for training is not explicitly mentioned in the provided references. [More Information Needed] on the exact hardware specifications.
- **Software Type:** The model Salesforce/blip2-opt-2.7b is trained using PyTorch deep learning framework, as evidenced by the code snippet in Reference 6:

```python
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")
```

The `from_pretrained` method is used to load the pre-trained model and processor, which is a common practice in PyTorch-based models.
- **Hours used:** According to the pre-training settings mentioned in the references:

We pre-train for 250k steps in the first stage and 80k steps in the second stage.

However, the exact amount of time used for training is not provided. To determine the total training time, more information would be needed, such as:
- The hardware used for training (GPU specs, number of GPUs, etc.)
- The time taken per training step
- Any additional details about the training process that could affect the total duration

Therefore, based on the given information, the specific amount of time used to train the Salesforce/blip2-opt-2.7b model cannot be determined. [More Information Needed]
- **Cloud Provider:** [More Information Needed]

The provided references do not mention the specific cloud provider that the model Salesforce/blip2-opt-2.7b was trained on. More information would be needed from the model developers or documentation to determine the cloud provider used for training this model.
- **Carbon Emitted:** [More Information Needed]

The provided references do not contain any information about the amount of carbon emitted during the training of the Salesforce/blip2-opt-2.7b model. To answer this question, we would need more specific details about the computational resources used, training time, and energy consumption during the model development process.
## Technical Specification

### Model Architecture and Objective

Model Architecture:
BLIP-2 consists of a frozen image encoder, a trainable Q-Former module, and a frozen language model (LLM). The Q-Former bridges the gap between the frozen image encoder and LLM. It has two transformer submodules that share the same self-attention layers: (1) an image transformer that interacts with the frozen image encoder for visual feature extraction, and (2) a text transformer that can function as both a text encoder and a text decoder.

The Q-Former extracts a fixed number of output features (queries) from the image encoder, independent of input image resolution. In the experiments, 32 queries are used, where each query has a dimension of 768 (same as the hidden dimension of the Q-Former). The output query representation (denoted as Z) has a size of 32 × 768, which is much smaller than the size of frozen image features.

Objectives:
BLIP-2 is pre-trained using image-text pairs with three objectives that share the same input format and model parameters:

1. Image-Text Contrastive Learning (ITC): Computes the pairwise similarity between each query output and the text representation, and selects the highest one as the image-text similarity. A unimodal self-attention mask is employed to avoid information leak between queries and text.

2. Image-Text Matching (ITM): A binary classification task that predicts whether an image-text pair is positive (matched) or negative (unmatched). A bi-directional self-attention mask is used where all queries and texts can attend to each other. Each output query embedding is fed into a two-class linear classifier to obtain a logit.

3. [More Information Needed]

These objectives work together to force the queries to extract visual information that is most relevant to the text. The pre-training aims to train the Q-Former such that the queries can learn to extract visual representation that is most informative of the text.

Sample Code:
```python
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")
```

### Compute Infrastructure

Here is the compute infrastructure information for the Salesforce/blip2-opt-2.7b model based on the provided references:

Pre-training:
- Used AdamW optimizer with β1 = 0.9, β2 = 0.98, weight decay of 0.05
- Cosine learning rate decay with peak learning rate of 1e-4 and linear warmup of 2k steps
- Minimum learning rate at second stage is 5e-5
- Used image size of 224x224 with random resized cropping and horizontal flipping augmentation
- Pre-trained for 250k steps in first stage, 80k steps in second stage
- Batch size of 2320/1680 for ViT-L/ViT-g in first stage, 1920/1520 for OPT/FlanT5 in second stage
- Converted frozen ViTs' and LLMs' parameters to FP16, except FlanT5 which used BFloat16

[More Information Needed] on the exact compute hardware (GPU/TPU types and counts) used for pre-training.

Inference:
The model can be run on GPU if available, otherwise falls back to CPU:

```
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
```

To load the model in 8-bit and automatically map to available devices:

```
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")
```

[More Information Needed] on recommended GPU hardware for optimal inference performance and throughput.

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

