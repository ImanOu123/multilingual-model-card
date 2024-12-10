# pip install -q transformers
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# checkpoint = "CohereForAI/aya-101"

# tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="/compute/babel-12-25/jiaruil5/.cache/")
# aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, cache_dir="/compute/babel-12-25/jiaruil5/.cache/")

# # Turkish to English translation
# tur_inputs = tokenizer.encode("Translate to English: Aya cok dilli bir dil modelidir.", return_tensors="pt")
# tur_outputs = aya_model.generate(tur_inputs, max_new_tokens=128)
# print(tokenizer.decode(tur_outputs[0]))
# # Aya is a multi-lingual language model

# # Q: Why are there so many languages in India?
# hin_inputs = tokenizer.encode("भारत में इतनी सारी भाषाएँ क्यों हैं?", return_tensors="pt")
# hin_outputs = aya_model.generate(hin_inputs, max_new_tokens=128)
# print(tokenizer.decode(hin_outputs[0]))
# # Expected output: भारत में कई भाषाएँ हैं और विभिन्न भाषाओं के बोली जाने वाले लोग हैं। यह विभिन्नता भाषाई विविधता और सांस्कृतिक विविधता का परिणाम है। Translates to "India has many languages and people speaking different languages. This diversity is the result of linguistic diversity and cultural diversity."


from transformers import AutoTokenizer, AutoModelForCausalLM

# model_id = "CohereForAI/aya-23-8B"
model_id = "CohereForAI/aya-expanse-8b"


tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/compute/babel-12-25/jiaruil5/.cache/")
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="/compute/babel-12-25/jiaruil5/.cache/", force_download=True)

# Format message with the command-r-plus chat template
messages = [{"role": "user", "content": "Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0][len(input_ids[0]):], skip_special_tokens=True)
print(gen_text)