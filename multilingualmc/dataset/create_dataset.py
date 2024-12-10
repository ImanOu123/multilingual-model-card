import os
import random
import json
random.seed(42)

dir_paper = "../data_ai_papers/text/"
dir_model_card = "../data_model_cards/original/claude3_json/"

papers = [i for i in os.listdir(dir_paper)]
model_cards = [i for i in os.listdir(dir_model_card)]

papers_test = random.sample(papers, 50)
papers = list(set(papers) - set(papers_test))

dataset = {
    "test": {
        "model_card": model_cards,
        "paper": papers_test
    },
    "dev": {
        "model_card": [],
        "paper": papers
    }
}

out_f = open("info.json", 'w')
json.dump(dataset, out_f, indent=2)