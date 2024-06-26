import requests
import json
from bs4 import BeautifulSoup
import re

def fetch_conference_paper_page(url):
    # Send a GET request to the page
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page: {url}")
    
    return response.text

def save_to_file(content, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)
        
def extract_awarded_papers(html_content, conf, year, man_best_paper=[], man_other_lst=[]):
    # parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all p tags - includes papers and information about papers
    paper_lst = soup.find_all('p')
    
    # extract title, link for pdf and award of paper
    awarded_papers = []
    for paper in paper_lst:
        title = paper.find("strong").text if paper.find("strong") else None
        link = paper.find("a", class_="badge badge-primary align-middle mr-1")['href'] if paper.find("a", class_="badge badge-primary align-middle mr-1") else None
        award = paper.find("i", class_="fas fa-award").parent['title'] if paper.find("i", class_="fas fa-award") else None

        # check if award associated with paper or part of manual search list
        tmp_title = re.sub(r'[^0-9a-zA-Z /]+', "", title.lower()) if title else None
        
        if award:
            awarded_papers.append({"title" : title, "venue" : conf, "year" : year, "award" : award, "link" : link})

        elif tmp_title in man_best_paper:
            awarded_papers.append({"title" : title, "venue" : conf, "year" : year, "award" : "Best Paper", "link" : link})
            
        else:
            for (ppr, award) in man_other_lst:
                if tmp_title == ppr:
                    awarded_papers.append({"title" : title, "venue" : conf, "year" : year, "award" : award, "link" : link})
        
        
    return awarded_papers
    
    
    
if __name__ == "__main__":
    # ACL CONFERENCE
    
    # source: https://aclweb.org/aclwiki/Best_paper_awards
    # man_paper_dict = {"2003" : ["Towards a Model of Face-to-Face Grounding"], 
    #                   "2006" : ["Semantic taxonomy induction from heterogenous evidence"],
    #                   "2007" : ["Learning synchronous grammars for semantic parsing with lambda calculus"],
    #                   "2008" : ["Forest Reranking: Discriminative Parsing with Non-Local Features", "A New String-to-Dependency Machine Translation Algorithm with a Target Dependency Language Model"],
    #                   "2009" : ["Concise Integer Linear Programming Formulations for Dependency Parsing", "Reinforcement Learning for Mapping Instructions to Actions", "K-Best A* Parsing"],
    #                   "2010" : ["Beyond NomBank: A Study of Implicit Arguments for Nominal Predicates", "SVD and Clustering for Unsupervised POS Tagging", "Extracting Social Networks from Literary Fiction"],
    #                   "2011" : ["Unsupervised Part-of-Speech Tagging with Bilingual Graph-Based Projections"],
    #                   "2012" : ["Bayesian Symbol-Refined Tree Substitution Grammars for Syntactic Parsing", "String Re-writing Kernel"],
    #                   "2013" : ["Grounded Language Learning from Video Described with Sentences", "A corpus-based evaluation method for Distributional Semantic Models"],
    #                   "2014" : ["Fast and Robust Neural Network Joint Models for Statistical Machine Translation", "Low-Rank Tensors for Scoring Dependency Structures"],
    #                   "2015" : ["Improving Evaluation of Machine Translation Quality Estimation", "Learning Dynamic Feature Selection for Fast Sequential Prediction"],
    #                   "2016" : ["Finding Non-Arbitrary Form-Meaning Systematicity Using String-Metric Learning for Kernel Regression"],
    #                   "2017" : ["Probabilistic Typology: Deep Generative Models of Vowel Inventories"],
    #                   "2018" : ["Finding syntax in human encephalography with beam search", "Learning to Ask Good Questions: Ranking Clarification Questions using Neural Expected Value of Perfect Information", "Let’s do it “again”: A First Computational Approach to Detecting Adverbial Presupposition Triggers"]
    #                 }
        
    # awarded_papers = [] 
     
    # # extract content from ACL conference pages
    # for yr in range(2000, 2024):
        
    #     # extract papers I want to manually add to list
    #     man_paper_lst = []
    #     if str(yr) in man_paper_dict:
    #         man_paper_lst = list(map(lambda x: re.sub(r'[^0-9a-zA-Z /]+', "", x.lower()), man_paper_dict[str(yr)]))

    #     content = fetch_conference_paper_page('https://aclanthology.org/events/acl-' + str(yr) + '/')
    #     awarded_papers += extract_awarded_papers(content, "ACL", yr, man_paper_lst)
        
    #     # manually add specific papers with no info in original ACL site
    #     if yr == 2014:
    #         awarded_papers.append({"title" : "Improving sparse word similarity models with asymmetric similarity measures", "venue" : "ACL", "year" : 2014, 
    #                                "award" : "Best Paper", "link" : "https://aclanthology.org/P14-2049.pdf"})
            
    
    # json.dump(awarded_papers, open("json/acl_best_papers.json", 'w'), indent=2, ensure_ascii=False)
    
    # ------------------------------------------------------------------------------------------------------
    
    # EMNLP CONFERENCE
    
    # sources: https://2023.emnlp.org/program/best_papers/, https://aisb.org.uk/conference-reports-empirical-methods-in-natural-language-processing-emnlp-2022/, https://2021.emnlp.org/blog/2021-10-29-best-paper-awards, https://2020.emnlp.org/blog/2020-11-19-best-papers, https://aclweb.org/aclwiki/Best_paper_awards
    # best_paper_dict = {"2023" : ["Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning", "Faster Minimum Bayes Risk Decoding with Confidence-based Pruning", "Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs Through a Global Prompt Hacking Competition", "PaperMage: A Unified Toolkit for Processing, Representing, and Manipulating Visually-Rich Scientific Documents", "Personalized Dense Retrieval on Global Index for Voice-enabled Conversational Systems"],
    #                   "2022" : ["Abstract Visual Reasoning with Tangram Shapes", "Topic-Regularized Authorship Representation Learning"],
    #                   "2021" : ["Visually Grounded Reasoning across Languages and Cultures", "CHoRaL: Collecting Humor Reaction Labels from Millions of Social Media Users"],
    #                   "2020" : ["Digital voicing of Silent Speech"],
    #                   "2002" : ["Discriminative Training Methods for Hidden Markov Models: Theory and Experiments with Perceptron Algorithms", "Using the Web to Overcome Data Sparseness"],
    #                   "2003" : ["Training Connectionist Models for the Structured Language Model"],
    #                   "2004" : ["Max-Margin Parsing"],
    #                   "2005" : ["Non-Projective Dependency Parsing using Spanning Tree Algorithms"],
    #                   "2007" : ["Modelling Compression with Discourse Constraints"],
    #                   "2009" : ["Unsupervised semantic parsing"],
    #                   "2010" : ["Dual Decomposition for Parsing with Non-Projective Head Automata"],
    #                   "2011" : ["A Probabilistic Forest-to-String Model for Language Generation from Typed Lambda Calculus Expressions"],
    #                   "2012" : ["A Coherence Model Based on Syntactic Patterns"],
    #                   "2013" : ["Breaking Out of Local Optima with Count Transforms and Model Recombination: A Study in Grammar Induction"],
    #                   "2014" : ["Modeling Biological Processes for Reading Comprehension"],
    #                   "2015" : ["Broad-coverage CCG Semantic Parsing with AMR", "Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems"],
    #                   "2016" : ["Improving Information Extraction by Acquiring External Evidence with Reinforcement Learning", "Global Neural CCG Parsing with Optimality Guarantees"],
    #                   "2017" : ["Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints", "Depression and Self-Harm Risk Assessment in Online Forums"],
    #                   "2018" : ["Linguistically-Informed Self-Attention for Semantic Role Labeling"],
    #                   "2019" : ["Specializing Word Embeddings (for Parsing) by Information Bottleneck"]}
        
        
    # other_paper_dict = {"2021" : [("MindCraft: Theory of Mind Modeling for Situated Dialogue in Collaborative Tasks", "Outstanding Paper"), ("SituatedQA: Incorporating Extra-Linguistic Contexts into QA", "Outstanding Paper"), ("When Attention Meets Fast Recurrence: Training Language Models with Reduced Compute", "Outstanding Paper"), ("Datasets: A Community Library for Natural Language Processing","Best Demonstration Paper")],
    #                     "2020" : [("GLUCOSE: GeneraLized and COntextualized Story Explanations", "Honorable Mention"), ("Spot The Bot: A Robust and Efficient Framework for the Evaluation of Conversational Dialogue Systems", "Honorable Mention"), ("Visually Grounded Compound PCFGs", "Honorable Mention"), ("Transformers: State-of-the-art Natural Language Processing", "Best Demonstration Award")]}
    
     
    # # extract content from EMNLP conference pages
    # awarded_papers = [] 
    # for yr in range(2000, 2024):
        
    #     # extract papers I want to manually add to list
    #     best_paper_lst = []
    #     if str(yr) in best_paper_dict:
    #         best_paper_lst = list(map(lambda x: re.sub(r'[^0-9a-zA-Z /]+', "", x.lower()), best_paper_dict[str(yr)]))
        
    #     other_paper_lst = []
    #     if str(yr) in other_paper_dict:
    #         other_paper_lst = list(map(lambda x: (re.sub(r'[^0-9a-zA-Z /]+', "", x[0].lower()), x[1]), other_paper_dict[str(yr)]))
        
    #     content = fetch_conference_paper_page('https://aclanthology.org/events/emnlp-' + str(yr) + '/')
    #     awarded_papers += extract_awarded_papers(content, "EMNLP", yr, best_paper_lst, other_paper_lst)
        
    # json.dump(awarded_papers, open("json/emnlp_best_papers.json", 'w'), indent=2, ensure_ascii=False)
    
    # ------------------------------------------------------------------------------------------------------
    
    # NAACL CONFERENCE
    
    # naacl_antho_urls = {"2024" : ["https://aclanthology.org/volumes/2024.naacl-long/", "https://aclanthology.org/volumes/2024.naacl-short/", "https://aclanthology.org/volumes/2024.naacl-demo/"], 
    #                     "2022" : ["https://aclanthology.org/volumes/2022.naacl-main/", "https://aclanthology.org/volumes/2022.naacl-demo/"], 
    #                     "2021" : ["https://aclanthology.org/volumes/2021.naacl-main/", "https://aclanthology.org/volumes/2021.naacl-demos/"], 
    #                     "2019" : ["https://aclanthology.org/volumes/N19-1/", "https://aclanthology.org/volumes/N19-4/"],
    #                     "2018" : ["https://aclanthology.org/volumes/N18-1/", "https://aclanthology.org/volumes/N18-2/", "https://aclanthology.org/volumes/N18-5/"],
    #                     "2016" : ["https://aclanthology.org/volumes/N16-1/", "https://aclanthology.org/volumes/N16-3/"],
    #                     "2015" : ["https://aclanthology.org/volumes/N15-1/", "https://aclanthology.org/volumes/N15-3/"],
    #                     "2013" : ["https://aclanthology.org/volumes/N13-1/"], 
    #                     "2012" : ["https://aclanthology.org/volumes/N12-1/"],
    #                     "2010": ["https://aclanthology.org/volumes/N10-1/"], 
    #                     "2009" : ["https://aclanthology.org/volumes/N09-1/", "https://aclanthology.org/volumes/N09-2/"],
    #                     "2007" : ["https://aclanthology.org/volumes/N07-1/", "https://aclanthology.org/volumes/N07-2/"],
    #                     "2006" : ["https://aclanthology.org/volumes/N06-1/", "https://aclanthology.org/volumes/N06-2/"], 
    #                     "2004" : ["https://aclanthology.org/volumes/N04-1/", "https://aclanthology.org/volumes/N04-4/"],
    #                     "2003" : ["https://aclanthology.org/volumes/N03-1/", "https://aclanthology.org/volumes/N03-2/"], 
    #                     "2001" : ["https://aclanthology.org/volumes/N01-1/"], 
    #                     "2000" : ["https://aclanthology.org/volumes/A00-1/"]}
    
    # # sources: https://www.cs.columbia.edu/2024/six-papers-from-the-nlp-speech-group-accepted-to-naacl-2024/, https://aclweb.org/aclwiki/Best_paper_awards
    
    # best_paper_dict = {"2004" : ["Catching the Drift: Probabilistic Content Models, with Applications to Generation and Summarization"],
    #                    "2006" :	["Context-Free Grammar Induction Based on Structural Zeros", "Prototype-Driven Learning for Sequence Models"],
    #                    "2007" :	["Combining Outputs from Multiple Machine Translation Systems"],
    #                    "2009" : ["Unsupervised Morphological Segmentation with Log-Linear Models", "New Features for Statistical Machine Translation"],
    #                    "2010" : ["Coreference Resolution in a Modular, Entity-Centered Model", "“cba to check the spelling”: Investigating Parser Performance on Discussion Forum Posts"],
    #                    "2012" : ["Vine Pruning for Efficient Multi-Pass Dependency Parsing", "Trait-Based Hypothesis Selection for Machine Translation", "Cross-lingual Word Clusters for Direct Transfer of Linguistic Structure"],
    #                    "2013" : ["The Life and Death of Discourse Entities: Identifying Singleton Mentions", "Automatic Generation of English Respellings"],
    #                    "2015" : ["Unsupervised Morphology Induction Using Word Embeddings", "'You’re Mr. Lebowski, I’m the Dude': Inducing Address Term Formality in Signed Social Networks", "Retrofitting Word Vectors to Semantic Lexicons"],
    #                    "2016" : ["Feuding Families and Former Friends; Unsupervised Learning for Dynamic Fictional Relationships", "Learning to Compose Neural Networks for Question Answering"],
    #                    "2018" :	["Deep contextualized word representations"],
    #                    "2019" : ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding}"],
    #                    "2024" : ["Teaching Language Models to Self-Improve through Interactive Demonstrations"]}
    # other_paper_dict = {}
     
    # # extract content from NAACL conference pages
    # awarded_papers = [] 
    # for yr in range(2000, 2025):
        
    #     # extract papers to manually add to list
    #     best_paper_lst = []
    #     if str(yr) in best_paper_dict:
    #         best_paper_lst = list(map(lambda x: re.sub(r'[^0-9a-zA-Z /]+', "", x.lower()), best_paper_dict[str(yr)]))
        
    #     other_paper_lst = []
    #     if str(yr) in other_paper_dict:
    #         other_paper_lst = list(map(lambda x: (re.sub(r'[^0-9a-zA-Z /]+', "", x[0].lower()), x[1]), other_paper_dict[str(yr)]))

    #     if str(yr) in naacl_antho_urls:
    #         for url in naacl_antho_urls[str(yr)]:
    #             content = fetch_conference_paper_page(url)
    #             awarded_papers += extract_awarded_papers(content, "NAACL", yr, best_paper_lst, other_paper_lst)
        
    # json.dump(awarded_papers, open("json/naacl_best_papers.json", 'w'), indent=2, ensure_ascii=False)
    
    # ------------------------------------------------------------------------------------------------------
    
    # EACL CONFERENCE
    
    # eacl_antho_urls = {"2024" : ["https://aclanthology.org/volumes/2024.eacl-long/", "https://aclanthology.org/volumes/2024.eacl-short/", "https://aclanthology.org/volumes/2024.eacl-demo/"], 
    #                     "2023" : ["https://aclanthology.org/volumes/2023.eacl-main/", "https://aclanthology.org/volumes/2023.eacl-demo/"], 
    #                     "2021" : ["https://aclanthology.org/volumes/2021.eacl-main/", "https://aclanthology.org/volumes/2021.eacl-demos/"], 
    #                     "2017" : ["https://aclanthology.org/volumes/E17-1/", "https://aclanthology.org/volumes/E17-2/", "https://aclanthology.org/volumes/E17-3/"],
    #                     "2014" : ["https://aclanthology.org/volumes/E14-1/", "https://aclanthology.org/volumes/E14-2/"],
    #                     "2012" : ["https://aclanthology.org/volumes/E12-1/", "https://aclanthology.org/volumes/E12-2/"],
    #                     "2009" : ["https://aclanthology.org/volumes/E09-1/", "https://aclanthology.org/volumes/E09-2/"],
    #                     "2006" : ["https://aclanthology.org/volumes/E06-1/", "https://aclanthology.org/volumes/E06-2/"], 
    #                     "2003" : ["https://aclanthology.org/volumes/E03-1/", "https://aclanthology.org/volumes/E03-2/"]}
    
    # # sources: https://aclweb.org/mirror/eacl2012/
    # best_paper_dict = {"2012" : ["Spectral Learning for Non-Deterministic Dependency Parsing"]}
    # other_paper_dict = {}
     
    # # extract content from NAACL conference pages
    # awarded_papers = [] 
    # for yr in range(2000, 2025):
        
    #     # extract papers to manually add to list
    #     best_paper_lst = []
    #     if str(yr) in best_paper_dict:
    #         best_paper_lst = list(map(lambda x: re.sub(r'[^0-9a-zA-Z /]+', "", x.lower()), best_paper_dict[str(yr)]))
        
    #     other_paper_lst = []
    #     if str(yr) in other_paper_dict:
    #         other_paper_lst = list(map(lambda x: (re.sub(r'[^0-9a-zA-Z /]+', "", x[0].lower()), x[1]), other_paper_dict[str(yr)]))

    #     if str(yr) in eacl_antho_urls:
    #         for url in eacl_antho_urls[str(yr)]:
    #             content = fetch_conference_paper_page(url)
    #             awarded_papers += extract_awarded_papers(content, "EACL", yr, best_paper_lst, other_paper_lst)
        
    # json.dump(awarded_papers, open("json/eacl_best_papers.json", 'w'), indent=2, ensure_ascii=False)
    
    # ------------------------------------------------------------------------------------------------------
    
    # LREC CONFERENCE
    
    lrec_antho_urls = {"2024" : ["https://aclanthology.org/volumes/2024.lrec-main/"], 
                        "2022" : ["https://aclanthology.org/volumes/2022.lrec-1/"], 
                        "2020" : ["https://aclanthology.org/volumes/2020.lrec-1/"], 
                        "2018" : ["https://aclanthology.org/volumes/L18-1/"],
                        "2016" : ["https://aclanthology.org/volumes/L16-1/"],
                        "2014" : ["https://aclanthology.org/volumes/L14-1/"],
                        "2012" : ["https://aclanthology.org/volumes/L12-1/"],
                        "2010" : ["https://aclanthology.org/volumes/L10-1/"],
                        "2008" : ["https://aclanthology.org/volumes/L08-1/"],
                        "2006" : ["https://aclanthology.org/volumes/L06-1/"], 
                        "2004" : ["https://aclanthology.org/volumes/L04-1/"],
                        "2002" : ["https://aclanthology.org/volumes/L02-1/"],
                        "2000" : ["https://aclanthology.org/volumes/L00-1/"]}
    
    # sources: 
    best_paper_dict = {}
    other_paper_dict = {"2024" : ("When Your Cousin Has the Right Connections: Unsupervised Bilingual Lexicon Induction for Related Data-Imbalanced Languages", "Best Student Paper Award")}
     
    # extract content from NAACL conference pages
    awarded_papers = [] 
    for yr in range(2000, 2025):
        
        # extract papers to manually add to list
        best_paper_lst = []
        if str(yr) in best_paper_dict:
            best_paper_lst = list(map(lambda x: re.sub(r'[^0-9a-zA-Z /]+', "", x.lower()), best_paper_dict[str(yr)]))
        
        other_paper_lst = []
        if str(yr) in other_paper_dict:
            other_paper_lst = list(map(lambda x: (re.sub(r'[^0-9a-zA-Z /]+', "", x[0].lower()), x[1]), other_paper_dict[str(yr)]))

        if str(yr) in lrec_antho_urls:
            for url in lrec_antho_urls[str(yr)]:
                content = fetch_conference_paper_page(url)
                awarded_papers += extract_awarded_papers(content, "LREC", yr, best_paper_lst, other_paper_lst)
        
    json.dump(awarded_papers, open("json/lrec_best_papers.json", 'w'), indent=2, ensure_ascii=False)