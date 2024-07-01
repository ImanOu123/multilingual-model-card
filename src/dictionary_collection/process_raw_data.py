import re

unrelated_terms = {
    "6060_initiative": [
        "For", "re",
        "online", "automatically", "Therefore", "Moreover", "Furthermore", "manually", "namely", "Additionally", "Hence", "pairwise",
        "automatic", "previous", "unseen", "Similar", "large", "simultaneous", "human", "motivated", "structured", "natural", "social", "visual", "spatial",
        "Detecting", "compared", "propose", "approach", "meaning", "spoken", "aspect", "corresponding", "shared", "named", "linked", "written",
        "English", "newswire", "French",
        "news", "number", "question", "information", "analysis", "problem", "answer", "questions", 
    ]
}

to_lowercase_terms = {
    "6060_initiative": [
        'Modeling', 'Morphology', 'Dialogue', 'Semantic', 'Clustering', 'Architectures'
    ]
}

def format_wiki_terms(term):
    # remove content inside the parenthesis
    return re.sub(r'\(.*?\)', '', term).strip()

