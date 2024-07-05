import re
import json
from nltk import word_tokenize, sent_tokenize

class DocProcessor():
    def __init__(self, paper_path):
        self.doc = []
        self.doc.extend(
            self.read_doc(
                paper_path
            )
        )
    
    def format_str(self, doc):
        return [{"heading": d['heading'].replace(r"\{|\}", ""), "content": d['content'].replace(r"\{|\}", "")} for d in doc]
    
    def read_doc(self, doc_path):
        print(f"Loading the doc {doc_path}..")
        doc = None
        with open(doc_path, 'r') as file:
            doc = json.load(file)
            res = []
            # heading: xxx, content: xxx
            info = {
                'heading': None,
                'content': None,
                'source': 'paper',
                # 'code_blocks': 
            }
            
            title = doc.get("title", '')
            if title != '':
                info['heading'] = 'title'
                info['content'] = title
                res.append(info.copy())
            
            authors = doc.get("authors", '')
            if authors != '':
                info['heading'] = 'authors'
                info['content'] = authors
                res.append(info.copy())
            
            abstract = doc.get('abstract', '')
            if abstract != '':
                info['heading'] = 'abstract'
                info['content'] = abstract
                res.append(info.copy())
            
            for section in doc.get("sections", []):
                heading = section.get("heading", "")
                txt = section.get("text", "")
                if heading != "" and txt != "":
                    info['heading'] = heading
                    info['content'] = txt
                    res.append(info.copy())
            return self.format_str(res)

def split_paragraph(paragraph, max_words=64):
    # Step 1: Split the paragraph into sentences
    sentences = sent_tokenize(paragraph)
    
    # Step 2: Handle newline characters within sentences
    split_sentences = []
    for sentence in sentences:
        sub_sentences = re.split(r'(\n+)', sentence)
        split_sentences.extend(sub_sentences)
    
    chunks = []
    current_chunk = []

    def add_sentence_to_chunk(sentence, chunk):
        words = word_tokenize(sentence)
        if len(words) > max_words:
            # If the sentence itself is longer than max_words, split it further
            sub_chunks = split_long_sentence(sentence, max_words)
            chunk.extend(sub_chunks)
        else:
            chunk.append(sentence)
        return chunk

    def split_long_sentence(sentence, max_words):
        
        words = word_tokenize(sentence)
        sub_chunks = []
        current_sub_chunk = []
        for word in words:
            current_sub_chunk.append(word)
            if len(current_sub_chunk) >= max_words:
                sub_chunks.append(' '.join(current_sub_chunk))
                current_sub_chunk = []
        if current_sub_chunk:
            sub_chunks.append(' '.join(current_sub_chunk))
        return sub_chunks

    # Step 3: Join sentences into chunks
    for sentence in split_sentences:
        current_chunk = add_sentence_to_chunk(sentence, current_chunk)
        current_chunk_str = ' '.join(current_chunk)
        if len(word_tokenize(current_chunk_str)) > max_words:
            # Remove the last added sentence to keep the chunk under max_words
            removed_sentence = current_chunk.pop()
            chunks.append(' '.join(current_chunk))
            current_chunk = [removed_sentence]

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks