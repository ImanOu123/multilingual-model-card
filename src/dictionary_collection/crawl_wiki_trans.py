import requests
from bs4 import BeautifulSoup
import markdownify
import sys
import json
from crawl_wikipedia import fetch_wikipedia_page, save_to_file

def extract_glossary_links(html_content):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    # print(soup)
    
    # Find all dl tags with class "glossary"
    glossary_lsts = soup.find_all('dl', class_='glossary')
    
    # Extract the glossary terms and their links
    glossary_entries = []

    for lst in glossary_lsts[:1]:
        
        glossary_terms = lst.find_all('dt')

        for term in glossary_terms[1:]:
            if term.find("a") and term.find("a").has_attr('href'):
                term_text = term.get_text(separator=' ', strip=True)
                term_link = term.find("a")['href']
                
                glossary_entries.append({
                    'term' : term_text, 
                    'link' : "https://en.wikipedia.org/" + term_link
                })
        
    return glossary_entries

def extract_translated_terms(html_content):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all li tags with "interlanguage-link" in class
    lang_lst = soup.find_all('li', class_=['interlanguage-link'])
    
    glossary_trans = {}
    
    for lang in lang_lst:
        # Find languages linked to wiki page of terms
        language = lang.find('a')['lang'] if lang.find('a').has_attr('href') else None
        
        # Extract term in other language
        title = None
        if lang.find('a').has_attr('href'):
            title = lang.find('a')['title']
            hy_idx = title.index("–")
            title = title[:hy_idx].strip()
        
        # Check if one of the desired languages is available
        if language and language in ['ar', 'zh', 'ru', 'ja', 'fr']:
            glossary_trans[language] = title
            
    return glossary_trans

if __name__ == '__main__':
    # Wikipedia page URL
    url = sys.argv[1]  # Replace with your desired Wikipedia page
    
    # Fetch the Wikipedia page
    html_content = fetch_wikipedia_page(url)
    
    save_to_file(html_content, sys.argv[2])
    
    # Extract terms and the links to their own wiki pages
    entries_link_dict = extract_glossary_links(html_content)
    
    # Extract the terms and their corresponding translations
    glossary_trans_dict = {}
    for entry_link in entries_link_dict:
        html_content = fetch_wikipedia_page(entry_link['link'])
        glossary_trans_dict[entry_link['term']] = extract_translated_terms(html_content)
    json.dump(glossary_trans_dict, open(sys.argv[3], 'w'), indent=2, ensure_ascii=False)

# Usage
# python3 crawl_wiki_trans.py https://en.wikipedia.org/wiki/Glossary_of_artificial_intelligence raw_data/wiki_ai.html raw_data/wiki_ai_trans.json
# python3 crawl_wiki_trans.py https://en.wikipedia.org/wiki/Glossary_of_computer_science raw_data/wiki_cs.html raw_data/wiki_cs_trans.json