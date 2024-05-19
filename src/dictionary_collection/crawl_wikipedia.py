import requests
from bs4 import BeautifulSoup
import markdownify
import sys
import json

def fetch_wikipedia_page(url):
    # Send a GET request to the Wikipedia page
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page: {url}")
    
    return response.text

def parse_html_to_markdown(html_content):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract the content within the main content div
    content_div = soup.find('div', {'id': 'bodyContent'})
    if not content_div:
        raise Exception("Could not find the main content div")
    
    # Convert HTML to Markdown using markdownify
    markdown_content = markdownify.markdownify(str(content_div), heading_style="ATX")
    return markdown_content

def save_to_file(content, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

def extract_glossary_terms(html_content):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all dt tags with class "glossary"
    glossary_terms = soup.find_all('dt', class_='glossary')
    
    # Extract the glossary terms and their definitions
    glossary_entries = []
    for term in glossary_terms:
        term_text = term.get_text(separator=' ', strip=True)
        definition = term.find_next_sibling('dd', class_='glossary')
        if definition:
            definition_text = definition.get_text(separator=' ', strip=True)
            glossary_entries.append({
                "term": term_text, 
                "explanation": definition_text
            })
    
    return glossary_entries

if __name__ == "__main__":
    # Wikipedia page URL
    url = sys.argv[1]  # Replace with your desired Wikipedia page
    
    # Fetch the Wikipedia page
    html_content = fetch_wikipedia_page(url)
    
    save_to_file(html_content, sys.argv[2])
    # # Convert HTML to Markdown
    # markdown_content = parse_html_to_markdown(html_content)
    
    # # Save the content to a markdown file
    # save_to_file(markdown_content, sys.argv[2])
    
    # print("Wikipedia page saved as markdown file: wikipedia_page.md")

    # extract glossary terms
    entries_dict = extract_glossary_terms(html_content)
    json.dump(entries_dict, open(sys.argv[3], 'w'), indent=2)


# Usage: 
# python3 crawl_wikipedia.py https://en.wikipedia.org/wiki/Glossary_of_artificial_intelligence data/wiki_ai.html data/wiki_ai.json
# python3 crawl_wikipedia.py https://en.wikipedia.org/wiki/Glossary_of_computer_science data/wiki_cs.html data/wiki_cs.json