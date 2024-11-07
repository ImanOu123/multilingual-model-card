import yaml
import json

yaml_data = yaml.safe_load(open("original/papers.yml", 'r').read())

json_data = []
for entry in yaml_data:
    venue = entry.get('venue')
    year = entry.get('year')
    place = entry.get('place')
    date = entry.get('date')
    sub = entry.get('sub')

    for award in entry.get('awards', []):
        json_entry = {
            'venue': venue,
            'year': year,
            'place': place,
            'date': date,
            'sub': sub,
            'award': award.get('award'),
            'title': award.get('title'),
            'author': award.get('author'),
            'link': award.get('link'),
            'source': award.get('source')
        }
        json_data.append(json_entry)

with open('json/papers.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=2)