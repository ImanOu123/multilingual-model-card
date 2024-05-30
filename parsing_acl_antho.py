import requests
from xml.etree import ElementTree
import re
import json

def process_antho_xml_file(url, filename):
    f = open(filename, "a")
    response = requests.get(url)

    if response.status_code == requests.codes.ok:
        tree = ElementTree.fromstring(response.content)

        for subtree in tree.findall(".//paper"):

            titleDict = {}
            abstractDict = {}

            # iterate through the subtree of the paper tag to extract different language titles and abstracts
            for child in subtree:
                
                # ignore text from 'fixed-case' subelems but still get text from other subelems such as 'a' for hyperlinks
                for elem in child.findall("fixed-case"):
                    child.remove(elem)
                
                text = "".join(child.itertext())

                # look for whether title or abstract
                title = re.search("title_", child.tag)
                abstract = re.search("abstract_", child.tag)
                
                # save to respective dictionary
                if child.tag == "title":
                    titleDict["en"] = text


                elif child.tag == "abstract":
                    abstractDict["en"] = text

                elif title:
                    titleDict[child.tag[title.span()[1]:]] = text

                elif abstract:
                    abstractDict[child.tag[abstract.span()[1]:]] = text
                
            # add dictionaries to file
            f.write(str(titleDict) + "\n" + str(abstractDict) + "\n")
    else:
        print('Content was not found.')

    f.close()


# extract file names of xml files
dirURL = "https://api.github.com/repos/BKHMSI/acl-anthology/git/trees/master?recursive=1"
resp = requests.get(dirURL)
res = resp.json()

for file in res["tree"]:
    path = file["path"]
    xmlFiles = re.search("data\/xml\/", path)
    if xmlFiles:
        # clear file
        f = open("data/" + path[xmlFiles.span()[1]:-4] + ".txt", "w")
        f.close()

        url = 'https://raw.githubusercontent.com/BKHMSI/acl-anthology/master/data/xml/' + path[xmlFiles.span()[1]:]

        # add dictionaries as lines in txt file
        process_antho_xml_file(url, "data/" + path[xmlFiles.span()[1]:-4] + ".txt")
        print("file processed: ", path[xmlFiles.span()[1]:-4])