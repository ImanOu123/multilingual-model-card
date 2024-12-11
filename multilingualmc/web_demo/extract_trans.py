import json
import csv

inputFile = "2024.naacl-long.97_abs"

with open(inputFile + '_seamless.jsonl', 'r') as json_file:
    seamless_json_list = [json.loads(line) for line in json_file]

with open(inputFile + '.jsonl', 'r') as json_file:
    refined_json_list = [json.loads(line) for line in json_file]
    
arabicFileSeamless = open("arabic_" + inputFile + "_seamless", "w")
arabicFileRefined = open("arabic_" + inputFile + "_refined", "w")

arabic_seamless_lst = list(map(lambda d: d["text_Arabic"], seamless_json_list))
arabicFileSeamless.write("\n".join(arabic_seamless_lst))

arabic_refined_lst = list(map(lambda d: d["text_Arabic"], refined_json_list))
arabicFileRefined.write("\n".join(arabic_refined_lst))
