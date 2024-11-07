import scipdf

res = scipdf.parse_pdf_to_dict('https://arxiv.org/pdf/1810.04805.pdf', grobid_url="http://localhost:8075")

print(res)