def get_json_list(path):
    import json
    f = open(path, 'r')
    info = []
    for line in f.readlines():
        info.append(json.loads(line))
    return info

def write_line_to_file(in_str, f):
    f.write(in_str)
    f.write("\n")
    f.flush()