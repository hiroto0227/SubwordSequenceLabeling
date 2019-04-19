import re

def config_file_to_dict(input_file):
    print("============= load config file ==================")
    config = {}
    fins = open(input_file,'r').readlines()
    for line in fins:
        if len(line) > 0 and line[0] == "#":
            continue
        if "=" in line:
            item, value = line.strip().split('#', 1)[0].split('=', 1)
            value_type = str2type(value)
            print(f"{item}={value}: {value_type}")
            if value_type == bool:
                config[item] = str2bool(value)
            elif value_type == float:
                config[item] = float(value)
            elif value_type == int:
                config[item] = int(value)
            elif value_type == list:
                config[item] = str2list(value)
            else:
                config[item] = value
    return config

def str2type(string):
    if re.fullmatch("(True|False)", string):
        return bool
    elif re.fullmatch("[0-9]+", string):
        return int
    elif re.fullmatch("[0-9]+\.[0-9]+", string):
        return float
    elif re.fullmatch("\[.+\]", string):
        return list
    else:
        return str

def str2bool(string):
    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False

def str2list(string):
    if re.match("\[.+\]", string):
        if "," in string:
            return [s.replace(" ", "") for s in string[1:-1].split(",")]
        else:
            return [string[1:-1]]
    else:
        print("parse Error. {} has no comma or brancket.".format(string))
        return []
