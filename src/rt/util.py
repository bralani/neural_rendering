import json
def readJson(path="a.json"):
    with open(path) as f:
        datas = json.load(f)
    print(len(datas))
    for data in datas:
        print(data)
        
if __name__ == "__main__":
    readJson()