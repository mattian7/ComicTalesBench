import os
import json
import openai
import regex
import argparse
import numpy as np
import jsonlines
from tqdm import tqdm
from utils import *
from models import *


parser = argparse.ArgumentParser()

parser.add_argument('--data', type = str, default = "./dataset/test.jsonl")
parser.add_argument('--difficultly', type = int, default = 0, choices=[0,1], help="eval difficultlyi, which 0 means more hint for llm generation")
parser.add_argument('--task', type=str, default = "structure", choices=["structure","comprehension"])
parser.add_argument('--model', type = str, defualt = "gpt-4o")

args = parser.parse_args()

def main():
    base_url = "https://sjtullm.zeabur.app"
    api_key = "sk-R8CCeQvk5mcHaJiS569f3aC8B10743DfB75171D68dBe8f8e"
    if "gpt" in args.model:
        model = GPT(base_url, api_key, args.model)
        outfile = "./output/gpt4_judgments.json"
    else:
        model = None
        outfile = None

    structure_file = "./dataset/structure.jsonl"
    comprehension_file = "./dataset/comprehension.jsonl"
    log = []
    data = []
    if args.task == "structure":
        with jsonlines.open(structure_file, 'r') as f:
            for line in f:
                data.append(line)
    else:
        with jsonlines.open(comprehension_file, 'r') as f:
            for line in f:
                data.append(line)


    prompt = get_prompt(args)
    for d in tqdm(data):
        image_url = d["image_url"]
        story = model.forward(prompt, image_url)
        tmp = {"image_url": image_url, "story": story}
        log.append(tmp)
        continue

    with jsonlines.open(outfile, 'a') as file:
        for l in log:
            file.write (json.dumps(l,indent=4))
            

if __name__ == "__main__":
    main()