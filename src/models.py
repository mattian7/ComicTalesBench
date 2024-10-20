import os
import json
import regex
import argparse
import numpy as np
from tqdm import tqdm
import openai
from openai import OpenAI

class GPT:
    def __init__(self, base_url, key, model_name):
        self.model = model_name
        openai.api_base = base_url
        openai.api_key = key

    def forward(self, prompt, image_url, tempertature=0.7):
        client = OpenAI()

        completion = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        }
                    },],
            }],
            tempertature=tempertature,
        )

        return completion.choices[0].message