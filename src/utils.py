import os
import json
import openai
import regex
import argparse
import numpy as np
from tqdm import tqdm


EVAL_PROMPT = """You are an ImageTaskEvaluatorGPT, an expert language model at judging whether or not a response adequately addresses an instruction in the context of an image. More specifically, you will be given the following:

1. An instruction: This is a question, an imperative request, or something similar about the image which requires a response.
2. A ground-truth response: This is the ground-truth response to the instruction in the context of the image annotated by the human annotator.
3. A predicted response: This response attempts to address the instruction in the context of the image without having access to the ground-truth response.

Your job is judge whether the predicted response is correct given the ground-truth response and the instruction.
 
Some things to remember:
- Even though you are just a language model, the instructions mostly require an objective answer i.e., the ground-truth response and instruction should be sufficient for you to judge the correctness of the predicted response. You do not need to have access to the complete image description.
- You are capable of judging response quality, accounting for important factors like correctness, relevance, fluency, specificity, etc. 
- You think step-by-step, and ultimately respond with your "Judgement: " as "Yes" or "No". Here, "Yes" implies that the predicted response is correct according to you, and "No" implies that the predicted response is not correct.
- Many times the predicted responses provide long explanations for their decision. In such cases, focus on whether the ground-truth response can be inferred from the predicted response or not. 

Instruction: {instruction}
Ground-truth Response: {groundtruth}
Predicted Response: {prediction}"""

GEN_PROMPT = """You are an ImageAnalysisAssistant, an expert multimodal large modle at telling a story from a series of pictures. More specifically, you will be give 6 pictures, each picture is one of a snapshot of the story, which happens from left to right.

Your job is to tell the story in a coherent and complete way, please notice the Goal, Attempt, and Outcome of the character.
"""

HINT = """Some things to remember:
- The whole story can be devided into three part. The 1st and 2nd picture consist the first story, 3rd and 4th picture consist the sencond part, 5th and 6th picture consist the third part.
- The Goal, Attempt and Outcomes are different in each part. Please use the method of empathy to explore the character's motivations.
- Organize your output into a coherent and complete story."""


def get_prompt(args):
    if args.difficulty == 0:
        prompt = GEN_PROMPT + '\n' + HINT
    else:
        prompt = GEN_PROMPT
    return prompt




