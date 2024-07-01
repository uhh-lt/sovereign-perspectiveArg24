import random
import pandas as pd
import sys
sys.path.append('.\\')
from .utils import read_gold_data

def rank_gpt_prompt(query : str, arguments : list, indices : int = None):
    prompt_str = f"The following are passages related to question {query}"
    if indices is None:
        indices = range(len(arguments))
    for i in range(len(indices)):
        prompt_str += f"\n[{indices[i]}] {arguments[i]}"
    prompt_str += "Rank these passages based on their relevance to the question."
    return prompt_str

def rank_gpt_prompt_demand_list(query : str, arguments : list, indices : int = None):
    prompt_str = rank_gpt_prompt(query, arguments, indices)
    prompt_str += " Return a python list containing all argument ids as integers ranked from best to worst."
    return prompt_str

def dict_scores_prompt(query : str, arguments : list):
    prompt = f"Given the question \"{query}\" and a list of arguments with ids. The task is to rank the arguments according to the question. The higher the score the more relevant it is to the question:"
    for i in range(len(arguments)):
        prompt += f"\n{i} - {arguments[i]}"
    prompt += "\nReturn a python dict with every single argument id and the scores only! No text!!! e.g. {{1: 0.9, 2: 0.3}}"
    return prompt

def load_train_data():
    data = read_gold_data('data-release')
    corpus = data["corpus"]
    perspective_queries_train = data["perspective"]["train"]
    train_predictions = pd.read_json("predictions/perspective_filtered_sbert_train_predictions.jsonl", lines=True)
    return corpus, perspective_queries_train, train_predictions

def predict_best_one_prompt(query_text : str, relevant_candidate_strings : list, used = list):
    prompt_str = f"Given are the question \"{query_text}\" and some arguments with ids:"
    for i in range(len(relevant_candidate_strings)):
        if i not in used:
            prompt_str += f"\n[{i}] {relevant_candidate_strings[i]}"
    prompt_str += f"\nFind the argument most directly adresses the question. Return the Integer argument id for the best argument only."
    return prompt_str

def binary_relevance_prompt(query_text : str, arguments : list):
    prompt_str = f"Given this question: \"{query_text}\".\nDetermine if the following argument perfectly answers the question: "
    for argument in arguments:
        prompt_str += f"{argument}\n"
    prompt_str += "If the argument perfectly answers the question return Integer 1, if it does not really answer it return Integer 0."
    return prompt_str

def compare_topics_prompt(text : str, topics : list):
    prompt_str = f"Given a question or an argument, classify it into one of the provided topics."
    prompt_str += f"\nQuestion/Argument: {text}"
    for i in range(len(topics)):
        prompt_str += f"\n[{i}: {topics[i]}]"
    prompt_str += "\nReturn the integer id of the most relevant topic only."
    return prompt_str

def implicit_demographics_prompt(demographic : str, arguments : list):
    prompt_str = f"The task is to rank arguments, if they fit the sociocultural property: {demographic}-\n"
    for i in range(len(arguments)):
        prompt_str += f"[{i}]: {arguments[i]}\n"
    prompt_str += f"Return a python dict with all argument ids between 0 and {len(arguments) - 1} and a score between 0 if the argumemt does not fit the demographic and 1 if it fits very good."
    return prompt_str

def implicit_demographic_list_prompt(demographic : str, arguments : list):
    prompt_str = f"The task is to rank arguments, if they fit the sociocultural property: {demographic}-\n"
    for i in range(len(arguments)):
        prompt_str += f"[{i}]: {arguments[i]}\n"
    prompt_str += "Rank these passages based on their relevance to the sociocultural property."
    return prompt_str