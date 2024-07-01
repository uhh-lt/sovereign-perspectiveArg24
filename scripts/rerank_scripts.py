import pandas as pd
import time
import ast
import sys
sys.path.append('.\\')
from . import prompts as pr
from llmusage.hugchat_api_usage import HuggingChat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import jsonlines
import numpy as np

def simple_rerank(dev_predictions : pd.DataFrame, corpus : pd.DataFrame, perspective_queries_dev : pd.DataFrame, mistral_huggingchat : HuggingChat, number_of_candidates : int, build_prompt, instruction_index : int = 0):
    answers = []
    no_rerank_ids = []
    for index, row in dev_predictions.iterrows():
        if index % 1 == 0:
            print(index)
        query_text = perspective_queries_dev.loc[perspective_queries_dev["query_id"] == row["query_id"]]["text"].values[0]
        relevant_candidate_strings = []
        relevant_candidates_ids = []
        for i in range(len(row["relevant_candidates"][:number_of_candidates])):
            relevant_candidate_strings.append(corpus.loc[corpus["argument_id"] == row["relevant_candidates"][i]]["argument"].values[0])
            relevant_candidates_ids.append(row["relevant_candidates"][i])
        unsuccessful = True
        unsuccessful_ct = 0
        while unsuccessful and unsuccessful_ct < 100:
            time.sleep(min(30, unsuccessful_ct*2))
            try:
                prompt = build_prompt(query_text, relevant_candidate_strings) if instruction_index is None else build_prompt(query_text, relevant_candidate_strings)
                model_answer = mistral_huggingchat.prompt(prompt)
                new_order_indices = ast.literal_eval(model_answer)
                mistral_huggingchat.delete_conversations()
            except Exception as e:
                print('exception in request or while parsing: ')
                print(e)
                if "many" in str(e):
                        time.sleep(10)
                unsuccessful_ct += 1
                try:
                    mistral_huggingchat.delete_conversations()
                except Exception as e2:
                    print('exception while deleting: ')
                    print(e2)
                    unsuccessful_ct += 1
                    time.sleep(1)
            else:
                try:
                    if isinstance(new_order_indices, dict):
                        new_order_indices = sorted(new_order_indices, key=lambda x: new_order_indices[x], reverse=True)
                    new_order = []
                    new_order_indices = list(new_order_indices)
                    for j in range(len(new_order_indices)):
                        if isinstance(new_order_indices[j], list):
                            new_order_indices[j] = new_order_indices[j][0]
                    all_indices = set(range(number_of_candidates))
                    missing_indices = list(set(all_indices) - set(new_order_indices))
                    new_order_indices.extend(missing_indices)
                    for j in new_order_indices:
                        #if isinstance(j, int):
                            new_order.append(relevant_candidates_ids[j])
                    if new_order:
                        dev_predictions.at[index, 'relevant_candidates'][:number_of_candidates] = new_order
                    unsuccessful = False
                    unsuccessful_ct = 0
                except Exception as e3:
                    print('exception while sorting or writing to json: ')
                    print(e3)
                    no_rerank_ids.append(row["query_id"])
                    unsuccessful_ct += 1
            if unsuccessful_ct >= 40:
                no_rerank_ids.append(row["query_id"])
                break
    return dev_predictions

def sliding_window_rerank(dev_predictions : pd.DataFrame, corpus : pd.DataFrame, perspective_queries_dev : pd.DataFrame, mistral_huggingchat : HuggingChat, number_of_candidates : int, window_size : int, step_size : int, build_prompt):
    answers = []
    no_rerank_ids = []
    for index, row in dev_predictions.iterrows():
        if index % 10 == 0:
            print(index)
        query_text = perspective_queries_dev.loc[perspective_queries_dev["query_id"] == row["query_id"]]["text"].values[0]
        relevant_candidate_strings = []
        relevant_candidates_ids = []
        for i in range(len(row["relevant_candidates"][:number_of_candidates])):
            relevant_candidate_strings.append(corpus.loc[corpus["argument_id"] == row["relevant_candidates"][i]]["argument"].values[0])
            relevant_candidates_ids.append(row["relevant_candidates"][i])
        end_index = len(relevant_candidate_strings) - 1
        reranked_indices = list(range(len(relevant_candidate_strings)))
        while end_index - window_size + step_size > 0:
            unsuccessful = True
            unsuccessful_ct = 0
            while unsuccessful and unsuccessful_ct < 100:
                if unsuccessful_ct > 3:
                    time.sleep(unsuccessful_ct * 2)
                try:
                    #prompt = pr.irina_suggestion_prompt(query_text, relevant_candidate_strings)
                    #new_order_indices = ast.literal_eval(mistral_huggingchat.prompt(prompt))
                    start_index = max(0, end_index - window_size)
                    window_indices = reranked_indices[start_index:end_index]
                    window = [relevant_candidate_strings[x] for x in window_indices]
                    prompt = build_prompt(query_text, window, indices=window_indices)
                    window_answers = (ast.literal_eval(mistral_huggingchat.prompt(prompt)))
                    for j in range(len(window_answers)):
                        if isinstance(window_answers[j], list):
                            window_answers[j] = window_answers[j][0]
                    if isinstance(window_answers, list):
                        missing_answers = list(set(window_indices) - set(window_answers))
                        window_answers.extend(missing_answers)
                    reranked_indices[start_index:end_index] = window_answers
                    mistral_huggingchat.delete_conversations()
                except Exception as e:
                    print(e)
                    if "many" in str(e):
                        time.sleep(10)
                    unsuccessful_ct += 1
                    try:
                        mistral_huggingchat.delete_conversations()
                    except Exception as e2:
                        unsuccessful_ct += 1
                        time.sleep(1)
                else:
                    #answers.append(new_order_indices)
                    #new_order = []
                    #for j in new_order_indices:
                    #    if isinstance(j, list):
                    #        j = j[0]
                    #        new_order.append(relevant_candidates_ids[j])
                    #dev_predictions.at[index, 'relevant_candidates'][:number_of_candidates] = new_order
                    end_index -= step_size
                    unsuccessful = False
                    unsuccessful_ct = 0
                if unsuccessful_ct >= 100:
                    no_rerank_ids.append(f"{row["query_id"]},{end_index - window_size}, {end_index}")
                    break
        # reorder ids based on indices
        new_order = []
        for i in range(len(reranked_indices)):
            new_order.append(relevant_candidates_ids[reranked_indices[i]])
        if index == 0:
            print(new_order)
        dev_predictions.at[index, 'relevant_candidates'][:number_of_candidates] = new_order
    return dev_predictions

# prompts the model the question over and over again until successful, or if unsuccessful too many times returns None
def prompt_model(mistral_huggingchat : HuggingChat, prompt : str):
    unsuccessful_ct = 0
    time.sleep(min(2*unsuccessful_ct, 61))
    while unsuccessful_ct < 100:
        try:
            answer = ast.literal_eval(mistral_huggingchat.prompt(prompt))
            mistral_huggingchat.delete_conversations()
        except Exception as e:
            unsuccessful_ct += 1
            if "many" in str(e):
                time.sleep(10)
            try:
                mistral_huggingchat.delete_conversations()
            except Exception as e2:
                unsuccessful_ct += 1
                if "many" in str(e2):
                    time.sleep(10)
                    if unsuccessful_ct > 3:
                        time.sleep(2* unsuccessful_ct)
        else:
            return answer
    return None

def combined_scores_rerank(dev_predictions : pd.DataFrame, corpus : pd.DataFrame, perspective_queries_dev : pd.DataFrame, mistral_huggingchat : HuggingChat, number_of_candidates : int, build_prompt, json_path : str = None, implicit : bool = False):
    sbert_encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    not_reranked = []
    already_reranked = 0
    for index, row in dev_predictions.iterrows():
        if index >= already_reranked or json_path is None:
            print(index)
            query_text = perspective_queries_dev.loc[perspective_queries_dev["query_id"] == row["query_id"]]["text"].values[0]
            query_demographic = str(perspective_queries_dev.loc[perspective_queries_dev["query_id"] == row["query_id"]]["demographic_property"].values[0])
            relevant_candidate_strings = []
            relevant_candidates_ids = []
            for i in range(len(row["relevant_candidates"][:number_of_candidates])):
                relevant_candidate_strings.append(corpus.loc[corpus["argument_id"] == row["relevant_candidates"][i]]["argument"].values[0])
                relevant_candidates_ids.append(row["relevant_candidates"][i])
            if implicit:
                prompt = build_prompt(query_demographic, relevant_candidate_strings)
            else:
                prompt = build_prompt(query_text, relevant_candidate_strings)
            print(prompt)
            model_answer = prompt_model(mistral_huggingchat, prompt)
            if isinstance(model_answer, dict):
                # recalculate cosine similarity
                query_embedding = sbert_encoder.encode(query_text)
                candidate_embeddings = sbert_encoder.encode(relevant_candidate_strings)
                similarites = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings)
                try:
                    for i in range(len(similarites[0])):
                        if i in model_answer:
                            model_answer[i] = similarites[0][i] + model_answer[i]
                        else:
                            print(model_answer)
                            model_answer[i] = similarites[0][i]
                    new_order_indices = sorted(model_answer, key=lambda x: model_answer[x], reverse=True)
                    new_order = []
                    for i in new_order_indices:
                        new_order.append(relevant_candidates_ids[i])
                    if new_order:
                        dev_predictions.at[index, 'relevant_candidates'][:number_of_candidates] = new_order
                        if not json_path is None:
                            new_order_dict = dev_predictions.iloc[index].to_dict()
                            with jsonlines.open(json_path, mode='a') as writer:
                                writer.write(new_order_dict)
                except Exception as e:
                    print(e)
                    not_reranked.append(row)
    return dev_predictions, not_reranked
            
def predict_best_one_rerank(dev_predictions : pd.DataFrame, corpus: pd.DataFrame, perspective_queries_dev : pd.DataFrame, huggingchat : HuggingChat, number_of_candidates : int, build_prompt, json_path : str = None):
    not_reranked = []
    # change this to be either 0 or amount of lines in json_path file
    already_reranked = 0
    for index, row in dev_predictions.iterrows():
        if index >= already_reranked or json_path is None:
            top_20 = []
            print(index)
            query_text, relevant_candidate_strings, relevant_candidate_ids = retrieve_query_and_args(row, perspective_queries_dev, corpus, number_of_candidates)
            for i in range(20):
                prompt = build_prompt(query_text, relevant_candidate_strings, top_20)
                time.sleep(1)
                model_answer = prompt_model(huggingchat, prompt)
                if isinstance(model_answer, list):
                    model_answer = model_answer[0]
                if isinstance(model_answer, int):
                    top_20.append(model_answer)
            new_order_indices = top_20 + list(set(range(number_of_candidates)) - set(top_20))
            new_order = [relevant_candidate_ids[i] for i in new_order_indices]
            dev_predictions.at[index, 'relevant_candidates'][:number_of_candidates] = new_order
            if json_path is not None:
                new_order_dict = dev_predictions.iloc[index].to_dict()
                with jsonlines.open(json_path, mode='a') as writer:
                    writer.write(new_order_dict)
    return dev_predictions, not_reranked


def retrieve_query_and_args(row : pd.DataFrame, perspective_queries_dev : pd.DataFrame, corpus : pd.DataFrame, number_of_candidates : int):
    query_text = perspective_queries_dev.loc[perspective_queries_dev["query_id"] == row["query_id"]]["text"].values[0]
    relevant_candidate_strings = []
    relevant_candidates_ids = []
    for i in range(len(row["relevant_candidates"][:number_of_candidates])):
        relevant_candidate_strings.append(corpus.loc[corpus["argument_id"] == row["relevant_candidates"][i]]["argument"].values[0])
        relevant_candidates_ids.append(row["relevant_candidates"][i])
    return query_text, relevant_candidate_strings, relevant_candidates_ids

def binary_relevance_rerank(dev_predictions : pd.DataFrame, corpus : pd.DataFrame, perspective_queries_dev : pd.DataFrame, huggingchat : HuggingChat, number_of_candidates : int, build_prompt, json_path : str = None):
    not_reranked = []
    # change this to be either 0 or amount of lines in json_path file
    already_reranked = 0
    for index, row in dev_predictions.iterrows():
        if index >= already_reranked or json_path is None:
            print(index)
            binary_relevance_scores = []
            query_text, relevant_candidate_strings, relevant_candidate_ids = retrieve_query_and_args(row, perspective_queries_dev, corpus, number_of_candidates)
            for argument in relevant_candidate_strings:
                prompt = build_prompt(query_text, [argument])
                model_answer = prompt_model(huggingchat, prompt)
                if isinstance(model_answer, list):
                    model_answer = model_answer[0]
                if model_answer == 1:
                    binary_relevance_scores.append(1)
                else:
                    binary_relevance_scores.append(0)
            good_indices = [index for index, value in enumerate(binary_relevance_scores) if value == 1]
            bad_indices = list(set(range(number_of_candidates)) - set(good_indices))
            # new_order = two_group_rerank(...)
            new_order_indices = good_indices + bad_indices
            new_order = [relevant_candidate_ids[i] for i in new_order_indices]
            dev_predictions.at[index, 'relevant_candidates'][:number_of_candidates] = new_order
            if json_path is not None:
                new_order_dict = dev_predictions.iloc[index].to_dict()
                with jsonlines.open(json_path, mode='a') as writer:
                    writer.write(new_order_dict)
    return dev_predictions, not_reranked

def compare_topics_rerank(dev_predictions : pd.DataFrame, corpus : pd.DataFrame, perspective_queries_dev : pd.DataFrame, huggingchat : HuggingChat, number_of_candidates : int, build_prompt, json_path : str = None):
    not_reranked = []
    # change this to be either 0 or amount of lines in json_path file but also only if not_reranked is empty
    already_reranked = 0
    topics_list = get_topics_list(corpus)
    print(topics_list)
    for index, row in dev_predictions.iterrows():
        if index >= already_reranked or json_path is None:
            print(index)
            arg_topic_relevance_scores = []
            query_text, relevant_candidate_strings, relevant_candidate_ids = retrieve_query_and_args(row, perspective_queries_dev, corpus, number_of_candidates)
            query_prompt = build_prompt(query_text, topics_list)
            query_topic_id = prompt_model(huggingchat, query_prompt)
            if isinstance(query_topic_id, list):
                query_topic_id = query_topic_id[0]
            for argument in relevant_candidate_strings:
                arg_prompt = build_prompt(argument, topics_list)
                arg_model_answer = prompt_model(huggingchat, arg_prompt)
                if isinstance(arg_model_answer, list):
                    arg_model_answer = arg_model_answer[0]
                if arg_model_answer == query_topic_id:
                    arg_topic_relevance_scores.append(1)
                else:
                    arg_topic_relevance_scores.append(0)
            good_indices = [index for index, value in enumerate(arg_topic_relevance_scores) if value == 1]
            bad_indices = list(set(range(number_of_candidates)) - set(good_indices))
            # new_order = two_group_rerank(...)
            new_order_indices = good_indices + bad_indices
            new_order = [relevant_candidate_ids[i] for i in new_order_indices]
            dev_predictions.at[index, 'relevant_candidates'][:number_of_candidates] = new_order
            if json_path is not None:
                new_order_dict = dev_predictions.iloc[index].to_dict()
                with jsonlines.open(json_path, mode='a') as writer:
                    writer.write(new_order_dict)
    return dev_predictions, not_reranked

def predict_query_topic_rerank(dev_predictions : pd.DataFrame, corpus : pd.DataFrame, perspective_queries_dev : pd.DataFrame, huggingchat : HuggingChat, number_of_candidates : int, build_prompt, json_path : str = None):
    not_reranked = []
    already_reranked = 0
    topics_list = get_topics_list(corpus)
    for index, row in dev_predictions.iterrows():
        if index >= already_reranked or json_path is None:
            if index % 10 == 0:
                print(index)
            arg_topic_relevance_scores = []
            query_text, relevant_candidate_strings, relevant_candidate_ids = retrieve_query_and_args(row, perspective_queries_dev, corpus, number_of_candidates)
            query_prompt = build_prompt(query_text, topics_list)
            query_topic_id = prompt_model(huggingchat, query_prompt)
            if isinstance(query_topic_id, list):
                query_topic_id = query_topic_id[0]
            for argument_id in relevant_candidate_ids:
                if topics_list[query_topic_id] == corpus.loc[corpus['argument_id'] == argument_id]['topic'].values[0]:
                    arg_topic_relevance_scores.append(1)
                else:
                    arg_topic_relevance_scores.append(0)
            good_indices = [index for index, value in enumerate(arg_topic_relevance_scores) if value == 1]
            bad_indices = list(set(range(number_of_candidates)) - set(good_indices))
            # new_order = two_group_rerank(...)
            new_order_indices = good_indices + bad_indices
            new_order = [relevant_candidate_ids[i] for i in new_order_indices]
            dev_predictions.at[index, 'relevant_candidates'][:number_of_candidates] = new_order
            if json_path is not None:
                new_order_dict = dev_predictions.iloc[index].to_dict()
                with jsonlines.open(json_path, mode='a') as writer:
                    writer.write(new_order_dict)
    return dev_predictions, not_reranked


def get_topics_list(corpus : pd.DataFrame):
    topics_list = corpus['topic'].values.tolist()
    return list(set(topics_list))

def rerank_scores(predictions : pd.DataFrame, corpus : pd.DataFrame, queries : pd.DataFrame, huggingchat : HuggingChat, number_of_candidates : int, build_prompt, json_path : str = None):
    not_reranked = []
    rerank_scores = []
    for index, row in predictions.iterrows():
        if index % 5 == 0:
            print(index)
        empty_rerank_scores = np.zeros(len(corpus))
        query_text, relevant_candidate_strings, relevant_candidate_ids = retrieve_query_and_args(row, queries, corpus, number_of_candidates)
        prompt = build_prompt(query_text, relevant_candidate_strings)
        model_answer = prompt_model(huggingchat, prompt)
        if isinstance(model_answer, dict):
            for key, value in model_answer.items():
                if isinstance(key, list):
                    key = key[0]
                if key in range(len(relevant_candidate_ids)):
                    argument_id = relevant_candidate_ids[key]
                    arg_index = np.argmax(corpus["argument_id"].values == argument_id)
                    empty_rerank_scores[arg_index] += value
        rerank_scores.append(empty_rerank_scores)
        if json_path is not None:
            rerank_row_dict = {
                'query_id': int(queries.iloc[index]["query_id"]),
                'rerank_scores': empty_rerank_scores.tolist()
            }
            with jsonlines.open(json_path, mode='a') as writer:
                writer.write(rerank_row_dict)
    return rerank_scores

def implicit_demographic_scores(predictions : pd.DataFrame, corpus : pd.DataFrame, queries : pd.DataFrame, huggingchat : HuggingChat, number_of_candidates : int, build_prompt, json_path : str = None, skip_until : int = 0):
    not_reranked = []
    rerank_scores = []
    for index, row in predictions.iterrows():
        if index % 5 == 0:
            print(index)
        if index >= skip_until:    
            empty_rerank_scores = np.zeros(len(corpus))
            query_text, relevant_candidate_strings, relevant_candidate_ids = retrieve_query_and_args(row, queries, corpus, number_of_candidates)
            query_demographic = str(queries.loc[queries["query_id"] == row["query_id"]]["demographic_property"].values[0])
            prompt = build_prompt(query_demographic, relevant_candidate_strings)
            model_answer = prompt_model(huggingchat, prompt)
            try:
                if isinstance(model_answer, dict):
                    for key, value in model_answer.items():
                        if isinstance(key, list):
                            key = key[0]
                        if key in range(len(relevant_candidate_ids)):
                            argument_id = relevant_candidate_ids[key]
                            arg_index = np.argmax(corpus["argument_id"].values == argument_id)
                            empty_rerank_scores[arg_index] += value
            except Exception as e:
                print(e)
            rerank_scores.append(empty_rerank_scores)
            if json_path is not None:
                rerank_row_dict = {
                    'query_id': int(queries.iloc[index]["query_id"]),
                    'demographic_scores': empty_rerank_scores.tolist()
                }
                with jsonlines.open(json_path, mode='a') as writer:
                    writer.write(rerank_row_dict)
    return rerank_scores

def implicit_rerank_list(dev_predictions : pd.DataFrame, corpus : pd.DataFrame, queries : pd.DataFrame, mistral_huggingchat : HuggingChat, number_of_candidates : int, build_prompt):
    answers = []
    no_rerank_ids = []
    for index, row in dev_predictions.iterrows():
        if index % 50 == 0:
            print(index)
        query_demographic = str(queries.loc[queries["query_id"] == row["query_id"]]["demographic_property"].values[0])
        relevant_candidate_strings = []
        relevant_candidates_ids = []
        for i in range(len(row["relevant_candidates"][:number_of_candidates])):
            relevant_candidate_strings.append(corpus.loc[corpus["argument_id"] == row["relevant_candidates"][i]]["argument"].values[0])
            relevant_candidates_ids.append(row["relevant_candidates"][i])
        prompt = build_prompt(query_demographic, relevant_candidate_strings)
        model_answer = prompt_model(mistral_huggingchat, prompt)
        try:
            print(type(model_answer))
            if type(model_answer) == list:
                new_order_indices = model_answer
            else:
                new_order_indices = ast.literal_eval(model_answer)
        except Exception as e:
            print(e)
        mistral_huggingchat.delete_conversations()
        try:
            if isinstance(new_order_indices, dict):
                new_order_indices = sorted(new_order_indices, key=lambda x: new_order_indices[x], reverse=True)
            new_order = []
            new_order_indices = list(new_order_indices)
            for j in range(len(new_order_indices)):
                if isinstance(new_order_indices[j], list):
                    new_order_indices[j] = new_order_indices[j][0]
            all_indices = set(range(number_of_candidates))
            missing_indices = list(set(all_indices) - set(new_order_indices))
            new_order_indices.extend(missing_indices)
            for j in new_order_indices:
                new_order.append(relevant_candidates_ids[j])
            if new_order:
                dev_predictions.at[index, 'relevant_candidates'][:number_of_candidates] = new_order
        except Exception as e3:
            print('exception while sorting or writing to json: ')
            print(e3)
            no_rerank_ids.append(row["query_id"])
    return dev_predictions