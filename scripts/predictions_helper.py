import pandas as pd
import numpy as np
from scripts.utils import read_gold_data

def calculate_predictions(corpus : pd.DataFrame, queries : pd.DataFrame, similarity_scores_df : pd.DataFrame = None, topic_scores_df : pd.DataFrame = None, demographic_scores_df : pd.DataFrame = None, rerank_scores_df : pd.DataFrame = None, prediction_path : str = None, sbert_weight : float = 1.0, topic_weight : float = 1.0, demographic_weight : float = 1.0, llm_weight : float = 1.0):
    if similarity_scores_df is None:
        for scores in (topic_scores_df, demographic_scores_df, rerank_scores_df):
            if scores is not None:
                num_rows = (len(scores))
                array_length = len(np.array(scores.iloc[0, 1]))
                similarity_scores = np.zeros((num_rows, array_length))
                break
    else:
        similarity_scores_df['similarity_scores'] = similarity_scores_df['similarity_scores'].apply(np.array)
        similarity_scores = np.vstack(similarity_scores_df["similarity_scores"])
    if topic_scores_df is None:
        topic_scores = np.zeros(similarity_scores.shape)
    else:
        topic_scores_df["topic_scores"] = topic_scores_df["topic_scores"].apply(np.array)
        topic_scores = np.vstack(topic_scores_df["topic_scores"])
    if topic_scores.shape != similarity_scores.shape:
        raise ValueError("The topic scores do not have the same shape as the similarity scores.") 
    if demographic_scores_df is None:
        demographic_scores = np.zeros(similarity_scores.shape)
    else:
        demographic_scores_df["demographic_scores"] = demographic_scores_df["demographic_scores"].apply(np.array)
        demographic_scores = np.vstack(demographic_scores_df["demographic_scores"])
    if demographic_scores.shape != similarity_scores.shape:
        raise ValueError("The demographic scores do not have the same shape as the similarity scores.")
    if rerank_scores_df is None:
        rerank_scores = np.zeros(similarity_scores.shape)
    else:
        rerank_scores_df["rerank_scores"] = rerank_scores_df["rerank_scores"].apply(np.array)
        rerank_scores = np.vstack(rerank_scores_df["rerank_scores"])
    if rerank_scores.shape != similarity_scores.shape:
        raise ValueError("The rerank score does not have the same shape as the similarity score.")
    scores = sbert_weight * similarity_scores + topic_weight * topic_scores + demographic_weight * demographic_scores + llm_weight * rerank_scores
    predictions = [
        {
            "query_id": queries.iloc[i]["query_id"],
            "relevant_candidates": [
                corpus.iloc[candidate_index]["argument_id"]
                for candidate_index in candidates.argsort()[::-1][:1000]
            ]
        }
        for i, candidates in enumerate(scores)
    ]
    predictions_df = pd.DataFrame(predictions)
    if prediction_path is not None:
        predictions_df.to_json(prediction_path, orient="records", lines=True)
    return predictions_df

def calculate_predictions_dif(corpus : pd.DataFrame, queries : pd.DataFrame, similarity_scores_df : pd.DataFrame, topic_scores_df : pd.DataFrame = None, demographic_scores_df : pd.DataFrame = None, rerank_scores_df : pd.DataFrame = None, prediction_path : str = None):
    similarity_scores_df['similarity_scores'] = similarity_scores_df['similarity_scores'].apply(np.array)
    similarity_scores = np.vstack(similarity_scores_df["similarity_scores"])
    if topic_scores_df is None:
        topic_scores = np.zeros(similarity_scores.shape)
    else:
        topic_scores_df["topic_scores"] = topic_scores_df["topic_scores"].apply(np.array)
        topic_scores = np.vstack(topic_scores_df["topic_scores"])
    if topic_scores.shape != similarity_scores.shape:
        raise ValueError("The topic scores do not have the same shape as the similarity scores.") 
    if demographic_scores_df is None:
        demographic_scores = np.zeros(similarity_scores.shape)
    else:
        demographic_scores_df["demographic_scores"] = demographic_scores_df["demographic_scores"].apply(np.array)
        demographic_scores = np.vstack(demographic_scores_df["demographic_scores"])
    if demographic_scores.shape != similarity_scores.shape:
        raise ValueError("The demographic scores do not have the same shape as the similarity scores.")
    if rerank_scores_df is None:
        rerank_scores = np.zeros(similarity_scores.shape)
    else:
        rerank_scores_df["rerank_scores"] = rerank_scores_df["rerank_scores"].apply(np.array)
        rerank_scores = np.vstack(rerank_scores_df["rerank_scores"])
    if rerank_scores.shape != similarity_scores.shape:
        raise ValueError("The rerank score does not have the same shape as the similarity score.")
    for i, row in enumerate(similarity_scores):
        indices = np.argpartition(row, -8)[-8:]
        sorted_indices = indices[np.argsort(row[indices])]
        eighth_highest_value = row[sorted_indices[-1]]
        if eighth_highest_value > 0.8:
            rerank_scores[i] = np.zeros_like(rerank_scores[i])
    scores = similarity_scores + topic_scores + demographic_scores + rerank_scores
    predictions = [
        {
            "query_id": queries.iloc[i]["query_id"],
            "relevant_candidates": [
                corpus.iloc[candidate_index]["argument_id"]
                for candidate_index in candidates.argsort()[::-1][:1000]
            ]
        }
        for i, candidates in enumerate(scores)
    ]
    predictions_df = pd.DataFrame(predictions)
    if prediction_path is not None:
        predictions_df.to_json(prediction_path, orient="records", lines=True)
    return predictions_df