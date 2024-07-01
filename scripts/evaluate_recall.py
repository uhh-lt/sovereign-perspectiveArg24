import pandas as pd
import numpy as np

"""
Evaluate the recall for given ground truth dataframe, a given predictions dataframe and a filepath.
Calculated are recall values at 1, 4, 8, 16, 20, 30 and 100 and also relative recall values for each k
which is each recall value divided by the revall at 100. In the output directory there will be the averaged recalls
over all predictions and also the recalls for each individual line in a jsonl file.
"""
k_values = [1, 4, 8, 16, 20, 30, 100]
def calculate_recall(ground_truth_df : pd.DataFrame, predictions_df : pd.DataFrame, output_dir : str):
    recalls_lists = []
    for k in k_values:
        for index, row in predictions_df.iterrows():
            query_id = row["query_id"]
            query_candidates = ground_truth_df.loc[ground_truth_df["query_id"] == query_id]["relevant_candidates"].values
            prediction_candidates = np.array(row["relevant_candidates"][:k])
            true_positives_at_k = np.intersect1d(prediction_candidates, query_candidates[0])
            recall_at_k = len(true_positives_at_k) / len(query_candidates[0])
            recalls_lists.append({
                "query_id": query_id,
                "k": k,
                "recall_at_k": recall_at_k
            })
    all_recalls_result_df = pd.DataFrame(recalls_lists)
    all_recalls_results_pivot = all_recalls_result_df.pivot_table(index='query_id', columns='k', values='recall_at_k')
    average_recalls_result = pd.DataFrame(recalls_lists).groupby(["k"]).mean().reset_index()
    average_recalls_result = average_recalls_result.drop(columns=["query_id"])
    average_recalls_result.to_csv(f"{output_dir}/averaged_recalls.csv", index=False)
    all_recalls_results_pivot.to_csv(f"{output_dir}/alle_recalls.csv")
