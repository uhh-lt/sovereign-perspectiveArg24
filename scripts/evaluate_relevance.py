import numpy as np


def dcg(relevance_scores):
    """
    Calculates the Discounted Cumulative Gain (DCG) for a list of relevance scores.

    DCG is a measure of ranking quality that considers the position of relevant items in the ranking list.
    The relevance scores are discounted logarithmically, proportionate to the position of each item,
    emphasizing the importance of higher-ranked items. The function computes the DCG value by summing
    the discounted relevance scores, where higher scores indicate better ranking quality.

    Parameters:
    - relevance_scores (list of int): A binary list (1s and 0s) representing the relevance
      of each ranked item. A score of 1 indicates the item is relevant to the query, and 0 indicates
      it is not relevant. The order of scores corresponds to the ranking order of items.

    Returns:
    - float: The computed DCG value for the given list of relevance scores. A higher DCG value indicates
      that relevant items appear earlier in the ranking.
    """
    # The rank of an item is its index in the list + 1 (since indexing starts at 0)
    ranks = np.arange(1, len(relevance_scores) + 1)
    # Discounted gains: relevance / log2(rank+1), smaller ranks are more important
    discounted_gains = relevance_scores / np.log2(ranks + 1)
    return np.sum(discounted_gains)


def idcg(relevance_scores, k_range):
    """
    Calculates the Ideal Discounted Cumulative Gain (IDCG) for various k in a ranked list.
    IDCG represents the best possible DCG score that could be achieved if the items
    were ranked in the optimal order, based on their relevance scores.

    The function calculates IDCG by first sorting the relevance scores in descending order to simulate
    an ideal ranking scenario where the most relevant items appear first. It then computes the DCG
    for the top-k items for each k value specified in the input, representing the maximum achievable
    DCG score for those top-k rankings.

    Parameters:
    - relevance_scores (list of int): A binary list (1s and 0s) representing the relevance
      of each ranked item. A score of 1 indicates the item is relevant to the query, and 0 indicates
      it is not relevant. The order of scores corresponds to the ranking order of items.

    - k_range (list or numpy.ndarray): A list or array of integers specifying the cut-off points (k values)
      at which to calculate the IDCG.

    Returns:
    - dict: A dictionary where each key corresponds to a k value from `k_range`, and each value is the
      calculated IDCG score for that k.
    """
    # IDCG is calculated by sorting the relevance scores in descending order
    sorted_scores = sorted(relevance_scores, reverse=True)
    k2idcg = {}
    for k in k_range:
        topk = sorted_scores[:k]  # Consider only the top-k scores
        k2idcg[k] = dcg(topk)
    return k2idcg


def ndcg(relevance_scores_predictions, relevance_scores_global, k_range):
    """
    Calculates the Normalized Discounted Cumulative Gain (nDCG) for a ranked list
    of items based on their relevance scores. nDCG measures the effectiveness of the model, taking into account
    the position of relevant items in the ranking. It normalizes the Discounted Cumulative
    Gain (DCG) of the ranked list against the Ideal DCG (IDCG), which represents the
    maximum possible DCG given the relevance scores.

    Parameters:
    - relevance_scores (list of int): A binary list (1s and 0s) representing the relevance
      of each ranked item. A score of 1 indicates the item is relevant to the query, and 0 indicates
      it is not relevant. The order of scores corresponds to the ranking order of items.

    - relevance_scores_global (list or numpy.ndarray): A binary list (1s and 0s) representing the
      relevance scores representing the "ideal" or ground truth relevance of items.
      This list is used to compute the IDCG for normalization.

    - k_range (list of int): A list or array of integers specifying the cut-off
      points (k values) at which to calculate the nDCG.

    Returns:
    - dict: A dictionary where each key corresponds to a k value from `k_range`, and each
      value is the calculated nDCG score for that k.

    Example:
    \\>>> relevance_scores_predictions = [3, 2, 3, 0, 1, 2]
    \\>>> relevance_scores_global = [3, 3, 2, 2, 1, 0]
    \\>>> k_range = [3, 5]
    \\>>> nDCG_scores = ndcg(relevance_scores_predictions, relevance_scores_global, k_range)
    """
    k2idcg = idcg(relevance_scores_global, k_range)
    k2ndcg = {}
    for k in k_range:
        topk = relevance_scores_predictions[:k]  # Consider only the top-k scores
        actual_dcg = dcg(topk)
        optimal_dcg = k2idcg[k]
        k2ndcg[k] = actual_dcg / optimal_dcg if optimal_dcg > 0 else 0
    return k2ndcg


def precision_at_k(relevance_scores, k):
    """
    Calculate precision at rank k.

    Parameters:
    - relevance_scores: A list of binary relevance scores (1 for relevant, 0 for not relevant),
                        ordered by the ranking of items.
    - k: The rank at which to calculate precision.

    Returns:
    - Precision at k: The proportion of relevant items in the top-k ranked items.
    """
    # Ensure k does not exceed the length of the relevance_scores list
    k = min(k, len(relevance_scores))

    # Slice the list to consider only the top-k items
    top_k_scores = relevance_scores[:k]

    # Calculate precision as the fraction of relevant items in the top-k
    precision = sum(top_k_scores) / k if k > 0 else 0
    return precision
