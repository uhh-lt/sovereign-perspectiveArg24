import math

from scipy.special import kl_div
from collections import Counter


def get_perspectives(argument_ids, corpus, socio_variable):
    """
    Retrieves the perspectives associated with a specified socio-cultural variable for a list of arguments.

    This function iterates through a list of argument identifiers and extracts the perspective of each argument
    with respect to a given socio-cultural variable (e.g., political affiliation, demographic information) from
    a provided dataframe. The dataframe is expected to contain various socio-cultural variables as columns,
    with each row representing a unique argument. The function returns a list of perspectives corresponding
    to each argument's socio-cultural attribute.

    Parameters:
    - argument_ids (list of int/str): A list containing the identifiers of the arguments for which
      perspectives are to be retrieved. Each identifier should correspond to a row index in the `corpus` dataframe.

    - corpus (pandas.DataFrame): The dataframe containing the arguments and their associated socio-cultural variables.
      Each row in this dataframe represents an argument, and each column represents a different socio-cultural variable.

    - socio_variable (str): The name of the socio-cultural variable column in the `corpus` dataframe for which
      perspectives are being requested. This variable specifies the particular aspect of socio-cultural context
      (e.g., 'political_view', 'age_group') for the arguments.

    Returns:
    - list: A list containing the perspectives for each argument identified by `argument_ids` with respect to the
      specified `socio_variable`. The order of perspectives in the returned list corresponds to the order of
      `argument_ids`.

    Example:
    \\>>> argument_ids = [201912, 20192312, 2019123]
    \\>>> socio_variable = 'political_spectrum'
    \\>>> perspectives = get_perspectives(argument_ids, corpus, socio_variable)
    \\>>> print(perspectives)
    ['liberal', 'conservative', 'liberal']
    """
    perspectives = [corpus.at[arg_id, socio_variable] for arg_id in argument_ids]
    return perspectives


def calculate_dcg(relevance_scores, alpha, perspectives):
    """
    Compute the Discounted Cumulative Gain (DCG) adjusted for perspective redundancy.
    This function evaluates the ranking quality by incorporating the novelty and diversity
    of perspectives among the ranked items. Each item's contribution to the DCG is penalized
    based on the redundancy of its perspective, encouraging diversity in the ranking.

    Parameters:
    - relevance_scores (list of int): A binary list (1s and 0s) representing the relevance
      of each ranked item. A score of 1 indicates the item is relevant to the query, and 0 indicates
      it is not relevant. The order of scores corresponds to the ranking order of items.

    - perspectives (list of str): A list containing the 'perspective' associated with each ranked
      item. Perspectives represent categorical attributes that describe the origin or viewpoint
      of each item, such as political affiliation or demographic group. The order of perspectives
      corresponds to the ranking order of items.

    - alpha (float): A parameter between 0 and 1 that controls the penalty for redundancy.
      A higher alpha increases the penalty for items with perspectives that have already appeared
      in the ranking, thereby encouraging more diversity. Specifically, alpha modifies the
      contribution of each item to the DCG score by reducing it for items with previously seen
      perspectives.

    Returns:
    - float: The adjusted DCG score, which quantifies the ranking quality by accounting for both
      relevance and diversity. Higher scores indicate better ranking quality, with adjustments
      made to penalize redundancy and promote diversity in perspectives.

    Example:
    \\>>> relevance_scores = [1, 0, 1, 1]
    \\>>> perspectives = ['politics', 'health', 'politics', 'economy']
    \\>>> alpha = 0.5
    \\>>> score = compute_adjusted_DCG(relevance_scores, perspectives, alpha)
    \\>>> print(score)
    """
    dcg = 0.0
    seen_perspectives = set()
    for i, (relevance, perspective) in enumerate(zip(relevance_scores, perspectives)):
        # Apply penalty for redundancy
        penalty = (1 - alpha) if perspective in seen_perspectives else 1
        seen_perspectives.add(perspective)
        # Calculate DCG
        dcg += (relevance * penalty) / math.log2(
            (i + 1) + 1)  # we add 1 because the index starts at 0 but the rank at 1
    return dcg


def calculate_idcg(ground_truth_relevance, alpha, ground_truth_perspectives, k_range):
    """
    Computes the Ideal Discounted Cumulative Gain (IDCG) for a given ranking order,
    taking into account the diversity of perspectives among the top-ranked items.
    The IDCG represents the maximum possible DCG score that could be achieved if
    the items were arranged in the most optimal way, ensuring that all uniquer relevant
    perspectives are included in the top-k items.

    Parameters:
    - ground_truth_relevance (list of int): A list of binary relevance scores (1 or 0)
      for each item in the ground truth data. A score of 1 indicates that the item is
      relevant, while a score of 0 indicates non-relevance. The list has to be ordered
      by relevance, that means the first items will all be relevant since it is the ground truth.
      The list is then padded with random non-relevant items to the length of the ranking.
    - alpha (float): A parameter between 0 and 1 that adjusts the penalty for redundancy
      among the perspectives. A higher alpha value imposes a greater penalty on redundancy,
      promoting diversity in the ranking.
    - ground_truth_perspectives (list of str): A list of perspectives associated with each
      item in the ground truth data. Each perspective represents a socio-cultural or
      demographic attribute of the item, such as its political affiliation or stance.
    - k_range (list of int): A list of integers specifying the ranks at which the IDCG
      should be calculated. For example, [5, 10, 15] would compute the IDCG at the top
      5, top 10, and top 15 items.

    Returns:
    - dict: A dictionary where each key corresponds to a rank from `k_range` and each value
      is the IDCG score at that rank. This score quantifies the maximum achievable ranking
      quality, considering both relevance and the diversity of perspectives up to that rank.
    """

    # check if there are non-relevant arguments
    if 0 in ground_truth_relevance:
        # find the index of the first non-relevant argument
        index_first_non_relevant = ground_truth_relevance.index(0)
    else:
        # all arguments are relevant
        index_first_non_relevant = len(ground_truth_relevance) - 1
        
    unique_perspectives = set(ground_truth_perspectives[:index_first_non_relevant])
    # check to cover all unique perspectives first
    covered_perspectives = set()

    indices_to_keep_first = []
    for idx, (relevance, perspective) in enumerate(zip(ground_truth_relevance, ground_truth_perspectives)):
        if relevance > 0 and perspective in unique_perspectives and perspective not in covered_perspectives:
            # add index to the list of indices to keep first
            indices_to_keep_first.append(idx)
            # add perspective to the set of covered perspectives
            covered_perspectives.add(perspective)
        elif len(covered_perspectives) == len(unique_perspectives):
            # Once all unique perspectives are covered we know how to re-arrange the list
            break
            # create a new list of perspectives, that contain first the indices_to_keep_first and then the rest
    indices_to_keep_last = set(range(len(ground_truth_relevance))) - set(indices_to_keep_first)

    ideal_sorted_perspectives = [ground_truth_perspectives[i] for i in indices_to_keep_first] + [
        ground_truth_perspectives[i] for i in indices_to_keep_last]
    ideal_sorted_relevance = [ground_truth_relevance[i] for i in indices_to_keep_first] + [ground_truth_relevance[i] for
                                                                                           i in indices_to_keep_last]
    k2idcg = {}
    for k in k_range:
        ideal_relevance_scores = ideal_sorted_relevance[:k]
        ideal_perspectives = ideal_sorted_perspectives[:k]
        ideal_dcg = calculate_dcg(ideal_relevance_scores, alpha, ideal_perspectives)
        k2idcg[k] = ideal_dcg

    return k2idcg


def alpha_ndcg(relevance_scores_predictions, perspectives_predictions, relevance_scores_global, perspectives_global,
               alpha, k_range):
    """
    Calculates the alpha-Normalized Discounted Cumulative Gain (alpha-nDCG), a metric
    that evaluates the quality of a ranking by considering both relevance and the diversity
    of perspectives among the ranked items. The alpha parameter introduces a penalty for
    redundancy.

    Parameters:
    - relevance_scores (list of int): A list of binary scores indicating the relevance of
      each item in the ranking. A score of 1 signifies that the item is relevant, while a
      score of 0 indicates non-relevance. The order of scores corresponds to the ranking order.

    - perspectives (list of str): A list identifying the perspective of each item in the
      ranking. Perspectives are categorical attributes that describe the origin, viewpoint,
      or demographic attribute of each item (e.g., political affiliation, demographic group).
      The order of perspectives aligns with the ranking order.

    - alpha (float): A coefficient between 0 and 1 used to adjust the penalty for perspective
      redundancy. A higher value of alpha increases the penalty for redundancy, promoting
      greater diversity within the ranking.

    Returns:
    - float: The alpha-nDCG score for the given ranking. This score ranges from 0 to 1, where
      1 represents an ideal ranking with maximal relevance and diversity. The score
      is computed by normalizing the DCG score with the IDCG score.

    Example:
    \\>>> relevance_scores = [1, 0, 1, 1, 0]
    \\>>> perspectives = ['politics', 'health', 'politics', 'economy', 'health']
    \\>>> alpha = 0.5
    \\>>> print(compute_alpha_nDCG(relevance_scores, perspectives, alpha))
    The function prints the calculated alpha-nDCG score based on the input parameters.
    """
    # Calculate IDCG for the whole list (all arguments are in this one)
    k2idcg = calculate_idcg(relevance_scores_global, alpha, perspectives_global, k_range)
    k2alpha_ndcg = {}
    # Calculate DCG, but cut off the relevance_scores and perspectives at k:
    for k in k_range:
        topk_relevance = relevance_scores_predictions[:k]
        topk_perspectives = perspectives_predictions[:k]
        dcg = calculate_dcg(topk_relevance, alpha, topk_perspectives)
        idcg = k2idcg[k]
        alpha_ndcg = dcg / idcg if idcg > 0 else 0
        k2alpha_ndcg[k] = alpha_ndcg

    # Calculate alpha-nDCG
    return k2alpha_ndcg


def calculateNormalizedDiscountedKLDivergence(ranked_perspectives, gold_propotion, protected_group, cut_off_points, k):
    """
        Calculates the Normalized Discounted Kullback-Leibler (rKL) Divergence for a ranked list of items, focusing on a specific
        protected group. This metric evaluates the fairness of the ranking by comparing the distribution of the protected
        group in top-k ranked items against a gold standard proportion. The divergence is calculated at specified cut-off
        points and then averaged, with each point discounted by the logarithm of its rank, to assess how well the ranking
        reflects the representation of the protected group.

        Parameters:
        - ranked_perspectives (list of str): A list of perspectives associated with each item in the ranking. Each
          perspective should indicate the group to which the item belongs.
        - gold_proportion (float): The expected or gold standard proportion of the protected group in the entire dataset
          or in an ideal fair distribution.
        - protected_group (str): The identifier for the protected group whose representation is being evaluated for fairness.
        - cut_off_points (list of int): A list of ranks at which to calculate the KL divergence. These points determine
          where in the ranking the distribution of the protected group is evaluated.
        - k (int): The maximum rank to consider for the calculation. If a cut-off point exceeds this value, it is ignored.

        Returns:
        - float: The normalized KL divergence score, which quantifies the deviation of the protected group's distribution
          in the ranking from the gold standard proportion. Lower scores indicate a distribution closer to the expected
          proportion, suggesting a fairer ranking.

        Example:
        \\>>> ranked_perspectives = ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'B']
        \\>>> gold_proportion = 0.5
        \\>>> protected_group = 'B'
        \\>>> cut_off_points = [4, 8]
        \\>>> k = 8
        \\>>> score = calculateNormalizedKLDivergence(ranked_perspectives, gold_proportion, protected_group, cut_off_points, k)
        # Output: A float value representing the normalized KL divergence score.
        """
    # gold_dist needs to be precomputed once based on all perspectives
    rKL_sum = 0  # Sum of discounted KL divergences
    Z = 0  # Normalization factor
    for cut_point in cut_off_points:
        if cut_point > k:
            break
        # calculate the proportion of protected group members in the top-k
        top_i_classes = ranked_perspectives[:cut_point]
        P = Counter(top_i_classes)[protected_group] / cut_point
        Q = gold_propotion
        # Calculate the KL divergence at this cut point
        KL_div = kl_div(P, Q)
        # Update the rKL sum, discounting by log2(i)
        rKL_sum += KL_div / math.log2(cut_point)

        # Update the normalization factor
        Z += 1 / math.log2(cut_point)

    # Calculate the final rKL by normalizing
    rKL = rKL_sum / Z
    return rKL
