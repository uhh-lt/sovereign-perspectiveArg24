import os
from collections import defaultdict

import pandas as pd


def read_gold_data(data_path):
    """
        Reads and loads the gold standard data from a specified directory path.
        Input the path to the root directory where the shared task data is stored at. It should contain subdirectories
        for different the different scenarios, each of which includes JSON Lines (.jsonl) files
        for development (dev), training (train). The root directory should also contain a JSON Lines file for the corpus.

        Parameters:
        - data_path (str): The path to the root directory.

        Returns:
        - dict: A nested dictionary containing the loaded data for each scenario. The top-level keys are the scenario names
          (e.g., 'baseline', 'perspective'). Each scenario
          key maps to another dictionary with three keys: 'dev' and 'train' corresponding to the development set,
          training set. The corpus is stored in 'corpus.jsonl'. The values for 'dev' and 'train' are pandas DataFrames
          with query_ids and a lists of argument_ids. The 'corpus' key maps to a pandas DataFrame built from
          the 'corpus.jsonl' file with argument_ids, their texts, and demographic profiles.
    """
    # if the path is relative, convert it to an absolute path, otherwise it will throw an error
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    data_path = os.path.abspath(os.path.join(project_root, data_path))

    try:

        baseline_train = pd.read_json(os.path.join(data_path, "baseline-queries", "queries_train.jsonl"), lines=True,
                                      orient="records")
        baseline_dev = pd.read_json(os.path.join(data_path, "baseline-queries", "queries_dev.jsonl"), lines=True,
                                    orient="records")

        perspective_dev = pd.read_json(os.path.join(data_path, "perspective-queries", "queries_dev.jsonl"), lines=True,
                                       orient="records")
        perspective_train = pd.read_json(os.path.join(data_path, "perspective-queries", "queries_train.jsonl"),
                                         lines=True,
                                         orient="records")

        corpus = pd.read_json(os.path.join(data_path, "corpus.jsonl"), lines=True, orient="records")
        # add important political issues as separate columns
        corpus = add_political_issues_profile(corpus)

        # add socio-cultural properties as separate columns
        corpus = convert_corpus_with_distinct_demographic_properties(corpus)
        # check if there is a file with 'test' in the directory
        if os.path.exists(os.path.join(data_path, "baseline-queries", "queries_test.jsonl")):
            baseline_test = pd.read_json(os.path.join(data_path, "baseline-queries", "queries_test.jsonl"), lines=True,
                                         orient="records")
            perspective_test = pd.read_json(os.path.join(data_path, "perspective-queries", "queries_test.jsonl"),
                                            lines=True,
                                            orient="records")
            return {"baseline": {"dev": baseline_dev, "train": baseline_train, "test": baseline_test},
                    "perspective": {"dev": perspective_dev, "train": perspective_train, "test": perspective_test},
                    "corpus": corpus}
    except ValueError:
        print("The data path is not correct. Please provide the correct path to the data directory.")
        return

    # put all data in a dictionary
    data = {"baseline": {"dev": baseline_dev, "train": baseline_train},
            "perspective": {"dev": perspective_dev, "train": perspective_train},
            "corpus": corpus}
    return data


def convert_corpus_with_distinct_demographic_properties(corpus):
    """
    Convert the demographic_properties column into separate columns for each socio-cultural property
    :param corpus: the corpus dataframe
    :return: the corpus dataframe with the socio-cultural properties converted into separate columns
    """
    df_demographic = pd.json_normalize(corpus['demographic_profile'])
    corpus = corpus.join(df_demographic)
    return corpus


def add_political_issues_profile(corpus):
    """
    Transform the list of important issues for each person into a separate variable, that specifies whether the issue
    is important or neutral for the person.
    :param corpus: the corpus dataframe
    :return: the corpus dataframe with the new variables as separate columns
    """
    political_issues = {'Liberale Gesellschaft', 'Ausgebauter Umweltschutz', 'Restriktive Finanzpolitik', 'Law & Order',
                        'Liberale Wirtschaftspolitik', 'Restriktive Migrationspolitik', 'Ausgebauter Sozialstaat',
                        'Offene Aussenpolitik'}
    for argument_id, row in corpus.iterrows():
        profile = row["demographic_profile"]
        important_issues = profile["important_political_issues"]
        for issue in political_issues:
            if issue not in important_issues:
                profile[issue] = "neutral"
            else:
                profile[issue] = "important"
        # update the row with the new profile
        corpus.at[argument_id, "demographic_profile"] = profile
    return corpus


def retrieve_argumentIDs_duplicate_texts(corpus):
    """
    Retrieve the argument_ids of arguments that have the same text. This method creates a dictionary that maps each
    argument_id to all other argument_ids that share the same text.
    :param corpus: the corpus dataframe
    :return: a dictionary that maps each argument_id to all other argument_ids that have the same text (as a set of IDs)
    """
    # Find duplicate texts
    duplicates = defaultdict(set)
    for argid, text in corpus[corpus.duplicated(subset=["argument"], keep=False)][["argument"]].itertuples():
        duplicates[argid].update(corpus[corpus["argument"] == text].index)
    # Remove self-references
    for argid, other_ids in duplicates.items():
        other_ids.remove(argid)
    return duplicates
