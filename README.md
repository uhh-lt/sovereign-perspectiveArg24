# sövereign-perspectiveArg24

This repository describes the code that was developed by team Sövereign in the context of the Perspective Argument Retrieval 2024 Shared Task. The main goal of the task is the incorporation of socio-cultural and -demographic information in an information retrieval process. 

More information about the shared task can be found under the following link

https://blubberli.github.io/perspective-argument-retrieval.github.io/

or in the shared task paper:

```bibtex
@inproceedings{falk-etal-2024-overview,
title = "{Overview of PerspectiveArg2024: The First Shared Task on Perspective Argument Retrieval}", 
author = "Falk, Neele and Waldis, Andreas and Gurevych, Iryna", 
booktitle = "Proceedings of the 11th Workshop on Argument Mining", 
month = aug, 
year = "2024", 
address = "Bangkok", 
publisher = "Association for Computational Linguistics" 
}
```

## Task Description
The Perpsective Argument Retrieval shared task revolves around incorporating socio-cultural and -demographic information in an information retrieval process. It is made up of 3 scenarios:

Scenario 1 (baseline scenario) is a simple information retrieval scenario. For each given query, a list of relevant arguments needs to be returned. No social or demographic properties need to be considered in this task.

Scenario 2 (explicit scenario) includes socio-cultural and -demographic information in both the query and each argument in the corpus. 

Scenario 3 (implicit scenario) includes the socio-cultural and -demographic information in the query only, the given socio-cultural and -demographic information in the argument corpus cannot be used. 

## Dataset
The dataset is based on the x-stance dataset. It contains political question  and arguments from the Swiss Federal elections. Overall five different sets of questions exist for the train set, the dev set and each of the three evaluation rounds from the shared task. The corpus of arguments changes only slighty between each round, as some new arguments, specifically relevant for the questions from the current evaluation round, are added. None of the datasets are included in this repository! To download the data for further testing, please head to the shared task repository:

https://github.com/Blubberli/perspective-argument-retrieval

For each train, dev or test set, there are two sets of queries: aside from unique query_ids, the set of queries for scenario 1 only contains a bunch of unique political questions. The set of queries for scenarios 2 and 3 contains the same questions as the set of baseline queries, but every question is included multiple times, paired up with a different social or demographic trait each time.

Aside from unique argument_ids, the corpus of arguments includes the argument text, a map, that includes all socio-cultural and -demographic features for the argument and also a topic and a stance parametre. The last two could both be used to improve results across all three scenarios.

## Solution Pipeline

Our task solution involves six different steps and is described in detail in our system description paper:

...

Firstly, depending on each scenario, our solution involves three or four different scores. These scores are all further described in the system description paper. For each of these scores, this repository contains one jupyter notebook, that allows the computation. 

For scenario 1 we use SBERT relevance scores, topic scores and llm scores. For scenario 2 we use SBERT relevance scores, topic scores, explicit demographic scores and llm scores. For scenario 3 we use SBERT relevance scores, topic scores and implicit demographic scores. 

To combine our scores into a prediction, we use a logistic regression to learn a weight for each of the scores. To train the logistic regression, we compute all relevant scores for the current scenario, and the gold scores, which are based on the ground truth relevant arguments, for the train set.

Both the llm scores and the implicit demographic scores are calculated by an llm. For our solution we used the HuggingChat API: 

https://github.com/Soulter/hugging-chat-api

For each of these scores, we give the llm 50 arguments as an input, hence we calculate predictions with all other given scores before predicting these scores. This means for scenario 1 we calculate a prediction with SBERT relevance score and topic score, then use the top 50 from this prediction to calculate llm scores, and then use SBERT relevance score, topic score and llm score to calculate a final prediction. For each prediction we always calculate weights using a logistic regression. All arguments, that are not included in the llm input get an llm score / implicit demographic score of 0.

## Use Sövereign solution
This describes how to use all attached notebooks to compute scores and predictions. Generally, all code passages, where anything needs to be changed or inserted are marked with comments in the code. 

### 1. Compute SBERT relevance scores
Use the sbert_relevance_scores.ipynb notebook to compute sbert scores. Make sure to set the destination folder in the end. When wanting to compute scores for one of the test sets, also make sure to set the correct data path name and to uncomment the corresponding code. This script originates from the organizers of the Perspective Argument Retrieval shared task and was only modified slightly.

### 2. Compute topic scores
Use the topic_scores.ipynb notebook to compute topic scores. Make sure to set the data path, folder name and file name and to uncomment code segments when computing for a test set. 

### 3. Compute explicit demographic scores
Use the explicit_demographic_socres.ipynb notebook to compute explicit demographic scores. Make sure to set data path and folder name, and to uncomment corresponding code segments when wanting to calculate scores for a test set. 

### 4. Compute gold scores for logistic regression
To compute the gold scores for a set of queries in the right format to use with the logistic regression, use gold_relevance_scores.ipynb notebook. Don't forget to set data release path name, the correct set of queries and the output path to a jsonl path, where the gold scores should be saved. When computing the scores for a test set, also don't forget to uncomment the corresponding code.

### 5. Compute weights with logistic regression
Use the logistic_regression.ipynb notebook to compute logistic regression weights. Don't forget to insert all paths to the relevant scores and some filepaths to save the formatted matrices. The current version of logistic_regression.ipynb is set up to calculate the weights for a final prediction for scenario 2, where all four scores exist. To calculate weights for any other prediction, those lines regarding the missing scores need to be commented, and some other small changed might be necessary.

### 6. Calculate predictions 
Use the calculate_predictions.ipynb notebook to compute the predictions for a set of queries based on pre-calculated scores. Don't forget to set the data path name, the paths to all relevant scores, to comment the lines including "read_json" for not needed scores, to set weights for all scores and a valid path to save the predictions. The notebook computes predictions in the format expected from the shared task, which is the best 1000 argument ids in descending order for each query id. The ranking is computed as the sum of all given scores, each weighted by the corresponding weight.

### 7. Compute implicit demographic scores / implicit llm scores
To compute the implicit demographic scores for task scenario 3, use the implicit_demographic_scores.ipynb notebook. Don't forget to set the data release path name, the correct set of queries and the path to an empty json file to save each score right after it was generated by the llm. Because the implicit demographic relevance is only scored for the top 50 most relevant arguments of each query, also set the path to a prediction, that will serve to determine these top 50. As we used the Huggingchat API to generate llm scores for scenarios 1 and 2 and the implicit demographic scores for scenario 3, a Huggingface account is needed to access the models through this API. Username or email adress and password need to be inserted in the notebook to grant access to the API. Please make sure to not share any code containing sensitive information. 

Watch out: we were a little inconsistent when naming these scores: sometimes they were called implicit llm scores, as they are predicted by an llm, and sometimes they were called implicit demographic scores, as the llm is prompted to score based on demographics only in thes scenario. If any issues still occur due to this, please notify us!

### 8. Compute explicit llm scores
To compute the llm scores for task scenario 1 or 2, use the llm_scores.ipynb notebook. Don't forget to set the data release path name, the correct set of queries, the huggingface username and password (make sure to not share these sensitive information online!) and the path to an empty json file to save scores right after computation. 

## Predictions
Aside from our method, this repository also already contains all predictions, that come up in our system description paper. The predictions are in the format expected by the shared task, which is the top 1000 highest ranking arguments for each query.

## Evaluation
To evaluate the results of any prediction, the Shared task organizers offer a few scripts. To invoke the evaluation you can use the notebook evaluation.ipynb. 

<br><br>
## 
All scripts related to evalutaion or utils were created by the organizers of the Perspective Argument Retrieval shared task with only some slight changes made by us in some of them. They can also be found in this repository:

https://github.com/Blubberli/perspective-argument-retrieval