import json
import os
from collections import Counter

import pandas as pd
import tabulate


def get_average_results_dataframe():
    inputpath = "/Users/johannesfalk/PycharmProject/cacl-ocl/results_feature_selection/regression_analysis"

    # get all json files in the input path
    json_files = [f for f in os.listdir(inputpath) if f.endswith('.json')]
    print(json_files)
    results = [json.load(open(f"{inputpath}/{file}")) for file in json_files]
    # convert into dataframes
    results = pd.DataFrame(results)
    # add column split
    results["split"] = list(range(1, 11))
    # print the head of the DataFrame
    # add a mean row to the DataFrame
    results.loc["mean"] = results.mean()
    # add deviation
    results.loc["std"] = results.std()
    # round to 3 digits
    results = results.round(3)
    print(tabulate.tabulate(results, headers="keys", tablefmt="psql"))
    # save to csv
    results.to_csv(f"{inputpath}/average_results.csv")


def get_best_features():
    inputpath = "/Users/johannesfalk/PycharmProject/cacl-ocl/results_feature_selection/regression_analysis"
    feature_files = [f for f in os.listdir(inputpath) if "features" in f]
    # read in the feature files
    features = [open(f"{inputpath}/{file}").readlines() for file in feature_files]
    # strip the newline character
    features = [list(map(str.strip, feature)) for feature in features]
    # get the number of features per model
    numbers_of_features = [len(feature) for feature in features]
    print("number of features per model", Counter(numbers_of_features))
    # count occurences of each feature
    feature_counts = Counter([feature for sublist in features for feature in sublist])
    # total number of features that are in the feature files
    print("total number of features", len(feature_counts))
    print(feature_counts)
    freq_each_count = Counter(feature_counts.values())
    print("number of features with a certain frequency", freq_each_count)





if __name__ == '__main__':
    get_average_results_dataframe()
