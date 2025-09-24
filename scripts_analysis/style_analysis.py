import os
from argparse import ArgumentParser

import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from feature_analysis_shap import get_processed_dataframe
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


def get_paper_vectors(feature_df):
    """
    This function gets the average feature vector for each paper.
    A paper is defined by the acl_id. The feature vectors are averaged over all the windows in the paper.
    :param feature_df:
    :return:
    """
    tqdm.pandas()
    # get average feature vector for each paper
    feature_df_t1 = feature_df[feature_df["after"] == 0]
    feature_df_t2 = feature_df[feature_df["after"] == 1]
    paper_vectors_t1 = feature_df_t1.groupby("acl_id").progress_apply(lambda x: x.mean())
    paper_vectors_t2 = feature_df_t2.groupby("acl_id").progress_apply(lambda x: x.mean())
    print("number of papers in t1", paper_vectors_t1.shape)
    print("number of papers in t2", paper_vectors_t2.shape)
    return paper_vectors_t1, paper_vectors_t2


def ten_fold_cross_validation(paper_vectors):
    # columns_to_ignore = ["abstract", "full_text", "title", "text", "ID", "after"]
    columns_to_ignore = ["after"]
    # split the data into 10 folds
    ten_fold_cv = KFold(n_splits=10, shuffle=True, random_state=42)
    fold2similarities = {}
    for fold, (train_index, test_index) in enumerate(ten_fold_cv.split(paper_vectors)):
        test_data = paper_vectors.iloc[test_index]
        # drop "after"
        test_data = test_data.drop(columns=columns_to_ignore)
        # compute a similarity matrix between papers
        cosine_similarity_matrix = cosine_similarity(test_data)
        # flatten but exclude the diagonal
        similarities = cosine_similarity_matrix[np.triu_indices(cosine_similarity_matrix.shape[0], k=1)]
        fold2similarities[fold] = similarities
    return fold2similarities


def create_box_plot(similarities_t1, similarities_t2, outpath, fold):
    # Set style
    sns.set(style="whitegrid", palette="Set2")

    # Plot Boxplot & Violin plot
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=[similarities_t1, similarities_t2], inner="quartile", linewidth=1.2,
                   palette=["#1f77b4", "#ff7f0e"])
    sns.boxplot(data=[similarities_t1, similarities_t2], width=0.2, fliersize=2, linewidth=1.2,
                boxprops=dict(alpha=0.6), palette=["#1f77b4", "#ff7f0e"])

    # Formatting
    plt.xticks([0, 1], ["t1", "t2"], fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.title("Comparison of Paper Similarities (t1 vs t2)", fontsize=14, fontweight="bold")

    plt.savefig(os.path.join(outpath, f"boxplot_{fold}.png"), dpi=300, bbox_inches="tight")

    # Clear plot to prevent overlap in future plots
    plt.clf()


# Function to compute Cohen's d
def cohen_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2)


# Function to perform bootstrapping
def bootstrap_test(x, y, n_bootstraps=1000):
    diff_means = []
    for _ in range(n_bootstraps):
        sample1 = np.random.choice(x, size=len(x), replace=True)
        sample2 = np.random.choice(y, size=len(y), replace=True)
        diff_means.append(np.mean(sample1) - np.mean(sample2))

    ci = np.percentile(diff_means, [2.5, 97.5])
    return np.mean(diff_means), ci


# Function to perform all tests
def run_stat_tests(x, y, n_bootstraps=1000):
    print("Running statistical tests...")
    ts, p_val_ttest = ttest_ind(x, y)
    # 1. Bootstrap test
    boot_mean_diff, boot_ci = bootstrap_test(x, y, n_bootstraps)
    print(f"Bootstrap mean difference: {boot_mean_diff:.4f}")
    print(f"Bootstrap 95% CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")
    bootstrap_mean_diff = boot_mean_diff
    bootstrap_ci_lower = boot_ci[0]
    bootstrap_ci_upper = boot_ci[1]

    # 2. Mann-Whitney U test
    u_stat, p_val = mannwhitneyu(x, y)
    print(f"Mann-Whitney U test p-value: {p_val:.4f}")

    # 3. Cohen's d
    effect_size = cohen_d(x, y)
    print(f"Cohen's d (effect size): {effect_size:.4f}")
    return {"bootstrap_mean_diff": bootstrap_mean_diff, "bootstrap_ci_lower": bootstrap_ci_lower,
            "bootstrap_ci_upper": bootstrap_ci_upper, "mannwhitneyu_p_val": p_val, "cohen_d": effect_size,
            "ttest_p_val": p_val_ttest, "mean_t1": np.mean(x), "mean_t2": np.mean(y)}


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    # add reading in data t1 and t2
    argument_parser.add_argument("--t1", type=str, help="Path to the first dataset")
    argument_parser.add_argument("--t2", type=str, help="Path to the second dataset")
    argument_parser.add_argument("--output", type=str, help="Path to the output directory")
    argument_parser.add_argument("--compare_highest", action="store_true",
                                 help="Whether to compare the highest similarity scores")
    args = argument_parser.parse_args()
    print("readung in the data...")
    # read in the data as parquet
    t1 = pd.read_parquet(args.t1)
    print("Read in the first dataset")
    t2 = pd.read_parquet(args.t2)
    print("Read in the second dataset")
    print("size of t1", t1.shape)
    print("size of t2", t2.shape)
    t1["after"] = 0
    t2["after"] = 1
    # scale and standardize the features
    feat_df_all = get_processed_dataframe(pd.concat([t1, t2]))
    # get document vectors
    paper_vectors_t1, paper_vectors_t2 = get_paper_vectors(feat_df_all)
    # perform 10-fold cross validation
    fold2similarities_t1 = ten_fold_cross_validation(paper_vectors_t1)
    fold2similarities_t2 = ten_fold_cross_validation(paper_vectors_t2)
    results = []
    for fold, similarities_t1 in fold2similarities_t1.items():
        similarities_t2 = fold2similarities_t2[fold]
        if args.compare_highest:
            # sort similarities in descending order
            similarities_t1 = np.sort(similarities_t1)[::-1]
            similarities_t2 = np.sort(similarities_t2)[::-1]
            # take the top 2000 similarities
            sample_t1 = similarities_t1[:1000]
            sample_t2 = similarities_t2[:1000]
        else:
            # take a good sample size for statistical tests
            sample_t1 = np.random.choice(similarities_t1, 1000)
            sample_t2 = np.random.choice(similarities_t2, 1000)
        statistics_dict = run_stat_tests(sample_t1, sample_t2)
        statistics_dict["fold"] = fold
        results.append(statistics_dict)
        # create box plot
        # take smaller sample of similarities to plot, take a random sample of 1000
        create_box_plot(sample_t1, sample_t2, args.output, fold)
    # save the results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output, "t_test_results.csv"), index=False)
