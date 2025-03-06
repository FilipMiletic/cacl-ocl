import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import tabulate
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import shap
from scipy.stats import ttest_ind
import seaborn as sns


def get_feature_impact(df):
    """
    Get the coefficient for one feature on the dependent variable. Get a metric of how important a feature is / how much
    it impacts the dependent variable.
    """
    feature2importance = {}
    # Iterate through each feature in the dataframe
    for feature in tqdm(df.columns.drop(["after", "acl_id"])):
        # Train a logistic regression model
        model = LogisticRegression()
        model.fit(df[[feature]], df["after"])

        # Get the coefficient (traditional feature importance)
        coefficient = model.coef_[0][0]
        # get the feature importance
        explainer = shap.Explainer(model, df[[feature]])
        shap_values = explainer(df[[feature]])

        # Get mean absolute SHAP value for feature importance
        mean_shap_value = np.abs(shap_values.values).mean()

        # Compute McFadden's R2
        log_likelihood_model = -log_loss(df["after"], model.predict_proba(df[[feature]])[:, 1])
        log_likelihood_null = -log_loss(df["after"], np.full_like(df["after"], df["after"].mean()))
        mcfadden_r2 = 1 - (log_likelihood_model / log_likelihood_null)
        feature2importance[feature] = {"Coefficient": coefficient, "Mean SHAP": mean_shap_value,
                                       "McFadden R2": mcfadden_r2}

    # sort the features by importance
    feature_importance_df = pd.DataFrame.from_dict(feature2importance, orient="index")
    # sort the features by importance
    feature_importance_df = feature_importance_df.sort_values(by="Mean SHAP", ascending=False)
    return feature_importance_df


def create_dendogram(args, feature_df, feature2importance, n_features=10):
    top_features = list(feature2importance.keys())
    top_features = top_features[:n_features]
    # get the spearman correlation matrix
    spearman_corr = feature_df[top_features].corr(method="spearman")
    distance_matrix = 1 - spearman_corr
    # convert the distance matrix to a condensed form
    condensed_distance_matrix = squareform(distance_matrix)
    # do hierarchical clustering
    linkage_matrix = sch.linkage(condensed_distance_matrix, method="ward")
    plt.figure(figsize=(20, 10))
    dendrogram = sch.dendrogram(linkage_matrix, labels=spearman_corr.columns, leaf_rotation=90)
    plt.title("Dendrogram of Top {} Features".format(n_features))
    plt.xlabel("Features")
    plt.ylabel("Distance")
    outpath = os.path.join(args.output, "dendrogram.png")
    # plot the dendrogram
    plt.savefig(outpath, dpi=300)


def get_processed_dataframe(feature_df):
    """
    Preprocess the dataframe by removing columns that have only one value, columns that have only NaN values, and
    columns that have np.inf or -np.inf values. Then scale the features using StandardScaler.
    Remove the columns that are not needed for the analysis.
    """
    # Drop text-based columns that are not needed for analysis
    columns_to_ignore = {"abstract", "full_text", "title", "text"}
    feature_df = feature_df.drop(columns=columns_to_ignore.intersection(feature_df.columns), errors="ignore")
    feature_df_current = feature_df.drop(columns=["after", "acl_id"], errors="ignore")
    # Identify and drop columns with only NaN values or a single unique value
    cols_to_drop = [col for col in feature_df_current.columns
                    if feature_df_current[col].nunique() == 1 or feature_df_current[col].isnull().all()]
    feature_df_current = feature_df_current.drop(columns=cols_to_drop, errors="ignore")

    # Replace np.inf and -np.inf with NaN, then fill NaN values with the column mean
    feature_df_current.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df_current.fillna(feature_df_current.mean(), inplace=True)

    # Ensure 'after' and 'acl_id' are not included in scaling
    non_scaling_cols = {"after", "acl_id"}
    scaling_features = [col for col in feature_df_current.columns if col not in non_scaling_cols]

    # Scale the numerical features
    scaler = StandardScaler()
    feature_df_scaled = pd.DataFrame(scaler.fit_transform(feature_df_current[scaling_features]),
                                     columns=scaling_features, index=feature_df_current.index)
    # Final safety check: Ensure no NaN values exist in the processed dataframe
    feature_df_scaled.fillna(0, inplace=True)
    # Add back 'after' and 'acl_id' if they exist
    for col in non_scaling_cols.intersection(feature_df.columns):
        feature_df_scaled[col] = feature_df[col]

    return feature_df_scaled


def get_paper_vectors(feature_df):
    tqdm.pandas()
    # get average feature vector for each paper
    feature_df_t1 = feature_df[feature_df["after"] == 0]
    feature_df_t2 = feature_df[feature_df["after"] == 1]
    paper_vectors_t1 = feature_df_t1.groupby("acl_id").progress_apply(lambda x: x.mean())
    paper_vectors_t2 = feature_df_t2.groupby("acl_id").progress_apply(lambda x: x.mean())
    print("number of papers in t1", paper_vectors_t1.shape)
    print("number of papers in t2", paper_vectors_t2.shape)
    return paper_vectors_t1, paper_vectors_t2


def compare_paper_similarity(args, paper_vectors_t1, paper_vectors_t2):
    # Compute cosine similarity matrices
    cosine_similarity_t1 = cosine_similarity(paper_vectors_t1)
    cosine_similarity_t2 = cosine_similarity(paper_vectors_t2)

    # Flatten similarity matrices for comparison
    similarities_t1 = np.ravel(cosine_similarity_t1)
    similarities_t2 = np.ravel(cosine_similarity_t2)

    # Perform t-test
    t_stat, p_value = ttest_ind(similarities_t1, similarities_t2)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Display t-statistic and p-value on the plot
    plt.annotate(f"t = {t_stat:.2f}\np = {p_value:.2e}", xy=(0.5, 0.9), xycoords="axes fraction",
                 ha="center", fontsize=12, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))

    # Save figure
    outpath = output_dir / "cosine_similarity.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    # Save t-test results to file
    with open(output_dir / "t_test_results.txt", "w") as f:
        f.write(f"t-statistic: {t_stat:.6f}\n")
        f.write(f"p-value: {p_value:.6e}\n")

    # Print results
    print(f"t-statistic: {t_stat:.6f}")
    print(f"p-value: {p_value:.6e}")
    print(f"Results saved to {outpath} and t_test_results.txt")


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    # add reading in data t1 and t2
    argument_parser.add_argument("--t1", type=str, help="Path to the first dataset")
    argument_parser.add_argument("--t2", type=str, help="Path to the second dataset")
    argument_parser.add_argument("--output", type=str, help="Path to the output file")
    args = argument_parser.parse_args()

    # read in the data as parquet
    t1 = pd.read_parquet(args.t1)
    print("Read in the first dataset")
    t2 = pd.read_parquet(args.t2)
    print("Read in the second dataset")
    print("size of t1", t1.shape)
    print("size of t2", t2.shape)
    t1["after"] = 0
    t2["after"] = 1
    feat_df_all = get_processed_dataframe(pd.concat([t1, t2]))
    print("Feature Processing Done")

    # feature2importance = get_feature_impact(feat_df_all)
    # print(tabulate.tabulate(feature2importance, headers="keys", tablefmt="pretty"))
    # save the feature importance into output path + file name
    outpath = os.path.join(args.output, "feature_importance.csv")
    # feature2importance.to_csv(outpath)
    print("Feature Importance Done")
    compare_paper_similarity(args, *get_paper_vectors(feat_df_all))
    print("cosine similarity done")
    # create_dendogram(args, feat_df_all, feature2importance)
    # print("dendogram done")
