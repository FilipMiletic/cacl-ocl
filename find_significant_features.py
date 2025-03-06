import pandas as pd
import tabulate
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import scipy.cluster.hierarchy as sch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler



def get_feature_impact(df):
    feature2importance = {}
    for feature in tqdm(df.columns):
        # train a logistic regression model
        # get the feature importance
        # plot the feature importance
        model = LogisticRegression()
        model.fit(df[[feature]], df["after"])
        importance = model.coef_[0]
        # summarize feature importance
        feature2importance[feature] = importance
    # sort the features by importance
    feature2importance = dict(sorted(feature2importance.items(), key=lambda item: item[1], reverse=True))
    return feature2importance


def feature_selection(feature_df):
    # drop columns that have nan values
    feature_df = feature_df.dropna(axis=1)
    # drop columns that have only one value
    feature_df = feature_df.loc[:, feature_df.nunique() > 1]
    feature2importance = get_feature_impact(feature_df)
    # sort the feature2importance by  absolute value
    feature2importance = dict(sorted(feature2importance.items(), key=lambda item: abs(item[1]), reverse=True))
    # only get the top 75
    feature2importance = dict(list(feature2importance.items())[:75])
    # do hierarchical clustering using spearman correlation and the top 75 features
    # get the top 75 features
    top_features = list(feature2importance.keys())
    # get the spearman correlation matrix
    spearman_corr = feature_df[top_features].corr(method="spearman")
    distance_matrix = 1 - spearman_corr
    # convert the distance matrix to a condensed form
    condensed_distance_matrix = squareform(distance_matrix)
    # do hierarchical clustering
    linkage_matrix = sch.linkage(condensed_distance_matrix, method="ward")
    plt.figure(figsize=(20, 10))
    dendrogram = sch.dendrogram(linkage_matrix, labels=spearman_corr.columns, leaf_rotation=90)
    # plot the dendrogram
    plt.savefig("dendrogram.png", dpi=300)

    # do agglomerative cluster


if __name__ == '__main__':
    df_t1 = "cacl_t1_features.parquet"
    df_t2 = "cacl_t2_features.parquet"
    # read the data into a pandas dataframe
    df_t1 = pd.read_parquet(df_t1)
    print(list(df_t1.columns))
    #df_t2 = pd.read_parquet(df_t2)
    #
