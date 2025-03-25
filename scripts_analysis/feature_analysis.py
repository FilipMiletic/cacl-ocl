import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import shap
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.model_selection import KFold


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


def bidirectional_elimination_logistic(X, y, significance_level=0.05, shap_cutoff=0.01, output_path='./output'):
    """
    Perform Bidirectional Elimination for a Logistic Regression Model with SHAP Filtering and R2 Calculation.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        significance_level (float): P-value threshold for feature elimination.
        shap_cutoff (float): Minimum SHAP importance value to retain a feature.
        output_path (str): Path to save model outputs.

    Returns:
        dict: Dictionary containing model summary, SHAP values, and R2 values.
    """

    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")

    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series")

    # Step 1: SHAP Filtering (Preliminary)
    print("Running SHAP for preliminary feature selection...")

    # Fit logistic regression model for SHAP values
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X).values

    # Calculate mean absolute SHAP value for each feature
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.Series(mean_shap_values, index=X.columns).sort_values(ascending=False)

    # Select features based on SHAP importance
    selected_features = shap_importance[shap_importance > shap_cutoff].index.tolist()
    X_filtered = X[selected_features]
    print(f"Selected {len(selected_features)} features based on SHAP importance.")

    # Remove multicollinearity
    correlation = X_filtered.corr().abs()
    high_corr = set()
    for i in range(len(correlation.columns)):
        for j in range(i):
            if correlation.iloc[i, j] > 0.95:
                colname = correlation.columns[i]
                high_corr.add(colname)

    X_filtered = X_filtered.drop(high_corr, axis=1)
    print(f"Dropped {len(high_corr)} features due to multicollinearity.")

    # Drop constant columns
    constant_columns = X_filtered.columns[X_filtered.nunique() <= 1]
    X_filtered = X_filtered.drop(constant_columns, axis=1)
    print(f"Dropped {len(constant_columns)} constant columns.")
    # Step 2: Bidirectional Elimination
    print("Starting bidirectional elimination...")

    remaining_features = list(X_filtered.columns)
    selected_features = []
    improved = True

    selected_features = []
    improved = True
    seen = set()
    iteration = 0
    max_iter = 100
    progressbar = tqdm(total=max_iter)
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        progressbar.update(1)

        # Forward Step
        best_pval = float('inf')
        best_feature = None

        for feature in remaining_features:
            if feature in seen:
                continue

            model = sm.Logit(y, sm.add_constant(X_filtered[selected_features + [feature]])).fit(disp=0)
            pval = model.pvalues[feature]

            if pval < significance_level and pval < best_pval:
                best_pval = pval
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            seen.add(best_feature)
            improved = True

        # Backward Step
        while len(selected_features) > 0:
            model = sm.Logit(y, sm.add_constant(X_filtered[selected_features])).fit(disp=0)
            pvalues = model.pvalues.drop('const')
            worst_pval = pvalues.max()

            # Add small tolerance to avoid oscillation due to floating-point issues
            if worst_pval > significance_level + 1e-6:
                worst_feature = pvalues.idxmax()

                # Break oscillation: If the feature is repeatedly added/removed, stop
                if worst_feature in seen:
                    break

                selected_features.remove(worst_feature)
                remaining_features.append(worst_feature)
                improved = True
            else:
                break

    # Fit Final Model
    final_model = sm.Logit(y, sm.add_constant(X_filtered[selected_features])).fit(disp=0)

    print("Selected features:", selected_features)

    # Step 4: Calculate R² Values
    predictions = final_model.predict(sm.add_constant(X_filtered[selected_features]))
    log_likelihood = final_model.llf
    null_log_likelihood = sm.Logit(y, np.ones(len(y))).fit(disp=0).llf
    n = len(y)

    # McFadden's R²
    r2_mcfadden = 1 - (log_likelihood / null_log_likelihood)
    # Cox & Snell R²
    r2_cox_snell = 1 - np.exp((2 / n) * (null_log_likelihood - log_likelihood))
    # Nagelkerke R²
    r2_nagelkerke = r2_cox_snell / (1 - np.exp(2 * null_log_likelihood / n))

    r2_values = {
        'McFadden_R2': r2_mcfadden,
        'Cox_Snell_R2': r2_cox_snell,
        'Nagelkerke_R2': r2_nagelkerke
    }

    # Step 5: Save Outputs
    output = {
        'final_model_summary': final_model.summary().as_text(),
        'shap_importance': shap_importance.to_dict(),
        'selected_features': selected_features,
        'r2_values': r2_values
    }
    return output


def write_results(outpath, fold, shap_importance, model_summary, r2_values, final_features):
    """
    Write the results to a file.
    """
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    with open(os.path.join(outpath, f"shap_importance_{fold}.csv"), "w") as f:
        json.dump(shap_importance, f, indent=4)
    with open(os.path.join(outpath, f"model_summary_{fold}.txt"), "w") as f:
        f.write(model_summary)
    with open(os.path.join(outpath, f"r2_values_{fold}.json"), "w") as f:
        json.dump(r2_values, f, indent=4)
    with open(os.path.join(outpath, f"final_features_{fold}.txt"), "w") as f:
        f.write("\n".join(final_features))


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    # add reading in data t1 and t2
    argument_parser.add_argument("--t1", type=str, help="Path to the first dataset")
    argument_parser.add_argument("--t2", type=str, help="Path to the second dataset")
    argument_parser.add_argument("--output", type=str, help="Path to the output directory")
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
    # take a sample of 3000 from each
    t1 = t1.sample(3000)
    t2 = t2.sample(3000)
    feat_df_all = get_processed_dataframe(pd.concat([t1, t2]))
    # get 10-fold cross validation
    nfolds = 10
    kfold_split = KFold(n_splits=nfolds, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in tqdm(enumerate(kfold_split.split(feat_df_all))):
        print(f"Running fold {fold + 1}/{nfolds}...")
        X_train, X_test = feat_df_all.iloc[train_index], feat_df_all.iloc[test_index]
        y_train, y_test = X_train["after"], X_test["after"]
        X_train = X_train.drop(columns=["after", "acl_id"])
        X_test = X_test.drop(columns=["after", "acl_id"])
        # if size is too big, downsample to 3000 samples per class
        count_class_0, count_class_1 = y_train.value_counts()
        if count_class_0 > 3000:
            X_train = pd.concat([X_train[y_train == 0].sample(3000), X_train[y_train == 1].sample(3000)])
            y_train = pd.concat([y_train[y_train == 0].sample(3000), y_train[y_train == 1].sample(3000)])
        results = bidirectional_elimination_logistic(X_train, y_train, output_path=args.output)
        write_results(args.output, fold, results["shap_importance"], results["final_model_summary"],
                      results["r2_values"], results["selected_features"])
        print("Results saved successfully.")
