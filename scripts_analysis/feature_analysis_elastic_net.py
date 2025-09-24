import argparse
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from tabulate import tabulate
from feature_analysis_shap import get_processed_dataframe


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def compute_log_likelihood(y_true, y_pred_proba):
    """Compute log-likelihood from predicted probabilities."""
    eps = 1e-15  # avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    return np.sum(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))


def pseudo_r2_measures(y_true, y_pred_proba):
    """Return both McFadden and Nagelkerke pseudo-R²."""
    n = len(y_true)
    ll_full = compute_log_likelihood(y_true, y_pred_proba)
    ll_null = compute_log_likelihood(y_true, np.repeat(np.mean(y_true), n))
    r2_mcfadden = 1 - (ll_full / ll_null)
    r2_nagelkerke = r2_mcfadden / (1 - ll_null / n)
    return r2_mcfadden, r2_nagelkerke


def robust_feature_selection(X, y, N_OUTER=10, N_INNER=5, SEED=42):
    """
    Perform robust feature selection using nested cross-validation with Elastic Net logistic regression.
    Store the averaged coefficients and selection frequencies of features.
    Store the average R2 and AUC across outer folds.
    """
    features = X.columns.tolist()
    mcfadden_scores, nagelkerke_scores, auc_scores = [], [], []
    feature_counts = pd.Series(0, index=features)
    coef_sums = pd.Series(0.0, index=features)

    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)

    # parameter grid
    Cs = np.logspace(-2, 4, 20)
    l1_ratios = np.linspace(0, 0.25, 3)
    param_grid = {"C": Cs, "l1_ratio": l1_ratios}

    for outer_train_idx, outer_test_idx in tqdm(outer_cv.split(X, y), total=N_OUTER, desc="Outer CV"):
        X_train, X_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_train, y_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

        logreg = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=1000,
            tol=1e-3,
            random_state=SEED
        )

        inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
        grid = GridSearchCV(
            estimator=logreg,
            param_grid=param_grid,
            cv=inner_cv,
           # scoring="neg_log_loss",
            scoring="roc_auc",
            n_jobs=-1,
            verbose=3  # <--- fold-by-fold logs
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # --- Performance on outer test set ---
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_scores.append(auc)

        r2_mcf, r2_nag = pseudo_r2_measures(y_test, y_pred_proba)
        mcfadden_scores.append(r2_mcf)
        nagelkerke_scores.append(r2_nag)

        # --- Track features ---
        coefs = best_model.coef_.flatten()
        print("coefficients of the best model in this fold:")
        print(pd.Series(coefs, index=features).sort_values(ascending=False).head(10))

        for feat, coef in zip(features, coefs):
            if coef != 0:
                feature_counts[feat] += 1
                coef_sums[feat] += coef

    print("\n=== Nested CV Results ===")
    print(f"Mean AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    print(f"Mean McFadden R²: {np.mean(mcfadden_scores):.3f} ± {np.std(mcfadden_scores):.3f}")
    print(f"Mean Nagelkerke R²: {np.mean(nagelkerke_scores):.3f} ± {np.std(nagelkerke_scores):.3f}")

    selection_freq = feature_counts / N_OUTER
    avg_coefs = coef_sums / feature_counts.replace(0, np.nan)

    summary = (
        pd.DataFrame({"Selection_Freq": selection_freq, "Avg_Coef": avg_coefs})
        .fillna(0)
        .sort_values("Selection_Freq", ascending=False)
    )
    print("\n=== Feature Stability Summary ===")
    print(summary.head(20))
    summary.to_csv("feature_stability.csv", index=True)


def find_highly_correlated_features(df1, df2, drop_cols=None, threshold=0.9):
    """Identify sets of highly correlated features in two dataframes."""

    def combine_sets(pairs):
        sets = []
        for a, b in pairs:
            found = False
            for s in sets:
                if a in s or b in s:
                    s.update([a, b])
                    found = True
                    break
            if not found:
                sets.append({a, b})
        return sets

    def get_high_corr(df):
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
        corr_matrix = df.corr().abs()
        high_corr = np.where(corr_matrix > threshold)
        pairs = [
            (corr_matrix.columns[x], corr_matrix.columns[y])
            for x, y in zip(*high_corr)
            if x != y and x < y
        ]
        return combine_sets(pairs)

    c1 = get_high_corr(df1)
    c2 = get_high_corr(df2)

    print("Highly correlated features in df1:", c1)
    print("Highly correlated features in df2:", c2)
    return c1, c2


def main(args):
    df1 = pl.read_parquet(args.input_t1).to_pandas()
    df2 = pl.read_parquet(args.input_t2).to_pandas()
    df1 = df1.sample(30000)
    df2 = df2.sample(30000)
    print("size of T1 dataframe: ", df1.shape)
    print("size of T2 dataframe: ", df2.shape)

    feature_df1 = get_processed_dataframe(df1).drop(columns=["acl_id", "ID"], errors="ignore")
    feature_df2 = get_processed_dataframe(df2).drop(columns=["acl_id", "ID"], errors="ignore")

    feature_df1["after"] = 0
    feature_df2["after"] = 1

    feature_df = pd.concat([feature_df1, feature_df2], axis=0)
    #print(tabulate(feature_df.head(), headers='keys', tablefmt='psql'))
    print("target distribution:")
    print(feature_df.after.value_counts())

    # Drop constants & NaNs
    constant_columns = feature_df.columns[feature_df.nunique() <= 1]
    feature_df = feature_df.drop(columns=constant_columns, errors="ignore")
    print(f"Dropped {len(constant_columns)} constant columns.")

    nan_columns = feature_df.columns[feature_df.isna().any()]
    feature_df = feature_df.drop(columns=nan_columns, errors="ignore")
    print(f"Dropped {len(nan_columns)} NaN columns.")
    """
    # Drop highly correlated
    correlated_sets1, correlated_sets2 = find_highly_correlated_features(
        feature_df1, feature_df2, drop_cols=["after"], threshold=0.9
    )
    to_drop = {col for s in correlated_sets1 + correlated_sets2 for col in list(s)[1:]}
    feature_df = feature_df.drop(columns=to_drop, errors="ignore")
    print(f"Dropped {len(to_drop)} highly correlated columns.")
    """
    target = feature_df["after"]
    feature_df = feature_df.drop(columns=["after"], errors="ignore")
    print("value counts of the target variable:")
    print(target.value_counts())

    robust_feature_selection(feature_df, target, N_OUTER=args.outer, N_INNER=args.inner, SEED=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust feature selection with nested CV.")
    parser.add_argument("--input_t1", type=str, required=True, help="Path to T1 parquet file")
    parser.add_argument("--input_t2", type=str, required=True, help="Path to T2 parquet file")
    parser.add_argument("--outer", type=int, default=5, help="Number of outer CV folds")
    parser.add_argument("--inner", type=int, default=5, help="Number of inner CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(args)
