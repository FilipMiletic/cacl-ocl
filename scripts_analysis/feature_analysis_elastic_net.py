import argparse
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

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

    for outer_train_idx, outer_test_idx in tqdm(
        outer_cv.split(X, y), total=N_OUTER, desc="Outer CV"
    ):
        X_train, X_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_train, y_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

        print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
        print(f"Train class distribution: {np.bincount(y_train) / len(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test) / len(y_test)}")

        model = LogisticRegressionCV(
            Cs=np.logspace(-6, 6, 30),
            cv=N_INNER,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=np.linspace(0, 1, 6),
            scoring="neg_log_loss",
            tol=1e-3,
            max_iter=1000,
            random_state=SEED,
        )
        model.fit(X_train, y_train)

        # Performance
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))

        r2_mcf, r2_nag = pseudo_r2_measures(y_test, y_pred_proba)
        mcfadden_scores.append(r2_mcf)
        nagelkerke_scores.append(r2_nag)

        # Feature tracking
        coefs = model.coef_.flatten()
        print(coefs)
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

    feature_df1 = get_processed_dataframe(df1).drop(columns=["acl_id", "ID"], errors="ignore")
    feature_df2 = get_processed_dataframe(df2).drop(columns=["acl_id", "ID"], errors="ignore")

    feature_df1["after"] = 0
    feature_df2["after"] = 1

    feature_df = pd.concat([feature_df1, feature_df2], axis=0)

    # Drop constants & NaNs
    constant_columns = feature_df.columns[feature_df.nunique() <= 1]
    feature_df = feature_df.drop(columns=constant_columns, errors="ignore")
    print(f"Dropped {len(constant_columns)} constant columns.")

    nan_columns = feature_df.columns[feature_df.isna().any()]
    feature_df = feature_df.drop(columns=nan_columns, errors="ignore")
    print(f"Dropped {len(nan_columns)} NaN columns.")

    # Drop highly correlated
    correlated_sets1, correlated_sets2 = find_highly_correlated_features(
        feature_df1, feature_df2, drop_cols=["after"], threshold=0.9
    )
    to_drop = {col for s in correlated_sets1 + correlated_sets2 for col in list(s)[1:]}
    feature_df = feature_df.drop(columns=to_drop, errors="ignore")
    print(f"Dropped {len(to_drop)} highly correlated columns.")

    target = feature_df["after"]
    feature_df = feature_df.drop(columns=["after"], errors="ignore")

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
