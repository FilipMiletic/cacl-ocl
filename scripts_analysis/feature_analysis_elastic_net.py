from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from tqdm.asyncio import tqdm

from feature_analysis_shap import get_processed_dataframe
import numpy as np
import pandas as pd


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
    mcfadden_scores = []
    nagelkerke_scores = []
    auc_scores = []
    feature_counts = pd.Series(0, index=features)
    coef_sums = pd.Series(0.0, index=features)

    # ---------------------------------------------------------------------
    # OUTER LOOP
    # ---------------------------------------------------------------------
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)

    for outer_train_idx, outer_test_idx in tqdm(outer_cv.split(X, y), total=N_OUTER, desc="Outer CV"):
        # outer cross validation split, train on outer_train, test on outer_test
        X_train, X_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_train, y_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

        # Inner CV is handled by LogisticRegressionCV
        model = LogisticRegressionCV(
            # Cs=10,  # grid of inverse regularization values
            Cs=np.logspace(-6, 6, 30),
            cv=N_INNER,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=np.linspace(0, 1, 6),  # test different alpha values
            scoring="neg_log_loss",
            tol=1e-3,
            max_iter=1000,
            random_state=SEED)
        model.fit(X_train, y_train)

        # -----------------------------------------------------------------
        # Performance on outer test set
        # -----------------------------------------------------------------
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_scores.append(auc)
        # McFadden & Nagelkerke pseudo-R²
        r2_mcf, r2_nag = pseudo_r2_measures(y_test, y_pred_proba)
        mcfadden_scores.append(r2_mcf)
        nagelkerke_scores.append(r2_nag)

        # -----------------------------------------------------------------
        # Track feature selection + coefficients
        # -----------------------------------------------------------------
        coefs = model.coef_.flatten()
        print(coefs)
        for feat, coef in zip(features, coefs):
            if coef != 0:  # selected feature
                feature_counts[feat] += 1
                coef_sums[feat] += coef

    # Summarize results
    print("\n=== Nested CV Results ===")
    print(f"Mean AUC (outer test sets): {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    print(f"Mean McFadden R² (outer test sets): {np.mean(mcfadden_scores):.3f} ± {np.std(mcfadden_scores):.3f}")
    print(f"Mean Nagelkerke R² (outer test sets): {np.mean(nagelkerke_scores):.3f} ± {np.std(nagelkerke_scores):.3f}")
    # Selection frequency
    selection_freq = feature_counts / N_OUTER

    # Average coefficient values (where selected)
    avg_coefs = coef_sums / feature_counts.replace(0, np.nan)

    # Combine into summary dataframe
    summary = pd.DataFrame({
        "Selection_Freq": selection_freq,
        "Avg_Coef": avg_coefs
    }).fillna(0).sort_values("Selection_Freq", ascending=False)

    print("\n=== Feature Stability Summary ===")
    print(summary.head(20))  # show top 20 features

    # Save results
    summary.to_csv("feature_stability.csv", index=True)


def find_highly_correlated_features(df1, df2, drop_cols=None, threshold=0.9):
    """
    Identify sets of highly correlated features in two dataframes.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Input dataframes containing features.
    drop_cols : list of str, optional
        Columns to drop before correlation analysis.
    threshold : float, default=0.9
        Correlation threshold above which features are considered highly correlated.

    Returns
    -------
    correlated_sets1, correlated_sets2 : list of sets
        Lists of sets, where each set contains highly correlated feature names.
    """

    def combine_sets(pairs):
        """Combine overlapping pairs into sets of correlated features."""
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
        """Return sets of highly correlated features for a dataframe."""
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
        corr_matrix = df.corr().abs()
        high_corr = np.where(corr_matrix > threshold)
        pairs = [(corr_matrix.columns[x], corr_matrix.columns[y])
                 for x, y in zip(*high_corr) if x != y and x < y]
        return combine_sets(pairs)

    correlated_sets1 = get_high_corr(df1)
    correlated_sets2 = get_high_corr(df2)

    print("Highly correlated features in df1:")
    print(correlated_sets1)
    print("Highly correlated features in df2:")
    print(correlated_sets2)

    return correlated_sets1, correlated_sets2


df1 = pd.read_parquet("/Users/johannesfalk/PycharmProject/cacl-ocl/cacl_t1_features_sample.parquet")
df2 = pd.read_parquet("/Users/johannesfalk/PycharmProject/cacl-ocl/cacl_t2_features_sample.parquet")
# subsample only 2000 rows for speed
# df1 = df1.sample(n=500, random_state=42)
# df2 = df2.sample(n=500, random_state=42)
feature_df1 = get_processed_dataframe(df1)
feature_df2 = get_processed_dataframe(df2)
print(feature_df1.head())
# cut off acl_id

feature_df1 = feature_df1.drop(columns=["acl_id", "ID"], errors="ignore")
feature_df2 = feature_df2.drop(columns=["acl_id", "ID"], errors="ignore")
feature_df1["after"] = 0
feature_df2["after"] = 1
# combine both dataframes
feature_df = pd.concat([feature_df1, feature_df2], axis=0)

constant_columns = feature_df.columns[feature_df.nunique() <= 1]
feature_df = feature_df.drop(constant_columns, axis=1)
print(f"Dropped {len(constant_columns)} constant columns.")
# drop any NaN columns
nan_columns = feature_df.columns[feature_df.isna().any()]
feature_df = feature_df.drop(nan_columns, axis=1)
print(f"Dropped {len(nan_columns)} NaN columns.")
# drop highly correlated features
correlated_sets1, correlated_sets2 = find_highly_correlated_features(feature_df1, feature_df2, drop_cols=["after"],
                                                                     threshold=0.9)
to_drop = set()
for s in correlated_sets1 + correlated_sets2:
    to_drop.update(list(s)[1:])  # keep one feature, drop the rest
feature_df = feature_df.drop(columns=to_drop, errors="ignore")
print(f"Dropped {len(to_drop)} highly correlated columns.")
print(feature_df.shape)
target = feature_df["after"]
print(target.shape)
print(target.value_counts())
featuer_df = feature_df.drop(columns=["after"], errors="ignore")
# do feature selection
robust_feature_selection(featuer_df, target, N_OUTER=5, N_INNER=5, SEED=42)
