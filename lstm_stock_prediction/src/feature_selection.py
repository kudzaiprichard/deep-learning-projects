import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_correlation_with_target(df, target="Close", threshold=0.3):
    """
    Get features correlated with the target above a threshold.

    Args:
        df: DataFrame with features
        target: Target column name
        threshold: Minimum absolute correlation

    Returns:
        Series of correlations above threshold, sorted by absolute value
    """
    correlations = df.corr()[target].drop(target)
    strong = correlations[correlations.abs() >= threshold].sort_values(
        key=abs, ascending=False
    )

    print(f"Features with |correlation| >= {threshold} to '{target}': {len(strong)}")
    return strong


def remove_highly_correlated(df, threshold=0.95, protect_columns=None):
    """
    Remove features that are highly correlated with each other.

    Args:
        df: DataFrame with features
        threshold: Correlation threshold for removal
        protect_columns: List of columns to never remove

    Returns:
        DataFrame with redundant features removed, list of dropped columns
    """
    if protect_columns is None:
        protect_columns = []

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = []
    for column in upper.columns:
        if column in protect_columns:
            continue
        if any(upper[column] > threshold):
            to_drop.append(column)

    df_reduced = df.drop(columns=to_drop)
    print(f"Removed {len(to_drop)} highly correlated features (threshold={threshold})")
    if to_drop:
        print(f"Dropped: {to_drop}")

    return df_reduced, to_drop


def plot_feature_correlations(df, target="Close", top_n=20):
    """
    Plot top feature correlations with target.

    Args:
        df: DataFrame with features
        target: Target column
        top_n: Number of top features to show
    """
    correlations = df.corr()[target].drop(target).sort_values(key=abs, ascending=True)
    top_features = correlations.tail(top_n)

    plt.figure(figsize=(10, 8))
    colors = ['green' if v > 0 else 'red' for v in top_features.values]
    top_features.plot(kind='barh', color=colors)
    plt.title(f'Top {top_n} Feature Correlations with {target}')
    plt.xlabel('Correlation')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, features=None, figsize=(12, 10)):
    """
    Plot correlation heatmap for selected features.

    Args:
        df: DataFrame with features
        features: List of features to include (None = all)
        figsize: Figure size
    """
    if features:
        corr = df[features].corr()
    else:
        corr = df.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()


def select_features(df, target="Close", corr_threshold=0.3, redundancy_threshold=0.95,
                    must_include=None):
    """
    Run the full feature selection pipeline.

    Args:
        df: DataFrame with engineered features
        target: Target column
        corr_threshold: Min correlation with target
        redundancy_threshold: Max correlation between features
        must_include: Features to always include

    Returns:
        List of selected feature names
    """
    if must_include is None:
        must_include = ["Close"]

    print("Starting feature selection pipeline...")
    print("-" * 40)

    # Step 1: Get features correlated with target
    strong_corr = get_correlation_with_target(df, target, corr_threshold)
    candidate_features = list(strong_corr.index)

    # Ensure must_include features are in the list
    for f in must_include:
        if f not in candidate_features and f in df.columns:
            candidate_features.insert(0, f)

    # Ensure target is included
    if target not in candidate_features:
        candidate_features.insert(0, target)

    print(f"\nCandidate features: {len(candidate_features)}")

    # Step 2: Remove redundant features
    df_candidates = df[candidate_features]
    df_reduced, dropped = remove_highly_correlated(
        df_candidates,
        threshold=redundancy_threshold,
        protect_columns=must_include
    )

    selected = list(df_reduced.columns)

    print("-" * 40)
    print(f"Final selected features ({len(selected)}):")
    for i, f in enumerate(selected, 1):
        corr_val = df.corr()[target].get(f, 0)
        print(f"  {i}. {f} (corr: {corr_val:.3f})")

    return selected