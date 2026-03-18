import pandas as pd
import numpy as np


def check_data_quality(df):
    """
    Print a data quality report.

    Args:
        df: DataFrame with stock data

    Returns:
        Dictionary with quality metrics
    """
    report = {
        "total_rows": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": df.isnull().sum().sum(),
        "duplicates": df.index.duplicated().sum(),
        "date_range": (df.index[0], df.index[-1]),
        "dtypes": df.dtypes.to_dict()
    }

    print("=" * 50)
    print("DATA QUALITY REPORT")
    print("=" * 50)
    print(f"Total rows: {report['total_rows']}")
    print(f"Date range: {report['date_range'][0].date()} to {report['date_range'][1].date()}")
    print(f"Total missing values: {report['total_missing']}")
    print(f"Duplicate dates: {report['duplicates']}")
    print("\nMissing per column:")
    for col, count in report['missing_values'].items():
        print(f"  {col}: {count}")
    print("=" * 50)

    return report


def handle_missing_values(df, method="ffill"):
    """
    Handle missing values in the dataset.

    Args:
        df: DataFrame with stock data
        method: 'ffill' (forward fill), 'bfill' (back fill),
                'interpolate' (linear interpolation), 'drop' (remove rows)

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    before = df.isnull().sum().sum()

    if method == "ffill":
        df = df.ffill()
    elif method == "bfill":
        df = df.bfill()
    elif method == "interpolate":
        df = df.interpolate(method="linear")
    elif method == "drop":
        df = df.dropna()

    # Handle any remaining NaN at edges
    df = df.bfill().ffill()

    after = df.isnull().sum().sum()
    print(f"Missing values: {before} → {after} (method: {method})")

    return df


def remove_duplicates(df):
    """
    Remove duplicate date entries.

    Args:
        df: DataFrame with DateTimeIndex

    Returns:
        DataFrame with duplicates removed
    """
    df = df.copy()
    before = len(df)
    df = df[~df.index.duplicated(keep='first')]
    after = len(df)
    print(f"Duplicates removed: {before - after}")
    return df


def detect_outliers(df, column="Close", method="iqr", threshold=1.5):
    """
    Detect outliers in a column.

    Args:
        df: DataFrame with stock data
        column: Column to check for outliers
        method: 'iqr' (interquartile range) or 'zscore'
        threshold: IQR multiplier or z-score threshold

    Returns:
        Boolean Series (True = outlier)
    """
    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        outliers = (df[column] < lower) | (df[column] > upper)

    elif method == "zscore":
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        outliers = z_scores.abs() > threshold

    print(f"Outliers detected in '{column}': {outliers.sum()} ({method}, threshold={threshold})")
    return outliers


def handle_outliers(df, column="Close", method="iqr", threshold=1.5, action="clip"):
    """
    Handle outliers by clipping or removing.

    Args:
        df: DataFrame with stock data
        column: Column to handle outliers
        method: 'iqr' or 'zscore'
        threshold: Detection threshold
        action: 'clip' (cap values) or 'remove' (drop rows)

    Returns:
        DataFrame with outliers handled
    """
    df = df.copy()
    outliers = detect_outliers(df, column, method, threshold)

    if action == "clip":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        df[column] = df[column].clip(lower=lower, upper=upper)
        print(f"Outliers clipped to [{lower:.2f}, {upper:.2f}]")

    elif action == "remove":
        df = df[~outliers]
        print(f"Outlier rows removed: {outliers.sum()}")

    return df


def detect_anomalous_returns(df, column="Close", max_daily_pct=15.0):
    """
    Detect anomalous daily price changes (e.g., stock splits, errors).

    Args:
        df: DataFrame with stock data
        column: Price column
        max_daily_pct: Maximum allowed daily percentage change

    Returns:
        DataFrame of anomalous rows
    """
    df = df.copy()
    df['pct_change'] = df[column].pct_change().abs() * 100
    anomalies = df[df['pct_change'] > max_daily_pct]

    print(f"Anomalous daily changes (>{max_daily_pct}%): {len(anomalies)}")
    if len(anomalies) > 0:
        print(anomalies[[column, 'pct_change']])

    df.drop(columns=['pct_change'], inplace=True)
    return anomalies


def clean_pipeline(df, missing_method="interpolate", outlier_threshold=3.0):
    """
    Run the full cleaning pipeline.

    Args:
        df: Raw DataFrame
        missing_method: Method for handling missing values
        outlier_threshold: IQR threshold for outlier detection

    Returns:
        Cleaned DataFrame
    """
    print("Starting cleaning pipeline...")
    print("-" * 40)

    df = remove_duplicates(df)
    df = handle_missing_values(df, method=missing_method)
    detect_anomalous_returns(df)

    print("-" * 40)
    print(f"Cleaning complete. Final shape: {df.shape}")
    return df