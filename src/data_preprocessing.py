import argparse
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process and split credit application data.")
    parser.add_argument('--data-dir', type=str, default='raw_data', help='Directory containing raw data files.')
    parser.add_argument('--output-dir', type=str, default='processed_data', help='Directory to save processed data files.')
    parser.add_argument('--credit-applications', type=str, default='credit_applications.csv', help='Filename of credit applications data.')
    parser.add_argument('--user-features', type=str, default='customers.csv', help='Filename of user features data.')
    parser.add_argument('--feature-cols', type=str, default='', help='Comma-separated feature columns in the data.')
    parser.add_argument('--label-col', type=str, default='credit_application', help='Name of the label column in the data.')
    parser.add_argument('--train-data-ratio', type=float, default=0.8, help='Fraction of data to use for the training set.')
    parser.add_argument('--test-data-ratio', type=float, default=0.1, help='Fraction of data to use for the test set.')
    return parser.parse_args()

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    log_format = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger

def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies log transformation to specified numeric columns to reduce skewness.
    
    Args:
        df (pd.DataFrame): DataFrame containing the features to transform.
        
    Returns:
        pd.DataFrame: DataFrame with log-transformed features.
    """
    numeric_cols = ['total_nr_trx', 'nr_debit_trx', 'volume_debit_trx', 'nr_credit_trx', 'volume_credit_trx']
    df[numeric_cols] = df[numeric_cols].apply(lambda x: np.log1p(x))
    return df

def apply_standard_scaling(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Fits a standard scaler on the training data and applies the transformation to the training, validation, and test datasets.
    
    Args:
        train_df (pd.DataFrame): Training data DataFrame.
        val_df (pd.DataFrame): Validation data DataFrame.
        test_df (pd.DataFrame): Testing data DataFrame.
        feature_cols (list): List of column names to scale.
        
    Returns:
        tuple: Tuple containing the scaled training, validation, and test DataFrames.
    """
    scaler = StandardScaler()
    train_df_scaled = scaler.fit_transform(train_df[feature_cols])
    val_df_scaled = scaler.transform(val_df[feature_cols])
    test_df_scaled = scaler.transform(test_df[feature_cols])

    # Update the original dataframes with the scaled values
    train_df[feature_cols] = train_df_scaled
    val_df[feature_cols] = val_df_scaled
    test_df[feature_cols] = test_df_scaled

    return train_df, val_df, test_df

def plot_distributions(df: pd.DataFrame, columns: list, output_dir: str, prefix: str = "") -> None:
    """
    Plots the distribution of each specified column in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list): List of column names to plot.
        output_dir (str): Directory to save the plot images.
        prefix (str): Prefix for the plot file names.
    """
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"{prefix}{column}_distribution.png"))
        plt.close()

def preprocess_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str, apply_log: bool = False, apply_scaling: bool = False) -> None:
    """
    Preprocesses the data by applying log transformation and standard scaling based on flags.
    Plots the feature distributions before and after transformations.
    
    Args:
        train_df (pd.DataFrame): Training data DataFrame.
        val_df (pd.DataFrame): Validation data DataFrame.
        test_df (pd.DataFrame): Testing data DataFrame.
        output_dir (str): Directory to save processed files and plots.
        apply_log (bool): Flag to apply log transformation.
        apply_scaling (bool): Flag to apply standard scaling.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot original distributions
    plot_distributions(train_df, ['total_nr_trx', 'nr_debit_trx', 'volume_debit_trx', 'nr_credit_trx', 'volume_credit_trx'], output_dir, "original_")
    
    if apply_log:
        # Apply log transformation
        train_df = apply_log_transform(train_df)
        val_df = apply_log_transform(val_df)
        test_df = apply_log_transform(test_df)
        # Plot distributions after log transformation
        plot_distributions(train_df, ['total_nr_trx', 'nr_debit_trx', 'volume_debit_trx', 'nr_credit_trx', 'volume_credit_trx'], output_dir, "log_")
    
    if apply_scaling:
        # Apply standard scaling
        feature_cols = ['total_nr_trx', 'nr_debit_trx', 'volume_debit_trx', 'nr_credit_trx', 'volume_credit_trx']
        train_df[feature_cols] = apply_standard_scaling(train_df[feature_cols])
        val_df[feature_cols] = apply_standard_scaling(val_df[feature_cols])
        test_df[feature_cols] = apply_standard_scaling(test_df[feature_cols])
        # Plot distributions after standard scaling
        plot_distributions(train_df, feature_cols, output_dir, "scaled_")
    
    # Continue with saving processed data as demonstrated previously...


def load_data(data_dir: str, credit_data: str, features_data: str, label_col: str) -> pd.DataFrame:
    """
    Loads and merges credit and features data.

    Args:
        data_dir (str): Directory containing the data files.
        credit_data (str): Filename of the credit data.
        features_data (str): Filename of the features data.
        label_col (str): Name of the label column.

    Returns:
        pd.DataFrame: Merged DataFrame of features and credit data.
    """
    credit_df = pd.read_csv(os.path.join(data_dir, credit_data), index_col=0)
    logging.info(f"Shape of credit data: {credit_df.shape}")
    logging.info(f"Percent of Credits in data: {credit_df[label_col].mean() * 100:.2f}%")

    features_df = pd.read_csv(os.path.join(data_dir, features_data), index_col=0)
    logging.info(f"Shape of features data: {features_df.shape}")

    assert len(features_df) == len(credit_df), "Credit data and features data must have the same number of rows."

    merged_df = features_df.merge(credit_df, on=["client_nr", "yearmonth"], how="left")
    logging.info(f"Shape of merged data: {merged_df.shape}")

    return merged_df

def split_data(df: pd.DataFrame, label_col: str, train_data_ratio: float, test_data_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train, validation, and test sets based on specified ratios.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        label_col (str): The name of the label column.
        train_data_ratio (float): The ratio of data to use for training.
        test_data_ratio (float): The ratio of data to use for testing.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train, validation, and test DataFrames.
    """
    val_data_ratio = 1 - train_data_ratio - test_data_ratio
    logging.info(f"Data split ratios - Train: {train_data_ratio}, Validation: {val_data_ratio}, Test: {test_data_ratio}")

    # Stratified sampling based on 'client_nr' and 'credit_application'
    client_label_dist = df.groupby('client_nr')[label_col].mean().reset_index()
    label_dist = df[label_col].mean()
    client_label_dist['stratum'] = (client_label_dist[label_col] > label_dist).astype(int)

    client_tv, client_test = train_test_split(client_label_dist, test_size=test_data_ratio, stratify=client_label_dist['stratum'], random_state=42)
    client_train, client_val = train_test_split(client_tv, test_size=val_data_ratio / (1 - test_data_ratio), stratify=client_tv['stratum'], random_state=42)

    train_df = df[df['client_nr'].isin(client_train['client_nr'])]
    val_df = df[df['client_nr'].isin(client_val['client_nr'])]
    test_df = df[df['client_nr'].isin(client_test['client_nr'])]

    logging.info(f"Data percentages - Train: {len(train_df) / len(df) * 100:.2f}%, Validation: {len(val_df) / len(df) * 100:.2f}%, Test: {len(test_df) / len(df) * 100:.2f}%")
    logging.info(f"Credit application percentages - Train: {train_df[label_col].mean() * 100:.2f}%, Validation: {val_df[label_col].mean() * 100:.2f}%, Test: {test_df[label_col].mean() * 100:.2f}%")

    return train_df, val_df, test_df

def get_features_and_labels(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """
    Processes the data, imputes missing values, and saves features and labels for training, validation, and testing.

    Args:
        train_df (pd.DataFrame): Training data DataFrame.
        val_df (pd.DataFrame): Validation data DataFrame.
        test_df (pd.DataFrame): Testing data DataFrame.
        output_dir (str): Directory to save the processed files.
    """
    # Impute missing 'CRG' values with the median
    crg_imputer = train_df['CRG'].median()
    for df in [train_df, val_df, test_df]:
        df['CRG'].fillna(crg_imputer, inplace=True)

    # Split features and labels
    X_train, y_train = train_df.drop(['client_nr', 'yearmonth', 'credit_application', 'nr_credit_applications'], axis=1), train_df['credit_application']
    X_val, y_val = val_df.drop(['client_nr', 'yearmonth', 'credit_application', 'nr_credit_applications'], axis=1), val_df['credit_application']
    X_test, y_test = test_df.drop(['client_nr', 'yearmonth', 'credit_application', 'nr_credit_applications'], axis=1), test_df['credit_application']

    # Save to CSV
    file_paths = {
        'train_features.csv': X_train, 'train_labels.csv': y_train,
        'val_features.csv': X_val, 'val_labels.csv': y_val,
        'test_features.csv': X_test, 'test_labels.csv': y_test
    }
    for file_name, data in file_paths.items():
        data.to_csv(os.path.join(output_dir, file_name), index=False)
        logging.info(f"Wrote {file_name} to {os.path.join(output_dir, file_name)}")

if __name__ == '__main__':
    logger = get_logger(__name__)
    args = parse_args()

    df = load_data(args.data_dir, args.credit_applications, args.user_features, args.label_col)
    train_df, val_df, test_df = split_data(df, args.label_col, args.train_data_ratio, args.test_data_ratio)
    get_features_and_labels(train_df, val_df, test_df, args.output_dir)
