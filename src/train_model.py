import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on processed data.")
    parser.add_argument('--train-features', type=str, default='train_features.csv', help='Path to the training features.')
    parser.add_argument('--train-labels', type=str, default='train_labels.csv', help='Path to the training labels.')
    parser.add_argument('--imbalance-strategy', type=str, default='none', choices=['none', 'weighted', 'oversample', 'undersample'], help='Strategy to handle class imbalance.')
    return parser.parse_args()

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logger

def load_data(features_path: str, labels_path: str) -> pd.DataFrame:
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    return X, y

def handle_imbalance(X, y, strategy: str):
    if strategy == 'weighted':
        # Handled during model training via scale_pos_weight
        return X, y
    elif strategy == 'oversample':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    elif strategy == 'undersample':
        under_sampler = RandomUnderSampler(random_state=42)
        X_res, y_res = under_sampler.fit_resample(X, y)
    else:
        X_res, y_res = X, y
    return X_res, y_res

def train_model(X, y, strategy: str):
    # Calculate scale_pos_weight for imbalanced dataset
    scale_pos_weight = (y == 0).sum() / (y == 1).sum() if strategy == 'weighted' else 1
    
    model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y.values.ravel())
    
    return model

def log_results(model, X, y, logger):
    preds = model.predict(X)
    report = classification_report(y, preds)
    accuracy = accuracy_score(y, preds)
    logger.info("Classification Report:\n" + report)
    logger.info(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    args = parse_args()
    logger = get_logger(__name__)
    
    # Load data
    X_train, y_train = load_data(args.train_features, args.train_labels)
    
    # Handle class imbalance
    X_train_bal, y_train_bal = handle_imbalance(X_train, y_train, args.imbalance_strategy)
    
    # Train model
    model = train_model(X_train_bal, y_train_bal, args.imbalance_strategy)
    
    # Log results
    log_results(model, X_train_bal, y_train_bal, logger)
