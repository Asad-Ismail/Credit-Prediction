import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, roc_curve
import logging
import argparse
import joblib
import numpy as np
from typing import Tuple, List, Any, Union

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for training a machine learning model.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a machine learning model on preprocessed data.")
    parser.add_argument('--train-features', type=str, default='processed_data/train_features.csv',
                        help='File path to the CSV containing training feature data.')
    parser.add_argument('--train-labels', type=str, default='processed_data/train_labels.csv',
                        help='File path to the CSV containing training label data.')
    parser.add_argument('--val-features', type=str, default='processed_data/val_features.csv',
                        help='File path to the CSV containing validation feature data.')
    parser.add_argument('--val-labels', type=str, default='processed_data/val_labels.csv',
                        help='File path to the CSV containing validation label data.')
    parser.add_argument('--model', type=str, default='logistic_regression',
                        choices=['xgboost', 'random_forest', 'logistic_regression'],
                        help='The type of model to train.')
    parser.add_argument('--hp-optimizer', action='store_true',default=False,
                        help='Flag to enable hyperparameter optimization.')
    parser.add_argument('--evaluation-metric', type=str, default='average_precision',
                        choices=['roc_auc', 'f1', 'precision', 'recall', 'average_precision'],
                        help='Evaluation metric to use for hyperparameter optimization and model evaluation.')
    parser.add_argument('--imbalance-strategy', type=str, default='weighted',
                        choices=['none', 'weighted', 'oversample', 'undersample'],
                        help='Approach to handle class imbalance in the dataset.')
    return parser.parse_args()

def get_logger(name: str) -> logging.Logger:
    """
    Configures and retrieves a logger instance.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logger

def save_model(model: Any, args: argparse.Namespace) -> None:
    """
    Saves the trained model to a file.

    Args:
        model (Any): The trained machine learning model to be saved.
        args (argparse.Namespace): The command-line arguments containing model and optimization details.
    """
    model_name = f"{args.model}{'_hp_optimized' if args.hp_optimizer else ''}.pkl"
    joblib.dump(model, model_name)
    logger = get_logger(__name__)
    logger.info(f"Model saved to {model_name}")

def evaluate(y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series) -> Tuple[float, float, float, np.ndarray, float]:
    """
    Calculate performance metrics for classification outcomes.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        y_pred_proba: Predicted probabilities for the positive class.

    Returns:
        A tuple containing the following evaluation metrics:
        - Recall
        - Precision
        - ROC AUC score
        - Confusion matrix
        - F1 score
    """
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)  
    
    return recall, precision, roc_auc, cm, f1

def load_data(train_features_path: str, train_labels_path: str, val_features_path: str, val_labels_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load training and validation datasets from specified file paths.

    Args:
        train_features_path: Path to the CSV file containing training features.
        train_labels_path: Path to the CSV file containing training labels.
        val_features_path: Path to the CSV file containing validation features.
        val_labels_path: Path to the CSV file containing validation labels.

    Returns:
        A tuple containing four pandas DataFrames:
        - Training features (X_train)
        - Training labels (y_train)
        - Validation features (X_val)
        - Validation labels (y_val)
    """
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_labels_path)
    X_val = pd.read_csv(val_features_path)
    y_val = pd.read_csv(val_labels_path)

    print(X_train.columns.tolist())
    return X_train, y_train, X_val, y_val

def handle_imbalance(X: pd.DataFrame, y: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a specified strategy to handle class imbalance in the dataset.

    Args:
        X: Feature data.
        y: Target labels.
        strategy: The strategy for handling class imbalance, which can be 'weighted', 'oversample', 'undersample', or 'none'.

    Returns:
        The resampled feature data and labels after applying the class imbalance strategy.
    """
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
        # 'none' strategy or any other unspecified strategy falls here
        X_res, y_res = X, y
    return X_res, y_res

def train_and_optimize_model(X: pd.DataFrame, y: pd.Series, args: Any) -> Any:
    """
    Trains and optimizes a machine learning model based on the specified arguments.

    Args:
        X: The input features for training the model.
        y: The target labels for training the model.
        args: Command-line arguments or other options that include the model type, hyperparameter optimization settings, and class imbalance strategy.

    Returns:
        The trained and potentially optimized model.
    """
    # Calculate the class weight for the positive class if the 'weighted' strategy is selected
    scale_pos_weight = ((y == 0).sum() / (y == 1).sum()).values if args.imbalance_strategy == 'weighted' else 1
    class_weights = {0: 1, 1: scale_pos_weight} if args.imbalance_strategy == 'weighted' else None
    
    if args.model == 'xgboost':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=class_weights if class_weights else 1)
        param_grid = {
            'max_depth': [3, 4, 5],
            'n_estimators': [200, 500, 700],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_lambda': [1, 0.1, 0]
        } if args.hp_optimizer else {}

    elif args.model == 'random_forest':
        model = RandomForestClassifier(class_weight=class_weights) if class_weights else RandomForestClassifier()
        param_grid = {
            'n_estimators': [200, 500, 700],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10]
        } if args.hp_optimizer else {}

    elif args.model == 'logistic_regression':
        model = LogisticRegression(solver='liblinear', class_weight=class_weights) if class_weights else LogisticRegression(solver='liblinear')
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2']
        } if args.hp_optimizer else {}

    # Perform hyperparameter optimization if enabled
    if args.hp_optimizer:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(model, param_grid, scoring=args.evaluation_metric, cv=cv, verbose=3)
        grid_search.fit(X, y.values.ravel())
        best_model = grid_search.best_estimator_
        logger = logging.getLogger(__name__)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best {args.evaluation_metric} score: {grid_search.best_score_}")
    else:
        best_model = model
        best_model.fit(X, y.values.ravel())

    return best_model

def log_results(model: Any, X: pd.DataFrame, y: pd.Series, logger: logging.Logger) -> None:
    """
    Logs the evaluation results of the trained model.

    Args:
        model: The trained machine learning model.
        X: The input features for evaluation.
        y: The true target labels for evaluation.
        logger: The logger object to log the evaluation results.
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    recall, precision, roc_auc, cm, f1 = evaluate(y, y_pred, y_pred_proba)
    logger.info(f"% Recall: {recall * 100}")
    logger.info(f"% Precision: {precision * 100}")
    logger.info(f"% F1 score: {f1 * 100}")
    logger.info(f"% ROC_AUC: {roc_auc * 100}")
    logger.info(f"CM : \n {cm}")

if __name__ == '__main__':
    args = parse_args()

    logger = get_logger(__name__)
    
    X_train, y_train, X_val, y_val = load_data(args.train_features, args.train_labels, args.val_features, args.val_labels)
    X_train_bal, y_train_bal = handle_imbalance(X_train, y_train, args.imbalance_strategy)
    
    best_model = train_and_optimize_model(X_train_bal, y_train_bal, args)
    
    log_results(best_model, X_val, y_val, logger)

    save_model(best_model,args)


