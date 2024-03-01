import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix, roc_curve
import logging
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on processed data.")
    parser.add_argument('--train-features', type=str, default='processed_data/train_features.csv', help='Path to the training features.')
    parser.add_argument('--train-labels', type=str, default='processed_data/train_labels.csv', help='Path to the training labels.')
    parser.add_argument('--val-features', type=str, default='processed_data/val_features.csv', help='Path to the training features.')
    parser.add_argument('--val-labels', type=str, default='processed_data/val_labels.csv', help='Path to the training labels.')
    parser.add_argument('--imbalance-strategy', type=str, default='weighted', choices=['none', 'weighted', 'oversample', 'undersample'], help='Strategy to handle class imbalance.')
    return parser.parse_args()

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logger


def evaluate_classification(y_true, y_pred, y_pred_proba):
    # Calculate recall
    recall = recall_score(y_true, y_pred)
    
    # Calculate precision
    precision = precision_score(y_true, y_pred)
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    '''
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    '''
    
    return recall, precision, roc_auc, cm

def load_data(train_features_path: str, train_labels_path: str, val_features_path: str, val_labels_path: str) -> pd.DataFrame:
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_labels_path)
    X_val = pd.read_csv(val_features_path)
    y_val = pd.read_csv(val_labels_path)
    return X_train, y_train, X_val, y_val

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
    scale_pos_weight = ((y == 0).sum() / (y == 1).sum()).values[0] if strategy == 'weighted' else 1
    
    model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y.values.ravel())
    
    return model

def log_results(model, X, y, logger):
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert probabilities to binary output
    recall, precision, roc_auc, cm = evaluate_classification(y, y_pred, y_pred_proba)
    logger.info(f"% Recall: {recall * 100} ")
    logger.info(f"% ROC_AUC: {roc_auc * 100 }")
    logger.info(f"CM : \n {cm}")

if __name__ == '__main__':
    args = parse_args()
    logger = get_logger(__name__)
    
    # Load data
    X_train, y_train, X_val, y_val = load_data(args.train_features, args.train_labels, args.val_features, args.val_labels)
    
    # Handle class imbalance
    X_train_bal, y_train_bal = handle_imbalance(X_train, y_train, args.imbalance_strategy)
    
    # Train model
    model = train_model(X_train_bal, y_train_bal, args.imbalance_strategy)
    
    # Log results
    log_results(model, X_val, y_val, logger)
