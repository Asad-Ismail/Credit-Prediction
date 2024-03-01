import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix, roc_curve
import logging
import argparse
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on processed data.")
    parser.add_argument('--train-features', type=str, default='processed_data/train_features.csv', help='Path to the training features.')
    parser.add_argument('--train-labels', type=str, default='processed_data/train_labels.csv', help='Path to the training labels.')
    parser.add_argument('--val-features', type=str, default='processed_data/val_features.csv', help='Path to the training features.')
    parser.add_argument('--val-labels', type=str, default='processed_data/val_labels.csv', help='Path to the training labels.')
    parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'random_forest', 'logistic_regression'], help='Model to train.')    
    parser.add_argument('--hp-optimizer', action='store_true', help='Enable hyperparameter optimization.')            
    parser.add_argument('--evaluation-metric', type=str, default='roc_auc', choices=['roc_auc', 'f1', 'precision', 'recall'], help='Metric for HP optimization and model selection.')
    parser.add_argument('--imbalance-strategy', type=str, default='weighted', choices=['none', 'weighted', 'oversample', 'undersample'], help='Strategy to handle class imbalance.')
    return parser.parse_args()

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logger

def save_model(model, args):
    model_name = args.model + ('_hp_optimized' if args.hp_optimizer else '') + '.pkl'
    joblib.dump(model, model_name)
    logger.info(f"Model saved to {model_name}")

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

def train_and_optimize_model(X, y, args):
    if args.model == 'xgboost':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        param_grid = {'max_depth': [3, 4, 5], 'n_estimators': [100, 200]} if args.hp_optimizer else {}
    elif args.model == 'random_forest':
        model = RandomForestClassifier()
        param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]} if args.hp_optimizer else {}
    elif args.model == 'logistic_regression':
        model = LogisticRegression(solver='liblinear')
        param_grid = {'C': [0.1, 1, 10]} if args.hp_optimizer else {}
    
    if args.hp_optimizer:
        cv = StratifiedKFold(n_splits=5)
        grid_search = GridSearchCV(model, param_grid, scoring=args.evaluation_metric, cv=cv, verbose=1)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best {args.evaluation_metric} score: {grid_search.best_score_}")
    else:
        best_model = model
        best_model.fit(X, y)
    
    return best_model

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
    
    X_train, y_train, X_val, y_val = load_data(args.train_features, args.train_labels, args.val_features, args.val_labels)
    X_train_bal, y_train_bal = handle_imbalance(X_train, y_train, args.imbalance_strategy)
    
    best_model = train_and_optimize_model(X_train_bal, y_train_bal, args)
    
    log_results(best_model, X_val, y_val, logger)
    save_model(best_model)


