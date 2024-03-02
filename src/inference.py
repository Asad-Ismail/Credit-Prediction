import pandas as pd
from typing import Tuple, Any
import joblib
import shap
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (recall_score, precision_score, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score)


def parse_args():
    parser = argparse.ArgumentParser(description="Model Inference.")
    parser.add_argument('--model-name', type=str, default='xgboost', help='Name of trained model.')
    parser.add_argument('--model-path', type=str, default='xgboost_hp_optimized.pkl', help='Path to the trained Model.')
    parser.add_argument('--val-features', type=str, default='processed_data/val_features.csv', help='Path to the training features.')
    parser.add_argument('--val-labels', type=str, default='processed_data/val_labels.csv', help='Path to the training labels.')
    return parser.parse_args()

def evaluate(y_true, y_pred, y_pred_proba,model_name):
    """
    Evaluate the model's performance and save the ROC and PR curves as image files.

    Parameters:
    y_true (array-like): True binary labels.
    y_pred (array-like): Binary predictions.
    y_pred_proba (array-like): Predicted probabilities for the positive class.
    roc_curve_path (str): File path to save the ROC curve image.
    pr_curve_path (str): File path to save the PR curve image.

    Returns:
    tuple: A tuple containing recall, precision, ROC AUC, and confusion matrix.
    """
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'figs/{model_name}_roc_curve.png')
    plt.close()

    # Plot PR curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)
    plt.figure()
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(f'figs/{model_name}_pr_curve.png')
    plt.close()

    # Plot confusion matrix
    classes=['Negative', 'Positive']
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False, xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'figs/{model_name}_cm.png')
    plt.close()
    
    return recall, precision, roc_auc, cm


def load_model(model_path: str) -> Any:
    """
    Load a pre-trained model from the specified file path.
    The model can be of any type that is compatible with joblib.

    Parameters:
    model_path (str): The file path to the trained model.

    Returns:
    Any: The loaded model.
    """
    return joblib.load(model_path)

def load_validation_data(features_path: str, labels_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load validation features and labels from the specified file paths.

    Parameters:
    features_path (str): The file path to the validation features.
    labels_path (str): The file path to the validation labels.

    Returns:
    Tuple[pd.DataFrame, pd.Series]: A tuple containing the validation features as a DataFrame and the labels as a Series.
    """
    X_val = pd.read_csv(features_path)
    y_val = pd.read_csv(labels_path).squeeze()
    return X_val, y_val

def predict(model: Any, X_val: pd.DataFrame) -> pd.Series:
    """
    Generate predictions using the provided model and validation features.

    Parameters:
    model (Any): The trained model.
    X_val (pd.DataFrame): The validation features.

    Returns:
    pd.Series: The predictions made by the model.
    """
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred,proba

def explain_model_predictions(model: Any, X_val: pd.DataFrame, output_image_path: str) -> None:
    """
    Compute and plot SHAP values for the validation set to explain the model's predictions.
    Save the plot as an image.

    Parameters:
    model (Any): The trained model.
    X_val (pd.DataFrame): The validation features.
    output_image_path (str): The file path to save the SHAP values plot image.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_val)

    shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    args=parse_args()
    
    # Load the model and validation data
    model = load_model(args.model_path)
    X_val, y_val = load_validation_data(args.val_features, args.val_labels)
    
    # Generate predictions
    pred,proba = predict(model, X_val)

    ## Evaluate Model
    evaluate(y_val, pred, proba,args.model_name)
    
    # Explain model predictions and save the plot as an image
    explain_model_predictions(model, X_val, output_image_path=f'figs/{args.model_name}_shap_summary_plot.png')