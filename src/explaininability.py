import pandas as pd
from typing import Tuple, Any
import joblib
import shap
import matplotlib.pyplot as plt

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
    y_val = pd.read_csv(labels_path).squeeze()  # Convert DataFrame to Series if y_val is a single column
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
    return model.predict(X_val)

def explain_model_predictions(model: Any, X_val: pd.DataFrame, output_image_path: str) -> None:
    """
    Compute and plot SHAP values for the validation set to explain the model's predictions.
    Save the plot as an image.

    Parameters:
    model (Any): The trained model.
    X_val (pd.DataFrame): The validation features.
    output_image_path (str): The file path to save the SHAP values plot image.
    """
    # Initialize SHAP explainer and compute SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_val)

    # Plot the SHAP values for the validation set and save the plot as an image
    shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()

# Example usage:
if __name__ == "__main__":
    # Define file paths
    model_path = 'model.pkl'  # This can be an XGBoost or RandomForest model
    val_features_path = 'processed_data/val_features.csv'
    val_labels_path = 'processed_data/val_labels.csv'
    shap_output_image_path = 'shap_summary_plot.png'
    
    # Load the model and validation data
    model = load_model(model_path)
    X_val, y_val = load_validation_data(val_features_path, val_labels_path)
    
    # Generate predictions
    y_pred = predict(model, X_val)
    
    # Explain model predictions and save the plot as an image
    explain_model_predictions(model, X_val, shap_output_image_path)