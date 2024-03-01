# Prediction script

import pandas as pd
import joblib
import shap

# Load the model
model = joblib.load('xgb_model.pkl')

# Load validation data
X_val = pd.read_csv('processed_data/val_features.csv')
y_val = pd.read_csv('processed_data/val_labels.csv')

# Make predictions
y_pred = model.predict(X_val)


# Assuming the previous code snippets are part of this script

# Initialize SHAP explainer and compute SHAP values
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X_val)

# Plot the SHAP values for the validation set
shap.summary_plot(shap_values, X_val, plot_type="bar")
