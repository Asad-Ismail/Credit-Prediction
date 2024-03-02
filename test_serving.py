import requests
import pandas as pd

# URL of the Flask app's predict endpoint
# Adjust the URL if your Flask app is hosted on a different port or host
PREDICT_URL = 'http://localhost:5000/predict'

# Path to the CSV file with validation features
VAL_FEATURES_PATH = 'processed_data/val_features.csv'

# Load validation features from the CSV file
val_features = pd.read_csv(VAL_FEATURES_PATH)

# Convert the first row of the validation features to a JSON payload
# You can loop over all rows if you want to send multiple requests
payload = {'features': val_features.iloc.to_dict()}

# Send the POST request with the JSON payload and receive the response
response = requests.post(PREDICT_URL, json=payload)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    predictions = response.json()
    print("Predictions received from the model:", predictions)
else:
    print("Error:", response.status_code, response.text)