import requests
import pandas as pd
from typing import List, Dict, Any

def load_validation_features(csv_path: str) -> pd.DataFrame:
    """
    Load validation features from a CSV file into a Pandas DataFrame.

    Parameters:
    csv_path (str): The file path to the CSV containing validation features.

    Returns:
    pd.DataFrame: A DataFrame containing the validation features.
    """
    return pd.read_csv(csv_path)

def create_payloads(features_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame of features into a list of JSON payloads suitable for API requests.

    Parameters:
    features_df (pd.DataFrame): The DataFrame containing the features to convert.

    Returns:
    List[Dict[str, Any]]: A list of dictionaries, each representing a JSON payload for a single observation.
    """
    return [{'features': row.to_dict()} for _, row in features_df.iterrows()]

def send_prediction_request(url: str, payload: Dict[str, Any]) -> requests.Response:
    """
    Send a POST request to a prediction API endpoint with a JSON payload.

    Parameters:
    url (str): The URL of the prediction API endpoint.
    payload (Dict[str, Any]): The JSON payload containing the input features.

    Returns:
    requests.Response: The response object returned by the requests.post call.
    """
    return requests.post(url, json=payload)

def log_predictions(response: requests.Response) -> None:
    """
    Log the predictions received from the model or the error if the request was unsuccessful.

    Parameters:
    response (requests.Response): The response object from the prediction API request.
    """
    if response.status_code == 200:
        predictions = response.json()
        print("Predictions received from the model:", predictions)
    else:
        print("Error:", response.status_code, response.text)


if __name__ == "__main__":
    PREDICT_URL = 'http://localhost:5000/predict'  
    VAL_FEATURES_PATH = 'processed_data/val_features.csv'  

    # Load validation features and create payloads
    val_features = load_validation_features(VAL_FEATURES_PATH)
    payloads = create_payloads(val_features)

    # Randmoly send the 10th feature row
    response = send_prediction_request(PREDICT_URL, payloads[10])

    # Log the results
    log_predictions(response)