from flask import Flask, request, jsonify
import pandas as pd
import joblib
import argparse

app = Flask(__name__)

def load_model(model_path: str):
    """
    Load a pre-trained model from the specified file path.

    Parameters:
    model_path (str): The file path to the trained model.

    Returns:
    The loaded model.
    """
    return joblib.load(model_path)

def parse_args():
    """
    Parse command-line arguments.

    Returns:
    Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Model Serving")
    parser.add_argument('--model-path', type=str, default='xgboost_hp_optimized.pkl', help='Path to the trained model.')
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()

# Load the model when the application starts
model = load_model(args.model_path)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for making predictions. Expects a POST request with a JSON payload
    containing the input features. Returns the predictions as JSON.
    """
    # Parse the JSON payload
    data = request.get_json(force=True)
    
    # Validate that the JSON payload is in the correct format
    if not isinstance(data, dict) or 'features' not in data:
        return jsonify({'error': 'Invalid input format. Expecting a JSON object with a "features" key.'}), 400
    
    # Convert the features into a DataFrame
    features = pd.DataFrame([data['features']])
    
    # Make predictions
    predictions = model.predict(features)
    
    # Convert predictions to a list for JSON serialization
    predictions = predictions.tolist()
    
    # Return the predictions as a JSON object
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=args.debug)