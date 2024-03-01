from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model when the application starts
model = joblib.load('model.pkl')

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
    app.run(debug=True)