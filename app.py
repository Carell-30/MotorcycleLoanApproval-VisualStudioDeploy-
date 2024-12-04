from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model
model_path = 'motorcycle_loan_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = joblib.load(model_path)

# Initialize LabelEncoder for 'Loan Approval Status' column
label_encoder = LabelEncoder()
label_encoder.fit(['Approved', 'Rejected'])  # Assuming 2 classes for Loan Approval Status

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Convert incoming JSON to DataFrame
        input_data = pd.DataFrame([data])

        # Print received data for debugging
        print("Received DataFrame:")
        print(input_data)

        # Ensure the input data has the same features as the model
        model_features = model.feature_names_in_  # Fetch model's expected features
        print("Model features:", model_features)
        print("Input data columns:", input_data.columns)

        # Check for missing columns
        missing_columns = [col for col in model_features if col not in input_data.columns]
        if missing_columns:
            return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400

        # Check for extra columns that the model does not expect
        extra_columns = [col for col in input_data.columns if col not in model_features]
        if extra_columns:
            return jsonify({'error': f'Extra columns: {", ".join(extra_columns)}'}), 400

        # Prepare input data for prediction by ensuring it's in the correct order
        input_data = input_data[model_features]

        # Make prediction
        prediction = model.predict(input_data)
        prediction_label = label_encoder.inverse_transform(prediction)

        return jsonify({"prediction": prediction_label[0]})

    except KeyError as ke:
        return jsonify({'error': f'Missing key: {str(ke)}'}), 400
    except ValueError as ve:
        return jsonify({'error': f'Invalid value: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Use dynamic port for deployment, with a default of 5000 for local testing
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
