from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500
        
    try:
        # Get JSON data
        data = request.get_json()
        
        # Check if JSON is empty or missing
        if not data:
            return jsonify({'error': 'No input data provided.'}), 400
        
        # Define expected features
        # Assuming the model was trained on the standard 13 features:
        # age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                             
        # Create a dictionary with default values if some features are missing (optional safety)
        # But for best results, we should extract them directly or error if missing
        input_data = {}
        for feature in expected_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            input_data[feature] = [data[feature]]
            
        # Convert to DataFrame
        df = pd.DataFrame(input_data)
        
        # Predict
        prediction = model.predict(df)[0]
        
        # Convert prediction to text
        if int(prediction) == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease Detected"
            
        return jsonify({'prediction': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
