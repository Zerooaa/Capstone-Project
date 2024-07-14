from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

model = None
le = None

# Get the absolute path to the directory containing this script
base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, 'model.pkl')
le_path = os.path.join(base_dir, 'label_encoder.pkl')

if os.path.exists(model_path) and os.path.exists(le_path):
    try:
        model = joblib.load(model_path)
        le = joblib.load(le_path)
        print("Model and Label Encoder loaded successfully!")
    except Exception as e:
        print(f"Error loading model or label encoder: {e}")
else:
    print(f"Model path: {model_path} exists: {os.path.exists(model_path)}")
    print(f"Label encoder path: {le_path} exists: {os.path.exists(le_path)}")
    print("One or both of the required files are missing.")

recommendations = {
    0: 'Review Q1: Standard speed limit in a school zone',
    1: 'Review Q2: Yield to pedestrians in a crosswalk',
    2: 'Review Q3: Understanding traffic signals',
    3: 'Review Q4: Right-of-way rules',
    4: 'Review Q5: Safe following distance'
}

@app.route('/recommend', methods=['POST'])
def recommend():
    if not model or not le:
        return jsonify({'error': 'Model or Label Encoder not loaded'}), 500
    
    try:
        data = request.json
        user_responses = np.array([data['responses']])
        prediction = model.predict(user_responses)
        recommendation = le.inverse_transform(prediction)[0]

        specific_recommendations = []
        for i, response in enumerate(data['responses']):
            if response == 0:  # Assuming 0 means the question was answered incorrectly
                specific_recommendations.append(recommendations[i])

        return jsonify({'recommendation': recommendation, 'specific_recommendations': specific_recommendations})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
