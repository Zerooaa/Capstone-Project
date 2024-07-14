from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and label encoder
model = joblib.load('model.pkl')
le = joblib.load('label_encoder.pkl')

# Define the specific recommendations for each question
recommendations = {
    0: 'Review Q1: Standard speed limit in a school zone',
    1: 'Review Q2: Yield to pedestrians in a crosswalk',
    2: 'Review Q3: Understanding traffic signals',
    3: 'Review Q4: Right-of-way rules',
    4: 'Review Q5: Safe following distance'
}

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_responses = np.array([data['responses']])
    prediction = model.predict(user_responses)
    recommendation = le.inverse_transform(prediction)[0]

    specific_recommendations = []
    for i, response in enumerate(data['responses']):
        if response == 0:  # Assuming 0 means the question was answered incorrectly
            specific_recommendations.append(recommendations[i])

    return jsonify({'recommendation': recommendation, 'specific_recommendations': specific_recommendations})

if __name__ == '__main__':
    app.run(debug=True)
