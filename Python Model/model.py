# Testing script (run separately from your web application)
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import requests

# 1. Model Training and Evaluation
data = {
    'Q1': [1, 0, 1, 1, 0],
    'Q2': [0, 1, 0, 1, 1],
    'Q3': [1, 1, 0, 0, 1],
    'Q4': [0, 1, 1, 0, 1],
    'Q5': [1, 0, 1, 1, 0],

    'Recommendation': [
        'Review Q1: Standard speed limit in a school zone',
        'Review Q2: Yield to pedestrians in a crosswalk',
        'Review Q3: Understanding traffic signals',
        'Review Q4: Right-of-way rules',
        'Review Q5: Safe following distance'
    ]
}

df = pd.DataFrame(data)

le = LabelEncoder()
df['Recommendation'] = le.fit_transform(df['Recommendation'])

X = df.drop('Recommendation', axis=1)
y = df['Recommendation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

joblib.dump(model, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')

model_loaded = joblib.load('model.pkl')
le_loaded = joblib.load('label_encoder.pkl')

assert model_loaded.predict(X_test).all() == model.predict(X_test).all(), "Model loading failed!"
assert (le_loaded.classes_ == le.classes_).all(), "Label encoder loading failed!"
print("Model and Label Encoder loaded successfully!")

# 2. Endpoint Testing
test_data = {
    'responses': [0, 1, 0, 1, 0, 1, 1, 0, 1, 1]  # Example responses
}

response = requests.post('http://127.0.0.1:5000/recommend', json=test_data)
print(response.json())