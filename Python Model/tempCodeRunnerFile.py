import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def load_and_preprocess_data(file_path):
    # Load the dataset
    quiz_data = pd.read_csv(file_path)

    # Encode the 'Performance' column
    label_encoder = LabelEncoder()
    quiz_data['Performance'] = label_encoder.fit_transform(quiz_data['Performance'])

    return quiz_data, label_encoder

def train_performance_model(X, y, label_encoder):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a single-label RandomForestClassifier for performance classification
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = rf_classifier.predict(X_test_scaled)

    # Evaluate the model
    classification_report_result = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("Performance Classification Report:\n", classification_report_result)

    return rf_classifier, scaler

def train_incorrect_intention_model(X, overall_scores, threshold=100):
    # Create a binary target variable for incorrect answers
    incorrect_intention = (overall_scores < threshold).astype(int)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, incorrect_intention, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a RandomForestClassifier for incorrect intention classification
    rf_incorrect_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_incorrect_classifier.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_incorrect = rf_incorrect_classifier.predict(X_test_scaled)

    # Evaluate the model
    incorrect_classification_report = classification_report(y_test, y_pred_incorrect)
    print("Incorrect Intention Classification Report:\n", incorrect_classification_report)

    return rf_incorrect_classifier, scaler

def generate_insights(user_scores, threshold=15):
    # Generate insights based on user scores
    insights = []
    
    # Find the category with the lowest score
    lowest_category = min(user_scores, key=user_scores.get)
    lowest_score = user_scores[lowest_category]
    
    # Add detailed recommendation for the category with the lowest score
    if lowest_score < threshold:
        insights.append(f"Your lowest score is in {lowest_category} with a score of {lowest_score}. Consider reviewing the material related to this topic in more detail.")
    
    # Generate general insights for other categories with scores below the threshold
    for category, score in user_scores.items():
        if category != lowest_category and score < threshold:
            insights.append(f"Your score in {category} is below {threshold}. Consider reviewing the material related to this topic.")

    return insights

def main():
    file_path = 'Python Model/Quiz-Sheet1.csv'
    quiz_data, label_encoder = load_and_preprocess_data(file_path)

    # Split features and target variable
    X = quiz_data.drop(columns=['UserID', 'Overall Score', 'Performance'])
    y = quiz_data['Performance']

    # Train models
    performance_model, scaler = train_performance_model(X, y, label_encoder)
    incorrect_intention_model, _ = train_incorrect_intention_model(X, quiz_data['Overall Score'])

    # Save the models and scaler
    joblib.dump(performance_model, 'performance_classifier.pkl')
    joblib.dump(incorrect_intention_model, 'incorrect_intention_classifier.pkl')
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == "__main__":
    main()
