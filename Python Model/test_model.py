import pandas as pd
import joblib

# Existing imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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

def simulate_quiz_session():
    # Simulate user's answers to the quiz
    quiz_questions = {
        "General Knowledge": [
            {"question": "Throwing candy wrappers, or anything from your vehicle windows is:",
             "choices": ["Prohibited", "Allowed", "Depends on the location"],
             "correct_answer": "Prohibited"},
            {"question": "Public Service Law prohibits a public utility driver to converse with his passengers to assure utmost attention to the road, specifically while the vehicle is:",
             "choices": ["In motion", "Stopped", "Parked"],
             "correct_answer": "In motion"},
            {"question": "The speed limit prescribed by law does not apply to a driver who is:",
             "choices": ["Operating an emergency vehicle", "Driving a private car", "On a highway"],
             "correct_answer": "Operating an emergency vehicle"},
            {"question": "If you have a driver's license, you can drive:",
             "choices": ["Any vehicle for which you are licensed", "Any vehicle", "Only private cars"],
             "correct_answer": "Any vehicle for which you are licensed"},
            {"question": "While driving, what document should you always take with you?",
             "choices": ["Driver's license", "Insurance papers", "Car registration"],
             "correct_answer": "Driver's license"}
        ],
        "Emergencies": [
            {"question": "What will happen when your front tire blows out?",
             "choices": ["The vehicle will pull sharply to the side of the blowout", "The vehicle will stop immediately", "Nothing significant"],
             "correct_answer": "The vehicle will pull sharply to the side of the blowout"},
            {"question": "What should you do when an ambulance comes up behind you flashing red lights and/or sounding its siren?",
             "choices": ["Pull over to the side and stop", "Speed up", "Ignore it"],
             "correct_answer": "Pull over to the side and stop"},
            {"question": "If you are the first to arrive at the scene of an accident, which of the following should you do:",
             "choices": ["Call emergency services", "Drive away", "Take photos"],
             "correct_answer": "Call emergency services"},
            {"question": "When a vehicle is stalled or disabled, the driver must park the vehicle on the shoulder of the road and:",
             "choices": ["Turn on hazard lights", "Leave it as is", "Call a tow truck"],
             "correct_answer": "Turn on hazard lights"},
            {"question": "When a vehicle starts to skid, what should the driver do?",
             "choices": ["Steer in the direction of the skid", "Brake hard", "Accelerate"],
             "correct_answer": "Steer in the direction of the skid"}
        ],
        "Handling and Driving": [
            {"question": "If a driver passes a blind person, he:",
             "choices": ["Must stop", "Can continue driving", "Should honk"],
             "correct_answer": "Must stop"},
            {"question": "When you make an abrupt move especially when you are on a wet and possibly slippery road, the following action can cause you to skid and lose control:",
             "choices": ["Sudden braking", "Gradual braking", "Maintaining speed"],
             "correct_answer": "Sudden braking"},
            {"question": "It is not considered safe driving on an expressway when:",
             "choices": ["You drive below the minimum speed limit", "You drive at the speed limit", "You drive above the speed limit"],
             "correct_answer": "You drive below the minimum speed limit"},
            {"question": "On a wet road, you must:",
             "choices": ["Reduce your speed", "Increase your speed", "Maintain speed"],
             "correct_answer": "Reduce your speed"},
            {"question": "Describes the thinking of a defensive driver:",
             "choices": ["Anticipates potential hazards and drives accordingly", "Drives aggressively", "Ignores other drivers"],
             "correct_answer": "Anticipates potential hazards and drives accordingly"}
        ],
        "Parking": [
            {"question": "When parking downhill, you should turn your front wheels:",
             "choices": ["Towards the curb", "Away from the curb", "Straight"],
             "correct_answer": "Towards the curb"},
            {"question": "What light shall be used when vehicles are parked on the highway at night?",
             "choices": ["Parking lights", "Headlights", "No lights"],
             "correct_answer": "Parking lights"},
            {"question": "When you are parked at the side of the road at night, you must:",
             "choices": ["Turn on your parking lights", "Leave your headlights on", "Turn off all lights"],
             "correct_answer": "Turn on your parking lights"},
            {"question": "Parking is prohibited:",
             "choices": ["In front of a driveway", "On the side of the road", "In a parking lot"],
             "correct_answer": "In front of a driveway"},
            {"question": "The vehicle is parked if:",
             "choices": ["The vehicle is stationary and the engine is off", "The vehicle is stationary with the engine running", "The vehicle is moving slowly"],
             "correct_answer": "The vehicle is stationary and the engine is off"}
        ],
        "Road Position": [
            {"question": "It is not a safe place to overtake in an/a:",
             "choices": ["Intersection", "Straight road", "Highway"],
             "correct_answer": "Intersection"},
            {"question": "If the brake light of the vehicle in front of you is lit up, you should:",
             "choices": ["Prepare to slow down", "Speed up", "Ignore it"],
             "correct_answer": "Prepare to slow down"},
            {"question": "If the vehicle's headlight in front of you is blinding your eyes, what should you do?",
             "choices": ["Look to the right edge of the road", "Look straight ahead", "Close your eyes"],
             "correct_answer": "Look to the right edge of the road"},
            {"question": "What should you do to combat fatigue and sleepiness during a long road trip?",
             "choices": ["Take regular breaks", "Keep driving", "Drink coffee continuously"],
             "correct_answer": "Take regular breaks"},
            {"question": "Being passed is a normal part of driving and should not be taken as an insult to one’s ability, you should:",
             "choices": ["Maintain your speed and allow the vehicle to pass", "Speed up", "Slow down abruptly"],
             "correct_answer": "Maintain your speed and allow the vehicle to pass"}
        ],
        "Violations and Penalties": [
            {"question": "You were flagged down due to a noisy muffler of your motorcycle, what will you do?",
             "choices": ["Have it repaired immediately", "Ignore it", "Continue driving"],
             "correct_answer": "Have it repaired immediately"},
            {"question": "How many days are given for you to settle the case to get your license back if you get caught?",
             "choices": ["5 days", "10 days", "15 days"],
             "correct_answer": "5 days"},
            {"question": "It refers to an act penalizing a person under the influence of alcohol, dangerous drugs, and similar substances, and for other purposes:",
             "choices": ["Anti-Drunk and Drugged Driving Act", "Clean Air Act", "Traffic Code"],
             "correct_answer": "Anti-Drunk and Drugged Driving Act"},
            {"question": "A type of field sobriety test that requires the driver to walk heel-to-toe along a straight line for nine (9) steps, turn at the end, and return to the point of origin without any difficulty. What is this test called?",
             "choices": ["Walk and Turn Test", "One-Leg Stand Test", "Horizontal Gaze Nystagmus Test"],
             "correct_answer": "Walk and Turn Test"},
            {"question": "A privately registered car utilized for hire and carriage of passengers or cargoes is a colorum vehicle and is prohibited by law. Drivers caught operating such vehicles for the first time are penalized by:",
             "choices": ["A fine and impoundment of the vehicle", "A warning", "License suspension"],
             "correct_answer": "A fine and impoundment of the vehicle"}
        ]
    }

    user_answers = {}
    correct_answers = {}
    for category, questions in quiz_questions.items():
        user_answers[category] = {}
        correct_answers[category] = {}
        for question in questions:
            user_choice = input(f"{question['question']} Choices: {', '.join(question['choices'])}\nYour answer: ")
            user_answers[category][question['question']] = user_choice
            correct_answers[category][question['question']] = question['correct_answer']

    return user_answers, correct_answers

def calculate_scores(user_answers, correct_answers, points_per_question=5):
    user_scores = {}
    for category, questions in user_answers.items():
        score = 0
        for question, answer in questions.items():
            if answer == correct_answers[category][question]:
                score += points_per_question
        user_scores[category] = score
    return user_scores

def main():
    # Load pre-trained models and scaler
    performance_model = joblib.load('performance_classifier.pkl')
    incorrect_intention_model = joblib.load('incorrect_intention_classifier.pkl')
    scaler = joblib.load('scaler.pkl')

    # Simulate user's quiz session and get answers
    user_answers, correct_answers = simulate_quiz_session()

    # Calculate user scores based on their answers
    user_scores = calculate_scores(user_answers, correct_answers)

    # Generate insights based on user's scores
    insights = generate_insights(user_scores)
    for insight in insights:
        print(insight)

if __name__ == "__main__":
    main()
