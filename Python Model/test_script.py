import requests

# Test data
test_data = {
    'responses': [0, 1, 0, 1, 0]  # Example responses
}

response = requests.post('http://127.0.0.1:5000/recommend', json=test_data)
print(response.json())