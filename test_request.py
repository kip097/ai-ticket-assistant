import requests

url = "http://127.0.0.1:8000/predict"

sample_ticket = {
    "text": "Почему не работает сайт?",
    "top_k": 3
}

response = requests.post(url, json=sample_ticket)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
