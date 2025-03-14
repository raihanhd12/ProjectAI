import requests

# Get a specific point by ID
point_id = "00755132-b2b6-4525-810e-3e3f97e94e01"
response = requests.get(
    f"http://localhost:6333/collections/documents/points/{point_id}")

# The response will contain both the vector and payload
vector_data = response.json()
print(vector_data["result"]["vector"])  # The numerical array
print(vector_data["result"]["payload"]["text"])  # The original text
