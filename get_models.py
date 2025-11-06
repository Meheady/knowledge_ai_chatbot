
import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.environ.get("GITHUB_TOKEN")
if not token:
    raise ValueError("GITHUB_TOKEN not found in .env file")

endpoint = "https://models.github.ai/inference/v1/catalog/models"

headers = {
    "Authorization": f"Bearer {token}"
}

response = requests.get(endpoint, headers=headers)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)
