"""
It helps to test the LLM API endpoint with a simple request to ensure that the model is responding correctly. The test sends a message asking the model to reply with the word "ready" and prints the status code and response from the API.

"""


import os, requests
from dotenv import load_dotenv
load_dotenv()

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    json={"model": os.environ["MODEL_NAME"],
          "messages": [{"role": "user", "content": "Reply with the word ready."}]},
    timeout=30,
)
print(response.status_code)
print(response.json())