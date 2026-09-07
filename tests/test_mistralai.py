import os

from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

client = Mistral(
    api_key=os.environ["MISTRAL_API_KEY"]
)

response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {
            "role": "user",
            "content": "Dis bonjour."
        }
    ],
)

print(response.choices[0].message.content)