import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

for m in client.models.list():
    print(m.name)
