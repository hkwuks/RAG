from dotenv import load_dotenv
import os
from openai import AzureOpenAI,OpenAI
import requests
from bs4 import BeautifulSoup
import yaml

load_dotenv()

AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_ENDPOINT = ""
GPT4O_MODEL_NAME = "gpt-4o"
TEXT_EMBEDDING_3_LARGE_DEPLOYMENT_NAME = ""
AZURE_OPENAI_API_VERSION = ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oai = OpenAI(api_key=OPENAI_API_KEY)


url = 'http://en.wikipedia.org/wiki/Elon_Musk'
response = requests.get(url)
soup = BeautifulSoup(response.text,'html.parser')

if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.exists('data/elon.md'):
    elon = soup.text.split('\nSee also')[0]
    with open('data/elon.md','w') as f:
        f.write(elon)
else:
    with open('data/elon.md','r') as f:
        elon = f.read()

if not os.path.exists('data/graphrag'):


