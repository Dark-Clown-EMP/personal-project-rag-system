import os
from dotenv import load_dotenv
import nest_asyncio
from llama_parse import LlamaParse
nest_asyncio.apply()
load_dotenv()
if not os.getenv("LLAMA_CLOUD_API_KEY"):
    raise ValueError("LLAMA_CLOUD_API_KEY not found. Check your .env file!")
print ('Initializing parser...')

parser = LlamaParse(result_type = 'markdown', verbose = True)

print('Parsing...')

documents = parser.load_data('./data/bcbs189.pdf')

print(f"Parsed {len(documents)} pages.")
print("--- CONTENT OF PAGE 1 ---")
print(documents[0].text[:1000]) # Printing first 1000 chars
