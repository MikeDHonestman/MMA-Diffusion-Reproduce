from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key="AIzaSyCmIuYGM6LVnztvfJ48PPIHF1eTakUe5JY")

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="hi"
)
print(response.text) 
