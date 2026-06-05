import os
import textwrap
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
gemini_model = "gemini-2.5-flash"

def call_llm_with_full_text(itext):
    """Calls Gemini with the provided text input list."""
    text_input = '\n'.join(itext) if isinstance(itext, list) else itext
    prompt = f"Please elaborate on the following content:\n{text_input}"
    
    try:
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are an expert Natural Language Processing expert. You can explain, read the input and answer items.",
                temperature=0.1,
            ),
        )
        return response.text.strip()
    except Exception as e:
        return str(e)

def print_formatted_response(response):
    """Prints the response wrapped at 80 characters."""
    wrapper = textwrap.TextWrapper(width=80)
    wrapped_text = wrapper.fill(text=response)
    print("\nResponse:")
    print("--------------------")
    print(wrapped_text)
    print("--------------------\n")
