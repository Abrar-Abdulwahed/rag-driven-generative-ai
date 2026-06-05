# Human Feedback (E2)
from helpers.llm_helper import call_llm_with_full_text, print_formatted_response


query = ["define a rag store"]
print("Sending request to Gemini via .env config...")
llm_response = call_llm_with_full_text(query)
print_formatted_response(llm_response)