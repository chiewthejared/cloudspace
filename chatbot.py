# A program that handles input and output interaction between user and Vertex AI
# Vertex AI acts as a chatbot, fed with information from a Google Datastore containing documents with structured data (PDFs, JSONs, etc.)
# Created to follow the concepts of Google AgentSpace
# Created by Jared Chiew

import functions_framework
import json
import os
import traceback
import google.genai as genai
from google.generativeai.types import Tool, VertexAISearch

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT", "cspace-jared")
LOCATION = os.environ.get("GCP_LOCATION", "global")
MODEL_NAME = "gemini-1.5-flash-001"

# --- Tool and Model Configuration ---
# For this SDK, the tool configuration is a dictionary
tool = Tool(
    retrieval=genai_types.Retrieval(
        source=genai.VertexAISearch(
            datastore=f"projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/music-files-chatbot_1752550034674_gcs_store"
        )
    )
)

# --- Model Initialization ---
# Initialize the client first
genai.configure(
    api_key=None, # No API key needed when running on Google Cloud
    transport="vertex_ai",
    location=LOCATION,
    project_id=PROJECT_ID,
)

# Get the model, passing the tools directly to it
model = genai.GenerativeModel(
    MODEL_NAME,
    tools=[tool]
)


@functions_framework.http
def chatbot_webhook(request):
    """
    HTTP Cloud Function that processes incoming requests for the chatbot.
    """
    # Set CORS headers
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*'}

    # --- Request Parsing ---
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            raise ValueError("Invalid JSON in request body or empty body.")

        user_query = request_json.get("text")
        if not user_query:
            # Fallback for Dialogflow/Agent Builder format
            user_query = request_json.get("queryInput", {}).get("text", {}).get("text")

        if not user_query:
             raise ValueError("User query not found in request body.")

    except ValueError as e:
        print(f"Error parsing request: {e}")
        return json.dumps({"fulfillmentText": f"Error: {e}"}), 400, headers

    print(f"Received user query: '{user_query}'")

    # --- Model Invocation ---
    try:
        # Start a chat session to maintain context if needed (good practice)
        chat = model.start_chat()

        # Send the user's message to the model
        response = chat.send_message(user_query)

        answer_text = response.text
        print(f"Model raw response text: {answer_text}")

        # --- Response Formatting ---
        response_payload = {
            "fulfillmentText": answer_text
        }
        return json.dumps(response_payload), 200, {"Content-Type": "application/json", **headers}

    except Exception as e:
        print(f"An error occurred during model invocation: {e}")
        traceback.print_exc()
        error_message = "I apologize, but I encountered an internal error. Please check the function logs for details."
        error_payload = {"fulfillmentText": error_message}
        return json.dumps(error_payload), 500, {"Content-Type": "application/json", **headers}


