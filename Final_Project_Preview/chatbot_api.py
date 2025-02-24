from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Store API key securely
API_KEY = "AIzaSyDZw27lSK07rFzIT2_IKfMuaZ7qGthjVpg"  # Replace with your actual API key
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle chatbot queries and fetch AI-generated responses.
    """
    user_input = request.json.get("question")

    payload = {
        "contents": [{
            "parts": [{"text": user_input}]
        }]
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        data = response.json()

        bot_reply = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, I couldn't find an answer.")

        return jsonify({"answer": bot_reply})

    except Exception as e:
        return jsonify({"answer": f"Error fetching AI response: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
