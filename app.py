import os
from flask import Flask, request, jsonify, render_template
from google import genai

app = Flask(__name__)

# Hosted LLM instead of a local CTransformers model file.
# Vercel functions have no room/time to load a multi-GB local model,
# so we call Google's Gemini API directly instead. Get a key at
# https://aistudio.google.com/apikey (has a free tier) and set
# GOOGLE_API_KEY in your Vercel project's Environment Variables settings.
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

TEMPLATE = """Write an email with {style} style and includes topic: {email_topic}.

Sender: {sender}
Recipient: {recipient}

Email Text:
"""


def getLLMResponse(form_input, email_sender, email_recipient, email_style):
    formatted_prompt = TEMPLATE.format(
        email_topic=form_input,
        sender=email_sender,
        recipient=email_recipient,
        style=email_style,
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=formatted_prompt,
        config={"temperature": 0.01, "max_output_tokens": 1024},
    )
    return response.text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True) or {}
        email_topic = data.get("email_topic", "")
        sender = data.get("sender", "")
        recipient = data.get("recipient", "")
        style = data.get("style", "Formal")

        if not email_topic:
            return jsonify({"error": "email_topic is required"}), 400

        result_text = getLLMResponse(email_topic, sender, recipient, style)
        return jsonify({"email": result_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
