from http.server import BaseHTTPRequestHandler
import json
import os
from google import genai

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
        config={"temperature": 0.01, "max_output_tokens": 256},
    )
    return response.text


class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            email_topic = data.get("email_topic", "")
            sender = data.get("sender", "")
            recipient = data.get("recipient", "")
            style = data.get("style", "Formal")

            if not email_topic:
                raise ValueError("email_topic is required")

            result_text = getLLMResponse(email_topic, sender, recipient, style)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"email": result_text}).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
