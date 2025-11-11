from flask import Flask, request, Response
import requests, os, json

app = Flask(__name__)

@app.route("/literature_review", methods=["POST"])
def literature_review():
    """
    POST /literature_review
    Input: {"query": "Summarize recent works on diffusion models for 3D scene generation"}
    Output: SSE streaming Markdown text
    """
    query = request.json.get("query", "").strip()
    if not query:
        return Response("data: Error: query is empty\n\n", mimetype="text/event-stream")

    base_url = os.getenv("SCI_MODEL_BASE_URL")
    api_key = os.getenv("SCI_MODEL_API_KEY")
    model = os.getenv("SCI_LLM_MODEL", "deepseek-chat")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert research assistant. "
                    "When given a topic, write a structured, factual, and concise literature review "
                    "including key papers, methods, and trends. "
                    "Use Markdown formatting with sections like **Overview**, **Representative Works**, and **Trends**."
                )
            },
            {"role": "user", "content": query}
        ],
        "stream": True,
    }

    def generate():
        try:
            with requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload, stream=True, timeout=900) as r:
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        if decoded.startswith("data:"):
                            yield f"{decoded}\n\n"
        except Exception as e:
            yield f"data: [Error] {str(e)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/", methods=["GET"])
def healthcheck():
    """Simple healthcheck"""
    return {"status": "ok", "track": "literature_review"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
