from flask import Flask, request, jsonify
from backend.summarizer import summarize_text
from bias_detector import analyze_entity_sentiment
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data["text"]
    try:
        summary = summarize_text(text)
        bias = analyze_entity_sentiment(text)
        return jsonify({
            "summary": summary,
            "bias": bias
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
