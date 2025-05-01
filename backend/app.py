from flask import Flask, request, jsonify
from summarizer import summarize_text
from bias_detector import analyze_entity_sentiment
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


warmup_text="""
Apple unveiled its vision for how computing will change in the future.

On Monday, Apple announced the Vision Pro mixed reality headset during the company's Worldwide Developers Conference at its Cupertino, Calif., headquarters.

The headset incorporates both augmented and virtual reality, requiring wearers only use their eyes, hands and voice to control. The headset will launch early next year for $3,499.

“Just as the Mac introduced us to personal computing, and iPhone introduced us to mobile computing, Apple Vision Pro introduces us to spatial computing," said Apple CEO Tim Cook said in a statement.

The Vision Pro includes a three-dimensional interface where apps appear to pop up within the room you occupy, while also still viewing your surroundings if you choose. A knob called the digital crown, also found on Apple Watch, controls how immersed users want to feel while wearing the headset.

The device will also support Apple's Magic Keyboard and Trackpad for those who want more traditional computing options.

With the headset, users can have browsing tabs arranged around them, stretch screens for a movie theater experience, or join video calls with spatial audio that can mimic an in-person interaction, according to Apple.

The device uses advanced eye tracking to let wearers use their eyes to highlight an app or other field. You can tap a finger to select or flick your wrist to scroll. If you view a search bar, once it's highlighted, you can speak to enter a query.

What's different about Vision Pro compared to other VR and AR headsets is EyeSight, a feature allowing other people to see the eyes of the person wearing the headset. When the wearer is immersed in an activity, the headset gives visual cues to others when they are busy. Apple said the feature is meant to keep users connected to their surroundings.

Apple also introduced iOS 17, the next operating system for iPhones. Features will include Contact Posters to give phone contacts more personality, live transcriptions of voice mails, NameDrop for quickly sharing contact information, and improvements to autocorrect for text messaging.

"""


summarize_text(warmup_text)
analyze_entity_sentiment(warmup_text)


@app.route("/", methods=["GET"])
def healthcheck():
    return "Backend is up", 200

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
        print("ERROR OCCURRED")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)