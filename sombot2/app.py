from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lime.lime_text import LimeTextExplainer
import numpy as np
import google.generativeai as genai
import json
import re

# ---------------- Env / Keys ----------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env")

# ---------------- Flask ----------------
app = Flask(__name__, template_folder="templates")

# ---------------- Models ----------------
genai.configure(api_key=API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

# Emotion pipeline
emotion_model = pipeline(
    "text-classification",
    model="joeddav/distilbert-base-uncased-go-emotions-student",
    top_k=None
)

id2label = emotion_model.model.config.id2label
label_order = [id2label[i] for i in sorted(id2label.keys())]
label_to_idx = {lbl: i for i, lbl in enumerate(label_order)}

explainer = LimeTextExplainer(class_names=label_order, random_state=42)
vader_analyzer = SentimentIntensityAnalyzer()

conversation_history = []


# ---------------- Helpers ----------------
def vader_severity(text: str):
    compound = vader_analyzer.polarity_scores(text)["compound"]
    severity = int(round(1 + 9 * abs(compound)))
    return max(1, min(10, severity)), compound


def analyse_emotion(text: str):
    out = emotion_model(text[:512])
    # Handle both [[{...}]] and [{...}] formats
    scores = out[0] if isinstance(out[0], list) else out
    best = max(scores, key=lambda x: x["score"])
    mood = best["label"]
    return mood, scores


def predict_proba(texts):
    out = np.zeros((len(texts), len(label_order)), dtype=float)
    outputs = emotion_model(texts, truncation=True, max_length=512)
    for i, result in enumerate(outputs):
        scores = result if isinstance(result, list) else [result]
        for d in scores:
            out[i, label_to_idx[d["label"]]] = float(d["score"])
    return out


def suggest(convo_text: str, mood_text: str) -> str:
    prompt = f"""You are a supportive, trauma-informed mental wellness assistant.
Conversation:
{convo_text}

Detected mood: {mood_text}.
Give ONE short, kind, practical self-care suggestion (max 2 sentences)."""
    try:
        resp = gemini.generate_content(prompt)
        return (resp.text or "").strip() or "Take a short break and drink water."
    except Exception:
        return "Take a short break and drink water."


def explain_why_with_lime(mood: str, lime_words: list, suggestion: str) -> str:
    if lime_words:
        quoted = ", ".join(f'"{w}"' for w, _ in lime_words[:3])
        return f"This suggestion fits because the user expressed feelings through words like {quoted}, which support the detected mood of {mood}."
    else:
        return f"This suggestion fits the detected mood of {mood} based on the user's message."


# ---------------- Routes ----------------
@app.route("/")
def entry():
    return render_template("entry.html")

@app.route("/chat")
def chat_page():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Please type something so I can respond."})

    conversation_history.append({"role": "user", "content": message})

    # reply (for now skipping full safety for clarity)
    prompt = f"You are SOMBOT, a calm companion. User said: {message}. Reply kindly."
    try:
        resp = gemini.generate_content(prompt)
        reply = (resp.text or "").strip()
    except Exception:
        reply = "I'm here with you. Take a deep breath—you're not alone."

    conversation_history.append({"role": "bot", "content": reply})
    return jsonify({"reply": reply})


@app.route("/analyse", methods=["POST"])
def analyse():
    if not conversation_history:
        return jsonify({"analysis": "No conversation yet. Say something first."})

    user_text = " ".join(m["content"] for m in conversation_history if m["role"] == "user")

    try:
        mood, scores = analyse_emotion(user_text)
    except Exception as e:
        return jsonify({"analysis": f"Emotion model error: {e}"})

    severity, compound = vader_severity(user_text)

    try:
        pred_idx = label_to_idx[mood]
        exp = explainer.explain_instance(
            user_text,
            predict_proba,
            labels=[pred_idx],
            num_features=10,
            num_samples=1000,
        )
        weighted_words = exp.as_list(label=pred_idx)
        lime_text = ", ".join(f"{w}: {wt:+.3f}" for w, wt in weighted_words) if weighted_words else "No strong words found."
    except Exception as e:
        weighted_words = []
        lime_text = f"LIME error: {e}"

    suggestion = suggest(user_text, mood)
    why = explain_why_with_lime(mood, weighted_words, suggestion)

    pretty = (
        f"SOMBOT:\n"
        f"🧠 Mood detected: {mood}\n"
        f"📊 Severity: {severity}/10 (compound={compound:.2f})\n"
        f"💡 Suggestion: {suggestion}\n\n"
        f"🤔 Why? {why}\n\n"
        f"📝 LIME words: {lime_text}"
    )

    return jsonify({"analysis": pretty})


@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    app.run(debug=True)