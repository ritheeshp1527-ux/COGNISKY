from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
import os
from dotenv import load_dotenv
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lime.lime_text import LimeTextExplainer
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
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
app.config['SECRET_KEY'] = 'some_secure_secret_cognisky'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cognisky.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    chat_sessions = db.relationship('ChatSession', backref='author', lazy=True)
    cognitive_results = db.relationship('CognitiveSurveyResult', backref='author', lazy=True)

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    mood = db.Column(db.String(50), nullable=True)
    severity = db.Column(db.Integer, nullable=True)
    messages = db.relationship('Message', backref='chat', lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)

class CognitiveSurveyResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_taken = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    problem_solving = db.Column(db.Float, nullable=False)
    numerical = db.Column(db.Float, nullable=False)
    emotional_intelligence = db.Column(db.Float, nullable=False)
    hypothesis = db.Column(db.Float, nullable=False)
    memory = db.Column(db.Float, nullable=False)
    total_score = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(20), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


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

# global history removed in favor of db
# conversation_history = []


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
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template("entry.html")

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get("username").strip()
        password = request.form.get("password")
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'danger')
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            flash('Your account has been created! You are now able to log in', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get("username").strip()
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route("/logout")
def logout():
    logout_user()
    session.pop('active_chat_id', None)
    return redirect(url_for('entry'))

@app.route("/dashboard")
@login_required
def dashboard():
    sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.date_created.desc()).all()
    surveys = CognitiveSurveyResult.query.filter_by(user_id=current_user.id).order_by(CognitiveSurveyResult.date_taken.desc()).all()
    return render_template("dashboard.html", sessions=sessions, surveys=surveys)

@app.route("/survey", methods=["GET"])
@login_required
def survey():
    return render_template("survey.html")

@app.route("/submit_survey", methods=["POST"])
@login_required
def submit_survey():
    # Helper to calculate average directly from the raw form since User wants (sum/20)*5 mapping.
    # Actually, the user mapping: "Section Score = Sum of 4 questions. Normalized Score = (Score / 20) * 5".
    # Since questions are on Likert Scale (1..5), 4 questions max sum = 20. Thus (sum/20)*5 is simply the average of the 4 questions.
    def compute_category(avg):
        if avg < 2: return "Low"
        elif 2 <= avg <= 3.5: return "Medium"
        return "High"
        
    def calc_score(prefix, num_normal, num_reverse_idx):
        score = sum(int(request.form.get(f"{prefix}_{i}", 3)) for i in range(1, num_normal + 1))
        # Reverse scaling for the final trigger question
        score += (6 - int(request.form.get(f"{prefix}_{num_reverse_idx}", 3)))
        return score / float(num_normal + 1)
    
    ps = calc_score('ps', 8, 9)
    num = calc_score('num', 8, 9)
    ei = calc_score('ei', 8, 9)
    hyp = calc_score('hyp', 8, 9)
    mem = calc_score('mem', 8, 9)
    
    total = sum([ps, num, ei, hyp, mem]) / 5.0
    cat = compute_category(total)
    
    result = CognitiveSurveyResult(
        user_id=current_user.id,
        problem_solving=ps,
        numerical=num,
        emotional_intelligence=ei,
        hypothesis=hyp,
        memory=mem,
        total_score=total,
        category=cat
    )
    db.session.add(result)
    db.session.commit()
    return redirect(url_for('survey_result', result_id=result.id))

@app.route("/survey_result/<int:result_id>")
@login_required
def survey_result(result_id):
    res = CognitiveSurveyResult.query.get_or_404(result_id)
    if res.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    return render_template("survey_result.html", result=res)

@app.route("/chat")
@login_required
def chat_page():
    if 'active_chat_id' not in session:
        new_session = ChatSession(user_id=current_user.id)
        db.session.add(new_session)
        db.session.commit()
        session['active_chat_id'] = new_session.id
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Please type something so I can respond."})

    active_id = session.get('active_chat_id')
    if not active_id:
        return jsonify({"reply": "Session error. Please refresh the page."})
    
    user_msg = Message(session_id=active_id, role="user", content=message)
    db.session.add(user_msg)

    # reply (for now skipping full safety for clarity)
    prompt = f"You are COGNISKY, a calm companion. User said: {message}. Reply kindly."
    try:
        resp = gemini.generate_content(prompt)
        reply = (resp.text or "").strip()
    except Exception:
        reply = "I'm here with you. Take a deep breath—you're not alone."

    bot_msg = Message(session_id=active_id, role="bot", content=reply)
    db.session.add(bot_msg)
    db.session.commit()

    return jsonify({"reply": reply})


@app.route("/analyse", methods=["POST"])
@login_required
def analyse():
    active_id = session.get('active_chat_id')
    if not active_id:
        return jsonify({"analysis": "No active conversation to analyse."})
        
    user_messages = Message.query.filter_by(session_id=active_id, role="user").all()
    if not user_messages:
        return jsonify({"analysis": "No user conversation yet. Say something first."})

    chat_session = ChatSession.query.get(active_id)
    user_text = " ".join(m.content for m in user_messages)

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

    chat_session.mood = mood
    chat_session.severity = severity
    db.session.commit()

    return jsonify({
        "analysis": {
            "mood": mood,
            "severity": severity,
            "compound": round(compound, 2),
            "suggestion": suggestion,
            "why": why,
            "lime_words": [{"word": w, "weight": wt} for w, wt in weighted_words] if weighted_words else [],
            "lime_error": None if weighted_words else lime_text
        }
    })


@app.route("/reset", methods=["POST"])
@login_required
def reset():
    session.pop('active_chat_id', None)
    return jsonify({"status": "reset"})


with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)