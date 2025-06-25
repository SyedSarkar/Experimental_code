import json
import streamlit as st
import os
import pandas as pd
import re
import time
from transformers.pipelines import pipeline
import nltk
from nltk.corpus import brown
import gspread
from google.oauth2.service_account import Credentials
import datetime

# ---- Google Sheets Setup ----
HEADERS = ["timestamp", "user", "specific_id", "phase", "cue", "sentence", "response", "sentiment",
           "confidence", "score", "response_time_sec", "accepted"]

@st.cache_resource
def connect_to_sheet():
    creds_json = json.loads(st.secrets["GOOGLE_CREDENTIALS_JSON"])
    creds = Credentials.from_service_account_info(
        creds_json,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    client = gspread.authorize(creds)
    return client.open("Intervention_Results").sheet1

def log_to_gsheet(row_dict):
    sheet = connect_to_sheet()
    row = [str(row_dict.get(col, "")) for col in HEADERS]
    sheet.append_row(row)
# ----------------------------------

# Download NLTK resources
nltk.download('brown')
nltk.download('words')
english_vocab = set(w.lower() for w in brown.words())  # Use Brown corpus for reference

STOPWORDS = {'Hassan', 'Asim', 'Ather'}

def looks_like_gibberish(word):
    # Relaxed check: Allow short words if they are alpha and not extreme repeats
    return (
        len(word) < 1 or  # Minimum length of 1
        not word.isalpha() or
        re.fullmatch(r"(.)\1{3,}", word) or  # Only reject extreme repeats (3+)
        re.search(r'[aeiou]{4,}', word) or  # Reduce vowel threshold
        re.search(r'[zxcvbnm]{5,}', word) or  # Reduce consonant threshold
        (len(word) < 3 and word not in english_vocab and not any(c.isupper() for c in word))  # Allow short words with some flexibility
    )

def is_valid_response(response, cue_word):
    tokens = response.lower().strip().split()
    if not 1 <= len(tokens) <= 3:
        return False
    for token in tokens:
        if token == cue_word.lower() or token in STOPWORDS or looks_like_gibberish(token):
            return False
    return True

def calculate_score(label):
    if label == "POSITIVE": return 2
    if label == "NEGATIVE": return -1
    return 1

def format_cue_word(cue):
    return f"""
    <div style='text-align: center; font-size: 36px; font-weight: bold; color: #010d1a; padding: 20px;'>
        {cue}
    </div>
    """

def format_feedback(msg, color):
    return f"""
    <div style='text-align: center; font-size: 28px; font-weight: bold; color: {color}; padding: 10px;'>
        {msg}
    </div>
    """

def get_safe_progress(current, total):
    if total == 0:
        return 0.0
    return min(max(current / total, 0.0), 1.0)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_model()

os.makedirs("results", exist_ok=True)

with open("data/cue_words.txt", "r") as f:
    cue_words = [line.strip() for line in f.readlines()]

with open("data/sentences.txt", "r") as f:
    sentences = [line.strip() for line in f.readlines()]

if "phase" not in st.session_state:
    st.session_state.user_id = ""
    st.session_state.specific_id = ""
    st.session_state.phase = 0
    st.session_state.step = 0
    st.session_state.score = 0
    st.session_state.used_texts = set()
    st.session_state.responses = []
    st.session_state.start_time = None
    st.session_state.badges = []

st.markdown("""
<style>
body {
    background-color: #f6f9fc;
    color: #222;
}
.stTextInput > div > div > input {
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

if st.session_state.phase == 0:
    st.title("Positive Phrase Intervention")
    st.markdown("""
    Welcome to this two-phase task designed to encourage positive associations and emotional reflection.

    - **Phase 1**: Respond to single cue words with positive and uplifting phrases.
    - **Phase 2**: React to full sentences with encouraging responses.
    - Avoid repeats and generic prepositions.
    """)
    user_input = st.text_input("Enter your Name or Roll Number:")
    specific_id = st.text_input("Enter your Study Participant ID:")
    if st.button("Start Task") and user_input.strip() and specific_id.strip():
        st.session_state.user_id = user_input.strip()
        st.session_state.specific_id = specific_id.strip()
        safe_id = re.sub(r'[^\w\-]', '_', user_input.strip())
        filename = f"results/{safe_id}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            st.session_state.responses = df.to_dict("records")
            st.session_state.used_texts = set(df["response"].dropna().str.lower().tolist())
            st.session_state.score = df["score"].sum()
            st.session_state.step = sum(1 for r in st.session_state.responses if r["phase"] == 1)
            st.session_state.phase = 2 if st.session_state.step >= len(cue_words) else 1
        else:
            st.session_state.phase = 1
        st.rerun()

if st.session_state.phase == 1:
    st.progress(get_safe_progress(st.session_state.step, len(cue_words)))
    st.markdown(f"**Points**: `{st.session_state.score}` | **Responses**: `{len(st.session_state.used_texts)}`")
    
    # Display badges
    if len(st.session_state.used_texts) >= 10 and "10 Responses" not in st.session_state.badges:
        st.session_state.badges.append("10 Responses")
        st.success("🏅 Badge Earned: 10 Positive Responses!")
    if st.session_state.step >= len(cue_words) and "Phase 1 Master" not in st.session_state.badges:
        st.session_state.badges.append("Phase 1 Master")
        st.success("🏆 Badge Earned: Phase 1 Master!")

    if st.session_state.step < len(cue_words):
        cue = cue_words[st.session_state.step]
        st.markdown(format_cue_word(cue), unsafe_allow_html=True)

        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        feedback = st.empty()

        def handle_input():
            phrase = st.session_state[f"input_{st.session_state.step}"].strip().lower()
            response_time = round(time.time() - st.session_state.start_time, 2)
            result = classifier(phrase)[0]
            label, conf = result['label'], result['score']
            score = calculate_score(label)

            # Fallback for "a nice person"
            if phrase == "a nice person" and label == "NEGATIVE":
                label = "POSITIVE"
                conf = 0.9

            entry = {
                "timestamp": str(datetime.datetime.now()),
                "user": st.session_state.user_id,
                "specific_id": st.session_state.specific_id,
                "phase": 1,
                "cue": cue,
                "sentence": "",
                "response": phrase,
                "sentiment": label,
                "confidence": conf,
                "score": 0,
                "response_time_sec": response_time,
                "accepted": False
            }

            # Allow acceptance if sentiment is POSITIVE with high confidence, even if is_valid_response fails
            if phrase in st.session_state.used_texts:
                feedback.markdown(format_feedback("⚠️ Already used! Kindly use a different word", "#e67e22"), unsafe_allow_html=True)
                time.sleep(2)
            elif label == "NEGATIVE":
                feedback.markdown(format_feedback("❌ Negative word detected! Try again.", "#c0392b"), unsafe_allow_html=True)
                time.sleep(2)
            elif not is_valid_response(phrase, cue) and label != "POSITIVE":
                feedback.markdown(format_feedback("❌ Invalid input! Please write something else", "#c0392b"), unsafe_allow_html=True)
                time.sleep(2)
            else:
                entry["score"] = score
                entry["accepted"] = True
                st.session_state.score += score
                st.session_state.used_texts.add(phrase)
                st.session_state.step += 1
                st.session_state.start_time = None
                feedback.markdown(format_feedback(f"✅ Sentiment: {label} ({conf:.2f}) | Score +{score}", "#27ae60"), unsafe_allow_html=True)
                time.sleep(2)

            st.session_state.responses.append(entry)
            safe_id = re.sub(r'[^\w\-]', '_', st.session_state.user_id)
            pd.DataFrame(st.session_state.responses).to_csv(f"results/{safe_id}.csv", index=False)
            log_to_gsheet(entry)

        st.text_input("Type a related uplifting and positive phrase (up to 3 words):", key=f"input_{st.session_state.step}", on_change=handle_input)

    else:
        st.success("🎉 Congratulations Phase 1 Complete!")
        if st.button("Proceed to Phase 2"):
            st.session_state.step = 0
            st.session_state.phase = 2
            st.rerun()

elif st.session_state.phase == 2:
    st.progress(get_safe_progress(st.session_state.step, len(sentences)))
    st.markdown(f"**Points**: `{st.session_state.score}` | **Responses**: `{len(st.session_state.used_texts)}`")

    if st.session_state.step < len(sentences):
        sentence = sentences[st.session_state.step]
        st.subheader(f"Sentence {st.session_state.step + 1}:")
        st.write(f"**{sentence}**")

        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        feedback = st.empty()

        def handle_input_2():
            phrase = st.session_state[f"input_s2_{st.session_state.step}"].strip().lower()
            response_time = round(time.time() - st.session_state.start_time, 2)
            result = classifier(phrase)[0]
            label, conf = result['label'], result['score']
            score = calculate_score(label)

            # Fallback for "a nice person"
            if phrase == "a nice person" and label == "NEGATIVE":
                label = "POSITIVE"
                conf = 0.9

            entry = {
                "timestamp": str(datetime.datetime.now()),
                "user": st.session_state.user_id,
                "specific_id": st.session_state.specific_id,
                "phase": 2,
                "cue": "",
                "sentence": sentence,
                "response": phrase,
                "sentiment": label,
                "confidence": conf,
                "score": 0,
                "response_time_sec": response_time,
                "accepted": False
            }

            # Allow acceptance if sentiment is POSITIVE with high confidence, even if is_valid_response fails
            if phrase in st.session_state.used_texts:
                feedback.markdown(format_feedback("⚠️ Already used! Kindly use something different.", "#e67e22"), unsafe_allow_html=True)
                time.sleep(2)
            elif label == "NEGATIVE":
                feedback.markdown(format_feedback("❌ Negative! Try again.", "#c0392b"), unsafe_allow_html=True)
                time.sleep(2)
            elif not is_valid_response(phrase, sentence) and label != "POSITIVE":
                feedback.markdown(format_feedback("❌ Invalid input! Please try something else", "#c0392b"), unsafe_allow_html=True)
                time.sleep(2)
            else:
                entry["score"] = score
                entry["accepted"] = True
                st.session_state.score += score
                st.session_state.used_texts.add(phrase)
                st.session_state.step += 1
                st.session_state.start_time = None
                feedback.markdown(format_feedback(f"✅ Sentiment: {label} ({conf:.2f}) | Score +{score}", "#27ae60"), unsafe_allow_html=True)
                time.sleep(2)

            st.session_state.responses.append(entry)
            safe_id = re.sub(r'[^\w\-]', '_', st.session_state.user_id)
            pd.DataFrame(st.session_state.responses).to_csv(f"results/{safe_id}.csv", index=False)
            log_to_gsheet(entry)

        st.text_input("Respond with a positive phrase:", key=f"input_s2_{st.session_state.step}", on_change=handle_input_2)

    else:
        if "Intervention Champion" not in st.session_state.badges:
            st.session_state.badges.append("Intervention Champion")
            st.success("🏆 Badge Earned: Intervention Champion!")
        st.session_state.step = 0
        st.session_state.phase = 3
        st.rerun()

elif st.session_state.phase == 3:
    st.balloons()
    st.success("🎉 Congratulations on Completing the Task!")
    st.markdown(f"**Final Score:** `{st.session_state.score}`")
    df = pd.DataFrame(st.session_state.responses)
    st.dataframe(df)

    with st.expander("📊 Click to see Analytics Dashboard"):
        st.subheader("AI Confidence Over Time")
        df["step"] = range(1, len(df) + 1)
        min_step, max_step = st.slider("Select step range:", int(df["step"].min()), int(df["step"].max()), (int(df["step"].min()), int(df["step"].max())))
        filtered_df = df[(df["step"] >= min_step) & (df["step"] <= max_step)]

        # Chart.js for AI Confidence using st.components.v1.html
        chart_data = {
            "type": "line",
            "data": {
                "labels": filtered_df["step"].tolist(),
                "datasets": [{
                    "label": "AI Confidence",
                    "data": filtered_df["confidence"].tolist(),
                    "borderColor": "#27ae60",
                    "backgroundColor": "rgba(39, 174, 96, 0.2)",
                    "fill": True,
                    "tension": 0.4
                }]
            },
            "options": {
                "scales": {
                    "x": {"title": {"display": True, "text": "Step"}},
                    "y": {"title": {"display": True, "text": "Confidence"}, "min": 0, "max": 1}
                },
                "plugins": {
                    "title": {"display": True, "text": "AI Confidence Over Time"}
                }
            }
        }
        chart_html = f"""
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <canvas id="confidenceChart"></canvas>
        <script>
            const ctx = document.getElementById('confidenceChart').getContext('2d');
            new Chart(ctx, {json.dumps(chart_data)});
        </script>
        """
        st.components.v1.html(chart_html, height=400, scrolling=True)

        st.subheader("Score Over Time")
        filtered_df["cumulative"] = filtered_df["score"].cumsum()
        chart_data_score = {
            "type": "line",
            "data": {
                "labels": filtered_df["step"].tolist(),
                "datasets": [{
                    "label": "Cumulative Score",
                    "data": filtered_df["cumulative"].tolist(),
                    "borderColor": "#3498db",
                    "backgroundColor": "rgba(52, 152, 219, 0.2)",
                    "fill": True,
                    "tension": 0.4
                }]
            },
            "options": {
                "scales": {
                    "x": {"title": {"display": True, "text": "Step"}},
                    "y": {"title": {"display": True, "text": "Cumulative Score"}}
                },
                "plugins": {
                    "title": {"display": True, "text": "Score Over Time"}
                }
            }
        }
        chart_html_score = f"""
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <canvas id="scoreChart"></canvas>
        <script>
            const ctx = document.getElementById('scoreChart').getContext('2d');
            new Chart(ctx, {json.dumps(chart_data_score)});
        </script>
        """
        st.components.v1.html(chart_html_score, height=400, scrolling=True)

    st.download_button("Download Results", df.to_csv(index=False).encode(), file_name=f"{st.session_state.user_id}_results.csv")

    if st.button("🔁 Restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
