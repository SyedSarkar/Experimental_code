import json
import streamlit as st
import os
import pandas as pd
import re
import random
import time
from transformers.pipelines import pipeline
import nltk
from nltk.corpus import brown, words
import gspread
from google.oauth2.service_account import Credentials
import datetime
from spellchecker import SpellChecker

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
english_vocab = set(w.lower() for w in words.words())  # Use NLTK words corpus
brown_vocab = set(w.lower() for w in brown.words())

# Custom list of valid hyphenated words
VALID_HYPHENATED_WORDS = {
    "self-aware", "well-being", "self-esteem", "self-confidence", "self-assured",
    "well-adjusted", "high-spirited", "self-reliant", "self-worth", "well-intentioned"
}

STOPWORDS = {'Hassan', 'Asim', 'Ather'}

@st.cache_resource
def init_spell_checker():
    checker = SpellChecker()
    checker.word_frequency.load_words(list(VALID_HYPHENATED_WORDS))
    return checker

spell = init_spell_checker()

def looks_like_gibberish(word):
    # Handle hyphenated words
    if '-' in word:
        if word.lower() in spell:
            return False
        parts = word.lower().split('-')
        if len(parts) != 2:  # Allow only one hyphen
            return True
        return not all(part.isalpha() for part in parts)
    
    # Standard gibberish checks for non-hyphenated words
    return (
        len(word) < 1 or
        not word.isalpha() or
        re.fullmatch(r"(.)\1{3,}", word) or
        re.search(r'[aeiou]{4,}', word) or
        re.search(r'[zxcvbnm]{5,}', word) or
        (len(word) < 3 and word.lower() not in english_vocab and not any(c.isupper() for c in word))
    )

def is_valid_response(response, cue_word):
    if cue_word is None:
        return False
    tokens = [response.strip()] if '-' in response else response.lower().strip().split()
    if len(tokens) != 1:  # Enforce single-word response (including hyphenated)
        return False
    token = tokens[0].lower()
    if token == cue_word.lower() or token in STOPWORDS or looks_like_gibberish(token):
        return False
    return token in spell or token in english_vocab

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

# Load cue words from Excel
df_cue = pd.read_excel("cue_words.xlsx")
cue_words_by_block = {
    "Block_1": df_cue[df_cue["Block"] == "Block_1"]["Threatening Word"].dropna().tolist(),
    "Block_2": df_cue[df_cue["Block"] == "Block_2"]["Threatening Word"].dropna().tolist(),
    "Block_3": df_cue[df_cue["Block"] == "Block_3"]["Threatening Word"].dropna().tolist(),
    "Block_4": df_cue[df_cue["Block"] == "Block_4"]["Threatening Word"].dropna().tolist()
}
for block in cue_words_by_block:
    cue_words_by_block[block] = list(dict.fromkeys(cue_words_by_block[block]))  # Remove duplicates

if "phase" not in st.session_state:
    st.session_state.user_id = ""
    st.session_state.specific_id = ""
    st.session_state.phase = "Start"
    st.session_state.step = 0
    st.session_state.score = 0
    st.session_state.used_texts = set()
    st.session_state.responses = []
    st.session_state.start_time = None
    st.session_state.badges = []
    st.session_state.current_cue = None
    st.session_state.last_response_accepted = False

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

if st.session_state.phase == "Start":
    st.title("Positive Phrase Intervention")
    st.markdown("""
    Welcome to this four-level task designed to encourage positive associations with threatening words.

    - Respond to each threatening cue word with a single positive or neutral word.
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
            st.session_state.step = sum(1 for r in st.session_state.responses if r["phase"] == "Block_1")
            st.session_state.phase = "Block_2" if st.session_state.step >= len(cue_words_by_block["Block_1"]) else "Block_1"
        else:
            st.session_state.phase = "Block_1"
        st.session_state.current_cue = random.choice(cue_words_by_block[st.session_state.phase]) if st.session_state.phase != "Start" else None
        st.rerun()

if st.session_state.phase in ["Block_1", "Block_2", "Block_3", "Block_4"]:
    level = st.session_state.phase
    total_words = len(cue_words_by_block[level])
    st.progress(get_safe_progress(st.session_state.step, total_words))
    st.markdown(f"**Points**: `{st.session_state.score}` | **Responses**: `{len(st.session_state.used_texts)}`")
    
    # Display badges
    if len(st.session_state.used_texts) >= 10 and "10 Responses" not in st.session_state.badges:
        st.session_state.badges.append("10 Responses")
        st.success("🏅 Badge Earned: 10 Positive Responses!")
    if st.session_state.step >= total_words and f"Level {level[-1]} Master" not in st.session_state.badges:
        st.session_state.badges.append(f"Level {level[-1]} Master")
        st.success(f"🏆 Badge Earned: Level {level[-1]} Master!")

    if st.session_state.step < total_words:
        import random
        if st.session_state.current_cue is None or st.session_state.last_response_accepted:
            available_cues = [w for w in cue_words_by_block[level] if w not in [r["cue"] for r in st.session_state.responses if r["phase"] == level]]
            if not available_cues:
                available_cues = cue_words_by_block[level]  # Reset if all used (for testing)
            st.session_state.current_cue = random.choice(available_cues)
            st.session_state.last_response_accepted = False

        st.markdown(format_cue_word(st.session_state.current_cue), unsafe_allow_html=True)

        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        feedback = st.empty()

        def handle_input():
            try:
                phrase = st.session_state[f"input_{st.session_state.step}"].strip().lower()
                if st.session_state.current_cue is None:
                    feedback.markdown(format_feedback("❌ Error: Cue not set. Please restart.", "#c0392b"), unsafe_allow_html=True)
                    return

                response_time = round(time.time() - st.session_state.start_time, 2)
                result = classifier(phrase)[0]
                label, conf = result['label'], result['score']
                score = calculate_score(label)

                # Spell check
                misspelled = spell.unknown([phrase])
                if misspelled:
                    suggestion = spell.correction(phrase)
                    feedback.markdown(format_feedback(f"❌ Misspelled! Did you mean '{suggestion}'? Try again.", "#c0392b"), unsafe_allow_html=True)
                    #time.sleep(2)
                    return

                # Fallback for specific cases
                if phrase in ["hope", "peace", "strength"] and label == "NEGATIVE" and st.session_state.current_cue in ["Ridiculed", "Shame", "Rejected"]:
                    label = "POSITIVE"
                    conf = 0.9

                entry = {
                    "timestamp": str(datetime.datetime.now()),
                    "user": st.session_state.user_id,
                    "specific_id": st.session_state.specific_id,
                    "phase": level,
                    "cue": st.session_state.current_cue,
                    "sentence": "",
                    "response": phrase,
                    "sentiment": label,
                    "confidence": conf,
                    "score": 0,
                    "response_time_sec": response_time,
                    "accepted": False
                }

                if phrase in st.session_state.used_texts:
                    feedback.markdown(format_feedback("⚠️ Already used! Please try a different word.", "#e67e22"), unsafe_allow_html=True)
                    #time.sleep(2)
                elif not is_valid_response(phrase, st.session_state.current_cue):
                    feedback.markdown(format_feedback("❌ Invalid input! Please enter a single word.", "#c0392b"), unsafe_allow_html=True)
                    #time.sleep(2)
                elif label == "NEGATIVE" or (label != "POSITIVE" and not any(pos in phrase for pos in ["hope", "peace", "strength"] if pos != st.session_state.current_cue.lower())):
                    feedback.markdown(format_feedback("❌ Negative or unrelated! Try a positive/neutral word related to the cue.", "#c0392b"), unsafe_allow_html=True)
                    #time.sleep(2)
                else:
                    entry["score"] = score
                    entry["accepted"] = True
                    st.session_state.score += score
                    st.session_state.used_texts.add(phrase)
                    st.session_state.step += 1
                    st.session_state.last_response_accepted = True
                    feedback.markdown(format_feedback(f"✅ Correct | Score +{score}", "#27ae60"), unsafe_allow_html=True)
                    #time.sleep(2)

                st.session_state.responses.append(entry)
                safe_id = re.sub(r'[^\w\-]', '_', st.session_state.user_id)
                pd.DataFrame(st.session_state.responses).to_csv(f"results/{safe_id}.csv", index=False)
                log_to_gsheet(entry)
                st.session_state.start_time = None if entry["accepted"] else time.time()  # Reset only on failure
            except Exception as e:
                feedback.markdown(format_feedback(f"❌ Error: {str(e)}. Please restart.", "#c0392b"), unsafe_allow_html=True)
                st.session_state.start_time = time.time()

        st.text_input(f"Type a single positive/neutral word related to '{st.session_state.current_cue}':", key=f"input_{st.session_state.step}", on_change=handle_input)

    else:
        st.success(f"🎉 Well done! You've done a great job finishing Level {level[-1]}!")
        if level == "Block_4":
            st.session_state.phase = "End"
        else:
            next_level = f"Block_{int(level[-1]) + 1}"
            st.session_state.step = 0
            st.session_state.phase = next_level
        st.rerun()

elif st.session_state.phase == "End":
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
