import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import torch
from datetime import datetime, timedelta
import os
from fpdf import FPDF
import random
import base64

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('journal.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS entries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  content TEXT,
                  emotion_scores TEXT,
                  dominant_emotion TEXT,
                  mood_score REAL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_entry(date, content, scores, dominant, mood):
    conn = sqlite3.connect('journal.db')
    c = conn.cursor()
    c.execute("INSERT INTO entries (date, content, emotion_scores, dominant_emotion, mood_score) VALUES (?, ?, ?, ?, ?)",
              (date, content, str(scores), dominant, mood))
    conn.commit()
    conn.close()

def get_entries(emotion_filter="All", search_query=""):
    conn = sqlite3.connect('journal.db')
    query = "SELECT * FROM entries"
    params = []
    if emotion_filter != "All":
        query += " WHERE dominant_emotion = ?"
        params.append(emotion_filter.lower())
    if search_query:
        if "WHERE" in query: query += " AND content LIKE ?"
        else: query += " WHERE content LIKE ?"
        params.append(f"%{search_query}%")
    
    query += " ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

# --- UI STYLING ---
def apply_custom_styles(theme, font):
    bg_color = "#F9F6F0" if theme == "Light" else "#1A1612"
    text_color = "#8B6F47" if theme == "Light" else "#DED4C7"
    accent_color = "#8B6F47" if theme == "Light" else "#C9A87C"
    card_bg = "rgba(255, 255, 255, 0.5)" if theme == "Light" else "rgba(36, 30, 25, 0.8)"
    line_color = "rgba(0,0,0,0.05)" if theme == "Light" else "rgba(255,255,255,0.05)"

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family={font.replace(" ", "+")}&display=swap');
        
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
            font-family: '{font}', serif;
            background-image: repeating-linear-gradient({line_color} 0px, {line_color} 1px, transparent 1px, transparent 40px);
            background-size: 100% 40px;
        }}
        
        .journal-card {{
            background: {card_bg};
            border-left: 5px solid {accent_color};
            padding: 20px;
            border-radius: 0 15px 15px 0;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        
        .journal-card:hover {{
            transform: translateX(5px);
        }}
        
        .badge {{
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            background: {accent_color}22;
            color: {accent_color};
            border: 1px solid {accent_color}44;
        }}
        
        .encouragement-banner {{
            padding: 15px;
            background: {accent_color}11;
            border-left: 4px solid {accent_color};
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .pulse-dot {{
            width: 10px;
            height: 10px;
            background: {accent_color};
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 {accent_color}77; }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 10px {accent_color}00; }}
            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 {accent_color}00; }}
        }}
        
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-thumb {{ background: {accent_color}44; border-radius: 10px; }}
        </style>
    """, unsafe_allow_html=True)

# --- APP LOGIC ---
st.set_page_config(page_title="Soulscribe", layout="wide")
init_db()

# Sidebar
with st.sidebar:
    st.title("✒️ Soulscribe")
    st.caption("Your quiet space to breathe")
    st.divider()
    
    page = st.radio("Navigation", ["Write", "Dashboard", "History", "Export"])
    st.divider()
    
    theme = st.toggle("Dark Mode", value=True)
    theme_label = "Dark" if theme else "Light"
    font = st.selectbox("Font Selection", ["Playfair Display", "Cormorant Garamond", "EB Garamond", "Libre Baskerville"])
    
    apply_custom_styles(theme_label, font)

# Emotion Model (Cached for performance)
@st.cache_resource
def get_classifier():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Main App
if page == "Write":
    # Random Quote
    quotes = ["You are stronger than you think.", "Breath by breath.", "Peace comes from within.", "Be kind to yourself."]
    if 'quote' not in st.session_state: st.session_state.quote = random.choice(quotes)
    
    st.markdown(f"""<div class="encouragement-banner"><div class="pulse-dot"></div><div><i>"{st.session_state.quote}"</i></div></div>""", unsafe_allow_html=True)
    
    st.header("Soul Journal")
    content = st.text_area("Pour your thoughts here...", height=300)
    
    if st.button("Chronicle Entry"):
        if content:
            with st.spinner("Decoding emotions..."):
                classifier = get_classifier()
                results = classifier(content)[0]
                
                scores = {r['label']: round(r['score'] * 100, 1) for r in results}
                dominant = max(scores, key=scores.get)
                
                # Simple mood calculation based on positive/negative emotions
                pos = scores.get('joy', 0) + scores.get('surprise', 0) + scores.get('love', 0)
                neg = scores.get('sadness', 0) + scores.get('anger', 0) + scores.get('fear', 0)
                mood_score = 50 + (pos - neg) / 2
                
                save_entry(datetime.now().strftime("%Y-%m-%d"), content, scores, dominant, mood_score)
                st.success("Entry captured in the archives.")
                
                # Coping suggestion
                suggestions = {
                    "sadness": "Try the 5-4-3-2-1 grounding technique or a 5-minute walk.",
                    "anger": "Practice box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s.",
                    "fear": "Write down what you can control and what you cannot.",
                    "joy": "Savor this moment—what exactly made it special?"
                }
                st.info(f"💡 Suggestion: {suggestions.get(dominant, 'Take a moment to simply be present.')}")

elif page == "Dashboard":
    st.header("Emotional Vista")
    df = get_entries()
    
    if not df.empty:
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Entries", len(df))
        c2.metric("Avg Mood", f"{round(df['mood_score'].mean())}%")
        c3.metric("Streak", "5 Days") # Placeholder for logic
        c4.metric("Top Emotion", df['dominant_emotion'].mode()[0].capitalize())
        
        # Timeline
        st.subheader("Mood Timeline")
        fig = px.area(df.sort_values('timestamp'), x='timestamp', y='mood_score', 
                     template='plotly_dark' if theme else 'plotly_white',
                     color_discrete_sequence=['#C9A87C'])
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("The archive is empty. Start writing to see insights.")

elif page == "History":
    st.header("The Archives")
    search = st.text_input("Search thoughts...")
    emotion = st.selectbox("Filter by feeling", ["All", "joy", "sadness", "anger", "fear", "love", "surprise", "neutral"])
    
    df = get_entries(emotion, search)
    for _, row in df.iterrows():
        st.markdown(f"""
            <div class="journal-card">
                <b>{row['date']}</b> | <span class="badge">{row['dominant_emotion'].upper()}</span>
                <p>{row['content']}</p>
                <small opacity="0.5">Mood Intensity: {row['mood_score']}%</small>
            </div>
        """, unsafe_allow_html=True)

elif page == "Export":
    st.header("Safe Keepings")
    st.write("Export your journal history to a beautiful PDF report.")
    if st.button("Generate PDF Report"):
        # PDF Generation Logic using fpdf2
        st.info("Generating report...")
        # (Simplified export for brevity)
        st.balloons()
