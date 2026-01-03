import streamlit as st
import numpy as np
import joblib


# Load trained SVM model and scaler

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# Streamlit Page Configuration

st.set_page_config(
    page_title="Spotify Hit Song Predictor",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Spotify Hit Song Predictor")
st.write("Enter the audio features to predict whether the song is a **Hit** or **Not a Hit**.")

# User Input Fields

danceability = st.number_input("Danceability (0–1)", 0.0, 1.0, 0.5)
energy = st.number_input("Energy (0–1)", 0.0, 1.0, 0.5)
key = st.number_input("Key (0–11)", 0, 11, 5)
loudness = st.number_input("Loudness (dB)", -60.0, 5.0, -10.0)
mode = st.selectbox("Mode (0 = Minor, 1 = Major)", [0, 1])
speechiness = st.number_input("Speechiness (0–1)", 0.0, 1.0, 0.05)
acousticness = st.number_input("Acousticness (0–1)", 0.0, 1.0, 0.3)
instrumentalness = st.number_input("Instrumentalness (0–1)", 0.0, 1.0, 0.0)
liveness = st.number_input("Liveness (0–1)", 0.0, 1.0, 0.2)
valence = st.number_input("Valence (0–1)", 0.0, 1.0, 0.5)
tempo = st.number_input("Tempo (BPM)", 50.0, 250.0, 120.0)
duration_ms = st.number_input("Duration (milliseconds)", 50000, 500000, 200000)
time_signature = st.number_input("Time Signature", 1, 7, 4)
chorus_hit = st.number_input("Chorus Hit (seconds)", 0.0, 100.0, 40.0)
sections = st.number_input("Number of Sections", 1, 20, 10)


# Prediction

if st.button("Predict"):
    input_data = np.array([[danceability, energy, key, loudness, mode,
                            speechiness, acousticness, instrumentalness,
                            liveness, valence, tempo, duration_ms,
                            time_signature, chorus_hit, sections]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("HIT")
    else:
        st.error("NOT HIT")