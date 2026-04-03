import streamlit as st
import numpy as np
import tensorflow as tf

# -----------------------------
# Load model and vectorizer
# -----------------------------
st.set_page_config(page_title="Toxic Comment Detector", page_icon="💬", layout="centered")
st.title("💬 Toxic Comment Detector")
st.write("Type a comment below (like Instagram, Facebook, or YouTube) to check its toxicity levels:")

# Load saved model
model = tf.keras.models.load_model("toxic_comment_model.h5")
vectorizer = tf.keras.models.load_model("vectorizer.keras")

# Toxicity categories
class_names = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# -----------------------------
# Helper functions
# -----------------------------
def get_color(prob):
    """Return a color from green -> yellow -> red based on probability"""
    if prob < 0.3:
        return "#00FF00"  # green
    elif prob < 0.6:
        return "#FFFF00"  # yellow
    else:
        return "#FF0000"  # red

def display_bar(label, prob):
    """Display a progress bar with color"""
    st.markdown(f"**{label}: {prob*100:.1f}%**")
    st.progress(int(prob*100))

# -----------------------------
# User input
# -----------------------------
user_input = st.text_area("Enter your comment here:", height=100)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please type a comment to analyze!")
    else:
        # Vectorize comment
        vect_comment = vectorizer(np.array([user_input]))
        # Predict
        pred_prob = model.predict(vect_comment)[0]
        
        # Display results
        st.subheader("🔥 Toxicity Levels")
        hate_score_total = 0
        
        for i, label in enumerate(class_names):
            prob = pred_prob[i]
            hate_score_total += prob
            color = get_color(prob)
            
            # Custom bar with st.markdown using color
            st.markdown(f"""
            <div style="margin-bottom:5px;">
                <span style="font-weight:bold;">{label} - {prob*100:.1f}%</span>
                <div style="background-color:#ddd; border-radius:5px; width:100%; height:20px;">
                <div style="width:{prob*100}%; background-color:{color}; height:100%; border-radius:5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Overall hate level
        hate_level = (hate_score_total / len(class_names)) * 100
        st.markdown("---")
        st.subheader("💀 Overall Hate Level")
        st.markdown(f"""
        <div style="background-color:#ddd; border-radius:5px; padding:2px; margin-bottom:5px;">
            <div style="width:{hate_level}%; background-color:red; padding:5px 0; border-radius:5px; text-align:center; color:white;">
                Hate Level - {hate_level:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)