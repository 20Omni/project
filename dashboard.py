
import streamlit as st
import pandas as pd
import joblib
import datetime
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer

# Load BERT model and tokenizer
@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-task-classifier")
    model = BertForSequenceClassification.from_pretrained("bert-task-classifier")
    return tokenizer, model

tokenizer, bert_model = load_bert()
bert_model.eval()

# Load XGBoost model and label encoder
priority_model = joblib.load("priority_xgboost.pkl")
priority_encoder = joblib.load("priority_label_encoder.pkl")

# Load Task Label Encoder
task_label_encoder = joblib.load("task_label_encoder.pkl")

# Sample users from your dataset
users = ["User_1", "User_2", "User_3", "User_4"]

# Session state to keep workload count
if "user_workload" not in st.session_state:
    st.session_state.user_workload = {}

# Title
st.title("📋 AI-Powered Task Assignment Dashboard")

# Task Input Form
with st.form("task_form"):
    task_desc = st.text_area("📝 Enter Task Description")
    deadline = st.date_input("📅 Deadline", min_value=datetime.date.today())
    submitted = st.form_submit_button("🚀 Predict & Assign")

if submitted and task_desc.strip():
    # --- Priority Prediction with XGBoost ---
    tfidf = joblib.load("priority_tfidf_vectorizer.pkl")
    task_vector = tfidf.transform([task_desc])
    pred_priority_encoded = priority_model.predict(task_vector)[0]
    pred_priority = priority_encoder.inverse_transform([pred_priority_encoded])[0]

    # --- Task Classification with BERT ---
    inputs = tokenizer(task_desc, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_category = task_label_encoder.inverse_transform([predicted_class_id])[0]

    # --- Workload + Deadline-based Assignment ---
    today = datetime.date.today()
    days_left = (deadline - today).days
    deadline_score = max(0, 10 - days_left)  # Closer deadline = higher penalty

    user_scores = []
    for user in users:
        load = st.session_state.user_workload.get(user, 0)
        score = load + deadline_score
        user_scores.append((user, score))

    assigned_user = sorted(user_scores, key=lambda x: x[1])[0][0]
    st.session_state.user_workload[assigned_user] = st.session_state.user_workload.get(assigned_user, 0) + 1

    # --- Display Output ---
    st.success(f"✅ Task assigned to **{assigned_user}**")
    st.markdown(f"🔖 **Predicted Category**: `{predicted_category}`")
    st.markdown(f"🚦 **Predicted Priority**: `{pred_priority}`")
    st.markdown(f"📅 **Days Until Deadline**: `{days_left}`")

    st.subheader("📊 Current Workload")
    st.dataframe(pd.DataFrame.from_dict(st.session_state.user_workload, orient="index", columns=["Tasks Assigned"]))

# Reset Button
if st.button("🔁 Reset Workload"):
    st.session_state.user_workload = {}
    st.success("✅ Workload reset successfully!")

