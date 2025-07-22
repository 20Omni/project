import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime
import gdown

# 📁 Download BERT model folder from Google Drive (only once)
bert_folder = "bert-task-classifier"
if not os.path.exists(bert_folder):
    folder_id = "1_utCJCA8RJGuMPAmw8Xg0qj1laz07OX8"
    gdown.download_folder(id=folder_id, quiet=False, use_cookies=False)

# ✅ Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_folder)
model = BertForSequenceClassification.from_pretrained(bert_folder)
model.eval()

# ✅ Load updated files (with new names)
xgb_model = joblib.load("priority_xgboost (1).pkl")
priority_encoder = joblib.load("priority_label_encoder (1).pkl")
vectorizer = joblib.load("priority_tfidf_vectorizer (1).pkl")

# ✅ Load your CSV containing users and tasks
df = pd.read_csv("your_dataset.csv")  # Replace with actual filename
users = df['assigned_to'].unique()
user_load = df['assigned_to'].value_counts().to_dict()

# ✅ Load BERT label encoder (for category prediction)
task_labels = joblib.load("task_label_encoder.pkl")  # Use your actual filename

# 🎯 Streamlit App UI
st.title("🚀 AI Task Assignment Dashboard")

task_desc = st.text_area("📝 Enter Task Description")
deadline_input = st.date_input("📅 Select Deadline")

if st.button("🔮 Predict & Assign"):
    if not task_desc.strip():
        st.warning("Please enter a task description.")
    else:
        # 🧠 Category using BERT
        inputs = tokenizer(task_desc, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class = torch.argmax(logits, dim=1).item()
        predicted_category = task_labels.inverse_transform([predicted_class])[0]

        # ⚡ Priority using XGBoost
        tfidf_input = vectorizer.transform([task_desc])
        predicted_priority = priority_encoder.inverse_transform(xgb_model.predict(tfidf_input))[0]

        # 👤 Assign to user with least load + urgency
        today = datetime.today().date()
        days_left = (deadline_input - today).days
        deadline_score = max(0, 10 - days_left)

        scores = {user: user_load.get(user, 0) + deadline_score for user in users}
        assigned_user = min(scores, key=scores.get)

        # ✅ Final Output
        st.success(f"📂 **Predicted Category**: {predicted_category}")
        st.success(f"⚠️ **Predicted Priority**: {predicted_priority}")
        st.success(f"👤 **Assigned To**: {assigned_user}")


