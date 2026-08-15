import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Page configuration
st.set_page_config(page_title="Wine Quality Evaluator", page_icon="🍷", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 3rem; color: #8B0000; text-align: center; margin-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🍷 Wine Quality Classification</p>', unsafe_allow_html=True)

# --- 📌 FEATURE A: Dataset Upload (Test Data Only) ---
st.subheader("📤 Step 1: Upload Test Data")
uploaded_file = st.file_uploader("Upload your test data (CSV format)", type=['csv'])

# Pre-load training data (must be in the local repo)
@st.cache_data
def load_training_data():
    try:
        df = pd.read_csv('winequality-white.csv', sep=';')
        df['quality_binary'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
        X = df.drop(['quality', 'quality_binary'], axis=1)
        y = df['quality_binary']
        return X, y
    except Exception as e:
        st.error(f"Error loading training dataset: {e}")
        return None, None

X_train_full, y_train_full = load_training_data()

if uploaded_file is not None and X_train_full is not None:
    try:
        # Load uploaded test data
        test_df = pd.read_csv(uploaded_file)
        # Ensure column names match
        if 'quality' in test_df.columns:
            X_test = test_df.drop(['quality'], axis=1)
            y_test = test_df['quality'].apply(lambda x: 1 if x >= 7 else 0)
        else:
            X_test = test_df
            # Check if there's a binary target, else use placeholder
            if 'quality_binary' in test_df.columns:
                y_test = test_df['quality_binary']
            else:
                st.warning("Uploaded test data does not have 'quality' or 'quality_binary'. Assuming no ground truth for metrics (Predictions only).")
                y_test = None
        
        # --- 📌 FEATURE B: Model Selection Dropdown ---
        st.subheader("⚙️ Step 2: Select Model")
        model_options = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        }
        selected_model_name = st.selectbox("Choose a machine learning model:", list(model_options.keys()))
        model = model_options[selected_model_name]
        
        if st.button("🚀 Run Model Evaluation on Test Data"):
            with st.spinner(f"Training {selected_model_name} and evaluating on uploaded test data..."):
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_full)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train_full)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None
                
                st.markdown("---")
                st.subheader("📊 Results on Uploaded Test Data")
                
                # --- 📌 FEATURE C: Display Evaluation Metrics ---
                if y_test is not None:
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{acc:.3f}")
                    col2.metric("Precision", f"{prec:.3f}")
                    col3.metric("Recall", f"{rec:.3f}")
                    col4.metric("F1 Score", f"{f1:.3f}")
                    
                    # --- 📌 FEATURE D: Confusion Matrix & Classification Report ---
                    st.subheader("📉 Confusion Matrix & Classification Report")
                    
                    # Plot Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    st.pyplot(fig)
                    
                    # Show Classification Report
                    st.text("Classification Report:")
                    st.code(classification_report(y_test, y_pred, target_names=['Bad', 'Good'], zero_division=0))
                else:
                    st.warning("Cannot calculate metrics because uploaded CSV lacks ground truth labels. Showing predictions only.")
                    st.write("Predicted Labels on Test Set:", y_pred)
                
    except Exception as e:
        st.error(f"An error occurred processing your test data: {e}")

elif uploaded_file is None:
    st.info("👈 Please upload a CSV file containing your test data to begin.")
elif X_train_full is None:
    st.error("Could not load the training file (`winequality-white.csv`). Ensure it is in the GitHub repository.")
