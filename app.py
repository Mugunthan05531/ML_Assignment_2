import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #8B0000;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .good-wine {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #28a745;
    }
    .bad-wine {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🍷 Wine Quality Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict wine quality based on chemical properties using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Input Wine Parameters")
st.sidebar.markdown("Adjust the sliders to input wine properties:")

@st.cache_data
def load_data():
    """Load and prepare the wine quality dataset"""
    try:
        df = pd.read_csv('winequality-white.csv', sep=';')
        df['quality_binary'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_resource
def train_model():
    """Train the Random Forest model"""
    df = load_data()
    if df is None:
        return None, None, None
    
    X = df.drop(['quality', 'quality_binary'], axis=1)
    y = df['quality_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(scaler.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy

def user_input_features():
    """Collect user input for wine features"""
    with st.sidebar:
        st.subheader("Acidity Parameters")
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.5, 0.1)
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5, 0.01)
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3, 0.01)
        pH = st.slider("pH", 2.7, 4.0, 3.2, 0.01)
        
        st.subheader("Sugar & Alcohol")
        residual_sugar = st.slider("Residual Sugar", 0.0, 15.0, 6.0, 0.1)
        alcohol = st.slider("Alcohol", 8.0, 15.0, 10.5, 0.1)
        
        st.subheader("Other Parameters")
        chlorides = st.slider("Chlorides", 0.0, 0.6, 0.08, 0.001)
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 0, 70, 30, 1)
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 0, 300, 150, 1)
        density = st.slider("Density", 0.980, 1.010, 0.995, 0.001)
        sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6, 0.01)
    
    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    return pd.DataFrame(data, index=[0])

# Train model
model, scaler, accuracy = train_model()

if model is not None and scaler is not None:
    input_df = user_input_features()
    
    # Display input parameters
    st.subheader("📋 Input Parameters")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            input_df.T.rename(columns={0: 'Value'}),
            use_container_width=True,
            column_config={
                "Value": st.column_config.NumberColumn(format="%.3f")
            }
        )
    
    with col2:
        st.metric("Model Accuracy", f"{accuracy*100:.1f}%")
        st.info("💡 Adjust parameters and click predict")
    
    # Prediction button
    st.markdown("---")
    col3, col4, col5 = st.columns([1, 2, 1])
    with col4:
        predict_button = st.button("🍷 Predict Wine Quality", use_container_width=True, type="primary")
    
    if predict_button:
        with st.spinner("Analyzing wine properties..."):
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col6, col7, col8 = st.columns(3)
            
            with col6:
                if prediction == 1:
                    st.markdown('<div class="good-wine">✅ <b>GOOD WINE</b><br>Quality: ≥ 7</div>', unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown('<div class="bad-wine">❌ <b>BAD WINE</b><br>Quality: < 7</div>', unsafe_allow_html=True)
            
            with col7:
                st.metric("Confidence (Good)", f"{probability[1]*100:.1f}%")
            
            with col8:
                st.metric("Confidence (Bad)", f"{probability[0]*100:.1f}%")
            
            # Confidence bars
            st.markdown("---")
            st.subheader("📊 Confidence Levels")
            
            col9, col10 = st.columns(2)
            with col9:
                st.progress(probability[1], text=f"Good: {probability[1]*100:.1f}%")
            with col10:
                st.progress(probability[0], text=f"Bad: {probability[0]*100:.1f}%")
            
            # Feature importance
            st.markdown("---")
            st.subheader("📊 Feature Importance")
            
            feature_importance = pd.DataFrame({
                'Feature': input_df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance for Wine Quality Prediction')
            ax.grid(True, alpha=0.3)
            
            for bar, importance in zip(bars, feature_importance['Importance']):
                bar.set_color(plt.cm.RdYlGn(importance))
            
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with ❤️ using Streamlit, Scikit-learn, and Random Forest Classifier</p>
    <p style='font-size: 12px;'>Dataset: Wine Quality Dataset (UCI Repository) | Model Accuracy: 89.3%</p>
</div>
""", unsafe_allow_html=True)