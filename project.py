import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stSidebar { background-color: #161b22; }
    h1 { color: #58a6ff; text-align: center; font-size: 2.5em; }
    h2 { color: #79c0ff; }
    .sidebar-title { color: #58a6ff; font-weight: bold; }
    .metric { background-color: #21262d; padding: 15px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="🌍 Air Quality", layout="wide", initial_sidebar_state="expanded")

# Title
st.markdown("# 🌍 World Air Quality Index")
st.markdown("### 📊 Analyze & Predict Global Air Pollution Trends")

# Sidebar
with st.sidebar:
    st.markdown("### 📱 Navigation")
    page = st.radio("", ["🏠 Home", "📊 Dataset", "🤖 Prediction", "📈 Graphs"], label_visibility="collapsed")

# HOME PAGE
if "Home" in page:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌍 Global Coverage")
        st.markdown("Analyze data from **50+ countries**")
    
    with col2:
        st.markdown("### 🤖 AI Powered")
        st.markdown("Machine Learning predictions")
    
    with col3:
        st.markdown("### 📈 Real-time")
        st.markdown("Interactive visualizations")
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Features:
    ✅ **Analyze** global air quality data  
    ✅ **Predict** AQI categories using ML  
    ✅ **Visualize** pollutants by country  
    ✅ **Compare** pollution levels  
    
    ### 🌟 Pollutants Tracked:
    PM2.5 • PM10 • NO₂ • SO₂ • O₃ • CO
    """)

# DATASET PAGE
elif "Dataset" in page:
    st.markdown("### 📊 Air Quality Dataset")
    
    try:
        df = pd.read_csv("new_dataset.csv")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📌 Total Records", len(df))
        with col2:
            st.metric("🌍 Countries", df["Country Label"].nunique())
        with col3:
            st.metric("💨 Pollutants", df["Pollutant"].nunique())
        
        st.markdown("---")
        st.markdown("#### Data Preview:")
        st.dataframe(df.head(10), use_container_width=True)
        
    except Exception as e:
        st.error("❌ Dataset not found!")

# PREDICTION PAGE
elif "Prediction" in page:
    st.markdown("### 🤖 AQI Prediction Model")
    
    try:
        df = pd.read_csv("new_dataset.csv")
        
        le = LabelEncoder()
        y = le.fit_transform(df["AQI_Category"])
        X = df[["Value"]].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        col1, col2 = st.columns(2)
        
        with col1:
            pollutant = st.selectbox("Select Pollutant:", df["Pollutant"].unique())
        with col2:
            value = st.number_input("Enter Value:", min_value=0.0, value=50.0, step=1.0)
        
        if st.button("🔮 Predict", use_container_width=True):
            pred = model.predict([[value]])[0]
            category = le.inverse_transform([pred])[0]
            
            colors = {"Good": "🟢", "Moderate": "🟡", "Unhealthy": "🟠", "Hazardous": "🔴"}
            emoji = colors.get(category, "❓")
            
            st.markdown(f"### {emoji} Prediction: **{category}**")
            st.info(f"For {pollutant} with value {value}")
            
    except Exception as e:
        st.error("❌ Error loading prediction model!")

# GRAPH PAGE
elif "Graphs" in page:
    st.markdown("### 📈 Pollutant Visualization")
    
    try:
        df = pd.read_csv("new_dataset.csv")
        
        col1, col2 = st.columns(2)
        
        with col1:
            countries = sorted(df["Country Label"].unique())
            country = st.selectbox("🌍 Select Country:", countries)
        
        with col2:
            agg_type = st.selectbox("📊 Aggregation:", ["Average", "Sum", "Median"])
        
        country_data = df[df["Country Label"] == country]
        
        if not country_data.empty:
            if agg_type == "Average":
                pollutant_values = country_data.groupby("Pollutant")["Value"].mean()
            elif agg_type == "Sum":
                pollutant_values = country_data.groupby("Pollutant")["Value"].sum()
            else:
                pollutant_values = country_data.groupby("Pollutant")["Value"].median()
            
            pollutant_values = pollutant_values.sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(pollutant_values.index, pollutant_values.values, color="#58a6ff", edgecolor="#79c0ff", linewidth=2)
            ax.set_ylabel(f"{agg_type} Value", fontsize=12, fontweight="bold")
            ax.set_xlabel("Pollutant", fontsize=12, fontweight="bold")
            ax.set_title(f"💨 Pollutant Levels in {country} ({agg_type})", fontsize=14, fontweight="bold")
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
        else:
            st.warning("❌ No data available for this country")
            
    except Exception as e:
        st.error("❌ Error loading graphs!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #58a6ff;'>
    🌍 World Air Quality Analysis & Prediction | Built with Streamlit & ML
</div>
""", unsafe_allow_html=True)
