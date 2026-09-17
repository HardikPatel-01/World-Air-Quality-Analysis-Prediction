import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Page Config
st.set_page_config(
    page_title="🌍 Air Quality",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "World Air Quality Analysis"}
)

# Custom CSS - Beautiful Styling
st.markdown("""
<style>
    * { margin: 0; padding: 0; }
    body { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); }
    .main { background-color: #0a0e27; }
    .stSidebar { background: linear-gradient(180deg, #1a1f3a 0%, #252d4a 100%); }
    
    h1 { 
        color: #00d4ff; 
        text-align: center; 
        font-size: 3em; 
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        margin-bottom: 10px;
    }
    h2 { 
        color: #00ff88; 
        font-size: 1.8em;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    h3 { color: #00d4ff; }
    
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 1.2em;
        font-weight: bold;
        border: 2px solid #00d4ff;
        border-radius: 10px;
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.6);
        transform: scale(1.02);
    }
    
    .metric-box {
        background: linear-gradient(135deg, #1a1f3a 0%, #252d4a 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
    }
    
    .feature-box {
        background: linear-gradient(135deg, #1e2a4a 0%, #2a3555 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
        margin: 10px 0;
        box-shadow: 0 4px 10px rgba(0, 255, 136, 0.2);
    }
    
    .stSelectbox, .stNumberInput, .st-c7, .stRadio {
        background-color: #1a1f3a !important;
    }
    
    .stSelectbox > div > div, .stNumberInput > div > div {
        background-color: #252d4a !important;
        border: 1px solid #00d4ff !important;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 📱 MENU")
    st.markdown("---")
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "ℹ️ About", "📊 Dataset", "🤖 Predict", "📈 Graphs"],
        index=0
    )
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.info("🔧 All features active")

# Main Title
st.markdown("""
    <div style='text-align: center; margin: 30px 0;'>
        <h1>🌍 WORLD AIR QUALITY INDEX</h1>
        <p style='font-size: 1.2em; color: #00ff88;'>Analyze • Predict • Visualize</p>
    </div>
""", unsafe_allow_html=True)

# HOME PAGE
if "Home" in page:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-box'>
            <h3 style='color: #00d4ff;'>🌍 50+ Countries</h3>
            <p>Global air quality data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-box'>
            <h3 style='color: #00ff88;'>🤖 AI Powered</h3>
            <p>ML predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-box'>
            <h3 style='color: #ff00ff;'>⚡ Real-time</h3>
            <p>Interactive analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 🎯 KEY FEATURES")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-box'>
            ✅ <b>Analyze Data</b> - Global air quality dataset
        </div>
        <div class='feature-box'>
            ✅ <b>Predict AQI</b> - Machine Learning model
        </div>
        <div class='feature-box'>
            ✅ <b>Visualize</b> - Beautiful charts & graphs
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-box'>
            ✅ <b>Track Pollutants</b> - PM2.5, NO₂, SO₂, O₃, CO
        </div>
        <div class='feature-box'>
            ✅ <b>Country Analysis</b> - Region-wise trends
        </div>
        <div class='feature-box'>
            ✅ <b>Interactive UI</b> - Modern dashboard
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 💨 Tracked Pollutants")
    
    pollutants = {
        "PM2.5": "Fine Particulates",
        "PM10": "Particulate Matter",
        "NO₂": "Nitrogen Dioxide",
        "SO₂": "Sulfur Dioxide",
        "O₃": "Ozone",
        "CO": "Carbon Monoxide"
    }
    
    cols = st.columns(3)
    for i, (pollutant, desc) in enumerate(pollutants.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='feature-box'>
                <h4>{pollutant}</h4>
                <p style='font-size: 0.9em;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ABOUT PAGE
elif "About" in page:
    st.markdown("## 📖 About This Project")
    
    st.markdown("""
    ### 🎯 Mission
    Provide **real-time insights** into global air quality trends and predict future pollution levels
    using advanced machine learning.
    
    ### 🏗️ Architecture
    - **Frontend**: Streamlit (Beautiful UI)
    - **Backend**: Python (Data Processing)
    - **ML Model**: Random Forest Classifier
    - **Data**: Global Air Quality Database
    
    ### 📊 Dataset Features
    - 50+ Countries
    - 6+ Pollutants Tracked
    - Real-time Updates
    - Historical Data
    
    ### 🔬 Technical Stack
    - Pandas (Data Analysis)
    - Scikit-learn (ML)
    - Matplotlib (Visualization)
    - Streamlit (Web Framework)
    """)

# DATASET PAGE
elif "Dataset" in page:
    st.markdown("## 📊 Air Quality Dataset")
    
    try:
        df = pd.read_csv("new_dataset.csv")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-box' style='text-align: center;'>
                <h1 style='color: #00d4ff; margin: 0;'>{len(df)}</h1>
                <p>📌 Total Records</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-box' style='text-align: center;'>
                <h1 style='color: #00ff88; margin: 0;'>{df['Country Label'].nunique()}</h1>
                <p>🌍 Countries</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-box' style='text-align: center;'>
                <h1 style='color: #ff00ff; margin: 0;'>{df['Pollutant'].nunique()}</h1>
                <p>💨 Pollutants</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-box' style='text-align: center;'>
                <h1 style='color: #ffaa00; margin: 0;'>{df['AQI_Category'].nunique()}</h1>
                <p>📈 Categories</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(15), use_container_width=True, height=400)
        
    except:
        st.error("❌ Dataset file not found!")

# PREDICTION PAGE
elif "Predict" in page:
    st.markdown("## 🤖 AI-Powered AQI Prediction")
    
    try:
        df = pd.read_csv("new_dataset.csv")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📌 Select Pollutant")
            pollutant = st.selectbox("", df["Pollutant"].unique(), label_visibility="collapsed")
        
        with col2:
            st.markdown("### 🔢 Enter Value")
            value = st.number_input("", min_value=0.0, value=50.0, step=1.0, label_visibility="collapsed")
        
        st.markdown("---")
        
        if st.button("🔮 PREDICT NOW", use_container_width=True):
            le = LabelEncoder()
            y = le.fit_transform(df["AQI_Category"])
            X = df[["Value"]].values
            
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            pred = model.predict([[value]])[0]
            category = le.inverse_transform([pred])[0]
            
            emojis = {"Good": "🟢", "Moderate": "🟡", "Unhealthy": "🟠", "Hazardous": "🔴"}
            emoji = emojis.get(category, "❓")
            
            st.markdown(f"""
            <div style='text-align: center; background: linear-gradient(135deg, #1a1f3a 0%, #252d4a 100%); 
                        padding: 40px; border-radius: 15px; border: 2px solid #00d4ff; margin-top: 20px;
                        box-shadow: 0 0 30px rgba(0, 212, 255, 0.4);'>
                <h1 style='color: #00d4ff; font-size: 2.5em; margin: 0;'>{emoji} PREDICTION</h1>
                <h2 style='color: #00ff88; font-size: 3em; margin: 20px 0;'>{category}</h2>
                <p style='color: #ffffff; font-size: 1.2em;'>
                    {pollutant} with value <b>{value}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    except:
        st.error("❌ Error in prediction!")

# GRAPHS PAGE
elif "Graphs" in page:
    st.markdown("## 📈 Beautiful Visualizations")
    
    try:
        df = pd.read_csv("new_dataset.csv")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌍 Select Country")
            countries = sorted(df["Country Label"].unique())
            country = st.selectbox("", countries, label_visibility="collapsed")
        
        with col2:
            st.markdown("### 📊 Aggregation Type")
            agg_type = st.selectbox("", ["Average", "Sum", "Median"], label_visibility="collapsed")
        
        st.markdown("---")
        
        country_data = df[df["Country Label"] == country]
        
        if not country_data.empty:
            if agg_type == "Average":
                pollutant_values = country_data.groupby("Pollutant")["Value"].mean()
            elif agg_type == "Sum":
                pollutant_values = country_data.groupby("Pollutant")["Value"].sum()
            else:
                pollutant_values = country_data.groupby("Pollutant")["Value"].median()
            
            pollutant_values = pollutant_values.sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(14, 7))
            fig.patch.set_facecolor('#0a0e27')
            ax.set_facecolor('#1a1f3a')
            
            bars = ax.bar(range(len(pollutant_values)), pollutant_values.values, 
                         color=['#00d4ff', '#00ff88', '#ff00ff', '#ffaa00', '#ff3333', '#00ccff'],
                         edgecolor='#ffffff', linewidth=2, alpha=0.8)
            
            ax.set_xticks(range(len(pollutant_values)))
            ax.set_xticklabels(pollutant_values.index, color='#ffffff', fontsize=11, fontweight='bold')
            ax.set_ylabel(f"{agg_type} Value", color='#ffffff', fontsize=12, fontweight='bold')
            ax.set_title(f"💨 Pollutant Levels in {country} ({agg_type})", 
                        color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.2, linestyle='--')
            ax.tick_params(colors='#ffffff')
            
            for spine in ax.spines.values():
                spine.set_color('#00d4ff')
                spine.set_linewidth(2)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("❌ No data available for this country")
    
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #00d4ff; font-size: 0.9em;'>
    <b>🌍 World Air Quality Analysis & Prediction</b><br>
    Built with ❤️ using Streamlit & Machine Learning<br>
    <small>© 2026 | All Rights Reserved</small>
</div>
""", unsafe_allow_html=True)
