import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Page Config
st.set_page_config(page_title="Air Quality Index", layout="wide")

# Simple CSS
st.markdown("""
<style>
    h1 { color: #1f77b4; font-size: 2.5em; margin-bottom: 10px; }
    h2 { color: #1f77b4; font-size: 1.8em; margin-top: 15px; }
    h3 { color: #333; }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("### Navigation")
st.sidebar.markdown("---")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Buttons
if st.sidebar.button("🏠 HOME", use_container_width=True):
    st.session_state.page = "Home"

if st.sidebar.button("📊 DATASET", use_container_width=True):
    st.session_state.page = "Dataset"

if st.sidebar.button("🤖 PREDICTION", use_container_width=True):
    st.session_state.page = "Prediction"

if st.sidebar.button("📈 GRAPH", use_container_width=True):
    st.session_state.page = "Graph"

st.sidebar.markdown("---")
st.sidebar.info("💡 Select a page from above")

page = st.session_state.page

# HOME PAGE
if page == "Home":
    st.markdown("# 🌍 World Air Quality Index")
    st.markdown("### Analyze Global Air Quality & Predict AQI Categories")
    st.markdown("---")
    
    st.markdown("## Welcome!")
    st.write("""
    This application helps you analyze air quality data from around the world.
    
    ### Features:
    - 📊 **View Dataset** - Explore air quality data from 50+ countries
    - 🤖 **Predict AQI** - Use AI to predict air quality categories
    - 📈 **Visualize** - See pollutant levels by country
    
    ### Pollutants Tracked:
    PM2.5, PM10, NO₂, SO₂, O₃, CO
    """)
    
    st.markdown("---")
    st.markdown("### Getting Started")
    st.write("""
    1. Click **DATASET** to view the air quality data
    2. Click **PREDICTION** to predict AQI for specific values
    3. Click **GRAPH** to visualize pollutant levels by country
    """)

# DATASET PAGE
elif page == "Dataset":
    st.markdown("# 📊 Dataset")
    st.markdown("### Air Quality Data")
    st.markdown("---")
    
    try:
        df = pd.read_csv("new_dataset.csv")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Countries", df["Country Label"].nunique())
        with col3:
            st.metric("Pollutants", df["Pollutant"].nunique())
        
        st.markdown("---")
        st.markdown("### Data Preview (First 15 rows)")
        st.dataframe(df.head(15), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Dataset Info")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        
    except FileNotFoundError:
        st.error("❌ Dataset not found!")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# PREDICTION PAGE
elif page == "Prediction":
    st.markdown("# 🤖 Predict Air Quality Category")
    st.markdown("### Enter pollutant value to predict AQI category")
    st.markdown("---")
    
    try:
        df = pd.read_csv("new_dataset.csv")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pollutant = st.selectbox("Select Pollutant:", df["Pollutant"].unique())
        
        with col2:
            value = st.number_input("Enter Value:", min_value=0.0, value=50.0, step=1.0)
        
        st.markdown("---")
        
        if st.button("🔮 Predict", use_container_width=True):
            # Train model
            le = LabelEncoder()
            y = le.fit_transform(df["AQI_Category"])
            X = df[["Value"]].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict
            pred = model.predict([[value]])[0]
            category = le.inverse_transform([pred])[0]
            
            # Show result
            st.markdown("---")
            st.markdown("### Prediction Result")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pollutant", pollutant)
            with col2:
                st.metric("Predicted Category", category)
            
            st.success(f"✅ For {pollutant} with value **{value}**, the predicted AQI category is **{category}**")
    
    except FileNotFoundError:
        st.error("❌ Dataset not found!")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# GRAPH PAGE
elif page == "Graph":
    st.markdown("# 📈 Pollutant Visualization")
    st.markdown("### View pollutant levels by country")
    st.markdown("---")
    
    try:
        df = pd.read_csv("new_dataset.csv")
        
        col1, col2 = st.columns(2)
        
        with col1:
            countries = sorted(df["Country Label"].unique())
            country = st.selectbox("Select Country:", countries)
        
        with col2:
            agg_type = st.selectbox("Aggregation Type:", ["Average", "Sum", "Median"])
        
        st.markdown("---")
        
        country_data = df[df["Country Label"] == country]
        
        if not country_data.empty:
            # Calculate aggregation
            if agg_type == "Average":
                pollutant_values = country_data.groupby("Pollutant")["Value"].mean().sort_values(ascending=False)
            elif agg_type == "Sum":
                pollutant_values = country_data.groupby("Pollutant")["Value"].sum().sort_values(ascending=False)
            else:  # Median
                pollutant_values = country_data.groupby("Pollutant")["Value"].median().sort_values(ascending=False)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(pollutant_values.index, pollutant_values.values, color="#1f77b4", edgecolor="#000", linewidth=1.5)
            ax.set_xlabel("Pollutant", fontsize=12, fontweight="bold")
            ax.set_ylabel(f"{agg_type} Value", fontsize=12, fontweight="bold")
            ax.set_title(f"Pollutant Levels in {country} ({agg_type})", fontsize=14, fontweight="bold")
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            st.markdown("---")
            st.markdown("### Data Summary")
            summary_df = pd.DataFrame({
                "Pollutant": pollutant_values.index,
                f"{agg_type} Value": pollutant_values.values
            })
            st.dataframe(summary_df, use_container_width=True)
        
        else:
            st.warning(f"❌ No data available for {country}")
    
    except FileNotFoundError:
        st.error("❌ Dataset not found!")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    🌍 World Air Quality Analysis & Prediction<br>
    © 2026 | All Rights Reserved
</div>
""", unsafe_allow_html=True)
