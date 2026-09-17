import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="🌍 Air Quality", layout="wide")
st.title("🌍 World Air Quality Index")

# Sidebar
with st.sidebar:
    page = st.radio("📊 Pages:", ["Home", "Dataset", "Prediction", "Graph"])

# HOME PAGE
if page == "Home":
    st.subheader("👋 Welcome!")
    st.write("""
    🌍 World Air Quality Analysis & Prediction
    
    ✅ Analyze global air quality data
    ✅ Predict AQI categories  
    ✅ Visualize pollutants
    """)

# DATASET PAGE
elif page == "Dataset":
    st.subheader("📊 Dataset")
    try:
        df = pd.read_csv("new_dataset.csv")
        st.write(f"**Total rows:** {len(df)}")
        st.dataframe(df.head(20))
    except:
        st.warning("Dataset not found")

# PREDICTION PAGE
elif page == "Prediction":
    st.subheader("🤖 Predict AQI")
    try:
        df = pd.read_csv("new_dataset.csv")
        
        le = LabelEncoder()
        y = le.fit_transform(df["AQI_Category"])
        X = df[["Value"]].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(n_estimators=50)
        model.fit(X_train, y_train)
        
        value = st.number_input("Enter value:", min_value=0.0, value=50.0)
        
        if st.button("Predict"):
            pred = model.predict([[value]])[0]
            category = le.inverse_transform([pred])[0]
            st.success(f"**Prediction: {category}**")
    except:
        st.error("Error loading data")

# GRAPH PAGE
elif page == "Graph":
    st.subheader("📈 Graphs")
    try:
        df = pd.read_csv("new_dataset.csv")
        
        countries = df["Country Label"].unique()
        country = st.selectbox("Select Country:", countries)
        
        country_data = df[df["Country Label"] == country]
        pollutant_values = country_data.groupby("Pollutant")["Value"].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(pollutant_values.index, pollutant_values.values, color="skyblue")
        ax.set_ylabel("Average Value")
        ax.set_title(f"Pollutants in {country}")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except:
        st.error("Error loading data")
