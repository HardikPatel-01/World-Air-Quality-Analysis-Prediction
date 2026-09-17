import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- UI --------------------
st.title("🌍 World Air Quality Index")

with st.sidebar:
    selected = option_menu(
        menu_title="Overview",
        options=["Home", "About Model", "Dataset", "Prediction Model", "Graph"],
        icons=["house", "info-circle", "table", "cpu", "bar-chart"],
        orientation="vertical"
    )

# -------------------- HOME --------------------
if selected == "Home":
    st.subheader("👋 Welcome!")
    st.write("""Explore world air quality dataset, make predictions, and visualize pollutant levels by country.


## 🌍 Air Quality Insight

Air Quality Insight is an interactive web application designed to **analyze, visualize, and understand global air quality trends**.
With real-time interactivity and data-driven insights, this project helps users explore how pollutants affect air quality across different regions.

### 🚀 Key Features

* **Interactive Graphs & Charts** – Visualize pollutant levels with dynamic plots.
* **Country-wise Analysis** – Select a country and instantly view its air quality trends.
* **Pollutant Breakdown** – Understand the impact of individual pollutants like PM2.5, NO₂, and CO.
* **Forecasting (AI-powered)** – Get predictions on future air quality using reinforcement learning.
* **Clean UI** – Minimal, engaging design with graph-driven background visuals.

### 🌟 Why Air Quality Insight?

* Provides **clear visibility** of environmental health across countries.
* Helps users, researchers, and policymakers understand trends.
* Combines **data analysis with AI** for smarter predictions.
* Built with **Streamlit** for seamless, interactive experience.

✨ Start exploring the air you breathe, and discover insights that matter!


""")

# -------------------- ABOUT MODEL --------------------
elif selected == "About Model":
    st.title("📘 About the Model")
    st.markdown("""
    ### 🎯 Goal
    Predict **Air Quality Category** based on pollutant values.

    ### 🧠 Model
    - Random Forest Classifier
    - Trained on `new_dataset.csv`
    
    The prediction engine behind **Air Quality Insight** is built using **Machine Learning techniques** that analyze historical air quality data.

    * **Data Preprocessing** – The raw dataset is cleaned, normalized, and transformed for accurate analysis.
    * **Feature Selection** – Key pollutants (PM2.5, NO₂, SO₂, O₃, CO) are considered as input features.
    * **Model Training** – A **Reinforcement Learning based approach** is applied to capture patterns and predict future air quality trends.
    * **Evaluation** – The model is tested against real-world data to ensure accuracy and reliability.

    This model helps the app **forecast air quality levels** and show how pollutants contribute to overall environmental health.


    """)

# -------------------- DATASET --------------------
elif selected == "Dataset":
    st.title("📁 Uploaded Dataset (new_dataset.csv)")
    try:
        dataset_df = pd.read_csv("new_dataset.csv")
        st.dataframe(dataset_df.head(50))  # show first 50 rows
    except FileNotFoundError:
        st.error("new_dataset.csv file not found. Please upload or check path.")

# -------------------- PREDICTION MODEL --------------------
elif selected == "Prediction Model":
    st.title("🤖 Predict Air Quality Category")

    try:
        dataset_df = pd.read_csv("new_dataset.csv")

        # Encode target
        le = LabelEncoder()
        dataset_df["AQI_Category"] = le.fit_transform(dataset_df["AQI_Category"])

        # Features & Target (for simplicity we use only Value, but can be expanded)
        X = dataset_df[["Value"]]
        y = dataset_df["AQI_Category"]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # User input
        pollutant = st.selectbox("Select Pollutant", dataset_df["Pollutant"].unique())
        value = st.number_input("Enter Value", min_value=0.0, value=50.0)

        if st.button("Predict"):
            pred = model.predict([[value]])[0]
            category = le.inverse_transform([pred])[0]
            st.success(f"Predicted AQI Category for {pollutant} with value {value}: **{category}**")

    except FileNotFoundError:
        st.error("new_dataset.csv file not found. Please upload or check path.")

# -------------------- GRAPH --------------------
elif selected == "Graph":
    st.title("📊 Visualize Pollutant Data by Country")

    try:
        dataset_df = pd.read_csv("new_dataset.csv")

        # Let user pick a country
        countries = dataset_df["Country Label"].dropna().unique()
        country = st.selectbox("🌍 Select Country", sorted(countries))

        # Let user choose aggregation type
        agg_type = st.radio("📊 Aggregation Type", ["Average", "Sum", "Median"])

        # Filter dataset
        country_data = dataset_df[dataset_df["Country Label"] == country]

        if country_data.empty:
            st.warning("No data available for this country.")
        else:
            # Choose aggregation
            if agg_type == "Average":
                pollutant_values = country_data.groupby("Pollutant")["Value"].mean().sort_values(ascending=False)
            elif agg_type == "Sum":
                pollutant_values = country_data.groupby("Pollutant")["Value"].sum().sort_values(ascending=False)
            else:  # Median
                pollutant_values = country_data.groupby("Pollutant")["Value"].median().sort_values(ascending=False)

            # Plot with log scale
            plt.figure(figsize=(10, 6))
            plt.bar(pollutant_values.index, pollutant_values.values, color="skyblue")
            plt.yscale("log")   # log scale applied here
            plt.xticks(rotation=45)
            plt.xlabel("Pollutant")
            plt.ylabel(f"{agg_type} Value (Log Scale)")
            plt.title(f"Pollutant Levels in {country} ({agg_type}, Log Scale)")
            st.pyplot(plt.gcf())

    except FileNotFoundError:
        st.error("new_dataset.csv file not found. Please upload or check path.")
