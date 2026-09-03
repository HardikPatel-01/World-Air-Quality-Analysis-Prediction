# 🌍 AEROSENSE — World Air Quality Analysis & Prediction

AEROSENSE is an interactive **Data Analytics and Machine Learning web application** developed using Python and Streamlit to analyze global air quality and predict Air Quality Index (AQI) categories.

The application allows users to explore air pollution data by country, compare pollutant levels, and generate visual insights for pollutants such as **PM2.5, PM10, NO₂, SO₂, O₃, and CO**.

## 🚀 Key Features

- 🌍 Country-wise air quality analysis
- 📊 Interactive pollutant visualizations
- 📈 Average, Sum, and Median aggregation
- 🔬 Analysis of major air pollutants
- 🤖 AQI category prediction using Random Forest
- 📁 Dataset exploration
- 🖥️ Interactive Streamlit interface
- 📉 Log-scale visualization for comparing pollutant concentrations

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Streamlit**

## 🤖 Machine Learning

The project uses a **Random Forest Classifier** to predict the Air Quality Category from pollutant-related data. The dataset is preprocessed and the AQI category is encoded before model training.

## 📊 Data Analysis

The application provides country-level analysis where users can select a country and compare pollutant values using **Average, Sum, or Median** aggregation. Logarithmic-scale charts are used to make pollutants with significantly different value ranges easier to compare.

## 🔄 Project Workflow

**Data → Cleaning → AQI Classification → Exploratory Analysis → Visualization → Machine Learning → Prediction**

The AQI categories are generated during preprocessing and stored in the processed dataset for use by the prediction application.

## 🎯 Objective

The main objective of AEROSENSE is to transform complex environmental data into an easy-to-use analytical and predictive application that can help users understand air-quality patterns and pollutant levels across countries.

## 🔮 Future Scope

- Time-series forecasting of pollutant concentrations
- Integration of weather and seasonal data
- Real-time air-quality data streaming
- Public cloud deployment
- More advanced predictive models
