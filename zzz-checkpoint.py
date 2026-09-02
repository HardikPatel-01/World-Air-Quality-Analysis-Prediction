import pandas as pd

# Load dataset
df = pd.read_csv(
    r"G:\air quality project\world_air_quality_file.csv",
    delimiter=",",
    encoding="latin1"
)

print(df.head())
print("Columns in dataset:", df.columns.tolist())

# ✅ Convert Value column to numeric (force errors to NaN)
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")


# AQI function
def get_aqi_category(pollutant, value):
    if pd.isna(value):
        return "Unknown"

    pollutant = pollutant.upper()  # normalize case

    if pollutant == "NO2":
        if value <= 0.05:
            return "Good"
        elif value <= 0.1:
            return "Moderate"
        elif value <= 0.2:
            return "Unhealthy"
        else:
            return "Hazardous"

    elif pollutant == "SO2":
        if value <= 0.02:
            return "Good"
        elif value <= 0.1:
            return "Moderate"
        elif value <= 0.2:
            return "Unhealthy"
        else:
            return "Hazardous"

    elif pollutant == "PM2.5":
        if value <= 12:
            return "Good"
        elif value <= 35.4:
            return "Moderate"
        elif value <= 55.4:
            return "Unhealthy"
        else:
            return "Hazardous"

    elif pollutant == "PM10":
        if value <= 54:
            return "Good"
        elif value <= 154:
            return "Moderate"
        elif value <= 254:
            return "Unhealthy"
        else:
            return "Hazardous"

    elif pollutant == "PM1":
        # Not officially standardized; using similar cutoffs as PM2.5
        if value <= 12:
            return "Good"
        elif value <= 35:
            return "Moderate"
        elif value <= 55:
            return "Unhealthy"
        else:
            return "Hazardous"

    elif pollutant == "NOX":
        if value <= 0.05:
            return "Good"
        elif value <= 0.1:
            return "Moderate"
        elif value <= 0.2:
            return "Unhealthy"
        else:
            return "Hazardous"

    elif pollutant == "NO":
        if value <= 0.03:
            return "Good"
        elif value <= 0.06:
            return "Moderate"
        elif value <= 0.1:
            return "Unhealthy"
        else:
            return "Hazardous"

    elif pollutant == "CO":
        if value <= 4.4:
            return "Good"
        elif value <= 9.4:
            return "Moderate"
        elif value <= 12.4:
            return "Unhealthy"
        else:
            return "Hazardous"

    elif pollutant == "O3":
        if value <= 0.054:
            return "Good"
        elif value <= 0.07:
            return "Moderate"
        elif value <= 0.085:
            return "Unhealthy"
        else:
            return "Hazardous"

    elif pollutant in ["BC", "TEMPERATURE", "UM003", "RELATIVEHUMIDITY"]:
        return "Not Applicable"  # not AQI-related pollutants

    else:
        return "Unknown"


# ✅ Apply AQI calculation row by row
df["AQI_Category"] = df.apply(
    lambda row: get_aqi_category(row["Pollutant"], row["Value"]),
    axis=1
)

# Save result
df.to_csv("new_dataset.csv", index=False, encoding="utf-8")

print("✅ AQI categories added for all rows and saved as new dataset.csv")
