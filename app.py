import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Set page title and layout
st.set_page_config(page_title="Fire Type Classifier", layout="centered")

# App title and info
st.title("Fire Type Classification")
st.markdown("Predict fire type based on MODIS satellite readings and view historical locations of the same fire type in India.")

# --- Detailed Fire Type Information ---
# This section provides an in-depth explanation of each fire classification.
with st.expander("ℹ️ Detailed Fire Type Information"):
    st.markdown("""
    <b>This application uses a machine learning model to classify thermal anomalies detected by the MODIS satellite into one of four distinct categories. Understanding these classifications is essential for interpreting the map and the model's predictions.</b>
    <ul>
    <li><b>Vegetation Fire:</b> Wildfires, forest fires, and agricultural burns.</li>
    <li><b>Offshore Fire:</b> Gas flaring on oil/gas rigs in water bodies.</li>
    <li><b>Other Static Land Source:</b> Industrial flares, waste disposal, brick kilns.</li>
    <li><b>Volcanic Fire:</b> Lava flows and volcanic vents (rare in India).</li>
    </ul>
    """, unsafe_allow_html=True)

# --- Load the model, scaler, and the original dataset ---
try:
    # Assuming these files are in the same directory as app.py
    model = joblib.load("best_fire_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Please ensure 'best_fire_detection_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Load and combine all three fire datasets
csv_files = ['modis_2021_India.csv', 'modis_2022_India.csv', 'modis_2023_India.csv']
df_list = []
for file in csv_files:
    if os.path.exists(file):
        df_list.append(pd.read_csv(file))
    else:
        st.error(f"Error: Dataset file '{file}' not found. Please ensure all three files are in the same directory.")
        st.stop()

df_fire_data = pd.concat(df_list, ignore_index=True)

# Based on the CSV files, the 'type' column is already present with numerical labels.
if 'latitude' not in df_fire_data.columns or 'longitude' not in df_fire_data.columns:
    st.error("The 'latitude' or 'longitude' columns are missing from the combined dataset.")
    st.stop()
if 'type' not in df_fire_data.columns:
    st.error("The 'type' column is missing from the dataset.")
    st.stop()

# --- User Input Fields ---
brightness = st.number_input("Brightness (K)", value=300.0)
bright_t31 = st.number_input("Brightness T31 (K)", value=290.0)
frp = st.number_input("Fire Radiative Power (FRP) (MW)", value=15.0)
scan = st.number_input("Scan", value=1.0)
track = st.number_input("Track", value=1.0)
confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"])

# Map confidence to numeric value
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# Combine and scale input
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
try:
    scaled_input = scaler.transform(input_data)
except ValueError as e:
    st.error(f"Scaling error: {e}. The input features might not match the features used for training.")
    st.stop()

# --- Explanation of Prediction vs. Map ---
st.info("""
**How the prediction and map work:**

* **Prediction:** Clicking "Predict Fire Type," the app uses a trained model to analyze the provided inputs. The model makes a prediction about the fire type based on the specific data.
* **Historical Map:** The map is a separate component. It is loaded with all historical fire data from the 2021-2023 MODIS datasets. It does not plot new input data. Instead, it filters the existing historical data to display past fires that belong to the same category as the model's prediction.
""")

# --- Prediction and Map Display Logic ---
# Map the numerical labels to user-friendly names
fire_type_names = {
    0: "Vegetation Fire",
    1: "Offshore Fire",
    2: "Other Static Land Source",
    3: "Volcanic Fire"
}

fire_type_descriptions = {
    "Vegetation Fire": "This refers to fires burning natural or planted vegetation, such as forest fires, wildfires, and agricultural burns used to clear land.",
    "Offshore Fire": "These are fires detected over a body of water, most commonly from gas flares on offshore oil and gas rigs, rather than a natural fire.",
    "Other Static Land Source": "This is a non-spreading, stationary fire on land, often from human-made or industrial sources like industrial flares at factories, fires at landfills, or brick kilns.",
    "Volcanic Fire": "This is a fire originating from volcanic activity."
}


if st.button("Predict Fire Type"):
    with st.spinner('Predicting and fetching data...'):
        # Get the prediction (numerical label)
        prediction = model.predict(scaled_input)[0]

        # Get the corresponding fire type name
        predicted_fire_type_name = fire_type_names.get(prediction, "Unknown Fire Type")

        st.subheader("Prediction Result:")
        st.success(f"The predicted fire type is: **{predicted_fire_type_name}**")

        # Display the explanation for the predicted fire type in an expander
        if predicted_fire_type_name in fire_type_descriptions:
            with st.expander("Learn more about this fire type"):
                st.markdown(fire_type_descriptions[predicted_fire_type_name])

        # Optional: Display the numerical code for debugging purposes
        with st.expander("See model's numerical output (for developers)"):
            st.info(f"The model predicted the numerical code: `{prediction}`")


        # Filter the combined dataset to show only locations of the predicted fire type
        filtered_df = df_fire_data[df_fire_data['type'] == prediction]

        # Display the map if there's data to show
        if not filtered_df.empty:
            st.subheader(f"Historical Locations of '{predicted_fire_type_name}' in India")
            st.map(filtered_df, latitude='latitude', longitude='longitude', zoom=4)
        else:
            st.info(f"No historical data found for '{predicted_fire_type_name}' to display on the map.")
