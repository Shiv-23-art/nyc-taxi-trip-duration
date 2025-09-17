import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk

# Load trained Random Forest model

model = joblib.load("../models/random_forest_model.pkl")

# Streamlit App

st.title("🚖 NYC Taxi Trip Duration Prediction")

st.markdown("""
This app predicts the **duration of taxi trips in New York City** using a 
machine learning model trained on the 2016 Kaggle NYC Taxi dataset.
Enter trip details below and get an estimated duration!
""")

# Input Fields

st.subheader("📍 Enter Pickup and Dropoff Coordinates")
pickup_latitude = st.number_input("Pickup Latitude", value=40.7128, format="%.6f")
pickup_longitude = st.number_input("Pickup Longitude", value=-74.0060, format="%.6f")
dropoff_latitude = st.number_input("Dropoff Latitude", value=40.7769, format="%.6f")
dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.9813, format="%.6f")

st.subheader("🛠️ Trip Details")
pickup_hour = st.number_input("Pickup Hour (0-23)", min_value=0, max_value=23, value=9)
pickup_dayofweek = st.number_input("Pickup Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0)
pickup_month = st.number_input("Pickup Month (1-12)", min_value=1, max_value=12, value=1)
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=8, value=1)


# Feature Engineering: Distance

def haversine_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    km = 2*6371*np.arcsin(np.sqrt(a))
    return km

distance_km = haversine_np(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)

# Prepare input features
input_data = np.array([[distance_km, pickup_hour, pickup_dayofweek, pickup_month, passenger_count]])


# Prediction

if st.button("🚀 Predict Trip Duration"):
    try:
        prediction_log = model.predict(input_data)
        prediction_seconds = np.expm1(prediction_log)[0]

        minutes = int(prediction_seconds // 60)
        seconds = int(prediction_seconds % 60)

        st.success(f"🕒 Predicted Trip Duration: **{minutes} min {seconds} sec**")

        # Map Visualization
       
        st.subheader("🗺️ Pickup and Dropoff Locations")

        map_data = pd.DataFrame({
            'lat': [pickup_latitude, dropoff_latitude],
            'lon': [pickup_longitude, dropoff_longitude],
            'label': ['Pickup', 'Dropoff'],
            'color': [[0, 255, 0], [255, 0, 0]]  # green=pickup, red=dropoff
        })

        # Scatterplot layer
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position='[lon, lat]',
            get_color='color',
            get_radius=100,
            pickable=True
        )

        # Line connecting pickup to dropoff
        line_layer = pdk.Layer(
            "LineLayer",
            data=pd.DataFrame({
                'start_lat': [pickup_latitude],
                'start_lon': [pickup_longitude],
                'end_lat': [dropoff_latitude],
                'end_lon': [dropoff_longitude]
            }),
            get_source_position='[start_lon, start_lat]',
            get_target_position='[end_lon, end_lat]',
            get_color=[0, 0, 255],
            get_width=5
        )

        # View state
        view_state = pdk.ViewState(
            latitude=(pickup_latitude + dropoff_latitude)/2,
            longitude=(pickup_longitude + dropoff_longitude)/2,
            zoom=12,
            pitch=0
        )

        # Render map
        r = pdk.Deck(
            layers=[scatter_layer, line_layer],
            initial_view_state=view_state,
            tooltip={"text": "{label}"}
        )

        st.pydeck_chart(r)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
