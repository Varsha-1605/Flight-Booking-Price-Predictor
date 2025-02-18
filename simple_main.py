
import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('model/model.pkl')

# Define categorical mapping based on encoding
airline_map = {'AirAsia': 0, 'Air India': 1, 'Go First': 2, 'Indigo': 3, 'SpiceJet': 4, 'Vistara': 5}
source_city_map = {'Bangalore': 0, 'Chennai': 1, 'Delhi': 2, 'Hyderabad': 3, 'Kolkata': 4, 'Mumbai': 5}
destination_city_map = {'Bangalore': 0, 'Chennai': 1, 'Delhi': 2, 'Hyderabad': 3, 'Kolkata': 4, 'Mumbai': 5}
departure_time_map = {'Afternoon': 0, 'Early Morning': 1, 'Evening': 2, 'Late Night': 3, 'Morning': 4, 'Night': 5}
arrival_time_map = {'Afternoon': 0, 'Early Morning': 1, 'Evening': 2, 'Late Night': 3, 'Morning': 4, 'Night': 5}
stops_map = {'One': 0, 'Two or more': 1, 'Zero': 2}
class_map = {'Business': 0, 'Economy': 1}

# Streamlit UI
st.title("✈️ Flight Price Prediction")

# User Inputs
airline = st.selectbox("✈️ Airline", list(airline_map.keys()))
source_city = st.selectbox("📍 Source City", list(source_city_map.keys()))
destination_city = st.selectbox("📍 Destination City", list(destination_city_map.keys()))
departure_time = st.selectbox("🕐 Departure Time", list(departure_time_map.keys()))
arrival_time = st.selectbox("🕐 Arrival Time", list(arrival_time_map.keys()))
stops = st.selectbox("🔀 Number of Stops", list(stops_map.keys()))
flight_class = st.selectbox("💺 Flight Class", list(class_map.keys()))
duration = st.slider("⏳ Flight Duration (in hours)", min_value=1.0, max_value=49.0, step=0.5)
days_left = st.slider("📅 Days Left for Booking", min_value=1, max_value=49, step=1)

# Submit Button
if st.button("🚀 Predict Price"):
    # Encode user inputs
    airline_encoded = airline_map[airline]
    source_encoded = source_city_map[source_city]
    destination_encoded = destination_city_map[destination_city]
    departure_encoded = departure_time_map[departure_time]
    arrival_encoded = arrival_time_map[arrival_time]
    stops_encoded = stops_map[stops]
    class_encoded = class_map[flight_class]

    # Prepare input data for prediction
    input_data = np.array([
        airline_encoded, source_encoded, departure_encoded, stops_encoded, arrival_encoded,
        destination_encoded, class_encoded, duration, days_left
    ]).reshape(1, -1)

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    # Display result
    st.subheader(f"💰 Predicted Flight Price: **₹{predicted_price:.2f}**")
    
    # Optional: Show a comparison chart
    st.write("📊 Price Analysis")
    st.bar_chart({"Predicted Price": [predicted_price]})

