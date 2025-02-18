import streamlit as st
import numpy as np
import joblib
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page settings
st.set_page_config(
    page_title="Smart Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('model/model.pkl')

model = load_model()

# Define categorical mapping based on encoding
airline_map = {'AirAsia': 0, 'Air India': 1, 'Go First': 2, 'Indigo': 3, 'SpiceJet': 4, 'Vistara': 5}
source_city_map = {'Bangalore': 0, 'Chennai': 1, 'Delhi': 2, 'Hyderabad': 3, 'Kolkata': 4, 'Mumbai': 5}
destination_city_map = {'Bangalore': 0, 'Chennai': 1, 'Delhi': 2, 'Hyderabad': 3, 'Kolkata': 4, 'Mumbai': 5}
departure_time_map = {'Afternoon': 0, 'Early Morning': 1, 'Evening': 2, 'Late Night': 3, 'Morning': 4, 'Night': 5}
arrival_time_map = {'Afternoon': 0, 'Early Morning': 1, 'Evening': 2, 'Late Night': 3, 'Morning': 4, 'Night': 5}
stops_map = {'Zero': 2, 'One': 0, 'Two or more': 1}
class_map = {'Business': 0, 'Economy': 1}

# Enhanced CSS for modern UI
st.markdown("""
    <style>
        /* Global Styles */
        .stApp {
            background: linear-gradient(120deg, #1a1a2e 0%, #16213e 100%);
            color: #ffffff;
        }
        
        /* Header Styling */
        .main-header {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .title {
            font-size: 48px;
            font-weight: 800;
            background: linear-gradient(120deg, #FFD700, #FFA500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 18px;
            color: #b8b9ba;
            text-align: center;
        }
        
        /* Input Container Styling */
        .input-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Custom Select Box */
        .stSelectbox > div > div {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 10px !important;
            color: white !important;
        }
        
        /* Custom Slider */
        .stSlider > div > div {
            color: #FFD700 !important;
        }
        
        /* Prediction Result Container */
        .prediction-container {
            background: rgba(255, 215, 0, 0.1);
            border-radius: 15px;
            padding: 25px;
            margin-top: 30px;
            text-align: center;
            border: 2px solid #FFD700;
            animation: glow 2s infinite alternate;
        }
        
        @keyframes glow {
            from {
                box-shadow: 0 0 10px -10px #FFD700;
            }
            to {
                box-shadow: 0 0 20px 5px #FFD700;
            }
        }
        
        /* Custom Button */
        .stButton > button {
            background: linear-gradient(90deg, #FFD700, #FFA500) !important;
            color: black !important;
            font-weight: bold !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 15px 30px !important;
            font-size: 18px !important;
            transition: transform 0.3s ease !important;
            width: 100% !important;
        }
        
        .stButton > button:hover {
            transform: scale(1.05) !important;
        }
        
        /* Charts Container */
        .chart-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            margin-top: 30px;
        }
        
        /* Tooltip Styling */
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted white;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="main-header">
        <h1 class="title">✈️ Smart Flight Price Predictor</h1>
        <p class="subtitle">Advanced AI-powered flight price prediction system with real-time analytics</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2 = st.tabs(["🎯 Price Prediction", "📊 Analytics"])

with tab1:
    # Main prediction section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        airline = st.selectbox("✈️ Select Airline", list(airline_map.keys()))
        source_city = st.selectbox("🛫 Departure City", list(source_city_map.keys()))
        departure_time = st.selectbox("🕐 Preferred Departure Time", list(departure_time_map.keys()))
        stops = st.selectbox("🔄 Number of Stops", list(stops_map.keys()))
        duration = st.slider("⏱️ Flight Duration (hours)", 1.0, 49.0, 2.5, 0.5)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        destination_city = st.selectbox("🛬 Arrival City", list(destination_city_map.keys()))
        arrival_time = st.selectbox("🕒 Preferred Arrival Time", list(arrival_time_map.keys()))
        flight_class = st.selectbox("💺 Travel Class", list(class_map.keys()))
        days_left = st.slider("📅 Days until Flight", 1, 49, 7)
        st.markdown('</div>', unsafe_allow_html=True)

    # Predict button
    if st.button("🔮 Predict Flight Price"):
        with st.spinner("🔄 Analyzing market data and computing best prices..."):
            # Input data preparation
            airline_encoded = airline_map[airline]
            source_encoded = source_city_map[source_city]
            destination_encoded = destination_city_map[destination_city]
            departure_encoded = departure_time_map[departure_time]
            arrival_encoded = arrival_time_map[arrival_time]
            stops_encoded = stops_map[stops]
            class_encoded = class_map[flight_class]

            input_data = np.array([
                airline_encoded, source_encoded, departure_encoded, stops_encoded, 
                arrival_encoded, destination_encoded, class_encoded, duration, days_left
            ]).reshape(1, -1)

            # Make prediction
            predicted_price = model.predict(input_data)[0]
            
            # Generate confidence interval (mock data for demonstration)
            lower_bound = predicted_price * 0.95
            upper_bound = predicted_price * 1.05

            # Display prediction results
            st.markdown("""
                <div class="prediction-container">
                    <h2 style="color: #FFD700; font-size: 36px;">Predicted Price</h2>
                    <h1 style="font-size: 48px; margin: 20px 0;">₹{:,.2f}</h1>
                    <p style="color: #b8b9ba;">Expected price range: ₹{:,.2f} - ₹{:,.2f}</p>
                </div>
            """.format(predicted_price, lower_bound, upper_bound), unsafe_allow_html=True)

            # Price Trend Analysis
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📈 Price Trend Analysis")
            
            # Generate mock historical data
            dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
            historical_prices = [predicted_price * (1 + np.random.normal(0, 0.02)) for _ in range(30)]
            
            df_historical = pd.DataFrame({
                'Date': dates,
                'Price': historical_prices
            })

            fig = px.line(df_historical, x='Date', y='Price',
                         title='30-Day Price Trend',
                         template='plotly_dark')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Price Comparison
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("🔄 Price Comparison by Airlines")
            
            # Generate mock comparison data
            airlines = list(airline_map.keys())
            comparison_prices = [predicted_price * (1 + np.random.normal(0, 0.1)) for _ in airlines]
            
            fig_comparison = go.Figure(data=[
                go.Bar(x=airlines, y=comparison_prices,
                      marker_color=['#FFD700' if a == airline else '#4169E1' for a in airlines])
            ])
            fig_comparison.update_layout(
                title='Price Comparison Across Airlines',
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Show success animation
            st.balloons()

with tab2:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.subheader("📊 Flight Price Analytics")
    
    # Generate mock analytics data
    analytics_data = {
        'Time of Day': ['Morning', 'Afternoon', 'Evening', 'Night'],
        'Average Price': [np.random.normal(8000, 1000) for _ in range(4)]
    }
    
    fig_analytics = px.bar(analytics_data, x='Time of Day', y='Average Price',
                          title='Average Price by Time of Day',
                          template='plotly_dark')
    fig_analytics.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_analytics, use_container_width=True)
    
    # Route Analysis
    popular_routes = pd.DataFrame({
        'Route': ['Delhi-Mumbai', 'Bangalore-Delhi', 'Mumbai-Chennai', 'Hyderabad-Bangalore'],
        'Average Price': [np.random.normal(7000, 1000) for _ in range(4)],
        'Popularity': [np.random.randint(70, 100) for _ in range(4)]
    })
    
    st.subheader("🛫 Popular Routes Analysis")
    fig_routes = px.scatter(popular_routes, x='Average Price', y='Popularity', text='Route',
                           title='Route Analysis by Price and Popularity',
                           template='plotly_dark')
    fig_routes.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_routes, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding: 20px; background: rgba(255, 255, 255, 0.05); border-radius: 15px;">
        <p style="color: #b8b9ba;">Made with ❤️ by AI-Powered Flight Price Predictor</p>
        <p style="color: #b8b9ba; font-size: 12px;">Powered by Advanced Machine Learning</p>
    </div>
""", unsafe_allow_html=True)


