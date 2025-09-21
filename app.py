import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoder
@st.cache_resource
def load_model_and_encoder():
    model = joblib.load('Bengaluru_House_Data.pkl')
    encoder = joblib.load('location_encoder.pkl')
    return model, encoder

model, le = load_model_and_encoder()

# Load dataset for unique locations
data = pd.read_csv('Bengaluru_House_Data.csv')
locations = sorted(data['location'].dropna().unique())  # Get unique locations for dropdown

# App title and description
st.title("Bangalore House Price Predictor")
st.write("Enter house details to predict the price in INR Lakhs.")

# Input fields (matching model features)
total_sqft = st.number_input("Total Square Feet", min_value=300.0, max_value=30000.0, value=1000.0, step=10.0)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)
bhk = st.selectbox("Number of Bedrooms (BHK)", options=[1, 2, 3, 4, 5, 6])
location = st.selectbox("Location", options=locations)

# Prediction button
if st.button("Predict Price"):
    # Convert inputs to model-compatible format
    try:
        # Handle total_sqft (mimic notebook's convert_sqft)
        sqft = float(total_sqft)
        
        # Encode location
        location_encoded = le.transform([location])[0]
        
        # Create input DataFrame (match training feature order)
        input_data = pd.DataFrame({
            'total_sqft': [sqft],
            'bath': [bath],
            'balcony': [balcony],
            'BHK': [bhk],
            'location': [location_encoded]
        })
        
        # Predict and display
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Price: ₹{prediction:.2f} Lakhs")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")