import streamlit as st
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_model_and_encoder():
    model = joblib.load('Bengaluru_House_Data.pkl')
    le = joblib.load('location_encoder.pkl')  # LabelEncoder instance
    return model, le

model, le = load_model_and_encoder()

# Keep only string locations, ignore floats/NaN
locations = [loc for loc in le.classes_.tolist() if isinstance(loc, str)]
locations = sorted(locations)


st.title("Bangalore House Price Predictor")
st.write("Enter house details to predict the price in INR Lakhs.")

total_sqft = st.number_input("Total Square Feet", min_value=300.0, max_value=30000.0, value=1000.0, step=10.0)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)
bhk = st.selectbox("Number of Bedrooms (BHK)", options=[1, 2, 3, 4, 5, 6])
location = st.selectbox("Location", options=locations)

if st.button("Predict Price"):
    try:
        sqft = float(total_sqft)
        # encode location to the integer label your model expects
        location_encoded = le.transform([location])[0]

        input_data = pd.DataFrame({
            'total_sqft': [sqft],
            'bath': [bath],
            'balcony': [balcony],
            'BHK': [bhk],
            'location': [location_encoded]
        })

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Price: ₹{prediction:.2f} Lakhs")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
