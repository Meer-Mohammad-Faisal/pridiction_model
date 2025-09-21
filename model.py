# %% [markdown]
# # Banglore House Price Prediction
# 

# %%
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder 
import joblib

# %%
df = pd.read_csv('Bengaluru_House_Data.csv')
df

# %%
# checking data types
print(df.dtypes)

# %%
### 1️⃣ Data Cleaning
df = df.drop(['area_type', 'society', 'availability'], axis=1)  # Drop unnecessary columns

# %%
# Handle missing values
df = df.dropna(subset=['size', 'total_sqft', 'price'])  # Drop rows with missing critical values
df['bath'].fillna(df['bath'].median(), inplace=True)
df['balcony'].fillna(df['balcony'].median(), inplace=True)

# %%
# Convert 'size' (e.g., "2 BHK" → 2)
df['BHK'] = df['size'].apply(lambda x: int(x.split(' ')[0]))  # Extract first number
df = df.drop(['size'], axis=1)  # Drop old size column

# %%
# Convert 'total_sqft' to numeric (handling ranges like "600 - 1000")
def convert_sqft(value):
    try:
        if '-' in value:
            vals = list(map(float, value.split('-')))
            return (vals[0] + vals[1]) / 2
        return float(value)
    except:
        return None  # Return None for invalid values

# %%
df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna(subset=['total_sqft'])  # Drop rows where sqft couldn't be converted

# %% [markdown]
# # Using Feature Engineering

# %%
###  Feature Engineering
# Encode 'location' as numeric (Label Encoding)
le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])
joblib.dump(le, 'location_encoder.pkl')

# %%
###  Train Linear Regression Model
# Define input (X) and target (y)
X = df[['total_sqft', 'bath', 'balcony', 'BHK', 'location']]
y = df['price']

# %%
# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# %% [markdown]
# 

# %% [markdown]
# # Evaluting Model

# %%
# Evaluate model
score = model.score(X_test, y_test)  # R² score
print(f"Model Accuracy (R² Score): {score:.2f}")

# %% [markdown]
# # Making prediction

# %%
# Make a prediction (example: 1000 sqft, 2 bath, 1 balcony, 2 BHK, location=5)
predicted_price = model.predict([[1000, 2, 1, 2, 5]])
print(f"Predicted Price: ₹{predicted_price[0]:.2f} Lakhs")

# %%
print(f"predicted_price: ₹{model.predict([[1000, 1, 1, 1, 5]])} Lakhs")




#save model
joblib.dump(model, 'Bengaluru_House_Data.pkl')
print("Model saved as Bengaluru_House_Data.pkl")