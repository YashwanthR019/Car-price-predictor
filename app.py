import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load('car_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title('🚗 Used Car Price Predictor')
st.write('Fill in the details below to get an estimated price for your car.')

brand = st.selectbox('Brand', ['Audi', 'BMW', 'Bentley', 'Datsun', 'Ferrari', 'Force', 'Ford',
    'Honda', 'Hyundai', 'ISUZU', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land Rover',
    'Lexus', 'MG', 'Mahindra', 'Maruti', 'Maserati', 'Mercedes-AMG', 'Mercedes-Benz',
    'Mini', 'Nissan', 'Porsche', 'Renault', 'Rolls-Royce', 'Skoda', 'Tata', 'Toyota',
    'Volkswagen', 'Volvo'])

fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual', 'Trustmark Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])

vehicle_age = st.slider('Vehicle Age (years)', 0, 30, 5)
km_driven = st.number_input('Kilometres Driven', min_value=100, max_value=500000, value=30000, step=1000)
mileage = st.number_input('Mileage (km/l)', min_value=4.0, max_value=50.0, value=18.0)
engine = st.number_input('Engine (CC)', min_value=500, max_value=7000, value=1200)
max_power = st.number_input('Max Power (bhp)', min_value=30.0, max_value=700.0, value=80.0)
seats = st.slider('Seats', 2, 9, 5)

if st.button('Predict Price 🚀'):
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0

    input_data['vehicle_age'] = vehicle_age
    input_data['km_driven'] = km_driven
    input_data['mileage'] = mileage
    input_data['engine'] = engine
    input_data['max_power'] = max_power
    input_data['seats'] = seats

    input_data[f'brand_{brand}'] = 1
    input_data[f'fuel_type_{fuel_type}'] = 1
    input_data[f'seller_type_{seller_type}'] = 1
    input_data[f'transmission_type_{transmission}'] = 1

    # Main prediction
    prediction = model.predict(input_data)[0]

    # Confidence range using individual tree predictions
    tree_predictions = [tree.predict(input_data)[0] for tree in model.estimators_]
    lower = np.percentile(tree_predictions, 10)
    upper = np.percentile(tree_predictions, 90)

    st.success(f'### Estimated Price: ₹{prediction:,.0f}')
    st.write(f'That is approximately ₹{prediction/100000:.1f} Lakhs')
    st.info(f'📊 Confidence Range: ₹{lower:,.0f} — ₹{upper:,.0f}')
    st.write(f'_(between ₹{lower/100000:.1f}L and ₹{upper/100000:.1f}L)_')

# Feature importance chart
st.subheader('📊 What Affects Car Price the Most?')
feat_df = pd.DataFrame({'Feature': model_columns, 'Importance': model.feature_importances_})
feat_df = feat_df.sort_values('Importance', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue')
ax.set_xlabel('Importance Score')
ax.set_title('Top 10 Features That Affect Car Price')
ax.invert_yaxis()
plt.tight_layout()
st.pyplot(fig)
