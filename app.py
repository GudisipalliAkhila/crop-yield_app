import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------------
# Load the saved model and supporting files
# ---------------------------------------------------------
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

with open('unique_crops.pkl', 'rb') as f:
    unique_crops = pickle.load(f)

with open('unique_areas.pkl', 'rb') as f:
    unique_areas = pickle.load(f)

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")
st.title("🌾 Crop Yield Prediction App")
st.write("Enter the details below to predict crop yield (hg/hectare).")

# ---------------------------------------------------------
# Input fields
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    area = st.selectbox("Country/Area", unique_areas)
    item = st.selectbox("Crop Type", unique_crops)
    year = st.number_input("Year", min_value=1990, max_value=2030, value=2024)

with col2:
    rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0, value=1000.0)
    temp = st.number_input("Average Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
    pesticides = st.number_input("Pesticides Used (tonnes)", min_value=0.0, value=1000.0)

# ---------------------------------------------------------
# Predict button
# ---------------------------------------------------------
if st.button("Predict Yield", type="primary"):
    # Build a single-row dataframe matching the training structure
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Fill in numeric values
    input_df['Year'] = year
    input_df['average_rain_fall_mm_per_year'] = rainfall
    input_df['avg_temp'] = temp
    input_df['pesticides_tonnes'] = pesticides

    # Fill in one-hot encoded columns (Area_ and Item_)
    area_col = f'Area_{area}'
    item_col = f'Item_{item}'

    if area_col in input_df.columns:
        input_df[area_col] = 1
    if item_col in input_df.columns:
        input_df[item_col] = 1

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"### Predicted Yield: {prediction:,.0f} hg/hectare")
    st.caption(f"That's approximately {prediction/10000:,.2f} tonnes/hectare")

# ---------------------------------------------------------
# Footer note
# ---------------------------------------------------------
st.markdown("---")
st.caption("Built with Random Forest Regression | Data: FAO/World Bank crop yield dataset")
