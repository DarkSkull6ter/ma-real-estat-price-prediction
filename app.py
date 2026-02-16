import streamlit as st
import pandas as pd
import numpy as np
import joblib
# IMPORT LOGIC
from locations import CITY_DISTRICT_DATA

# ==========================================
# 1. SETUP & ASSET LOADING
# ==========================================
st.set_page_config(page_title="Morocco Real Estate AI", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load('morocco_house_model_v2.pkl')
    prestige_map = joblib.load('district_prestige_map.pkl')
    return model, prestige_map

# ==========================================
# 2. UI DESIGN 
# ==========================================
st.title("🇲🇦 Morocco House Price Predictor")
st.markdown("---")

with st.sidebar:
    st.header("📍 Location")
    city = st.selectbox("Select City", options=list(CITY_DISTRICT_DATA.keys()))
    district = st.selectbox("Select District", options=CITY_DISTRICT_DATA[city])
    category = st.selectbox("Property Type", ["Appartements", "Maisons", "Villas"])

col1, col2 = st.columns(2)

with col1:
    st.header("📐 Dimensions")
    surface = st.number_input("Surface Area (m²)", min_value=10, value=75)
    # Floor: Min 0, Max 20, Default 0 (Most common)
    floor = st.slider("Floor Level", 0, 20, 0)
    
with col2:
    st.header("🚪 Rooms")
    # Bedrooms: Min 1, Max 20, Default 2
    beds = st.slider("Bedrooms", 1, 20, 2)
    # Bathrooms: Min 0, Max 12, Default 2
    baths = st.slider("Bathrooms", 0, 12, 2)
    # Living Rooms: Min 0, Max 12, Default 1
    living = st.slider("Living Rooms", 0, 12, 1)

st.header("✨ Amenities")
c1, c2, c3 = st.columns(3)
elev = c1.checkbox("Elevator")
ac = c2.checkbox("Air Conditioning")
kitchen = c3.checkbox("Fitted Kitchen")

# ==========================================
# 3. PREDICTION ENGINE
# ==========================================
if st.button("💰 Estimate Market Price", use_container_width=True):
    try:
        model, prestige_map = load_assets()
        
        # Format inputs to match training data
        district_norm = district.lower()
        city_district = f"{city} - {district_norm}"
        
        raw_data = {
            'surface_habitable_m2': float(surface),
            'bedrooms': float(beds),
            'bathrooms': float(baths),
            'living_rooms': float(living),
            'floor': float(floor),
            'has_elevator': elev,
            'has_air_conditioning': ac,
            'has_fitted_kitchen': kitchen,
            'has_balcony': False, 'has_parking': False, 'has_security': False, 'has_terrace': False,
            'city': city,
            'category': category,
            'location_precision': 'Precise',
            'district_norm': district_norm,
            'city_district': city_district
        }
        input_df = pd.DataFrame([raw_data])
        
        # Engineering Replication
        fallback_prestige = sum(prestige_map.values()) / len(prestige_map)
        input_df['district_m2_value'] = prestige_map.get(district_norm, fallback_prestige)
        input_df['bath_per_bed'] = input_df['bathrooms'] / (input_df['bedrooms'] + 0.1)
        input_df['room_density'] = (input_df['bedrooms'] + input_df['living_rooms']) / input_df['surface_habitable_m2']
        input_df['modernity_score'] = int(elev) + int(ac) + int(kitchen)
        input_df['size_prestige_score'] = input_df['surface_habitable_m2'] * input_df['district_m2_value']
        
        # Log Transformations
        input_df['surface_habitable_m2'] = np.log1p(input_df['surface_habitable_m2'])
        
        # Prediction
        pred_log = model.predict(input_df)
        final_price = np.expm1(pred_log)[0]
        
        st.balloons()
        st.success(f"### Estimated Price: {final_price:,.0f} DH")
        st.info(f"Market Context: {input_df['district_m2_value'].iloc[0]:,.0f} DH/m² in {district}")

    except Exception as e:
        st.error(f"Error: {e}")