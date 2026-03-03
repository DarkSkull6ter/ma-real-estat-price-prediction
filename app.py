import streamlit as st
import pandas as pd
import numpy as np
import joblib
from locations import CITY_DISTRICT_DATA

# ==========================================
# 1. CHARGEMENT DES ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoders = joblib.load('models/target_encoders.pkl')
    return model, scaler, encoders

# Configuration
st.set_page_config(page_title="IA Immobilier Maroc", layout="centered")
st.title("🇲🇦 Morocco House Price Predictor")

# ==========================================
# 2. INTERFACE UTILISATEUR
# ==========================================
with st.sidebar:
    st.header("📍 Localisation")
    city = st.selectbox("Ville", options=list(CITY_DISTRICT_DATA.keys()))
    district = st.selectbox("Quartier", options=CITY_DISTRICT_DATA[city])
    category = st.selectbox("Type", ["Appartements", "Maisons", "Villas-Riads"])

col1, col2 = st.columns(2)
with col1:
    surface = st.number_input("Surface (m²)", min_value=10, value=80)
    floor = st.slider("Étage", 0, 20, 1)
with col2:
    beds = st.slider("Chambres", 1, 10, 2)
    baths = st.slider("Salles de bain", 1, 10, 1)
    living = st.slider("Salons", 0, 10, 1)

st.header("✨ Équipements")
c1, c2, c3, c4 = st.columns(4)
elev = c1.checkbox("Ascenseur")
ac = c2.checkbox("Climatisation")
kitchen = c3.checkbox("Cuisine équipée")
balcony = c4.checkbox("Balcon")

c5, c6, c7 = st.columns(3)
parking = c5.checkbox("Parking")
security = c6.checkbox("Sécurité")
terrace = c7.checkbox("Terrasse")

# ==========================================
# 3. MOTEUR DE PRÉDICTION
# ==========================================
if st.button("💰 Estimer le prix", use_container_width=True):
    try:
        model, scaler, encoders = load_assets()
        
        # A. Préparation des données brutes
        total_rooms = beds + living
        input_data = {
            'surface_habitable_m2': float(surface),
            'bedrooms': float(beds),
            'bathrooms': float(baths),
            'living_rooms': float(living),
            'floor': float(floor),
            'total_rooms': float(total_rooms),
            'room_to_surface_ratio': float(total_rooms / surface) if surface > 0 else 0.0,
            'bathroom_to_bedroom_ratio': float(baths / beds) if beds > 0 else 0.0,
            'total_amenities': float(sum([elev, balcony, ac, kitchen, parking, security, terrace])),
            'has_elevator': int(elev), 
            'has_balcony': int(balcony), 
            'has_air_conditioning': int(ac),
            'has_fitted_kitchen': int(kitchen), 
            'has_parking': int(parking), 
            'has_security': int(security), 
            'has_terrace': int(terrace),
            'city': str(city), 
            'category': str(category), 
            'district_norm': str(district).lower()
        }
        
        df = pd.DataFrame([input_data])

        # B. ÉTAPE 1 : Target Encoding (AVANT le Scaling, comme dans votre Colab)
        for col in ['city', 'category', 'district_norm']:
            df[col] = encoders[col].transform(df[[col]].astype(str))

        # C. ÉTAPE 2 : Scaling (Uniquement sur les colonnes numériques)
        # Selon votre Colab : numeric_cols = X.select_dtypes(include=np.number).columns
        # Cela inclut les colonnes numériques originales + les équipements (int) + les encodées
        numeric_cols = [
            'surface_habitable_m2', 'bedrooms', 'bathrooms', 'living_rooms', 'floor',
            'total_rooms', 'room_to_surface_ratio', 'bathroom_to_bedroom_ratio', 'total_amenities',
            'has_elevator', 'has_balcony', 'has_air_conditioning', 'has_fitted_kitchen',
            'has_parking', 'has_security', 'has_terrace', 'city', 'category', 'district_norm'
        ]
        
        # IMPORTANT : On réordonne pour correspondre au fit du Scaler
        df = df[numeric_cols]
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # D. ÉTAPE 3 : Inférence
        # Comme vous n'avez pas fait de log1p sur 'y' dans FLAML, on prédit directement
        final_price = model.predict(df)[0]

        # Résultat
        st.balloons()
        st.success(f"### Prix Estimé : {max(0, final_price):,.0f} DH")
        st.info("Note : Ce modèle a été entraîné avec FLAML (R²: 0.73)")

    except Exception as e:
        st.error(f"Erreur technique : {e}")