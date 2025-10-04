import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import plotly.express as px
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_PATH = 'models/xgb_model.json' # Ensure this path is correct
DATA_PATH = 'data/kepler_koi_data.csv' # Placeholder, only used for visualization/lookup

# Set Streamlit Page Configuration (Dark Mode Aesthetic)
st.set_page_config(
    page_title="AstroSearch: Exoplanet Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AstroSearch - NASA Space App Challenge 2024"
    }
)

# Function to load model and data (using Streamlit's cache for speed)
@st.cache_resource
def load_resources():
    """Loads the pre-trained XGBoost model and the original dataset for lookup."""
    try:
        # Load the XGBoost model
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)

        # Load the original data (for visualization and random lookup)
        # NOTE: Filter and preprocess this data the same way it was for training
        df = pd.read_csv(DATA_PATH)
        df_clean = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
        df_clean['is_exoplanet'] = df_clean['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)
        # Ensure only the 7 features are kept for simplicity
        features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_impact', 'koi_model_snr', 'koi_steff', 'koi_srad']
        df_clean = df_clean.dropna(subset=features) # simple dropna for visualization purposes

        return model, df_clean, features
    except FileNotFoundError:
        st.error("Error: Model or Data files not found. Check paths.")
        return None, None, None

# Load all necessary components
xgb_model, full_data, features = load_resources()

if xgb_model is not None:
    
    # --- UI/UX: TITLE AND INTRODUCTION ---
    st.title("🌌 AstroSearch: Exoplanet Classifier")
    st.markdown("""
        *Harnessing AI/ML to classify Kepler Objects of Interest (KOI) as either **Confirmed Exoplanet** or **False Positive**.*
        This tool uses a pre-trained **XGBoost Classifier** to analyze key transit parameters.
    """)
    st.divider()

    # --- MAIN CONTENT TABS ---
    tab1, tab2, tab3 = st.tabs(["🚀 Prediction Tool", "📊 Model Performance & Data", "📚 Exoplanet Glossary"])

    with tab1:
        st.header("1. Enter Exoplanet Parameters")
        
        # --- SIDEBAR: INPUT & RANDOM SELECTION ---
        st.sidebar.header("🎯 Parameter Input")
        
        # Random Exoplanet Selection
        if st.sidebar.button("✨ Load Random Confirmed Exoplanet"):
            random_exoplanet = full_data[full_data['is_exoplanet'] == 1].sample(1).iloc[0]
            st.session_state.initial_params = random_exoplanet[features].to_dict()
        
        # Initialize session state for inputs
        if 'initial_params' not in st.session_state:
             st.session_state.initial_params = full_data[features].median().to_dict()


        # Input Fields in the Sidebar
        col_input_1, col_input_2 = st.sidebar.columns(2)

        with col_input_1:
            period = st.number_input("Orbital Period (days)", format="%.4f", value=st.session_state.initial_params['koi_period'], min_value=0.0)
            duration = st.number_input("Transit Duration (hrs)", format="%.2f", value=st.session_state.initial_params['koi_duration'], min_value=0.0)
            depth = st.number_input("Transit Depth (ppm)", value=st.session_state.initial_params['koi_depth'], min_value=0.0)
            st.markdown("---")
            temp = st.slider("Stellar Temperature (K)", min_value=2000, max_value=12000, value=int(st.session_state.initial_params['koi_steff']))

        with col_input_2:
            impact = st.slider("Impact Parameter", min_value=0.0, max_value=2.0, format="%.2f", value=st.session_state.initial_params['koi_impact'])
            snr = st.number_input("Signal-to-Noise Ratio (SNR)", value=int(st.session_state.initial_params['koi_model_snr']), min_value=0)
            srad = st.number_input("Stellar Radius (Solar Radii)", format="%.2f", value=st.session_state.initial_params['koi_srad'], min_value=0.0)

        # --- PREDICTION LOGIC ---
        user_input_array = np.array([[period, duration, depth, impact, snr, temp, srad]])
        user_df = pd.DataFrame(user_input_array, columns=features)

        if st.sidebar.button("🔎 **CLASSIFY OBJECT**", type="primary"):
            
            # Predict
            prediction_proba = xgb_model.predict_proba(user_df)[:, 1][0]
            prediction_class = xgb_model.predict(user_df)[0]
            
            # Display Result
            col_result, col_vis = st.columns([1, 2])

            with col_result:
                st.subheader("Classification Result")
                if prediction_class == 1:
                    st.success(f"✅ **CONFIRMED EXOPLANET**", icon="🪐")
                else:
                    st.error(f"❌ **FALSE POSITIVE**", icon="🚧")
                
                st.metric("Confidence Score", f"{prediction_proba * 100:.2f}%")
                
                st.info("The prediction is based on the correlation of your input parameters with the confirmed and false positive transits in the Kepler data.")


            with col_vis:
                st.subheader("Data Context: Transit Depth vs. Period")
                
                # Plot User Point vs Historical Data (using Plotly for interactivity)
                fig = px.scatter(
                    full_data, 
                    x='koi_period', 
                    y='koi_depth', 
                    color='koi_disposition', 
                    log_x=True,
                    title='Historical KOI Data (Transit Depth vs. Period)',
                    color_discrete_map={'CONFIRMED': 'green', 'FALSE POSITIVE': 'red', 'CANDIDATE': 'yellow'},
                    hover_data=['koi_period', 'koi_depth', 'koi_disposition']
                )
                
                # Add the user's input point
                fig.add_scatter(x=[period], y=[depth], mode='markers', 
                                name='Your Input', marker=dict(size=15, color='white', symbol='star'))
                
                st.plotly_chart(fig, use_container_width=True)


    with tab2:
        st.header("2. Model Performance & Feature Analysis")
        
        # --- FEATURE IMPORTANCE CHART ---
        st.subheader("A. Feature Importance (How the Model Decides)")
        
        # NOTE: This requires running feature importance extraction during or after training
        # For a hackathon, hardcode typical importances if time is short, but dynamic is better.
        try:
            # Dynamic Feature Importance (Requires 'feature_importances_' from a fitted model)
            importances = xgb_model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                title="XGBoost Feature Importance Score",
                color='Importance',
                color_continuous_scale=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            st.caption("Transit Depth and Signal-to-Noise Ratio (SNR) are typically the strongest indicators.")
            
        except:
             st.warning("Feature importance data not fully available in the loaded model. Skipping chart.")


    with tab3:
        st.header("3. Exoplanet Terminology Glossary")
        st.markdown("A quick reference for the terms used in the Kepler datasets.")

        st.subheader("Key Terms")
        st.markdown(
            """
            * **Kepler Object of Interest (KOI):** Any star observed by the Kepler mission that shows transit-like features, making it a candidate for hosting a planet.
            * **Transit Method:** The primary method used by Kepler and TESS, where a planet passes in front of its star, causing a temporary, periodic dip in the star's brightness.
            * **Orbital Period:** The time it takes for the exoplanet to complete one full orbit around its star (measured in Earth days).
            * **Transit Duration:** The length of time the planet takes to cross the face of the star (measured in hours).
            * **Transit Depth:** The maximum dip in the star's light, often measured in parts per million (ppm). **This is a key indicator of planet size relative to the star.**
            * **Signal-to-Noise Ratio (SNR):** A measure of the strength of the transit signal relative to the noise (random fluctuations) in the light curve. **A higher SNR means a more confident detection.**
            * **Impact Parameter:** The distance between the center of the stellar disk and the center of the planet's path as it crosses the star, normalized by the stellar radius.
            """
        )
