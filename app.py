import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. Configuration and Setup ---
st.set_page_config(
    page_title="AstroSearch: Exoplanet Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained model and scaler
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/xgb_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or Scaler not found! Please run 'train_model.py' first.")
        st.stop()

# Function to load a subset of the original data for visualization
@st.cache_data
def load_visualization_data():
    # Load the full data and filter for CONFIRMED/FALSE POSITIVE for visualization context
    DATA_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&select=koi_disposition,koi_depth,koi_period&format=csv"
    df = pd.read_csv(DATA_URL, comment='#')
    df_vis = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    df_vis.dropna(subset=['koi_depth', 'koi_period'], inplace=True)
    return df_vis

# --- Load once at startup ---
xgb_model, scaler = load_assets()
df_vis = load_visualization_data()

# Feature mapping (must match the order used in train_model.py)
FEATURE_MAP = {
    'Orbital Period (days)': 'koi_period',
    'Transit Duration (hours)': 'koi_duration',
    'Transit Depth (ppm)': 'koi_depth',
    'Impact Parameter': 'koi_impact',
    'Signal-to-Noise Ratio (SNR)': 'koi_model_snr',
    'Stellar Effective Temp (K)': 'koi_steff',
    'Stellar Radius (Solar Radii)': 'koi_srad'
}

# --- 2. UI/UX Plan (Streamlit-based) ---
st.title("🌌 AstroSearch: Exoplanet Classifier")
st.markdown("""
    **Challenge:** Create an AI/ML model to automatically classify Kepler Objects of Interest (KOIs).
    **Model:** XGBoost Classifier trained on Kepler data.
    **Goal:** Input planetary parameters and predict if the signal is a **Confirmed Exoplanet** or a **False Positive**.
""")

st.markdown("---")

# Dark Mode Aesthetic (using Streamlit's built-in components)
# Input Panel in Sidebar
with st.sidebar:
    st.header("⚙️ Planetary Parameters Input")
    st.markdown("Adjust the sliders below to define the characteristics of the signal you want to classify.")
    
    # Input Widgets - Using suitable min/max values based on typical Kepler data
    input_data = {}
    
    input_data['Orbital Period (days)'] = st.slider(
        'Orbital Period (days)', 
        min_value=0.5, max_value=500.0, value=30.0, step=0.1, 
        help="Time for the planet to orbit the star once."
    )
    input_data['Transit Duration (hours)'] = st.slider(
        'Transit Duration (hours)', 
        min_value=0.5, max_value=24.0, value=5.0, step=0.1,
        help="Length of time the transit lasts."
    )
    input_data['Transit Depth (ppm)'] = st.number_input(
        'Transit Depth (ppm)', 
        min_value=10, max_value=100000, value=500, step=10,
        help="The drop in stellar brightness (parts per million)."
    )
    input_data['Impact Parameter'] = st.slider(
        'Impact Parameter', 
        min_value=0.0, max_value=1.5, value=0.5, step=0.01,
        help="The normalized distance from the center of the star to the center of the planet's transit path."
    )
    input_data['Signal-to-Noise Ratio (SNR)'] = st.number_input(
        'Signal-to-Noise Ratio (SNR)', 
        min_value=7.1, max_value=1000.0, value=100.0, step=1.0,
        help="A measure of the detection's strength."
    )
    input_data['Stellar Effective Temp (K)'] = st.slider(
        'Stellar Effective Temp (K)', 
        min_value=2500, max_value=10000, value=5777, step=100,
        help="Effective temperature of the host star."
    )
    input_data['Stellar Radius (Solar Radii)'] = st.slider(
        'Stellar Radius (Solar Radii)', 
        min_value=0.1, max_value=5.0, value=1.0, step=0.01,
        help="Radius of the host star relative to the Sun's radius."
    )

    # Convert the collected inputs into the model's required DataFrame format
    input_df = pd.DataFrame([input_data])
    input_df.columns = list(FEATURE_MAP.values()) # Ensure column names match trained model features

# --- 3. Prediction Workflow ---
if st.button('✨ Run AstroSearch Prediction', type='primary'):
    
    # 3.1. Preprocess Input
    # Scale the input data using the saved scaler
    scaled_input = scaler.transform(input_df)
    
    # 3.2. Run Prediction
    # Predict probability of being a CONFIRMED exoplanet (class 1)
    prediction_proba = xgb_model.predict_proba(scaled_input)[0][1]
    
    # Binary classification (0.5 threshold)
    prediction_class = 'CONFIRMED EXOPLANET' if prediction_proba >= 0.5 else 'FALSE POSITIVE'
    
    # 3.3. Result Display
    st.header("🎯 Prediction Result")
    
    col_result, col_conf = st.columns(2)
    
    # Use distinct icons/colors for impact
    if prediction_class == 'CONFIRMED EXOPLANET':
        emoji = "✅"
        color_style = "background-color: #28a745; color: white; padding: 15px; border-radius: 10px;" # Green
    else:
        emoji = "❌"
        color_style = "background-color: #dc3545; color: white; padding: 15px; border-radius: 10px;" # Red

    with col_result:
        st.markdown(f"""
        <div style="{color_style}; text-align: center;">
            <h3 style="color: white; margin: 0;">{emoji} CLASSIFICATION:</h3>
            <h1 style="color: white; margin: 0; font-size: 3em;">{prediction_class}</h1>
        </div>
        """, unsafe_allow_html=True)
        
    with col_conf:
        st.metric(
            label="Confidence Score (Probability of Confirmed)", 
            value=f"{prediction_proba*100:.2f}%", 
            delta_color="off"
        )
        st.info("The Confidence Score is the model's certainty that the signal is a **Confirmed Exoplanet**.")

    st.markdown("---")

    # --- 4. Visualization & Educational Features ---
    st.header("📊 Context & Exploration")
    
    # 4.1. Feature Importance Chart
    st.subheader("Model Feature Importance")
    # Get feature importances from the trained model
    feature_importances = pd.Series(xgb_model.feature_importances_, index=list(FEATURE_MAP.values()))
    fig_importance = px.bar(
        feature_importances.sort_values(ascending=True),
        orientation='h',
        title='Which Features Drive the Prediction?',
        labels={'value': 'Importance Score', 'index': 'Feature Name'}
    ).update_layout(xaxis_title='Importance Score', showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # 4.2. Scatter Plot of Data
    st.subheader("Your Input Relative to Historical Data")
    
    # Define the user point for the plot
    user_point = pd.DataFrame({
        'koi_period': [input_df['koi_period'].iloc[0]],
        'koi_depth': [input_df['koi_depth'].iloc[0]],
        'koi_disposition': ['YOUR INPUT']
    })
    
    # Combine historical data and user input
    plot_data = pd.concat([df_vis[['koi_period', 'koi_depth', 'koi_disposition']], user_point], ignore_index=True)
    
    # Create the scatter plot
    fig_scatter = px.scatter(
        plot_data, 
        x='koi_period', 
        y='koi_depth', 
        color='koi_disposition',
        log_x=True, # Period is better viewed on a log scale
        log_y=True, # Depth is often better viewed on a log scale
        hover_data=['koi_period', 'koi_depth'],
        title='Exoplanet Period vs. Transit Depth (Log Scale)',
        labels={'koi_period': 'Orbital Period (days, Log Scale)', 'koi_depth': 'Transit Depth (ppm, Log Scale)'},
        color_discrete_map={'CONFIRMED': '#1f77b4', 'FALSE POSITIVE': '#ff7f0e', 'YOUR INPUT': '#e30022'} # Custom colors
    )
    # Highlight the user's point more prominently
    fig_scatter.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')), selector=dict(name='YOUR INPUT'))

    st.plotly_chart(fig_scatter, use_container_width=True)

# --- 5. Exoplanet Glossary (Optional but great for education) ---
with st.expander("📚 Exoplanet Glossary"):
    st.markdown("""
    - **KOI (Kepler Object of Interest):** A star observed by the Kepler mission that appears to have a transiting planet candidate.
    - **Transit:** The event where a planet passes in front of its star, causing a measurable dip in the star's brightness.
    - **Transit Depth (ppm):** The percentage (in parts per million) of the star's light that is blocked by the planet. Larger planets cause deeper transits.
    - **Impact Parameter:** A measure of how centrally the transit crosses the star's disc. A value of 0 is a perfect center-crossing.
    - **SNR (Signal-to-Noise Ratio):** The ratio of the strength of the exoplanet signal (the transit) to the background noise. A high SNR means a more reliable detection.
    """)
