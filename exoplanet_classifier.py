import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import altair as alt

# --- Configuration and Constants ---
# NASA Exoplanet Archive TAP API URL
# We will query the 'cumulative' table (KOI) which contains disposition, period, radius, etc.
NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
DEFAULT_FEATURES = [
    'koi_period', 'koi_duration', 'koi_prad', 'koi_teq', 'koi_insol',
    'koi_srad', 'koi_slogg', 'koi_steff', 'koi_model_snr'
]
TARGET_COLUMN = 'koi_disposition'
BINARY_MAP = {'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}

@st.cache_data
def load_and_preprocess_training_data():
    """
    Simulates loading and preparing a classic Kepler dataset for training.
    
    NOTE: In a real-world scenario, you would download the full KOI cumulative table
    (e.g., from NASA Exoplanet Archive) and save it as a CSV. Here, we simulate 
    a small, clean DataFrame structure.
    """
    st.info("Simulating loading and preparing the NASA KOI training dataset...")

    # Create synthetic data structure based on common Kepler/KOI features
    n_samples = 500
    data = {
        'koi_disposition': np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], size=n_samples, p=[0.2, 0.3, 0.5]),
        'koi_period': np.random.lognormal(2, 1, size=n_samples),
        'koi_duration': np.random.rand(n_samples) * 10,
        'koi_prad': np.random.rand(n_samples) * 10 + 0.5,
        'koi_teq': np.random.normal(500, 200, size=n_samples),
        'koi_insol': np.random.lognormal(2, 0.5, size=n_samples),
        'koi_srad': np.random.normal(1, 0.5, size=n_samples),
        'koi_slogg': np.random.normal(4.5, 0.5, size=n_samples),
        'koi_steff': np.random.normal(5700, 500, size=n_samples),
        'koi_model_snr': np.random.rand(n_samples) * 100
    }
    df = pd.DataFrame(data)
    
    # Introduce some NaN values to test the Imputer
    df.loc[df.sample(frac=0.05).index, 'koi_slogg'] = np.nan
    df.loc[df.sample(frac=0.03).index, 'koi_period'] = np.nan
    
    # Target variable mapping: Binary Classification (Exoplanet=1, False Positive=0)
    df['target'] = df[TARGET_COLUMN].map(BINARY_MAP)
    
    # Imputation: Fill missing values with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    df[DEFAULT_FEATURES] = imputer.fit_transform(df[DEFAULT_FEATURES])
    
    # Scaling the features
    scaler = StandardScaler()
    df[DEFAULT_FEATURES] = scaler.fit_transform(df[DEFAULT_FEATURES])
    
    return df, imputer, scaler

@st.cache_resource
def train_adaboost_model(df_train):
    """
    Trains the AdaBoost Classifier.
    """
    X = df_train[DEFAULT_FEATURES]
    y = df_train['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize the base estimator (Decision Tree) and the AdaBoost Classifier
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    model = AdaBoostClassifier(
        estimator=base_estimator, 
        n_estimators=100, 
        learning_rate=0.5,
        algorithm='SAMME',
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['False Positive', 'Exoplanet'], output_dict=True)
    
    return model, accuracy, report

def fetch_exoplanet_data(koi_list):
    """
    Fetches data for a list of KOI IDs from the NASA Exoplanet Archive.
    """
    koi_ids = ','.join([f"'{koi}'" for koi in koi_list])
    
    # Select the columns we need, including the disposition for verification
    select_cols = ','.join(['kepid', 'koi_name', TARGET_COLUMN] + DEFAULT_FEATURES)
    
    # Construct the ADQL query
    query = f"SELECT {select_cols} FROM cumulative WHERE koi_name IN ({koi_ids})"
    
    params = {
        'query': query,
        'format': 'csv'
    }
    
    st.info(f"Querying NASA Exoplanet Archive for {len(koi_list)} objects...")
    
    try:
        response = requests.get(NASA_TAP_URL, params=params, timeout=30)
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        
        # Read the response text directly into a Pandas DataFrame
        df = pd.read_csv(StringIO(response.text))
        
        if df.empty or len(df) == 0:
            st.error("The NASA archive returned no data for the provided IDs. Please ensure the IDs are valid KOI (Kepler Object of Interest) names (e.g., K00001.01).")
            return None
        
        # Clean up column names by removing surrounding quotes and spaces if present
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        st.success(f"Successfully fetched {len(df)} data points.")
        return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from NASA Exoplanet Archive: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during data processing: {e}")
        return None

def analyze_prediction(row, prediction, probability):
    """Generates a detailed, justified explanation for the classification."""
    
    is_exoplanet = (prediction == 1)
    status = "Exoplanet (Confirmed/Candidate)" if is_exoplanet else "False Positive"
    
    # Determine the primary factor based on planetary radius and SNR
    prad = row['koi_prad'].iloc[0] if isinstance(row, pd.DataFrame) else row['koi_prad']
    snr = row['koi_model_snr'].iloc[0] if isinstance(row, pd.DataFrame) else row['koi_model_snr']
    
    justification = f"The model classified this object as **{status}** with a confidence of **{probability:.2%}**."
    
    if is_exoplanet:
        if prad < 2.0:
            justification += " The prediction is strongly supported by the small planetary radius ($R_p < 2.0$ $R_E$), which is characteristic of smaller, likely rocky exoplanets."
        elif prad >= 5.0:
            justification += " The object has a large planetary radius ($R_p \\approx {prad:.2f}$ $R_E$), typical of gas giants or ice giants."
        else:
            justification += f" The planetary radius is moderate ($R_p \\approx {prad:.2f}$ $R_E$), suggesting a Neptune-like or super-Earth body."
            
        justification += f" Additionally, the Signal-to-Noise Ratio (SNR) of $\\approx {snr:.1f}$ is high, indicating a strong transit signal."
    else:
        justification += " This classification is often driven by stellar properties and transit geometry."
        if snr < 10.0:
            justification += " The low Signal-to-Noise Ratio (SNR $\\approx {snr:.1f}$) suggests a weak or ambiguous transit signal, a common indicator of a false positive."
        else:
            justification += f" Despite a strong SNR ($\approx {snr:.1f}$), other factors (like high stellar surface gravity or temperature mismatch) likely pointed to a stellar eclipsing binary scenario, leading to the False Positive classification."

    return justification

def generate_feature_chart(df_results, features):
    """Creates a comparative Altair chart showing feature values."""
    
    df_chart = df_results[features].reset_index(drop=True).T
    df_chart.columns = ['Value']
    df_chart = df_chart.reset_index().rename(columns={'index': 'Feature'})
    
    # Filter for human-readable features for plotting
    df_chart = df_chart[df_chart['Feature'].isin(['koi_period', 'koi_prad', 'koi_teq', 'koi_slogg', 'koi_model_snr'])]
    
    base = alt.Chart(df_chart).encode(
        x=alt.X('Value', type="quantitative"),
        y=alt.Y('Feature', sort='-x', title="Key Parameter"),
        tooltip=['Feature', 'Value']
    ).properties(
        title='Key Feature Values (for Classification)'
    )

    chart = base.mark_bar(color='#4c78a8', opacity=0.8).encode(
        x=alt.X('Value', axis=alt.Axis(title='Value (Scaled/Raw)')),
        tooltip=[
            alt.Tooltip('Feature'), 
            alt.Tooltip('Value', format='.2f')
        ]
    )
    return chart

# --- Streamlit Application Layout ---

st.set_page_config(
    page_title="Exoplanet AI Classifier (AdaBoost)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🪐 NASA Exoplanet AI Classifier")
st.markdown("### Powered by AdaBoost & NASA Exoplanet Archive")

# --- Model Training Section (Sidebar) ---
with st.sidebar:
    st.header("ML Model Setup")
    
    if 'model' not in st.session_state:
        st.session_state.is_trained = False
        st.session_state.training_df, st.session_state.imputer, st.session_state.scaler = load_and_preprocess_training_data()
        
    if not st.session_state.is_trained:
        st.warning("Model not yet trained. Click below to initialize.")
        if st.button("Train AdaBoost Classifier"):
            with st.spinner("Training AdaBoost Classifier on simulated KOI data..."):
                try:
                    model, accuracy, report = train_adaboost_model(st.session_state.training_df)
                    st.session_state.model = model
                    st.session_state.accuracy = accuracy
                    st.session_state.report = report
                    st.session_state.is_trained = True
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.session_state.is_trained = False

    if st.session_state.is_trained:
        st.success("AdaBoost Model is Trained and Ready! (Binary Classification)")
        st.metric(label="Model Accuracy (Test Set)", value=f"{st.session_state.accuracy:.2%}")
        
        with st.expander("Training Report (Exoplanet vs. FP)"):
            st.code(
                f"""
                Precision (Exoplanet): {st.session_state.report['Exoplanet']['precision']:.2f}
                Recall (Exoplanet): {st.session_state.report['Exoplanet']['recall']:.2f}
                Precision (FP): {st.session_state.report['False Positive']['precision']:.2f}
                Recall (FP): {st.session_state.report['False Positive']['recall']:.2f}
                """
            )
        
        st.markdown("---")
        st.subheader("Advanced Algorithms")
        st.markdown("""
        For a complete solution, you would need to implement:
        - **KNN:** Classification based on nearest feature vectors.
        - **CNN:** Requires fetching and analyzing the raw light curves (flux vs. time series data).
        - **BLS/Radial Velocity:** Not ML models, but signal processing techniques often used for feature extraction.
        """)
        st.markdown("---")


# --- Main Application Logic (Exoplanet ID Upload) ---

st.header("1. Upload Kepler Object IDs (KOI/TOI/TIC)")
st.markdown("Upload a single-column CSV file containing Kepler Object of Interest (KOI) IDs, one ID per row, to fetch data and run the prediction. *(e.g., K00001.01)*")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None and st.session_state.is_trained:
    
    # 1. Read the uploaded CSV (expected to be a list of IDs)
    try:
        user_ids_df = pd.read_csv(uploaded_file, header=None, names=['ID'])
        koi_list = user_ids_df['ID'].astype(str).str.strip().tolist()
        
        if not koi_list:
            st.error("The uploaded file is empty or contains no valid IDs.")
        else:
            st.write(f"Found {len(koi_list)} IDs to analyze.")
            st.dataframe(user_ids_df.head(5), use_container_width=True)
            
            # 2. Fetch data from NASA Exoplanet Archive
            fetched_df = fetch_exoplanet_data(koi_list)
            
            if fetched_df is not None and not fetched_df.empty:
                st.header("2. Data Preparation & Prediction")
                
                # Check for required columns
                missing_cols = [col for col in DEFAULT_FEATURES if col not in fetched_df.columns]
                if missing_cols:
                    st.error(f"Fetched data is missing required features for the model: {', '.join(missing_cols)}. Cannot proceed.")
                else:
                    # 3. Preprocess the fetched data
                    data_to_predict = fetched_df[DEFAULT_FEATURES].copy()
                    
                    # Apply Imputation (using the imputer trained on the training data)
                    imputer = st.session_state.imputer
                    data_to_predict[DEFAULT_FEATURES] = imputer.transform(data_to_predict[DEFAULT_FEATURES])
                    
                    # Apply Scaling (using the scaler trained on the training data)
                    scaler = st.session_state.scaler
                    data_to_predict[DEFAULT_FEATURES] = scaler.transform(data_to_predict[DEFAULT_FEATURES])
                    
                    st.success("Data successfully preprocessed and scaled.")
                    
                    # 4. Run Prediction
                    model = st.session_state.model
                    predictions = model.predict(data_to_predict)
                    probabilities = model.predict_proba(data_to_predict)
                    
                    # Extract the probability for the 'Exoplanet' class (index 1 in the binary map)
                    exoplanet_prob = probabilities[:, 1]
                    
                    # 5. Compile Results
                    results = fetched_df[['koi_name', TARGET_COLUMN]].copy()
                    results.columns = ['ID', 'Archive Disposition']
                    results['ML Prediction'] = np.where(predictions == 1, 'EXOPLANET', 'FALSE POSITIVE')
                    results['Probability (Exoplanet)'] = exoplanet_prob
                    
                    # Merge fetched raw data for detailed analysis
                    final_results = pd.merge(results, fetched_df[DEFAULT_FEATURES], left_index=True, right_index=True)

                    st.header("3. ML Classification Results")
                    
                    # Display the final classification table
                    st.dataframe(
                        final_results[['ID', 'Archive Disposition', 'ML Prediction', 'Probability (Exoplanet)']]
                        .sort_values(by='Probability (Exoplanet)', ascending=False), 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("---")
                    
                    # Detailed Visualization and Explanation for the top prediction
                    st.subheader("Detailed Analysis of Top Candidate")
                    
                    top_candidate = final_results.sort_values(by='Probability (Exoplanet)', ascending=False).iloc[0]
                    
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.metric(
                            label=f"Top Predicted Object: {top_candidate['ID']}", 
                            value=top_candidate['ML Prediction'], 
                            delta=f"{top_candidate['Probability (Exoplanet)']:.2%} Confidence"
                        )
                        
                        # Generate Explanation
                        justification = analyze_prediction(top_candidate, top_candidate['ML Prediction'] == 'EXOPLANET', top_candidate['Probability (Exoplanet)'])
                        st.markdown(justification)
                        st.markdown(f"*(NASA Archive Disposition: {top_candidate['Archive Disposition']})*")
                        st.markdown("""
                        **Justification Principles:**
                        - **Exoplanet:** Low radius ($R_p < 2.0$ $R_E$) & High SNR.
                        - **False Positive:** High stellar gravity ($\log g$) or Low SNR.
                        """)

                    with col2:
                        # Generate Graph
                        chart = generate_feature_chart(top_candidate, DEFAULT_FEATURES)
                        st.altair_chart(chart, use_container_width=True)
                    
    except pd.errors.EmptyDataError:
        st.error("Could not read the CSV file. Please ensure it is correctly formatted.")
    except Exception as e:
        st.error(f"An unexpected error occurred during file processing: {e}")

elif uploaded_file is not None and not st.session_state.is_trained:
     st.error("Please train the AdaBoost model in the sidebar before uploading data.")
     
elif 'is_trained' in st.session_state and st.session_state.is_trained and uploaded_file is None:
    st.info("Upload a CSV file containing KOI/TOI/TIC IDs to begin classification.")
