import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Configuration ---
DATA_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&select=kepid,kepoi_name,koi_disposition,koi_period,koi_duration,koi_depth,koi_impact,koi_model_snr,koi_steff,koi_srad&format=csv"
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# --- 1. Data Loading and Cleaning ---
def load_and_clean_data(url):
    print("Loading data...")
    df = pd.read_csv(url, comment='#')
    
    # Define the key features as per your idea
    FEATURES = [
        'koi_period', 'koi_duration', 'koi_depth', 
        'koi_impact', 'koi_model_snr', 'koi_steff', 'koi_srad'
    ]
    
    # Filter for the binary classification task: CONFIRMED vs FALSE POSITIVE
    df_filtered = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    
    # Drop rows with any missing values in the required columns
    df_cleaned = df_filtered.dropna(subset=FEATURES + ['koi_disposition'])
    
    # Convert 'koi_disposition' to binary target (0: FALSE POSITIVE, 1: CONFIRMED)
    df_cleaned['exoplanet'] = (df_cleaned['koi_disposition'] == 'CONFIRMED').astype(int)
    
    print(f"Original Data Points: {len(df)}")
    print(f"Data Points used for Training (CONFIRMED/FALSE POSITIVE, no missing data): {len(df_cleaned)}")
    
    return df_cleaned, FEATURES

# --- 2. Training Pipeline ---
def train_and_save_model(df, features):
    X = df[features]
    y = df['exoplanet']
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2.1. Feature Scaling (essential for models like this)
    # Scale the features to a 0-1 range to normalize the inputs
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2.2. XGBoost Model Training
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', 
        use_label_encoder=False, 
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # 2.3. Evaluation
    y_pred = xgb_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete. Validation Accuracy: {accuracy:.4f}")
    
    # 2.4. Save Model and Scaler
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(xgb_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model and Scaler saved to '{MODEL_DIR}/'")
    
    return xgb_model

# --- Main Execution ---
if __name__ == "__main__":
    df_data, features = load_and_clean_data(DATA_URL)
    train_and_save_model(df_data, features)
