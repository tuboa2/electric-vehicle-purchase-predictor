"""
Production Inference Script for the Distilled EV Prediction Model
This script demonstrates how to load the 5MB distilled LightGBM model and generate predictions in milliseconds.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# We still need the feature engineering module to format incoming user data
from src.features import build_grandmaster_features

def init_model(model_path: str = "models/distilled_student.txt") -> lgb.Booster:
    """Loads the distilled model into memory once at server startup."""
    full_path = PROJECT_ROOT / model_path
    if not full_path.exists():
        raise FileNotFoundError(f"Distilled model not found at {full_path}")
    print(f"[*] Loading distilled model from {full_path}...")
    return lgb.Booster(model_file=str(full_path))

def predict_user(model: lgb.Booster, user_data: pd.DataFrame) -> np.ndarray:
    """
    Takes a raw DataFrame containing user attributes (e.g. from an API request),
    runs the feature engineering pipeline, and returns the probability of buying an EV.
    """
    # In production, we don't have a 'test' set, so we pass the user_data as 'train' 
    # to the feature engineering module. The module expects train, test, orig.
    # We pass empty dataframes for test and orig.
    empty_df = pd.DataFrame()
    X_features, _, _, cat_cols = build_grandmaster_features(user_data, empty_df, None)
    
    # Ensure categoricals are properly typed for LightGBM
    for c in cat_cols:
        if c in X_features.columns:
            X_features[c] = X_features[c].astype('category')
            
    # Drop target if it somehow sneaks in
    if "Will_Buy_EV" in X_features.columns:
        X_features = X_features.drop(columns=["Will_Buy_EV"])
        
    # Generate prediction in milliseconds
    probabilities = model.predict(X_features)
    return probabilities

if __name__ == "__main__":
    # --- Example Usage ---
    # Imagine a user submits this data through your Web App / API
    sample_user = pd.DataFrame({
        "Age": [34],
        "Gender": ["Male"],
        "Annual_Income_USD": [95000],
        "City_Type": ["Urban"],
        "Daily_Commute_km": [25.5],
        "Current_Car_Type": ["Combustion"],
        "Charging_Stations_Near_Home": [2],
        "Charging_Stations_Near_Work": [5],
        "Home_Charging_Possible": ["Yes"],
        "Subsidy_Available": ["Yes"],
        "Range_Anxiety_Level": [1],
        "Environmental_Concern_Level": [4]
    })
    
    print("[*] Initializing Production Inference Engine...")
    engine = init_model()
    
    print("[*] Processing incoming user request...")
    prob = predict_user(engine, sample_user)
    
    print(f"\n[+] Prediction Complete!")
    print(f"    User EV Purchase Probability: {prob[0]*100:.2f}%")
