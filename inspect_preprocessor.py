import joblib
import pandas as pd
import numpy as np

try:
    preprocessor = joblib.load('models/production/v1.0.0/preprocessor.pkl')
    
    print("\n--- Check Protocol Type ---")
    if 'protocol_type' in preprocessor.numerical_cols:
        print("ALERT: protocol_type IS in numerical_cols!")
    else:
        print("protocol_type is NOT in numerical_cols.")

    if 'protocol_type' in preprocessor.categorical_cols:
        print("protocol_type IS in categorical_cols.")
    else:
        print("protocol_type is NOT in categorical_cols.")

    print("\n--- First 5 Columns ---")
    for i in range(5):
        print(f"{i}: {preprocessor.feature_names[i]}")

except Exception as e:
    print(f"Error: {e}")
