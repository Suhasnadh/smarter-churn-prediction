
# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath, sheet_name="Telco_Churn")
    df['Churn'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
    df.drop(['CustomerID', 'Lat Long', 'City', 'State', 'Country', 'Zip Code',
             'Latitude', 'Longitude', 'Churn Label', 'Churn Value', 'Churn Score',
             'CLTV', 'Churn Reason'], axis=1, inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
