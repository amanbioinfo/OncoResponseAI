import pandas as pd
import numpy as np

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)

    # Drop rows without known target
    df = df.dropna(subset=['TARGET'])

    # Fill categorical missing values
    for col in [
        'Cancer Type (matching TCGA label)',
        'Microsatellite instability Status (MSI)'
    ]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Winsorize LN_IC50
    Q1, Q3 = df['LN_IC50'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df['LN_IC50'] = df['LN_IC50'].clip(
        Q1 - 1.5 * IQR,
        Q3 + 1.5 * IQR
    )

    return df


def encode_features(df):
    X = df.drop(['LN_IC50', 'COSMIC_ID', 'CELL_LINE_NAME'], axis=1)
    y = df['LN_IC50']

    X = pd.get_dummies(X, drop_first=True)

    return X, y
