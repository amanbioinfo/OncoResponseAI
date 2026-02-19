# src/train_model.py
import os
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from src.data_preprocessing import load_and_clean_data, encode_features


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "GDSC_DATASET.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_model():
    df = load_and_clean_data(DATA_PATH)
    X, y = encode_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",  # safer than gpu_hist on cloud
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("R²:", r2_score(y_test, preds))
    rmse = mean_squared_error(y_test, preds) ** 0.5
    print("RMSE:", rmse)

    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_best_model.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "feature_columns.pkl"))

if __name__ == "__main__":
    train_and_save_model()
