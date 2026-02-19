import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from data_preprocessing import load_and_clean_data, encode_features

DATA_PATH = "../data/GDSC_DATASET.csv"

def train_and_save_model():
    df = load_and_clean_data(DATA_PATH)
    X, y = encode_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base_model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='gpu_hist',
        random_state=42
    )

    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [5, 7],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    }

    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=15,
        scoring='neg_root_mean_squared_error',
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    print("R²:", r2_score(y_test, preds))
    print("RMSE:", mean_squared_error(y_test, preds, squared=False))

    joblib.dump(best_model, "../models/xgb_best_model.pkl")
    joblib.dump(X.columns.tolist(), "../models/feature_columns.pkl")


if __name__ == "__main__":
    train_and_save_model()
