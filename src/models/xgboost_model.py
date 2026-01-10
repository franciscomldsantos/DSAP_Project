import xgboost as xgb
import pandas as pd

def train_and_predict(train_data, test_data, features, target_col='fwd_ret_1m'):
    """
    Trains an XGBoost model on train_data and predicts on test_data.
    Returns test_data with a new 'score' column.
    """
    X_train = train_data[features]
    y_train = train_data[target_col]
    
    # XGBoost uses gradient boosting to sequentially improve model performance.
    # A low depth and learning rate are chosen to maintain robustness.
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict on Test Data
    X_test = test_data[features]
    
    # Return a copy to avoid SettingWithCopy warnings
    test_data_with_score = test_data.copy()
    test_data_with_score['score'] = model.predict(X_test)
    
    return test_data_with_score