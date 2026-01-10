from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def train_and_predict(train_data, test_data, features, target_col='fwd_ret_1m'):
    """
    Trains a Random Forest model on train_data and predicts on test_data.
    Returns test_data with a new 'score' column.
    """
    X_train = train_data[features]
    y_train = train_data[target_col]
    
    # Random Forest is an ensemble method that averages multiple decision trees.
    # Conservative hyperparameters (max_depth=3) are used to prevent overfitting on noisy financial data.
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=10,
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