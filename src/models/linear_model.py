from sklearn.linear_model import LinearRegression
import pandas as pd

def train_and_predict(train_data, test_data, features, target_col='fwd_ret_1m'):
    """
    Trains a Linear Regression model on train_data and predicts on test_data.
    Returns test_data with a new 'score' column.
    """
    X_train = train_data[features]
    y_train = train_data[target_col]
    
    # Simple OLS regression serves as the baseline model.
    # It assumes a linear relationship between momentum features and future returns.
    model = LinearRegression(fit_intercept=True, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predict on Test Data
    X_test = test_data[features]
    
    # Return a copy to avoid SettingWithCopy warnings
    test_data_with_score = test_data.copy()
    test_data_with_score['score'] = model.predict(X_test)
    
    return test_data_with_score