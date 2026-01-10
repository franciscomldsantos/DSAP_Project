from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd

def train_and_predict(train_data, test_data, features, target_col='fwd_ret_1m'):
    """
    Trains a Lasso Regression model (L1 regularization) on train_data and predicts on test_data.
    Returns test_data with a new 'score' column.
    """
    X_train = train_data[features]
    y_train = train_data[target_col]
    
    # Lasso requires standardized features to penalize coefficients correctly.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Alpha controls the regularization strength. Higher values force more coefficients 
    # to become exactly zero, effectively selecting only the most important features.
    model = Lasso(
        alpha=0.0001,
        max_iter=10000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    X_test = test_data[features]
    X_test_scaled = scaler.transform(X_test)
    
    test_data_with_score = test_data.copy()
    test_data_with_score['score'] = model.predict(X_test_scaled)
    
    return test_data_with_score
