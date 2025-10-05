import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta

def calculate_rmsle(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Log Error (RMSLE).
    """
    # Ensure non-negative predictions for log operation
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

def create_lgbm_datasets(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, cat_features: list) -> tuple[lgb.Dataset, lgb.Dataset]:
    """
    Prepares LightGBM Dataset objects, correctly tagging categorical features.
    """
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=lgb_train, free_raw_data=False)
    return lgb_train, lgb_val

def train_lgbm_model(lgb_train: lgb.Dataset, lgb_val: lgb.Dataset, n_estimators: int = 2000) -> lgb.Booster:
    """
    Trains the LightGBM model with predefined parameters and early stopping.
    """
    params = {
        "objective": "rmse",
        "metric": "rmse",
        "learning_rate": 0.036,
        "num_leaves": 196,
        "min_data_in_leaf": 494,
        "feature_fraction": 0.605,
        "bagging_freq": 1,
        "bagging_fraction": 0.606,
        "seed": 2025,
        "verbosity": -1,
        "n_jobs": -1
    }
    
    print("Starting LightGBM training...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=n_estimators,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )
    return model

def predict_and_save_submission(
    model: lgb.Booster, 
    X_test: pd.DataFrame, 
    test_feat_df: pd.DataFrame, 
    original_test_df: pd.DataFrame, 
    train_df: pd.DataFrame
):
    """
    Generates predictions on the test set, handles edge cases, and creates submission file.
    """
    # 1. Predict
    preds_test = model.predict(X_test, num_iteration=model.best_iteration)
    preds_test = np.clip(preds_test, 0, None)
    
    test_feat_df["pred_sales"] = preds_test
    
    # 2. Merge predictions back to the original test structure
    test_out = original_test_df.merge(
        test_feat_df[["unique_id", "date", "pred_sales"]], 
        on=["unique_id", "date"], 
        how="left"
    )
    
    # 3. Fallback: Fill missing predictions (e.g., due to missing lags/data issues) with median sales
    median_sales = train_df.groupby("unique_id")["sales"].median().to_dict()
    test_out["sales"] = test_out.apply(
        lambda r: median_sales.get(r["unique_id"], 0.0) if pd.isna(r["pred_sales"]) else r["pred_sales"], 
        axis=1
    )
    
    # 4. Final clip and save
    test_out["sales"] = test_out["sales"].clip(lower=0.0)
    submission = test_out[["id", "sales"]].copy()
    submission.to_csv("submission.csv", index=False)
    
    print(f"\nSubmission file saved with {submission.shape[0]} rows.")