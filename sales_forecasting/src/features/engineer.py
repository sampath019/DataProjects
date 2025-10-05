import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta

# Define paths and feature generation parameters
INPUT_DIR = "/kaggle/input/store-sales-time-series-forecasting"
LAGS = [1, 2, 3, 7, 14, 21, 28, 56]
ROLL_WINDOWS = [3, 7, 14, 28, 56]
CATEGORICAL_COLS = ["family", "city", "state", "type", "store_cluster"]
DATE_COLS = ["dow", "day", "month", "year", "weekofyear"]

def load_and_preprocess_raw_data(input_dir: str = INPUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads all raw data files, performs initial preprocessing, and merges exogenous features.
    
    Returns: (train_df, test_df, stores_df)
    """
    # 1. Load data
    train = pd.read_csv(os.path.join(input_dir, "train.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(input_dir, "test.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(input_dir, "stores.csv"))
    oil = pd.read_csv(os.path.join(input_dir, "oil.csv"), parse_dates=["date"])
    holidays = pd.read_csv(os.path.join(input_dir, "holidays_events.csv"), parse_dates=["date"])
    
    # 2. Rename and create unique ID
    stores = stores.rename(columns={"cluster": "store_cluster"})
    stores["store_nbr"] = stores["store_nbr"].astype(int)
    
    train["unique_id"] = train["store_nbr"].astype(str) + "_" + train["family"]
    test["unique_id"] = test["store_nbr"].astype(str) + "_" + test["family"]
    
    # 3. Oil: forward fill
    oil = oil.set_index("date").asfreq("D").ffill().reset_index().rename(columns={"dcoilwtico": "oil_price"})
    
    # 4. Holidays: national only
    holidays = holidays[holidays["locale"] == "National"].copy()
    holidays["holiday_flag"] = 1
    holidays = holidays[["date", "holiday_flag"]].drop_duplicates()
    
    # 5. Merge function
    def merge_exog(df):
        df = df.merge(oil, on="date", how="left")
        df = df.merge(holidays, on="date", how="left")
        df["holiday_flag"] = df["holiday_flag"].fillna(0)
        return df

    train = merge_exog(train)
    test = merge_exog(test)
    
    return train, test, stores

def create_features(train: pd.DataFrame, test: pd.DataFrame, stores: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Creates a full panel dataframe, generates date/time-series features, 
    encodes categorical variables, and returns the final dataframe along with feature list.
    
    Returns: (df_all, features_list)
    """
    test["sales"] = np.nan
    df_all = pd.concat([train, test], sort=False).reset_index(drop=True)
    df_all = df_all.sort_values(["unique_id", "date"]).reset_index(drop=True)

    # 1. Merge store metadata
    df_all = df_all.merge(stores, on="store_nbr", how="left")

    # 2. Basic date features
    df_all["dow"] = df_all["date"].dt.dayofweek
    df_all["day"] = df_all["date"].dt.day
    df_all["month"] = df_all["date"].dt.month
    df_all["year"] = df_all["date"].dt.year
    df_all["weekofyear"] = df_all["date"].dt.isocalendar().week.astype(int)

    # 3. Label encode categorical columns
    for c in CATEGORICAL_COLS:
        df_all[c] = df_all[c].astype(str)
        le = LabelEncoder()
        df_all[c] = le.fit_transform(df_all[c].fillna("NA"))

    # 4. Time Series Features (Lags, Rolling, EWMA)
    for lag in LAGS:
        df_all[f"lag_{lag}"] = df_all.groupby("unique_id")["sales"].shift(lag)

    for w in ROLL_WINDOWS:
        # Rolling Mean/Std (shift 1 to prevent data leakage)
        shift_col = df_all.groupby("unique_id")["sales"].shift(1)
        df_all[f"rmean_{w}"] = shift_col.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        df_all[f"rstd_{w}"] = shift_col.rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True)
        
        # EWMA
        df_all[f"ewm_{w}"] = df_all.groupby("unique_id")["sales"].shift(1).transform(lambda x: x.ewm(span=w, adjust=False).mean())

    # 5. Lagged Exogenous Features
    df_all["promo_lag_1"] = df_all.groupby("unique_id")["onpromotion"].shift(1)
    df_all["promo_roll_7"] = df_all.groupby("unique_id")["onpromotion"].shift(1).rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)

    df_all["days_to_holiday"] = df_all.groupby("unique_id")["holiday_flag"].shift(1).fillna(0)
    df_all["days_to_holiday_7"] = df_all.groupby("unique_id")["holiday_flag"].shift(1).rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)

    # 6. Final Missing Value Handling
    group_median = df_all.groupby("unique_id")["sales"].transform("median")
    lag_cols = [c for c in df_all.columns if "lag_" in c or "rmean_" in c or "rstd_" in c or "ewm_" in c or "promo_" in c or "days_to_holiday" in c]
    for col in lag_cols:
        # Fill NaN values created by lags/rolling with the group median, then 0 for series with only NaN/missing.
        df_all[col] = df_all[col].fillna(group_median).fillna(0.0)

    df_all["oil_price"] = df_all["oil_price"].fillna(df_all["oil_price"].median())

    # 7. Define the feature list
    feature_cols = [f for f in df_all.columns if f not in ["id", "date", "sales", "unique_id", "store_nbr", "onpromotion", "type"]]

    return df_all, feature_cols