import pandas as pd
import numpy as np
import re
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.utils import resample
# You may need to install this package separately: pip install category_encoders
from category_encoders import BinaryEncoder 

# List of columns to drop immediately as they are irrelevant or have too many missing values
DROP_LIST = ['examide', 'citoglipton', 'weight', 'encounter_id', 'patient_nbr', 
             'payer_code', 'medical_specialty']

def load_and_clean_data(filepath):
    """
    Loads the diabetes dataset, handles initial missing values ('?'),
    and converts the 'readmitted' column to a binary target variable.
    """
    print("Step 1/7: Loading and initial cleaning...")
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)
    
    # Drop irrelevant or high-NA columns
    df.drop(DROP_LIST, axis=1, inplace=True, errors='ignore')
    
    # Handle gender 'Unknown/Invalid' and drop rows where gender is invalid
    df.gender.replace('Unknown/Invalid', np.nan, inplace=True)
    df.dropna(subset=['gender'], how='all', inplace=True)
    
    # Convert target variable to binary: <30 days readmission (1) vs. No readmission/>30 days (0)
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    return df

def feature_engineer_diagnostics(df):
    """
    Implements the custom logic to map ICD9 codes (diag_1, diag_2, diag_3) 
    to broader disease categories (Circulatory, Respiratory, etc.).
    """
    print("Step 2/7: Engineering diagnostic features...")
    
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    
    # 1. Fill NaN diagnostics with 'NaN' string placeholder
    for col in diag_cols:
        df[col].fillna('NaN', inplace=True)
    
    # 2. Transform V/E codes and 'NaN' string into numeric codes
    def transform_func(value):
        value = re.sub("V[0-9]*", "0", value) # V codes to 0
        value = re.sub("E[0-9]*", "0", value) # E codes to 0
        value = re.sub('NaN', "-1", value) # Our custom NaN placeholder to -1
        return float(value)

    for col in diag_cols:
        df[col] = df[col].apply(transform_func)
    
    # 3. Map numeric codes to disease categories
    def transform_category(value):
        if (value >= 390 and value <= 459) or value == 785:
            return 'Circulatory'
        elif (value >= 460 and value <= 519) or value == 786:
            return 'Respiratory'
        elif (value >= 520 and value <= 579) or value == 787:
            return 'Digestive'
        elif value == 250:
            return 'Diabetes'
        elif value >= 800 and value <= 999:
            return 'Injury'
        elif value >= 710 and value <= 739:
            return 'Musculoskeletal'
        elif (value >= 580 and value <= 629) or value == 788:
            return 'Genitourinary'
        elif value >= 140 and value <= 239:
            return 'Neoplasms'
        elif value == -1:
            return 'NAN' # Missing category
        else:
            return 'Other'

    for col in diag_cols:
        df[col] = df[col].apply(transform_category)
        
    return df

def handle_missing_and_outliers(df):
    """
    Handles remaining missing values, removes rows based on discharge status, 
    and applies Local Outlier Factor (LOF) to filter data.
    """
    print("Step 3/7: Handling missing values and outliers...")
    
    # Missing Value Filling: Fill remaining 'race' NA with the mode
    df["race"].fillna(df["race"].mode()[0], inplace=True)
    
    # Remove observations with uninformative discharge IDs (original code step)
    df = df.loc[~df.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]
    
    # Identify numerical columns for LOF (based on original code logic)
    # The original code determined these after dropping initial columns. We infer them here:
    numerical_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                      'num_medications', 'number_outpatient', 'number_emergency', 
                      'number_inpatient', 'number_diagnoses']
    
    # Apply Local Outlier Factor (LOF)
    clf = LocalOutlierFactor(n_neighbors=2, contamination=0.1)
    df_scores = clf.fit_predict(df[numerical_cols])
    
    # Determine threshold based on 3rd smallest score (original code logic)
    threshold_value = np.sort(clf.negative_outlier_factor_)[2]
    
    # Filter the DataFrame to exclude outliers
    new_df = df[clf.negative_outlier_factor_ > threshold_value].reset_index(drop=True)
    
    return new_df

def encode_features(df):
    """
    Applies custom, Ordinal, Label, One-Hot, and Binary Encoding to all features.
    """
    print("Step 4/7: Encoding all categorical features...")
    
    df = df.copy()

    # 1. Custom Encoding for Drug Features (21 columns)
    drugs = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
             'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
             'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide', 
             'metformin-pioglitazone', 'metformin-rosiglitazone', 
             'glimepiride-pioglitazone', 'glipizide-metformin', 'troglitazone', 
             'tolbutamide', 'acetohexamide']

    for col in drugs:
        df[col] = df[col].replace(['No', 'Steady', 'Up', 'Down'], [0, 1, 1, 1]).astype(int)

    # 2. Custom Encoding for A1Cresult and max_glu_serum
    df['A1Cresult'] = df['A1Cresult'].replace(['>7', '>8', 'Norm', 'None'], [1, 1, 0, -99])
    df['max_glu_serum'] = df['max_glu_serum'].replace(['>200', '>300', 'Norm', 'None'], [1, 1, 0, -99])
    
    # 3. One-Hot Encoding for Race and ID columns
    df = pd.get_dummies(df, columns=['race'], prefix=["enc"])
    
    columns_ids = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    df[columns_ids] = df[columns_ids].astype('str')
    df = pd.get_dummies(df, columns=columns_ids)
    
    # 4. Ordinal Encoding for Age (Must be done after splitting, but defining here)
    # The 'age' column is already grouped in 10-year intervals and will be ordinal encoded later
    
    return df

def preprocess_pipeline(data_filepath):
    """
    Runs the full data preparation pipeline and returns the pre-encoded DataFrame.
    
    Args:
        data_filepath (str): Path to the raw diabetic_data.csv file.

    Returns:
        pd.DataFrame: The fully cleaned and mostly encoded DataFrame (before train/test split).
    """
    df = load_and_clean_data(data_filepath)
    df = feature_engineer_diagnostics(df)
    df = handle_missing_and_outliers(df)
    df = encode_features(df)
    
    return df

def split_and_resample_data(df):
    """
    Splits the data into train/test, performs necessary final encoding (Ordinal, Label, Binary),
    and applies undersampling to the training set.

    Args:
        df (pd.DataFrame): The fully preprocessed DataFrame from preprocess_pipeline.

    Returns:
        tuple: (X_train_resampled, X_val, X_test, y_train_resampled, y_val, y_test)
    """
    print("Step 5/7: Splitting, final encoding, and resampling...")
    
    X = df.drop(columns="readmitted", axis=1)
    Y = df.readmitted
    
    # Split into initial Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    
    # --- Final Feature Encoding (Needs fit/transform on training data) ---
    
    # Ordinal Encoding for Age
    ordinal_enc = OrdinalEncoder()
    X_train['age'] = ordinal_enc.fit_transform(X_train['age'].values.reshape(-1, 1))
    X_test['age'] = ordinal_enc.transform(X_test['age'].values.reshape(-1, 1))
    
    # Label Encoding for Diagnosis Categories
    diag_list = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_list:
        label_enc = LabelEncoder()
        # Ensure categories are consistent before fitting/transforming
        all_categories = pd.concat([X_train[col], X_test[col]]).astype('category').cat.categories
        label_enc.fit(all_categories)
        X_train[col] = label_enc.transform(X_train[col])
        X_test[col] = label_enc.transform(X_test[col])

    # Binary Encoding for other nominal columns
    binary = ['change', 'diabetesMed', 'gender']
    binary_enc = BinaryEncoder(cols=binary)
    X_train = binary_enc.fit_transform(X_train)
    X_test = binary_enc.transform(X_test)
    
    # --- Undersampling Majority Class ---
    
    print("Step 6/7: Applying Undersampling...")
    X_resample = pd.concat([X_train, y_train], axis=1)

    not_readmitted = X_resample[X_resample.readmitted == 0]
    readmitted = X_resample[X_resample.readmitted == 1]

    # Undersample the majority class (not_readmitted) to match the minority class size
    not_readmitted_sampled = resample(not_readmitted,
                                     replace=False,
                                     n_samples=len(readmitted),
                                     random_state=42)

    downsampled = pd.concat([not_readmitted_sampled, readmitted])
    
    y_train_resampled = downsampled.readmitted
    X_train_resampled = downsampled.drop('readmitted', axis=1)
    
    # --- Final Train-Validation Split ---
    
    print("Step 7/7: Splitting resampled data into Train and Validation sets...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_resampled, y_train_resampled, test_size=0.25, random_state=42
    )
    
    return X_train_final, X_val, X_test, y_train_final, y_val, y_test

if __name__ == '__main__':
    # Example usage: Replace with your actual data path
    # NOTE: You will need the ID_mapping.csv file and the diabetic_data.csv 
    # for this entire pipeline to work as intended.
    
    print("Data Pipeline Test Initiated.")
    
    # IMPORTANT: Update this path to where your file is located
    # data_path = "../input/diabetes/diabetic_data.csv" 
    # df_processed = preprocess_pipeline(data_path)
    # X_train, X_val, X_test, y_train, y_val, y_test = split_and_resample_data(df_processed)
    
    print("If run successfully, X_train, X_val, X_test, y_train, y_val, y_test would be ready for modeling.")