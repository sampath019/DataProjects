import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(data_path):
    """
    Loads the financial transaction dataset from the specified path.

    Args:
        data_path (str): The full path to the CSV file (e.g., /kaggle/input/...).

    Returns:
        pd.DataFrame: The loaded transaction data.
    """
    print(f"Loading dataset from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None

def preprocess_for_modeling(df):
    """
    Applies feature engineering, encoding, and scaling, preparing the data
    for the machine learning model.

    This implements the preprocessing steps from the notebook code:
    1. Label encoding for 'type' (transaction type).
    2. Selecting features: 'type', 'amount', 'isFlaggedFraud'.
    3. Scaling numerical features.

    Args:
        df (pd.DataFrame): The raw input DataFrame.

    Returns:
        tuple: (X_scaled, y) where X_scaled is the processed feature matrix 
               and y is the target variable.
    """
    if df is None:
        return None, None
    
    df_copy = df.copy()
    
    # 1. Label Encode categorical variable 'type'
    print("Applying Label Encoding to 'type' column...")
    df_copy["type"] = LabelEncoder().fit_transform(df_copy["type"])

    # 2. Select relevant features and target
    # Note: If your notebook code intends to include balance features later, 
    # you would add them to this list.
    features = ["type", "amount", "isFlaggedFraud"]
    
    # Check if all required features are present
    missing_features = [f for f in features if f not in df_copy.columns]
    if 'isFraud' not in df_copy.columns:
        print("Error: Target column 'isFraud' not found.")
        return None, None
    if missing_features:
        print(f"Error: Missing required features: {missing_features}")
        return None, None
        
    X = df_copy[features]
    y = df_copy["isFraud"]

    # 3. Scale numerical features
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Functions for visualization (EDA) are kept separate as they use visualization libraries

def perform_eda(df):
    """
    Executes the exploratory data analysis (EDA) visualization steps.
    Note: This function uses Matplotlib and Seaborn, which should be 
          imported and run directly where the visualizations are needed 
          (i.e., in the Jupyter Notebook itself).
    
    This function is primarily included here for structure, but its contents 
    are usually run directly in the notebook for interactive plotting.
    
    Args:
        df (pd.DataFrame): The raw input DataFrame.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n--- Running EDA Visualizations ---")
    
    # Check basic info & missing values
    print("Dataset Overview:\n", df.info(verbose=False), "\n")
    print("Missing Values:\n", df.isnull().sum(), "\n")

    # Fraud vs Non-Fraud Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x="isFraud", data=df, palette="coolwarm")
    plt.xlabel("Fraudulent Transaction")
    plt.ylabel("Count")
    plt.title("Fraud vs Non-Fraud Distribution")
    plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
    plt.show()

    # Transaction Type Distribution
    plt.figure(figsize=(10,5))
    sns.countplot(x="type", data=df, palette="magma", order=df["type"].value_counts().index)
    plt.xticks(rotation=45)
    plt.xlabel("Transaction Type")
    plt.ylabel("Count")
    plt.title("Distribution of Transaction Types")
    plt.show()

    # Fraudulent Transaction Amounts
    plt.figure(figsize=(8,5))
    sns.histplot(df[df["isFraud"] == 1]["amount"], bins=30, kde=True, color="red", alpha=0.6)
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.title("Distribution of Fraudulent Transaction Amounts (Log Scale)")
    plt.xscale("log")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df[["amount", "isFraud", "isFlaggedFraud"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Fraud & Transaction Amount")
    plt.show()
    print("EDA Complete.")


if __name__ == '__main__':
    # This block is for testing the functions independently
    print("src/processing.py loaded and ready for use.")