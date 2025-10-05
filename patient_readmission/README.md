### 🎯 Goal
To predict the likelihood of a patient with **diabetes** being readmitted to the hospital within **30 days** (`<30`) using comprehensive clinical data. This is framed as a binary classification problem critical for improving health outcomes and resource management.

### 📊 Dataset
The dataset includes 10 years (1999-2008) of clinical data from 130 US hospitals, featuring demographic, diagnostic, lab results, and medication information.

### 🛠️ Methodology and Key Steps
The pipeline implements extensive preprocessing and comparative modeling:

1.  **Exploratory Data Analysis (EDA):** Initial cleaning, handling of '?' as missing values, and visualization (`matplotlib`, `seaborn`, `missingno`).
2.  **Feature Engineering:**
    * **Custom ICD-9 Mapping:** Converting raw diagnostic codes (`diag_1`, `diag_2`, `diag_3`) into broad disease categories (e.g., Circulatory, Respiratory).
    * **Outlier Handling:** Utilizing **Local Outlier Factor (LOF)** to filter anomalous observations.
    * **Custom Encoding:** Simplifying 21 drug features (e.g., `metformin`) into a binary presence indicator (1/0).
3.  **Encoding:** Applying `OrdinalEncoder` for age, `LabelEncoder` for diagnostic categories, and `BinaryEncoder`/`OneHotEncoder` for other nominal features.
4.  **Imbalance Handling:** Applying **Undersampling** to the training data to balance the severe class imbalance of the `readmitted` target variable.
5.  **Comparative Modelling (Baseline):** Training and evaluating six different classifiers:
    * **Linear:** Logistic Regression
    * **Ensemble/Tree-based:** Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
6.  **Hyperparameter Tuning:** Selecting top features based on LightGBM importance and tuning the best-performing models using `GridSearchCV`.

### ⚙️ Libraries
`pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `matplotlib`, `seaborn`, `missingno`, `plotly`, `category-encoders`, `Pillow`.