### 🎯 Goal
To build a classification model capable of accurately identifying **fraudulent transactions** within a synthetic mobile money environment, focusing on maximizing **Precision** and **Recall**.

### 📊 Dataset
The **PaySim Mobile Money Simulator** dataset is used, which synthetically mirrors the behavior of aggregated real-world mobile money transactions over 30 days (744 time steps).

### 🔍 Analysis Focus
The primary goal is to distinguish legitimate transactions from fraudulent ones, which are exclusively observed in **CASH-OUT** and **TRANSFER** operations. The model relies on analyzing patterns in transaction type, amount, and the change in account balances of both the origin and destination accounts.

### ⚙️ Libraries
`pandas`, `numpy`, `scikit-learn` (implied for model training).