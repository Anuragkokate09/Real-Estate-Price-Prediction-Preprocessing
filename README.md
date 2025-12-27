# 🏠 **Real Estate Price Prediction & Data Preprocessing**
### *Automated Regression Pipeline for Property Valuation*

---

## 📝 **Project Overview**
This project demonstrates a foundational Machine Learning workflow for predicting real estate prices based on property area. By implementing a **Linear Regression** model, the project showcases how to transform raw, unorganized data into a predictive engine. The core focus is on **Data Preprocessing**, ensuring that the model receives clean, normalized features for maximum accuracy.

---

## 🛠️ **Machine Learning Pipeline**
The script (`Data Preprocessing 101.py`) follows a standardized Data Science lifecycle:

* **Data Cleaning:** * Automated removal of whitespace and case normalization for column headers.
    * Replacement of spaces with underscores to ensure programmatic compatibility.
* **Missing Value Analysis:** Systematic checking for null values to maintain data integrity.
* **Feature Engineering & Scaling:** * Implementation of `StandardScaler` to perform Z-score normalization.
    * Ensures the model treats the 'Area' feature with a mean of 0 and a variance of 1.
* **Model Training:** * Uses an **80/20 Train-Test split** to validate performance on unseen data.
    * Trains a `LinearRegression` model using Scikit-Learn.
* **Visual Evaluation:** Generates scatter plots of actual data paired with a smooth regression line for trend analysis.



---

## ⚙️ **Technical Stack**
* **Language:** `Python 3.x`
* **Machine Learning:** `Scikit-Learn` (LinearRegression, StandardScaler, train_test_split)
* **Data Manipulation:** `Pandas` & `NumPy`
* **Visualization:** `Matplotlib`

---

## 📊 **Model Execution & Results**
The pipeline provides real-time feedback during execution:
1.  **Preprocessing:** Displays the first few rows of scaled numerical values.
2.  **Validation:** Prints a side-by-side comparison of **Actual vs. Predicted** prices from the test set.
3.  **Visualization:** Outputs a Matplotlib chart showing the relationship between Area and Price.

---

## 🚀 **How to Run**
1. Ensure `area_price_dataset.csv` is present in the root directory.
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib numpy
3. Run the script:
   ```bash
   python "Data Preprocessing 101.py"
