# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 2. Load dataset
df = pd.read_csv("area_price_dataset.csv")

print("\nOriginal Data:")
print(df)

# Check for null values
print("\nChecking Null Values:")
print(df.isnull().sum())

# CLEAN COLUMN NAMES
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nColumn Names Cleaned Successfully!")
print(df.columns)

# 3. Split X and Y
X = df[['area']]
Y = df['price']

# 4. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Area Values (First 5):")
print(X_scaled[:5])

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# 6. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Trained Successfully!\n")

# 7. Predict for test data
y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {pred:.2f}")

# 8. Plotting Regression Line
plt.scatter(df["area"], df["price"], label="Actual Data")

# smooth line
area_range = np.linspace(df["area"].min(), df["area"].max(), 100)

# scale with correct column name
area_scaled = scaler.transform(pd.DataFrame({"area": area_range}))

price_line = model.predict(area_scaled)

plt.plot(area_range, price_line, linewidth=2, label="Regression Line")

plt.xlabel("Area (sq ft)")
plt.ylabel("Price (Lakhs)")
plt.title("Linear Regression Line: Area vs Price")
plt.legend()
plt.grid(True)
plt.show()

# 9. Single Prediction Input
input_area = float(input("\nEnter Area (sq ft): "))

scaled_area = scaler.transform(pd.DataFrame({'area': [input_area]}))

predicted_price = model.predict(scaled_area)

print(f"\nPredicted price for Area {input_area} sq.ft = ₹ {predicted_price[0]:.2f} lakhs")
