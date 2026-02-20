import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json

# Create model folder
os.makedirs("model", exist_ok=True)

# Load data
df = pd.read_csv("data/crop_yield.csv")

# Drop index column if exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Remove missing values
df = df.dropna()

print("Columns:", df.columns)

# Target column (CORRECT)
target = "hg/ha_yield"

# Features and label
X = df.drop(columns=[target])
y = df[target]

# Convert categorical variables
X = pd.get_dummies(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "model/crop_model.pkl")
print("\nModel saved successfully!")

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, importances))

# Sort by importance
sorted_features = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

with open("model/feature_importance.json", "w") as f:
    json.dump(sorted_features, f)

print("Feature importance saved successfully!")
