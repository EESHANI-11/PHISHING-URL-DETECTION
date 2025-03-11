import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib


# Load CSV file
df = pd.read_csv("phishing.csv")

# Remove "Index" and "class" columns
df.drop(columns=["Index", "class"], inplace=True)

# Extract features and target variable
X = df.values  # Features
y = pd.read_csv("phishing.csv")["class"].values  # Labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save the new model
with open("model.pkl", "wb") as file:
 joblib.dump(model, "pickle/model.pkl")

print("✅ Model retrained with 30 features!")
