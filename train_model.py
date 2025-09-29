import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load prepared data
df = pd.read_csv("prepared_data.csv")

# Features and label
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_20', 'EMA_50', 'SMA_200', 'RSI']
X = df[features]
y = df['Label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, "stock_model.pkl")
print("Model saved as stock_model.pkl")