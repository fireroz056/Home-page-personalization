import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load your data
# Assuming you've already loaded your dataset as 'df'
# For example, read from a CSV or any other data source
# df = pd.read_csv('your_data.csv')

# Example data structure (replace this with your actual data loading code)
df = pd.DataFrame({
    'price_sensitivity': [1, 2, 1, 0],  # Example values (0: Low, 1: Medium, 2: High)
    'meal_size_preference': [2, 1, 3, 2],
    'loyalty_status': [1, 0, 1, 0],
    'frequent_order_time': [12, 20, 9, 18],
    'past_orders': [5, 2, 8, 4],
    'average_order_value': [30, 45, 25, 50],
    'current_time': [1000, 1800, 1200, 1400],
    'current_weather': [0, 1, 0, 0],  # 0: Clear, 1: Rainy
    'cart_status': [1, 0, 1, 1],  # 0: Empty, 1: Items in cart
    'device_type': [1, 0, 1, 0],  # 0: Mobile, 1: Desktop
    'page_view': [2, 5, 3, 4],
    'current_promotion': [1, 0, 1, 0],  # 0: No promotion, 1: Promotion active
    'converted': [1, 0, 1, 0]  # Target variable (converted: 1 or 0)
})

# Separate features (X) and target variable (y)
X = df.drop(columns=["converted"])  # Drop the target column
y = df["converted"]  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file
joblib.dump(model, 'banner_prediction_model.pkl')
print("Model saved as banner_prediction_model.pkl")

