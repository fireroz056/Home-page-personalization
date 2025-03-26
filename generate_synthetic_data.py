import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Seed for reproducibility
np.random.seed(42)

# Simulate a synthetic dataset
n_samples = 10000  # Number of samples

# Features
user_ids = np.arange(1, n_samples + 1)
price_sensitivity = np.random.choice(['Low', 'Medium', 'High'], size=n_samples)
meal_size_preference = np.random.choice([1, 2, 3], size=n_samples)  # 1 = Small, 2 = Medium, 3 = Large
loyalty_status = np.random.choice([0, 1], size=n_samples)  # 0 = Non-loyal, 1 = Loyal
frequent_order_time = np.random.choice([0, 1, 2, 3, 4], size=n_samples)  # 0 = No, 1-4 = Time of the day
past_orders = np.random.randint(1, 20, size=n_samples)  # Number of past orders
average_order_value = np.random.randint(10, 100, size=n_samples)  # Avg order value in currency
current_time = np.random.choice(['Morning', 'Afternoon', 'Evening'], size=n_samples)
current_weather = np.random.choice(['Sunny', 'Rainy', 'Cloudy'], size=n_samples)
cart_status = np.random.choice(['Abandoned', 'Completed'], size=n_samples)
device_type = np.random.choice([0, 1], size=n_samples)  # 0 = Mobile, 1 = Desktop
page_view = np.random.randint(1, 10, size=n_samples)  # Number of pages viewed
current_promotion = np.random.choice([0, 1], size=n_samples)  # 0 = No, 1 = Yes
banner_shown = np.random.choice([0, 1], size=n_samples)  # 0 = Not shown, 1 = Shown

# Target Variable (Converted)
converted = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 30% conversion rate

# Create DataFrame
df = pd.DataFrame({
    'user_id': user_ids,
    'price_sensitivity': price_sensitivity,
    'meal_size_preference': meal_size_preference,
    'loyalty_status': loyalty_status,
    'frequent_order_time': frequent_order_time,
    'past_orders': past_orders,
    'average_order_value': average_order_value,
    'current_time': current_time,
    'current_weather': current_weather,
    'cart_status': cart_status,
    'device_type': device_type,
    'page_view': page_view,
    'current_promotion': current_promotion,
    'banner_shown': banner_shown,
    'converted': converted
})

# Data Preprocessing
df['price_sensitivity'] = df['price_sensitivity'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['current_time'] = df['current_time'].map({'Morning': 0, 'Afternoon': 1, 'Evening': 2})
df['current_weather'] = df['current_weather'].map({'Sunny': 0, 'Rainy': 1, 'Cloudy': 2})
df['cart_status'] = df['cart_status'].map({'Abandoned': 0, 'Completed': 1})

# Split the data into features and target variable
X = df.drop(columns=["converted", "user_id"])
y = df["converted"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example model training (Random Forest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
