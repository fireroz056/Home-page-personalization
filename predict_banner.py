import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('banner_prediction_model.pkl')  # Load the saved model

# Example of input data (replace this with your actual real-time user data)
sample_data = {
    'price_sensitivity': [1],  # Example values (0: Low, 1: Medium, 2: High)
    'meal_size_preference': [2],
    'loyalty_status': [1],  # 1: Loyal customer
    'frequent_order_time': [18],  # Hour of the day
    'past_orders': [10],
    'average_order_value': [50],
    'current_time': [1200],  # Time in minutes (e.g., 1200 = 20:00)
    'current_weather': [0],  # 0: Clear, 1: Rainy
    'cart_status': [1],  # 0: Empty, 1: Items in cart
    'device_type': [1],  # 1: Desktop
    'page_view': [3],  # Number of pages viewed in current session
    'current_promotion': [1],  # 1: Promotion active
}

# Convert the sample data to a DataFrame
input_data = pd.DataFrame(sample_data)

# Make predictions using the trained model
predictions = model.predict(input_data)

# Output the predicted banner
if predictions[0] == 1:
    print("Show Banner A")  # This banner is shown for high conversion likelihood
else:
    print("Show Banner B")  # This banner is shown for low conversion likelihood
