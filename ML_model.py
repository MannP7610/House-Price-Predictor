# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset (replace 'house_data.csv' with your dataset)
house_data = pd.read_csv('house_data.csv')

# Assuming 'features' are the variables used to predict the house price, and 'target' is the house price column
features = house_data[['feature1', 'feature2', 'feature3', ...]]  # Replace 'feature1', 'feature2', ... with actual feature names
target = house_data['price']  # Replace 'price' with actual target column name

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices using the trained model
predictions = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Example of using the trained model to predict the price of a new house
new_house_features = np.array([[feature1_value, feature2_value, feature3_value, ...]])  # Replace with actual feature values
predicted_price = model.predict(new_house_features)
print("Predicted Price for the new house:", predicted_price)
