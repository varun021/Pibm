import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the California Housing dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)  # Features
y = pd.Series(california.target)  # Target variable (House prices)

# Display the first few rows of the dataset
print(X.head())

# Create a multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

print(f"Intercept: {intercept}")
print("Coefficients:")
for feature, coef in zip(california.feature_names, coefficients):
    print(f"{feature}: {coef}")

# Make predictions using the model
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Plotting actual vs predicted values
plt.scatter(y, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Multiple Linear Regression - California Housing Dataset')
plt.legend()
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', lw=2)  # Diagonal line for reference
plt.savefig('California_Multiple_Linear_Regression.png')
print("Plot saved as 'California_Multiple_Linear_Regression.png'")
