import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('Salary_Data.csv')

# Display the first few rows to understand the dataset structure
print(data.head())

# Assuming the dataset has columns named 'YearsExperience' and 'Salary'
X = data[['YearsExperience']].values  # Independent variable
y = data['Salary'].values  # Dependent variable

# Create a linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the slope (coefficient) and intercept of the model
slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope (m): {slope}")
print(f"Intercept (b): {intercept}")

# Predicting salary values based on the model
y_pred = model.predict(X)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression on Salary Data')
plt.legend()

# Save the plot as an image
plt.savefig('Salary_Data_Regression.png')
print("Plot saved as 'Salary_Data_Regression.png'")
