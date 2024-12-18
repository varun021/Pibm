import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Select one feature for simple linear regression (e.g., the third feature)
X = diabetes.data[:, np.newaxis, 2]  # Reshape to 2D array for fitting
y = diabetes.target  # The target variable

# Create a linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the slope (coefficient) and intercept of the model
slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope (m): {slope}")
print(f"Intercept (b): {intercept}")

# Predict the target values
y_pred = model.predict(X)

# Plotting the regression line
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('Normalized Feature Value')
plt.ylabel('Diabetes Progression')
plt.title('Simple Linear Regression on Diabetes Dataset')
plt.legend()

# Save the plot as an image
plt.savefig('Diabetes_Simple_Linear_Regression.png')
print("Plot saved as 'Diabetes_Simple_Linear_Regression.png'")
