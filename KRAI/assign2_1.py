import numpy as np
import matplotlib.pyplot as plt

# Given input arrays (x and y values)
x = np.array([1, 2, 3, 4, 5])  # Independent variable
y = np.array([2, 3, 5, 7, 11])  # Dependent variable

# Calculate the means of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

# Calculate the coefficients (slope 'm' and intercept 'b')
n = len(x)
numerator = np.sum((x - mean_x) * (y - mean_y))
denominator = np.sum((x - mean_x) ** 2)
m = numerator / denominator
b = mean_y - m * mean_x

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# Predict y values
y_pred = m * x + b

# Plotting the regression line
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', label='Regression line')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Simple Linear Regression')
plt.legend()

# Save the plot as an image file
plt.savefig('simple_linear_regression.png')
print("Plot saved as 'simple_linear_regression.png'")
