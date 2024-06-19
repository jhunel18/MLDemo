import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset: x_train = age, y_train = glucose
x_train = np.array([43, 21, 25, 42, 57, 59]).reshape(-1, 1)
y_train = np.array([99, 65, 79, 75, 87, 81])

# Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Output the model details
r_sq = model.score(x_train, y_train)
print(f"Coefficient of Determination: {r_sq}")
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_}")

# Single test data point
x_test_single = np.array([30]).reshape(-1, 1)
y_test_single = np.array([80])  # Known test value for comparison

# Predict the response for the single test data point
y_test_single_pred = model.predict(x_test_single)
print(f"Predicted response: {y_test_single_pred}")

# If you want to calculate R^2 for the single test point (although it's not meaningful with a single point):
# y_test_pred needs to be in the same format as y_test for r2_score function
print(f"R^2 for the single test point: {r2_score(y_test_single, y_test_single_pred)}")
