import numpy as np
from sklearn.linear_model import LinearRegression

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

# Function to predict glucose level based on age
def predict_glucose(age):
    age_array = np.array([age]).reshape(-1, 1)
    glucose_pred = model.predict(age_array)
    return glucose_pred[0]

# Example usage
age_input = 30
glucose_prediction = predict_glucose(age_input)
print(f"Predicted glucose level for age {age_input}: {glucose_prediction}")
