# test_accuracy.py
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# Generate dummy regression data for testing
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
svr_model = SVR()
lr_model = LinearRegression()

# Train the models
svr_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Calculate accuracy
svr_accuracy = svr_model.score(X_test, y_test) * 100  # In percentage
lr_accuracy = lr_model.score(X_test, y_test) * 100

# Print out the accuracy
print(f"SVR Accuracy: {svr_accuracy:.2f}%")
print(f"Linear Regression Accuracy: {lr_accuracy:.2f}%")

