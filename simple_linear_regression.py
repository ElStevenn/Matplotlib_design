import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os


df = pd.read_csv(os.path.join(os.getcwd(), "datasets", "test.csv"))

X = np.array(df.x).reshape(-1, 1)
Y = np.array(df.y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)  # Split the data into train and test


# plt.show()

TheLinearRegression = LinearRegression() # Define Linear Regression

TheLinearRegression.fit(X=X_train, y=Y_train)

x_perds = TheLinearRegression.predict(X_test) # Predict 10 tiems y predicts
print(x_perds)

fig, ax = plt.subplots()
ax.scatter(X_train,Y_train,c="blue", label="Training data")

ax.scatter(x_perds, Y_test, c="red", label="Regression line")

plt.show()