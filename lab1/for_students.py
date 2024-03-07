import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
x_train = np.c_[np.ones_like(x_train),x_train]
theta_best = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)

# TODO: calculate error
y_predicted = x_train.dot(theta_best)

mse = np.mean((y_predicted-y_train)**2)
print("Error for closed-form solution",mse)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
x_train_stand = (x_train-np.mean(x_train))/np.std(x_train)

# TODO: calculate theta using Batch Gradient Descent
learing_rate = 0.01
itr = 1000
m=len(x_train_stand)
theta = np.random.randn(2)

for iteration in range(itr):
    grad = 2/m * x_train_stand.T.dot(x_train_stand.dot(theta)-y_train)
    theta -= learing_rate*grad

# TODO: calculate error
y_predicted_bgd = x_train_stand.dot(theta)
mse_bgd = np.mean((y_predicted_bgd-y_train)**2)
print("Error for Batch Gradient Descent",mse_bgd)
# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()