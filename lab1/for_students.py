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
X_train = np.c_[np.ones_like(x_train),x_train]
theta_best = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)


# TODO: calculate error
y_predicted = X_train.dot(theta_best)
mse = np.mean((y_predicted-y_train)**2)
print("Error in train set for closed-form solution",mse)

X_test = np.c_[np.ones_like(x_test),x_test]
y_predicted_test = X_test.dot(theta_best)
mse_test = np.mean((y_predicted_test-y_test)**2)
print("Error in test set for closed-form solution",mse_test)


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
X_train_stand = np.c_[np.ones_like(x_train_stand),x_train_stand]

y_train_stand = (y_train-np.mean(y_train))/np.std(y_train)
Y_train_stand = y_train_stand.reshape(-1, 1)                    #changing the size of an y_train_stand from array to column vector


# TODO: calculate theta using Batch Gradient Descent
learing_rate = 0.01
itr = 100
m=len(X_train_stand)
theta = np.random.randn(2,1)

for iteration in range(itr):
    grad = 2/m * X_train_stand.T.dot(X_train_stand.dot(theta)-Y_train_stand)
    theta -= learing_rate*grad


scaled_theta = theta.copy()
scaled_theta[1] = scaled_theta[1] * np.std(y_train) / np.std(x_train)
scaled_theta[0] = np.mean(y_train) - scaled_theta[1] * np.mean(x_train)
scaled_theta = scaled_theta.reshape(-1)


# TODO: calculate error
y_predicted_bgd = X_train.dot(scaled_theta)
mse_bgd = np.mean((y_predicted_bgd-y_train)**2)
print("Error in train set for Batch Gradient Descent",mse_bgd)

y_predicted_bgd_test = X_test.dot(scaled_theta)
mse_bgd_test = np.mean((y_predicted_bgd_test-y_test)**2)
print("Error in test set for Batch Gradient Descent",mse_bgd_test)


# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(scaled_theta[0]) + float(scaled_theta[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
