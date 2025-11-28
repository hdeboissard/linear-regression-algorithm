# importing relevant packages and modules
import numpy as np
import matplotlib.pyplot as plt
import requests

# introducing hardcoded data (can pull from the API later)
data = {
    1851: 27368800,
    1861: 28917900,
    1871: 31484700,
    1881: 34934500,
    1891: 37802400,
    1901: 38237000,
    1911: 42082000,
    1921: 44027000,
    1931: 46038000,
    1951: 50225000,
    1961: 52807000,
    1971: 55928000,
    1981: 56357000,
    1991: 57439000,
    2001: 59113000,
    2011: 63182000,
    2021: 67121301
}


# converting above data to two arrays
years = np.array(list(data.keys()), dtype=float)
population = np.array(list(data.values()), dtype=float)

# normalising the years data set
new_years = []
for year in years:
    new_years.append(year - 1851)

# renaming to X and Y
X = new_years
Y = population

# Mean Squared Error (MSE) loss:
# L = (1/N) * Σ (y_pred[i] - y_true[i])^2
# y_pred = x_true * m + b

# Gradients of the MSE loss:
# dL/dm = (2/N) * Σ_i [ x_i * (m*x_i + b - y_i) ]
# dL/db = (2/N) * Σ_i [(m*x_i + b - y_i)]

# finding derivative of loss function at constant m wrt b
def gradient_at_b(X, Y, m, b):
    N = len(X)
    diff = 0
    for i in range(N):
        y_value = Y[i]
        x_value = X[i]
        diff += y_value - (x_value * m + b)
    gradient = -1 * diff * 2 / N
    return gradient

# finding derivative of loss function at constant b wrt m
def gradient_at_m(X, Y, m, b):
    N = len(X)
    sum_val = 0
    for i in range(N):
        y_value = Y[i]
        x_value = X[i]
        sum_val += (x_value * (m * x_value + b - y_value))
    gradient = 2 * sum_val / N
    return gradient

# function to modify values of b and m to minimise loss
def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = gradient_at_b(x, y, m_current, b_current)
    m_gradient = gradient_at_m(x, y, m_current, b_current)
    b_new = b_current - b_gradient * learning_rate
    m_new = m_current - m_gradient * learning_rate
    return b_new, m_new

# using above function to iterate into more optimal values of b and m
def gradient_descent(x, y, learning_rate, num_iterations):
    b = 0
    m = 0
    for i in range(num_iterations):
        b, m = step_gradient(b, m, x, y, learning_rate)
    return b, m

b, m = gradient_descent(X, Y, learning_rate=0.00005, num_iterations = 1000000)
print('This is b: ', round(b))
print('This is m: ', round(m))

# graphical section for comparing gradient descent results to actual calculation

def find_line(x, y):

    x_len = len(x)
    y_len = len(y)

    if x_len != y_len:
        return 'Incompatible Data!'
    
    N = x_len

    def find_mean(numbers):
        count = 0
        for item in range(N):
            count += numbers[item]
        return count / N
    
    def find_sum(numbers):
        count = 0
        for item in range(N):
            count += numbers[item]
        return count
    
    def find_sum_both(x, y):
        count = 0
        for item in range(N):
            count += x[item] * y[item]
        return count
    
    def find_sum_sq(x):
        count = 0
        for item in range(N):
            count += (x[item]) ** 2
        return count
    
    x_mean = find_mean(x)
    y_mean = find_mean(y)

    x_sum = find_sum(x)
    y_sum = find_sum(y)
    xy_sum = find_sum_both(x, y)

    x_sq_sum = find_sum_sq(x)

    m = (N * xy_sum - x_sum * y_sum) / (N * x_sq_sum - (x_sum) ** 2)
    b = (y_sum - m * x_sum) / N

    return m, b
    
m_real, b_real = find_line(X, Y)
print('This is the real value of m: ', round(m_real))
print('This is the real value of b: ', round(b_real))


def create_list(num_points, exp_base):
    iterations = []
    for point in range(num_points + 1):
        iterations.append(exp_base ** point)
    return iterations

# defining how many points you are going to tkae and with what exp base

num_points = 9
exp_base = 5 

iterations = create_list(num_points, exp_base)

b_results = []
m_results = []

for i in range(num_points + 1):
    print('Currently on iteration: ', i)
    b, m = gradient_descent(X, Y, learning_rate=0.00005, num_iterations = iterations[i])
    b_results.append(float(b))
    m_results.append(float(m))


print(b_results)
print(m_results)
print(iterations)

# creating values to graph

# also creating the actual data plot:
plt.figure(figsize=(12, 5))
plt.scatter(years, Y,  color='red')
y_predicted = []
for i in range(len(X)):
    y_predicted.append(m * X[i] + b)
plt.plot(years, y_predicted, color='red')
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population at Different Years with fitted Regression Line')

# creating an array to plot the value of m and b
m_real_plotdata = []
for i in range(num_points + 1):
    m_real_plotdata.append(m_real)

b_real_plotdata = []
for i in range(num_points + 1):
    b_real_plotdata.append(b_real)

# plotting m
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(iterations, m_results, 'g-o', label='Gradient Descent m', linewidth=2)
plt.plot(iterations, m_real_plotdata, 'r--', label='Analytical m (exact)', linewidth=2)
plt.xscale('log')
plt.xlabel('Number of Iterations')
plt.ylabel('Gradient (m)')
plt.title('Convergence of Gradient (m)')
plt.legend()
plt.grid(True, alpha=0.3)

# plotting b
plt.subplot(1, 2, 2)
plt.plot(iterations, b_results, 'b-o', label='Gradient Descent b', linewidth=2)
plt.plot(iterations, b_real_plotdata, 'r--', label='Analytical b (exact)', linewidth=2)
plt.xscale('log')
plt.xlabel('Number of Iterations')
plt.ylabel('Intercept (b)')
plt.title('Convergence of Intercept (b)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
