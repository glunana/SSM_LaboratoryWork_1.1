import matplotlib.pyplot as plt
import numpy as np

def expression(x):
    return np.log10(x + 1) - np.exp(x) + 2.5

def derivative(x):
    return (1 - np.log(10) * np.exp(x) * (x + 1)) / (np.log(10) * (x + 1))

def equivalent_expression(x):
    return 10**(np.exp(x) - 2.5) - 1 if x > -1 else np.nan

def halving_method(a, b, epsilon):
    while True:
        x = (b + a) / 2
        if np.abs(b - a) < 2 * epsilon:
            break
        if np.abs(x) < epsilon:
            return x
        elif expression(x) * expression(a) < 0:
            b = x
        elif expression(x) * expression(b) < 0:
            a = x
    return x

def chord_method(a, b, epsilon):
    while True:
        x = a - expression(a) * (b - a) / (expression(b) - expression(a))
        if np.abs(expression(x)) < epsilon:
            break
        if expression(x) * expression(a) < 0:
            b = x
        elif expression(x) * expression(b) < 0:
            a = x
    return x

def newton_method(start_point, epsilon):
    x = start_point
    while True:
        x_next = x - expression(x) / derivative(x)
        if np.abs(x_next - x) < epsilon:
            break
        x = x_next
    return x_next

def simple_iteration_method(initial_guess, epsilon, max_iterations=1000):
    x = initial_guess
    iterations = 0
    while iterations < max_iterations:
        x_next = equivalent_expression(x)
        if np.isnan(x_next):
            break
        if np.abs(x_next - x) < epsilon:
            break
        x = x_next
        iterations += 1
    return x_next

# Графік
x_values = np.linspace(-2, 2, 500)

plt.figure(figsize=(10, 6))
plt.plot(x_values, np.log10(x_values + 1), label='log10(x + 1)')
plt.plot(x_values, np.exp(x_values) - 2.5, label='e^x - 2.5')

plt.title('log10(x + 1) = e^x - 2.5')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print("Метод половинного ділення x1: ", halving_method(-0.999, -0.8, 0.03))
print("Метод половинного ділення x2: ", halving_method(1, 1.2, 0.03))
print("Метод хорд x1: ", chord_method(-0.999, -0.8, 0.03))
print("Метод хорд x2: ", chord_method(1, 1.2, 0.03))
print("Метод Ньютона x1: ", newton_method(-0.999, 0.03))
print("Метод Ньютона x2: ", newton_method(1, 0.03))
print("Метод простої ітерації x1: ", simple_iteration_method(-0.999, 0.03))
print("Метод простої ітерації x2: ", simple_iteration_method(1.032, 0.03))