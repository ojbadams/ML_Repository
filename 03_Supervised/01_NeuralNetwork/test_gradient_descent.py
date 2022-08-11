from gradient_descent import GradientDescent
import numpy as np

# f(x) = x^2
def f(points):
    return 2 * points

gd = GradientDescent(f, 0.1, np.array([10]))
min_values = gd.fit()
print(f'Minimum value is {min_values}')