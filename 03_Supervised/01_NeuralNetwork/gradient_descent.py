import numpy as np

class GradientDescent:
    def __init__(self, function, learning_rate: float, initial_point: np.array):
        '''
        GradientDescent object, for a specific function iterate through to identify the minimum value of a function
            function - function - derivative of the function to find the minimum point of
            learning_rate - float - find the rate at which the array is updated
            initial_point - np.array - initial array of the GD method
        '''
        # Derivative of the original function
        self.f_derivative = function
        # Learning Rate
        self.lr = learning_rate 
        # Initial Point (guess)
        self.init_point = initial_point

    def _update_point(self, point: np.array) -> np.array:
        '''
        Newtons method
        ''' 
        return point - self.lr * self.f_derivative(point)

    def _stopping_value(self, new_point: np.array, old_point: np.array) -> bool:
        return np.abs(new_point - old_point) < 0.0001

    def fit(self):
        old_point = self.init_point
        new_point = self._update_point(old_point)

        while not self._stopping_value(new_point, old_point):
            old_point = new_point
            new_point = self._update_point(old_point)

        return new_point

