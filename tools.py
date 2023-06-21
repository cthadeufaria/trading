import numpy as np

class stochastic_gradient:
    """Class to optimize function using stochastic gradient."""
    def __init__(self, learning_rate, weights, gradient, type='ascent') -> None:
        self.learning_rate = learning_rate
        self.weights = weights
        self.gradient = gradient
        if type == 'ascent':
            self.sum = 1.
        elif type == 'descent':
            self.sum = -1.
        else:
            raise Exception("Parameter \"type\" must be either \"ascent\" or \"descent\".") 

    def optimize(self):
        pass
