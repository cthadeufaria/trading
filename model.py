"""Any mathematical model for defining current status."""
import numpy as np

class gmlang():
    """Class to define and update the model based on Guangming Lang paper."""
    def __init__(self, close_price, variance, delta=0.0001) -> None:
        self.mu = close_price
        self.P = variance
        self.Q = delta / (1 - delta)
        self.R = self.P

    def update(self, mu, variance, volume) -> None:
        self.mu = mu
        self.P = variance
        self.Q = self.Q
        self.R = self.P * ((volume[0]) / np.minimum(volume[0], volume[1]))
