"""Any mathematical model that define and quantify current status."""
import numpy as np

class GLang:
    """Class to define and update the model based on Guangming Lang paper."""
    def __init__(self, close_price, variance, delta=0.0001) -> None:
        self.mu = close_price
        self.P = variance
        self.Q = delta / (1 - delta)
        # self.R = self.P

    def update(self, last_volume, current_volume, P) -> None:
        if np.minimum(last_volume, current_volume) == 0:
            pass
        else:
            self.R = P * ((last_volume) / np.minimum(last_volume, current_volume))

class RuizCruz:
    """Class to define and update the mathematical model to be controlled."""
    def __init__(self, nak, pk, mk):
        self.vpk = pk*nak + mk
        self.mk = mk
        self.nak = nak

        self.vpk1 = self.vpk
        self.mk1 = mk
        self.nak1 = nak

    def update(self, pk, dpk, r, uk) -> None:
        self.nam = np.absolute(uk)
        self.alfa = np.sign(uk)

        self.vpk1 = self.vpk + dpk*self.nak + dpk*uk - r*pk*self.nam
        self.mk1 = self.mk - pk*uk - r*pk*self.nam
        self.nak1 = self.nak + uk

        self.vpk = self.vpk1
        self.mk = self.mk1
        self.nak = self.nak1
