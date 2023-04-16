import numpy as np

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
