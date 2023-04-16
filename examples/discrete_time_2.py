import numpy as np
import matplotlib.pyplot as plt
import random
from controller import PID
from model import RuizCruz

# PID parameters
kp = 0.005
ki = 0.0001
kd = 0.001

# PID object
pid = PID(kp, ki, kd)

# trading fee
r = 0.0025

# initial model parameters
nak = 0 # nº of shares hold of BTC/BUSD
pk = 24375.12340 # BTC/BUSD price
mk = 806.91204500 # nº of shares hold of BUSD

# math model parameters
rc = RuizCruz(nak, pk, mk)

# initial portfolio value setpoint reference
vref = 1.01*rc.vpk

def price_estimator(pk):
    delta = random.uniform(-0.005, 0.005)
    return pk*(delta + 1)

# simulation parameters
tstop = 150
ts = 1
n = int(tstop/ts)

# initial simulation results
data = []
data.append(rc.vpk)
pk_data = []
pk_data.append(pk*3.3/100)

uk = 0
nam = np.absolute(uk)

for k in range(n):
    # update asset's price
    pk1 = price_estimator(pk)
    dpk = pk1 - pk

    print('vref: ', vref, 'vpk: ', rc.vpk, 'error: ', pid.last_error, 'uk: ', uk, 'nak: ', rc.nak, 'mk: ', rc.mk, 'pk*uk: ', pk*uk, 'dpk:', dpk)

    # update and adjust controller output
    uk = pid.update(vref, rc.vpk, ts, r, pk, uk) # TODO review error function TODO test other ts's
    uk = pid.adjust(uk, rc.mk, pk, r, rc.nak)

    # buy/sell operation uk
     
    # update model 
    rc.update(pk, dpk, r, uk)

    data.append(rc.vpk)
    pk_data.append(pk*3.3/100)
    pk = pk1

t = np.arange(0, tstop + ts, ts)
plt.plot(t, data, '-*', label = 'VPK')
plt.plot(t, pk_data, '-*')
plt.grid()
plt.show()
