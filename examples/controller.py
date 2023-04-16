import numpy as np

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.last_error = 0
        self.integral = 0

    def update(self, setpoint, feedback, dt, r, pk, uk):
        error = self.Kp * (setpoint - feedback) + r*pk*uk 
        # error = setpoint - feedback 
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        self.output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        return self.output
    
    def adjust(self, uk, mk, pk, r, nak):
        print(uk >= mk / ((1 + r)*pk))
        uk = np.minimum(uk, mk / ((1 + r)*pk))
        uk = np.maximum(uk, -nak)
        return uk
    
    # def controller(ts, vpk, vref, uk_prev, e_prev, pk, dpk, mk, nak, r):
        # kp = 0.8
        # ti = 5
        # e = vref - vpk
        # # e = kp*e_prev + r*pk*np.absolute(uk_prev)
        # # uk = uk_prev + kp*(e - e_prev) + (kp/ti)*ts*e
        # uk = (e_prev - dpk*nak - kp*e_prev)/(dpk)
        # alfa = np.sign(uk)
        # nam = np.absolute(uk)

        # if alfa == 1:
        #     if nam > mk/((1 + r)*pk):
        #         uk = mk/((1 + r)*pk)
        # elif alfa == -1:
        #     if nam > nak:
        #         uk = -nak

        # nam = np.absolute(uk)

        # return uk, e, nam
