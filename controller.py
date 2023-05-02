"""Any input/output model used for creating a portfolio."""
import numpy as np

class KalmanFilter1D:
    """Class to implement 1 dimensional Kalman Filter."""
    def __init__(self) -> None:
        pass
    
    def update1D(mean1, var1, mean2, var2):
        ''' This function takes in two means and two squared variance terms,
            and returns updated gaussian parameters.'''
        # Calculate the new parameters
        new_mean = (var2*mean1 + var1*mean2)/(var2+var1)
        new_var = 1/(1/var2 + 1/var1)
        
        return [new_mean, new_var]

    def predict1D(mean1, var1, mean2, var2):
        ''' This function takes in two means and two squared variance terms,
            and returns updated gaussian parameters, after motion.'''
        # Calculate the new parameters
        new_mean = mean1 + mean2
        new_var = var1 + var2
        
        return [new_mean, new_var]


class KalmanFilter(KalmanFilter1D):
    """Class to define and update a Kalman Filter."""
    def __init__(self) -> None:
        pass
    
    def predict(self, X, P, A, Q, B, U) -> tuple[np.array, np.array]: 
        """      
        X : The mean state estimate of the previous step (k-1) - shape(m,1) 
        P : The state covariance of previous step (k-1) - shape(m,m) 
        A : The transition  matrix - shape(m,m) 
        Q : The process noise covariance matrix - shape(m,m) 
        B : The input effect matrix - shape(p, m) 
        U : The control input - shape(q,1)
        """ 
        X = A @ X + B @ U 
        P = A @ P @ A.T + Q 
        return(X, P) 

    def update(self, X, P, Y, H, R) -> tuple[np.array, np.array]: 
        """      
        K  : the Kalman Gain matrix 
        IS : the Covariance or predictive mean of Y  
        """
        IS = H @ P @ H.T + R  
        K = P @ H.T @ np.linalg.inv(IS) 
        X = X + K @ (Y - H @ X) 
        P = P - K @ IS @ K.T 
        # P = P - K @ H @ P 
        return (X, P)
    

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
