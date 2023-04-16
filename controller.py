"""Any predictive model used for creating a portfolio."""
import numpy as np

class KalmanFilter:
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