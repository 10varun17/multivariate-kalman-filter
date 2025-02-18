import numpy as np
from scipy.linalg import inv

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x0):
        """
        Initialize the Kalman Filter.

        Parameters:
        F : State transition matrix.
        H : Measurement matrix.
        Q : Process noise covariance matrix.
        R : Measurement noise covariance matrix.
        P : Initial estimate error covariance.
        x0 : Initial state estimate.
        """
        self.F = F          
        self.H = H          
        self.Q = Q          
        self.R = R         
        self.P = P          
        self.x = x0         
    
    def predict(self, u=None, B=None):
        """
        Prediction step of the Kalman Filter.
        """
        if u is None or B is None:
            self.x = self.F @ self.x
        else:
            self.x = self.F @ self.x + B @ u
        
        # Update the error covariance.
        self.P = self.F @ self.P @ self.F.T + self.Q 
        return self.x, self.P
    
    def update(self, z):
        """
        Update step of the Kalman Filter using measurement z.
        """
        # Compute the innovation covariance.
        S = self.H @ self.P @ self.H.T + self.R
        
        # Compute the Kalman Gain.
        K = self.P @ self.H.T @ inv(S)
        
        # Compute the measurement residual.
        y = z - self.H @ self.x
        
        # Update the state estimate.
        self.x = self.x + K @ y
        
        # Update the error covariance.
        I = np.eye(self.P.shape[0])
        self.P = I @ self.P - K @ self.H @ self.P
        
        return self.x, self.P, K
