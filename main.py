# main.py
import numpy as np
from multi_kf import KalmanFilter
from plotting import plot_results

def simulate_motion(F, Q, x0, num_steps):
    """
    Simulate the object's motion using the state transition matrix F and process noise Q.
    """
    states = [x0]
    x = x0
    for _ in range(num_steps - 1):
        w = np.random.multivariate_normal(mean=np.zeros(Q.shape[0]), cov=Q)
        x = F @ x + w
        states.append(x)
    return np.array(states)

def generate_measurements(H, R, true_states):
    """
    Generate noisy measurements from the true states using measurement matrix H and measurement noise R.
    """
    measurements = []
    for x in true_states:
        v = np.random.multivariate_normal(mean=np.zeros(R.shape[0]), cov=R)
        z = H @ x + v
        measurements.append(z)
    return np.array(measurements)

def main():
    dt = 0.1              
    total_time = 10.0     
    num_steps = int(total_time / dt)
    
    # Define the state vector: [x, y, x_dot, y_dot]
    x0 = np.array([0.0, 0.0, 1.0, 0.5])
    
    # Define the State Transition Matrix F for a constant velocity model.
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])
    
    # Assume no control inputs
    B = None
    u = None
    
    # Process noise covariance matrix Q.
    q = 0.1
    Q = q * np.eye(4)
    
    # Measurement matrix H.
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    
    # Measurement noise covariance matrix R.
    r = 1.0
    R = r * np.eye(2)
    
    # Initial estimate error covariance matrix P0.
    P0 = np.eye(4)
    
    # Simulate the true motion of the object.
    true_states = simulate_motion(F, Q, x0, num_steps)
    
    # Generate noisy measurements from the true states.
    measurements = generate_measurements(H, R, true_states)
    
    # Initialize the Kalman Filter.
    kf = KalmanFilter(F, H, Q, R, P0, x0)
    
    # Run the Kalman Filter for each measurement.
    estimates = []
    for z in measurements:
        # Predict step
        kf.predict(u, B)
        
        # Update step
        x_updated, P_updated, K = kf.update(z)
        
        # Store the current state estimate
        estimates.append(x_updated)
    
    estimates = np.array(estimates)
    
    # Plot the trajectories
    plot_results(true_states, measurements, estimates, dt)
    
if __name__ == "__main__":
    main()
