import matplotlib.pyplot as plt

def plot_results(true_states, measurements, estimates, dt):
    """
    Plot the true trajectory, noisy measurements, and Kalman filter estimates.
    """
    true_x = true_states[:, 0]
    true_y = true_states[:, 1]
    
    meas_x = measurements[:, 0]
    meas_y = measurements[:, 1]
    
    est_x = estimates[:, 0]
    est_y = estimates[:, 1]
    
    plt.figure(figsize=(10, 8))
    plt.plot(true_x, true_y, "g-", label="True Trajectory")
    plt.plot(meas_x, meas_y, "rx", label="Measurements")
    plt.plot(est_x, est_y, "b-", label="KF Estimated Trajectory")
    plt.title("2D Object Tracking with Kalman Filter")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("multi_kf.png")
    plt.show()
