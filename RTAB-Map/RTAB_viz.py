import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math 

def main():
    rtabmap_x = []
    rtabmap_y = []
    rtabmap_z = []

    ekf_x = []
    ekf_y = []
    ekf_z = []

    # Reading RTAB-Map data
    with open('rtabmap_trajectory.txt', 'r') as rtabmap_file:
        for line in rtabmap_file:
            x, y, z = line.split(',')
            rtabmap_x.append(float(x))
            rtabmap_y.append(float(y))
            rtabmap_z.append(float(z))

    with open('ekf_trajectory.txt', 'r') as ekf_file:
        for line in ekf_file:
            x, y, z = line.split(',')
            ekf_x.append(float(x))
            ekf_y.append(float(y))
            ekf_z.append(float(z))
            
    # Converting lists to numpy arrays for easier mathematical operations
    rtabmap_x = np.array(rtabmap_x)
    rtabmap_y = np.array(rtabmap_y)
    rtabmap_z = np.array(rtabmap_z)
    ekf_x = np.array(ekf_x)
    ekf_y = np.array(ekf_y)
    ekf_z = np.array(ekf_z)
    
    # Truncating data to the shortest length for fair comparison

    
    # Calculating RMSE for each axis

    # Plotting the trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(rtabmap_x, rtabmap_y, rtabmap_z, c='blue',label='RTAB-Map')
    ax.plot(ekf_x, ekf_y, ekf_z, c='red', label='EKF')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Trajectories Comparison')
    ax.legend()
    # plt.savefig('./src/scripts/trajectory.png')
    plt.show()


    min_length = min(len(rtabmap_x), len(ekf_x))
    rtabmap_x = rtabmap_x[:min_length]
    rtabmap_y = rtabmap_y[:min_length]
    rtabmap_z = rtabmap_z[:min_length]
    ekf_x = ekf_x[:min_length]
    ekf_y = ekf_y[:min_length]
    ekf_z = ekf_z[:min_length]

    rmse_x = math.sqrt(np.mean((rtabmap_x - ekf_x) ** 2))
    rmse_y = math.sqrt(np.mean((rtabmap_y - ekf_y) ** 2))
    rmse_z = math.sqrt(np.mean((rtabmap_z - ekf_z) ** 2))
    print("RMSE: ", rmse_x, rmse_y, rmse_z, "(x, y, z)")
    
    # Calculating 3D RMSE
    rmse_3d = math.sqrt(np.mean((rtabmap_x - ekf_x) ** 2 + (rtabmap_y - ekf_y) ** 2 + (rtabmap_z - ekf_z) ** 2))
    print("3D RMSE: ", rmse_3d)

if __name__ == '__main__':
    main()