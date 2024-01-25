import numpy as np
import math
import matplotlib.pyplot as plt

def polar_to_cartesian(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def point_line_distance(x, y, x1, y1, x2, y2):
    absolute_dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    normalization = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return absolute_dist / normalization

def split_and_merge(points, start, end, threshold):
    if start >= end:
        return []
    x1, y1 = points[start]
    x2, y2 = points[end]
    max_dist = 0
    farthest_index = start

    for i in range(start + 1, end):
        x, y = points[i]
        dist = point_line_distance(x, y, x1, y1, x2, y2)
        if dist > max_dist:
            max_dist = dist
            farthest_index = i

    if max_dist > threshold:
        half1 = split_and_merge(points, start, farthest_index, threshold)
        half2 = split_and_merge(points, farthest_index, end, threshold)
        return half1 + half2
    else:
        return [(start, end)]

# Test data in polar coordinates
rho_test = np.array([[10, 11, 11.7, 13, 14, 15, 16, 17, 17, 17, 16.5, 17, 17, 16, 14.5, 14, 13]]).T
n = rho_test.shape[0]
theta_test = (math.pi/180)*np.linspace(0, 85, n).reshape(-1,1)
x_test, y_test = polar_to_cartesian(rho_test, theta_test)

# Apply the Split-and-Merge algorithm
points = np.column_stack((x_test, y_test))
segment_indices = split_and_merge(points, 0, len(points) - 1, 1.5)

# Visualization
plt.figure(figsize=(10, 6))
for start_idx, end_idx in segment_indices:
    segment = points[start_idx:end_idx + 1]
    plt.plot(segment[:, 0], segment[:, 1], marker='o')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Split-and-Merge Line Detection')
plt.grid(True)
plt.show()
