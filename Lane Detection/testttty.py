import time
import math
import numpy as np
import cv2
import rospy
# from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology
import matplotlib.pyplot as plt
img = cv2.imread("0830.png")

'''sob'''
thresh_min=15
thresh_max=255
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
abs_sobelx = np.absolute(sobelx)
abs_sobely = np.absolute(sobely)
combined_sobel = cv2.addWeighted(abs_sobelx, 1, abs_sobely, 1, 0)
scaled_sobel = np.uint8(255 * combined_sobel / np.max(combined_sobel))
ret, sob_output = cv2.threshold(scaled_sobel, thresh_min, thresh_max, cv2.THRESH_BINARY)


'''Color'''
thresh=(100, 255)
col = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
s_channel = col[:, :, 2]
# l_channel = col[:, :, 1]
# ret,l_output = cv2.threshold(l_channel, 50, 255, cv2.THRESH_BINARY)
ret,col_output = cv2.threshold(s_channel, 25, 255, cv2.THRESH_BINARY)
# col_output = cv2.add(s_output,l_output)






'''Combine'''
combined_binary = np.zeros_like(sob_output)
combined_binary[(col_output == 255) | (sob_output == 255)] = 255
# Remove noise from the binary image
# combined_binary = combined_binary.astype(bool)
combined_binary = morphology.remove_small_objects(combined_binary, min_size=10, connectivity=2)


img_size = (img.shape[1], img.shape[0])
print(img_size)
# src = np.float32([[430, 450], [790, 450], [190, 695], [1000, 690]])
# src = np.float32([[580, 280], [800, 280], [400, 370], [900, 370]])
# src = np.float32([[475, 240], [700, 240], [200, 375], [800, 375]])
src = np.float32([[500, 255], [720, 255], [320, 375], [820, 375]])
src = np.float32([[545, 400], [720, 400], [240, 695], [1000, 690]])
# src = np.float32([[450, 250], [700, 250], [200, 360], [900, 360]])
dst = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])


M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# Apply perspective transform using cv2.warpPerspective
warped_img = cv2.warpPerspective(combined_binary, M, img_size)




# cv2.imshow('img', img)
# cv2.imshow('gray', gray)
# cv2.imshow('blurred', blurred)
# cv2.imshow('scaled_sobel', scaled_sobel)
# cv2.imshow('sob_output', sob_output)
# cv2.imshow('col', col)
# cv2.imshow('c_output', l_output)
# cv2.imshow('s_output', s_output)

cv2.imshow('col_output', col_output)
cv2.imshow('combined_binary', combined_binary)
cv2.imshow('warped_img', warped_img)
x = [src[0][0],src[1][0],src[2][0],src[3][0]]
y = [src[0][1],src[1][1],src[2][1],src[3][1]]
plt.plot(x, y, 'ro')
plt.imshow(img)
plt.show()





# Wait for a key press
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()

