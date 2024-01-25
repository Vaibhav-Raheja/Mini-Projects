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
img = cv2.imread("test.png")
# img = rospy.Subscriber('camera/image_raw', Image, queue_size=1)
# pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
# pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
# img = sub_image.copy()
# img = cv2.imread("test.png")
'''sob'''
thresh_min=30
thresh_max=255
# 1. Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 2. Gaussian blur the image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#3. Use cv2.Sobel() to find derivatives for both X and Y Axis
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
# 4. Use cv2.addWeighted() to combine the results
abs_sobelx = np.absolute(sobelx)
abs_sobely = np.absolute(sobely)
combined_sobel = cv2.addWeighted(abs_sobelx, 1, abs_sobely, 1, 0)
# 5. Convert each pixel to uint8, then apply threshold to get a binary image
scaled_sobel = np.uint8(255 * combined_sobel / np.max(combined_sobel))
# sob_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255
ret, sob_output = cv2.threshold(scaled_sobel, thresh_min, thresh_max, cv2.THRESH_BINARY)


'''Color'''
thresh=(100, 255)
col = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
# l_bound = np.array([0, 80, 0])
# u_bound = np.array([0, 255, 255])
# c_line = np.zeros_like(col)
# c_output = cv2.inRange(col, l_bound, u_bound)
# 2. Extract the S (saturation) channel

l_channel = col[:, :, 1]
ret, l_output = cv2.threshold(l_channel, 30, thresh[1], cv2.THRESH_BINARY)
s_channel = col[:, :, 2]
ret, s_output = cv2.threshold(s_channel, thresh[0], thresh[1], cv2.THRESH_BINARY)
# # 3. Apply threshold on S channel to get a binary image
col_output = np.zeros_like(col)
col_output = cv2.add(s_output,l_output)






'''Combine'''
combined_binary = np.zeros_like(sob_output)
combined_binary[(col_output == 255) | (sob_output == 255)] = 255
# Remove noise from the binary image
# combined_binary = combined_binary.astype(bool)
# combined_binary = morphology.remove_small_objects(combined_binary, min_size=10, connectivity=2)


img_size = (img.shape[1], img.shape[0])
print(img_size)
# # src = np.float32([[237, 280], [415, 280], [0, 400], [620, 400]])
# src = np.float32([[235, 285], [400, 280], [10, 420], [640, 420 ]])

# # src = np.float32([[450, 250], [700, 250], [200, 360], [900, 360]])
# offset = 75
# # dst = np.float32([[0, 0], [img.shape[1]-offset, offset], [0, img.shape[0]+offset], [img.shape[1]-offset, img.shape[0]+offset]])
dst = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
# img= img[0 : 400, 0 : 640]
src = np.float32([[260, 270], [385, 270], [0, 395], [640, 395]])
# dst = np.float32([[0, 0], [640, 0], [640, 400], [0, 400]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# Apply perspective transform using cv2.warpPerspective
warped_img = cv2.warpPerspective(combined_binary, M, img_size)
warped_img = cv2.copyMakeBorder(warped_img, 0, 0, 100, 50, cv2.BORDER_CONSTANT, None, value = 0)
warped_img = cv2.resize(warped_img, (640, 480))


cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.imshow('blurred', blurred)
cv2.imshow('scaled_sobel', scaled_sobel)
cv2.imshow('sob_output', sob_output)
cv2.imshow('gg', col)
cv2.imshow('h_output', l_output)
cv2.imshow('s_output', s_output)
cv2.imshow('col_output', col_output)
cv2.imshow('combined_binary', combined_binary)
cv2.imshow('warped_img', warped_img)

plt.imshow(img)
# plt.imshow(warped_img)
x = [src[0][0],src[1][0],src[2][0],src[3][0]]
y = [src[0][1],src[1][1],src[2][1],src[3][1]]
plt.plot(x, y, 'ro')
# bird = cv2.imread("bird_from_cv2.png")
# plt.imshow(bird)
plt.show()





# Wait for a key press
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()

