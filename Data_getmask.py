import cv2
import numpy as np

img = cv2.imread('data/mask.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])
red_mask = cv2.inRange(hsv, lower_red2, upper_red2)

result = np.zeros_like(img[:,:,0])
result[red_mask != 0] = 255

cv2.imwrite('data/mask2.png', result)

print("处理完成，已保存为mask2.png")