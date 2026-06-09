import cv2
import numpy as np

mask_mangrove = cv2.imread('data/mask_mangrove.png', cv2.IMREAD_GRAYSCALE)
mask_mangrove_resized = cv2.resize(mask_mangrove, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

dual_path = cv2.imread('data/DualPathClassifier.png', cv2.IMREAD_GRAYSCALE)
dual_path_resized = cv2.resize(dual_path, (mask_mangrove_resized.shape[1], mask_mangrove_resized.shape[0]), interpolation=cv2.INTER_LINEAR)

output = np.zeros((mask_mangrove_resized.shape[0], mask_mangrove_resized.shape[1], 3), dtype=np.uint8)

output[(mask_mangrove_resized >= 220) & (dual_path_resized >= 220)] = [255, 255, 255]
output[(dual_path_resized >= 100) & (dual_path_resized <= 200)] = [0, 255, 255]
output[(dual_path_resized >= 220) & (mask_mangrove_resized <= 220)] = [0, 0, 255]
# output[np.where((output[:, :, 0] == 0) & (output[:, :, 1] == 0) & (output[:, :, 2] == 0))] = [0, 0, 0]

mask_jpg = cv2.imread('data/mask2.jpg')
final_output = mask_jpg * 0.8 + output * 0.2

cv2.imwrite('data/output.png', final_output.astype(np.uint8))