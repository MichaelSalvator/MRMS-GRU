import numpy as np
import cv2
import random
random.seed(36)

mask_change = cv2.imread('postresearch/data/mask_change.png', 0)
mask_nochange = cv2.imread('postresearch/data/mask_nochange.png', 0)
merged_data = np.load('postresearch/data/merged_data.npy')

if mask_change is None:
    raise FileNotFoundError("Error: 'data/mask_change.png' 文件未找到，请检查路径。")
if mask_nochange is None:
    raise FileNotFoundError("Error: 'data/mask_nochange.png' 文件未找到，请检查路径。")

# 将两个 mask 调整为相同大小
# mask_change = cv2.resize(mask_change, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
# mask_nochange = cv2.resize(mask_nochange, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

positive_coords = np.where(mask_change == 255)
positive_coords = list(zip(positive_coords[0], positive_coords[1]))
num_positive = int(len(positive_coords) / 2)
positive_indices = np.random.choice(len(positive_coords), num_positive, replace=False)
selected_positive_coords = [positive_coords[i] for i in positive_indices]

negative_coords = np.where(mask_nochange == 255)
negative_coords = list(zip(negative_coords[0], negative_coords[1]))
negative_indices = np.random.choice(len(negative_coords), num_positive, replace=False)
selected_negative_coords = [negative_coords[i] for i in negative_indices]
num_negative = len(selected_negative_coords)

background_coords = np.where((mask_change == 0) & (mask_nochange == 0))
background_coords = list(zip(background_coords[0], background_coords[1]))
background_indices = np.random.choice(len(background_coords), num_positive*10, replace=False)
selected_background_coords = [background_coords[i] for i in background_indices]
num_background_coords = len(selected_background_coords)

all_coords = selected_positive_coords + selected_negative_coords + selected_background_coords
labels = np.array([2] * num_positive + [1] * num_negative + [0] * num_background_coords)

time, band, h, w = merged_data.shape
length_ratios = [1, 0.5, 0.25, 0.125]
length_names = ['full', 'half', 'quarter', 'eighth']

for ratio, name in zip(length_ratios, length_names):
    t_length = int(time * ratio)
    samples = np.zeros((t_length, band, len(all_coords)))
    for i, (y, x) in enumerate(all_coords):
        samples[:, :, i] = merged_data[:t_length, :, y, x]
    np.save(f'postresearch/data/val_samples_{name}.npy', samples)
    np.save(f'postresearch/data/val_labels_{name}.npy', labels)

vis_img = np.zeros((h, w, 3), dtype=np.uint8)
for y, x in positive_coords:
    cv2.circle(vis_img, (x, y), 3, (0, 0, 255), -1)
for y, x in selected_negative_coords:
    cv2.circle(vis_img, (x, y), 3, (0, 255, 255), -1)
for y, x in selected_background_coords:
    cv2.circle(vis_img, (x, y), 3, (255, 0, 0), -1)

cv2.imwrite('postresearch/data/sample_visualization.png', vis_img)

print(f"处理完成，样本数量: {len(all_coords)}，已保存不同长度的样本和标签以及可视化图像")