import cv2
import numpy as np
import matplotlib.pyplot as plt
path = '../source/health_bar_red.png'
img = cv2.imread(path)
print(img.shape)
# 创建一个只包含红色通道的掩码
red_mask = np.zeros_like(img)
red_mask[:, :, 2] = 255  # 红色通道在OpenCV中是第三个索引（BGR格式）

# 将蓝色和绿色通道设置为0
img[:, :, 0] = 0  # 蓝色通道
img[:, :, 1] = 0  # 绿色通道

# 将掩码应用到图像上，只保留红色通道
red_channel_image = cv2.bitwise_and(img, red_mask)
red_channel_image[ red_channel_image >= 170] = 255
red_channel_image[red_channel_image < 170 ] =0

# 计算255像素值的面积比例
total_pixels = red_channel_image[:,:,2].size
white_pixels = np.sum(red_channel_image == 255)
area_ratio = white_pixels / total_pixels
print(f"总面积: {total_pixels}")
print(f"红色像素: {white_pixels}")
print(f"血量: {area_ratio + 0.1}")
cv2.imshow('org', img)
cv2.imshow('resize_img',red_channel_image)
cv2.waitKey(0)