import cv2
import matplotlib.pyplot as plt
import numpy as np
def blue_border_detect(roi):
    #BGR
    roi[:,:,1] = 0
    roi[:,:,2] = 0
    #plt_image = cv2.cvtColor(resize_image,cv2.COLOR_BGR2RGB)
    # plt.imshow(plt_image)
    # plt.show()
    roi[roi >= 255] = 255
    roi[roi < 255] = 0
    roi = cv2.erode(roi, (5, 5), iterations=7)
    resize_image = cv2.dilate(roi,(5, 5), iterations=7)
    # cv2.imshow('res',resize_image)
    # plt.imshow(resize_image)
    # plt.show()
    return resize_image


def gray_border_detect(roi):
    # 定义颜色范围
    lower_range1 = np.array([100, 100, 100])
    upper_range1 = np.array([110, 110, 110])

    lower_range2 = np.array([55, 55, 50])
    upper_range2 = np.array([65, 65, 60])

    # 提取第一个颜色范围的像素点
    mask1 = cv2.inRange(roi, lower_range1, upper_range1)

    # 提取第二个颜色范围的像素点
    mask2 = cv2.inRange(roi, lower_range2, upper_range2)

    # 将两个掩码进行逻辑或操作，合并两个颜色范围
    mask = cv2.bitwise_or(mask1, mask2)

    # 将掩码应用到原图，提取特定颜色范围的像素点
    result = cv2.bitwise_and(roi, roi, mask=mask)
    result = cv2.erode(result, (5, 5), iterations=7)
    result = cv2.dilate(result,(5, 5), iterations=7)
    result[result > 0] = 255
    # cv2.imshow('s',result)
    # cv2.waitKey(0)
    return result
def detect_border(roi, border_type):
    if border_type == 'gray': return gray_border_detect(roi)
    elif border_type == 'blue': return blue_border_detect(roi)

def detect_border_type(game_frame):
    # 读取图像
    # game_frame = cv2.imread(game_frame)  # 替换为你的图像路径
    # game_frame = cv2.cvtColor(game_frame, cv2.COLOR_BGR2RGB)
    # plt.imshow(game_frame)
    # plt.show()
    game_frame = cv2.cvtColor(game_frame, cv2.COLOR_RGB2BGR)

    # 将BGR图像转换为HSV图像
    hsv = cv2.cvtColor(game_frame, cv2.COLOR_BGR2HSV)

    # 定义灰色和蓝色的HSV范围
    blue_lower = np.array([90, 150, 244])
    blue_upper = np.array([130, 180, 255])
    blue_lower2 = np.array([45, 88, 135])
    blue_upper2 = np.array([60, 110, 170])
    gray_lower = np.array([100, 100, 100])
    gray_upper = np.array([115, 115, 115])
    gray_lower2 = np.array([50, 50, 50])
    gray_upper2 = np.array([65, 65, 65])

    # 创建掩码
    gray_mask1 = cv2.inRange(hsv, gray_lower, gray_upper)
    gray_mask2 = cv2.inRange(hsv, gray_lower2, gray_upper2)
    blue_mask1 = cv2.inRange(hsv, blue_lower, blue_upper)
    blue_mask2 = cv2.inRange(hsv, blue_lower2, blue_upper2)

    # 合并掩码
    gray_mask = gray_mask1 | gray_mask2
    blue_mask = blue_mask1 | blue_mask2

    # 应用掩码并分析结果
    gray_pixels = np.sum(gray_mask)
    blue_pixels = np.sum(blue_mask)

    print('灰色像素:', gray_pixels)
    print('蓝色像素:', blue_pixels)

    if gray_pixels > blue_pixels:
        print("游戏 border 为 灰色")
        return 'gray'
    else:
        print("游戏 border 为 蓝色")
        return 'blue'


# detect_border_type('../source/border/border_gary.png')