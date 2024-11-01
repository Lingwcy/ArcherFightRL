
import cv2
import mss
from skimage.metrics import structural_similarity as ssim
import numpy as np
import time


from pynput.mouse import Controller,Button
def resize_image(image, size):
    # 调整图片大小
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def compare_images(img1_path, screen, size=(256, 256)):
    # 读取图片
    img1 = cv2.imread(img1_path)
    if img1 is None:
        raise FileNotFoundError(f"无法读取图像：{img1_path}")

    # 将mss的截图转换为opencv图像
    screen_image = np.array(screen)
    img2 = cv2.cvtColor(screen_image, cv2.COLOR_RGB2BGR)

    # 调整图片到相同大小
    img1_resized = resize_image(img1, size)
    img2_resized = resize_image(img2, size)

    # 将图片转换为灰度图
    img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

    # 计算结构相似性指数（SSIM）
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

def reset_game(mode, window_location):
    if mode == 'offline':
        reset_offline_game(window_location)
        return

def reset_offline_game(window_location):
    # 预先准备的角色死亡截图路径
    death_img_path = 'source/offline_channel/1.png'
    death_img_path2 = 'source/offline_channel/2.png'
    print('重启等待中....')
    time.sleep(5)
    # 截取的当前模拟器屏幕图像路径
    with mss.mss() as sct:
        sc_grab = sct.grab(window_location)
    # 注意：mss抓取的图像是RGB格式，且包含透明度通道，需要转换为BGR格式
    screen_image = cv2.cvtColor(np.array(sc_grab), cv2.COLOR_RGB2BGR)
    similarity = compare_images(death_img_path, screen_image)
    similarity2 = compare_images(death_img_path2, screen_image)
    # 设置阈值
    threshold = 0.6
    print('相似度:', similarity)
    if similarity > threshold or similarity2 > threshold:
        print("角色死亡，游戏结束，准备重启游戏...")
        # # 重启游戏的代码
        do_offline_reset_game_channel(window_location,1)
    else:
        print("游戏正在进行中...")
def find_game_window_position_right_bottom(window_location):
    # 游戏窗口的位置和大小（逻辑坐标）
    #window_location = {'top': 158, 'left': 576, 'width': 1930, 'height': 1157}
    # 计算窗口右下角的物理坐标
    right_bottom_x = window_location['left'] + window_location['width']
    right_bottom_y = window_location['top'] + window_location['height']
    return  right_bottom_x, right_bottom_y


def do_offline_reset_game_channel(window_location,scale_factor):
    local = {'left': window_location['left'] / scale_factor, 'top': window_location['top'] / scale_factor, 'width': window_location['width'] / scale_factor, 'height': window_location['height'] /scale_factor}
    mouse = Controller()
    window_width = local['width']
    window_height = local['height']
    # 过程 1.png
    x = int(local['left'] + 0.78*window_width)
    y = int(local['top'] + 0.75*window_height)
    mouse.position = (x,y)
    mouse.press(Button.left)
    time.sleep(0.5)
    mouse.release(Button.left)

    time.sleep(1)

    # 过程 2.png
    x = int(local['left'] + 0.8*window_width)
    y = int(local['top'] + 0.85*window_height)
    mouse.position = (x,y)
    mouse.press(Button.left)
    time.sleep(0.5)
    mouse.release(Button.left)

    time.sleep(1)

    # 过程 3.png
    x = int(local['left'] + 0.77*window_width)
    y = int(local['top'] + 0.75*window_height)
    mouse.position = (x,y)
    mouse.press(Button.left)
    time.sleep(0.5)
    mouse.release(Button.left)

    time.sleep(1)

    # 过程 4.png
    x = int(local['left'] + 0.28*window_width)
    y = int(local['top'] + 0.82*window_height)
    mouse.position = (x,y)
    mouse.press(Button.left)
    time.sleep(0.5)
    mouse.release(Button.left)
# #
# # print(scale_properties(window_location,1/2))
# do_offline_reset_game_channel(window_location,2)
# #find_game_window_position_right_bottom({'top': 198, 'left': 1328, 'width': 1184, 'height': 701})
# x, y = find_game_window_position_right_bottom(window_location)
# mouse = Controller()
# x = 196
# y = 159
# mouse.position = (x+1000 , y)
# mouse.position = (196, 159)
# mouse.press(Button.left)
# time.sleep(0.3)
# mouse.release(Button.left)
# reset_game('offline', {'top': 100, 'left': 100, 'width': 800, 'height': 600})
# 1244.0 731.5