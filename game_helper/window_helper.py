import mss
import cv2
import numpy as np
import pygetwindow as gw


def get_game_window_coordinates(title):
    """
    获取游戏窗口的坐标。

    Args:
    title (str): 游戏窗口的标题。

    Returns:
    dict: 包含窗口的left, top, width, height的字典。
    """
    game_windows = gw.getWindowsWithTitle(title)
    if game_windows:
        game_window = game_windows[0]
        return {'left': game_window.left, 'top': game_window.top, 'width': game_window.width,
                'height': game_window.height}
    else:
        return None


def draw_border_detect_roi(frame, window_width, window_height):
    """
    在游戏窗口中心画一个矩形，作为角色检测碰撞 border 的 roi。

    Args:
    frame (numpy.ndarray): 游戏窗口的图像帧。
    window_width (int): 游戏窗口的宽度。
    window_height (int): 游戏窗口的高度。
    """
    # 计算矩形的宽度和高度（基于游戏窗口宽高比0.25）
    rect_width = int(window_width * 0.11)
    rect_height = int(window_height * 0.16)

    # 计算矩形的中心位置
    center_x = int(window_width / 2)
    center_y = int(window_height / 2)

    # 计算矩形的左上角和右下角坐标
    rect_top_left = (center_x - int(rect_width / 2), center_y - int(rect_height / 2))
    rect_bottom_right = (center_x + int(rect_width / 2), center_y + int(rect_height / 2))

    # 在游戏窗口中心绘制矩形
    cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 255, 0), 2)

def draw_hp_detect_roi(frame, window_width, window_height):
    """
    检测角色 血条 的 roi。

    Args:
    frame (numpy.ndarray): 游戏窗口的图像帧。
    window_width (int): 游戏窗口的宽度。
    window_height (int): 游戏窗口的高度。
    """
    # 计算矩形的宽度和高度（基于游戏窗口宽高比0.25）
    rect_width = int(window_width * 0.09)
    rect_height = int(window_height * 0.03)

    # 计算矩形的中心位置
    center_x = int(window_width / 2)
    center_y = int(window_height / 2.27)

    # 计算矩形的左上角和右下角坐标
    rect_top_left = (center_x - int(rect_width / 2), center_y - int(rect_height / 2))
    rect_bottom_right = (center_x + int(rect_width / 2), center_y + int(rect_height / 2))
    # roi = frame[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]
    # cv2.imshow('roi',roi)
    # 在游戏窗口中心绘制矩形
    cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 255, 0), 2)
    cv2.waitKey(1)
def draw_score_detect_roi(frame, window_width, window_height):
    """
    检测角色 血条 的 roi。

    Args:
    frame (numpy.ndarray): 游戏窗口的图像帧。
    window_width (int): 游戏窗口的宽度。
    window_height (int): 游戏窗口的高度。
    """
    # 计算矩形的宽度和高度（基于游戏窗口宽高比0.25）
    rect_width = int(window_width * 0.06)
    rect_height = int(window_height * 0.03)

    # 计算矩形的中心位置
    center_x = int(window_width / 1.04)
    center_y = int(window_height / 4.5)

    # 计算矩形的左上角和右下角坐标
    rect_top_left = (center_x - int(rect_width / 2), center_y - int(rect_height / 2))
    rect_bottom_right = (center_x + int(rect_width / 2), center_y + int(rect_height / 2))
    # roi = frame[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]
    # cv2.imshow('roi',roi)
    # 在游戏窗口中心绘制矩形
    cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 255, 0), 2)
    cv2.waitKey(1)

def display_game_window_stream_with_info(title):
    """
    显示游戏窗口的视频流并在中心画矩形。

    Args:
    title (str): 游戏窗口的标题。
    """
    with mss.mss() as sct:
        while True:
            coordinates = get_game_window_coordinates(title)
            if not coordinates:
                print("游戏窗口未找到")
                return

            monitor = {
                'top': coordinates['top'],
                'left': coordinates['left'],
                'width': coordinates['width'],
                'height': coordinates['height']
            }
            screenshot = sct.grab(monitor)
            if screenshot is None:
                continue  # 如果截图为空，则跳过当前循环

            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 在游戏窗口中心画矩形
            draw_border_detect_roi(frame, monitor['width'], monitor['height'])
            draw_hp_detect_roi(frame, monitor['width'], monitor['height'])
            draw_score_detect_roi(frame, monitor['width'], monitor['height'])

            # 显示游戏窗口的视频流
            cv2.imshow('Game Window Stream', frame)

            # 检查按键，如果按下'q'则退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


# 使用方法
if __name__ == "__main__":
    game_title = "MuMu模拟器12"  # 替换为游戏窗口的实际标题
    display_game_window_stream_with_info(game_title)