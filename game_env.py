from time import sleep
from game_helper.detect import detect_border_type,detect_border
import cv2
import gymnasium
import mss
import numpy as np
import re
import pygetwindow as gw
from game_helper.reset_game import compare_images
import pytesseract
import win32gui
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from game_helper.keyboard_helper import *
from typing import Optional
from game_helper.reset_game import reset_game

pytesseract.pytesseract.tesseract_cmd = (
    r"D:\\Tesseract-OCR\\tesseract.exe"
)

class ArcherFightEnv(gymnasium.Env):
    MOVEMENT_KEYS = ["w", "a", "s", "d","wa","wd","ds","sa"]
    GAME_WINDOW_HEIGHT = 240*2
    GAME_WINDOW_WIDTH = 420*2
    GAME_WINDOW_TITLE = "MuMu模拟器12"

    def __init__(self):
        super(ArcherFightEnv, self).__init__()
        # 游戏窗口
        self.window_id = win32gui.FindWindow(None, self.GAME_WINDOW_TITLE)
        # 行为
        self.action_space = spaces.MultiDiscrete(
            [8, 10]  # 移动  # 单次移动时间(0-10)->(0-1s)
        )

        # 观测空间
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.GAME_WINDOW_HEIGHT,
                self.GAME_WINDOW_WIDTH,
                4,  # color channels
            ),
            dtype=np.uint8,
        )
        self.is_first_epoch = True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.display_game_window_stream_with_info()
        self.reward = 'undefined'
        self.done = False
        self.previous_health = 0
        self.previous_exp = 0
        self.previous_score = 0

        print(self.window_location)
        # 如果是重启游戏，则发起等待...，然后启动重启逻辑
        if not self.is_first_epoch:
            reset_game('offline',self.window_location)


        # 观测图像，原始图像
        obs = self.get_game_frame()

        # 判断 border 颜色类型
        self.border_type = detect_border_type(obs)
        # 使用观测图像进行ocr检测 拿取 reward
        self.get_current_reward(obs)
        return self.modify_observation(obs), {}

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        print("\n\nSTEP ###############################################")
        foreground_id = win32gui.GetForegroundWindow()
        while foreground_id != self.window_id:
            foreground_id = win32gui.GetForegroundWindow()
            sleep(1)
            print("等待鼠标瞄准游戏窗口...")

        self.display_game_window_stream_with_info()

        print('移动:',self.MOVEMENT_KEYS[actions[0]])
        # 移动时间
        move_time = actions[1] / 3
        print('移动时间:', move_time)
        # 执行 action
        key_down(self.MOVEMENT_KEYS[actions[0]])
        sleep(move_time)
        key_up(self.MOVEMENT_KEYS[actions[0]])
        # 从最新的观测来计算 reward
        obs = self.get_game_frame()


        self.reward = self.get_current_reward(obs)
        # 判断角色是否死亡
        if self.previous_health < 0.06:
            print("-----------------角色死亡判断------------------")
            # 预先准备的角色死亡截图路径
            death_img_path = 'source/offline_channel/1.png'
            print('死亡判断....')
            with mss.mss() as sct:
                sc_grab = sct.grab(self.window_location)
            screen_image = cv2.cvtColor(np.array(sc_grab), cv2.COLOR_RGB2BGR)
            similarity = compare_images(death_img_path, screen_image)
            threshold = 0.6
            print('相似度:', similarity)
            if similarity > threshold:
                print("角色死亡，游戏结束，准备重启游戏...")
                self.is_first_epoch = False
                self.done = True
            else:
                print("游戏正在进行中...")
        return self.modify_observation(obs), self.reward, self.done, False, {}


    def get_game_frame(self):
        """
        :argument 此方法为 ai gent 提供 observation
        :return: obs/status
        """
        with mss.mss() as sct:
            sc_grab = sct.grab(self.window_location)
            game_frame  = self.frame(sc_grab)

            # 确保图像是uint8类型
            #game_frame = np.array(game_frame, dtype=np.uint8)
            # 显示图像

        return game_frame
    def frame(self, sc_grab):
        # 将 mss.shot() 返回的图像数据转换为 OpenCV 图像格式
        img = np.array(sc_grab)
        if img.shape[2] == 4:  # 检查是否有 alpha 通道
            img = img[:, :, :3]  # 移除 alpha 通道
        game_frame = cv2.resize(
            img,
            (self.GAME_WINDOW_WIDTH, self.GAME_WINDOW_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        game_frame = cv2.cvtColor(game_frame , cv2.COLOR_BGR2RGB)  # 转换为 RGB
        return game_frame
    def get_current_reward(self , game_frame):
        """
        计算一个 step 的 reward 和 punishment
        :param game_frame:
        :return:
        """
        reward = 0
        reward += self.get_exp_gained(game_frame) * 100  #等级的奖励是 10 倍率
        reward += self.get_score_gained() * 1# 分数奖励是 1 倍率
        reward += self.get_health_gained() * 100
        reward -= self.get_border_gained() * 100 # 接触墙体 -> 0.1 接触边界 -> 0.3  倍率 1000
        print('本轮总得分:', reward)
        return reward
    def get_border_gained(self) -> float:
        roi = self.get_border_detect_roi()
        border_image = detect_border(roi,self.border_type)
        border_frame = cv2.cvtColor(border_image, cv2.COLOR_BGR2RGB)

        total_pixels = border_frame[:,:,2].size
        border_pixels = np.sum(border_frame == 255)
        score = border_pixels / total_pixels
        print('total_pixels',total_pixels)
        print('border_pixels',border_pixels)
        print('score',score)
        print('边界分数', -int(score * 100)) # 接触墙体 -> 0.1 接触边界 -> 0.3
        return score
    def get_exp_gained(self, game_frame) -> int:
        w, h = 17 * 2, 10 * 2
        x, y = 118 * 2, 16 * 2
        # 经验值ROI
        exp_game_frame = game_frame[y:y+h, x:x+w].copy()
        # Posterize
        # exp_game_frame[exp_game_frame >= 128] = 255
        # exp_game_frame[exp_game_frame < 128] = 0
        # exp_game_frame[:, :, 1] = 0
        # exp_game_frame[:, :, 2] = 0

        # # cv2.imshow("Captcha Check", exp_game_frame)
        # # cv2.waitKey(50)
        # 使用 Matplotlib 显示图像
        # exp_game_frame = cv2.cvtColor(exp_game_frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(exp_game_frame)
        # # 显示图像窗口
        # plt.show()

        exp: str = pytesseract.image_to_string(exp_game_frame)
        gained_exp = 0
        if len(exp) > 1 :
            numbers = re.findall(r'\d+', exp)
            level_number = int(numbers[0]) if numbers else None
            if level_number is None: return gained_exp
            gained_exp = level_number - self.previous_exp
            self.previous_exp = level_number
            print('等级奖励:',gained_exp * 100)
            return gained_exp
        return gained_exp
    def get_score_gained(self,) -> int:
        roi = self.get_score_detect_roi()
        score: str = pytesseract.image_to_string(roi)
        print('ocr检测分数：',score)
        gained_score = 0
        if len(score) > 1 :
            numbers = re.findall(r'\d+', score)
            score_number = int(numbers[0]) if numbers else None
            if score_number is None: return gained_score
            gained_score = score_number - self.previous_score
            self.previous_score = score_number
            print('分数奖励:', gained_score)
            return gained_score
        return gained_score
    def get_health_gained(self) -> float:
        roi = self.get_hp_detect_roi()
        # img = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        red_mask = np.zeros_like(roi)
        red_mask[:, :, 2] = 255
        roi[:, :, 0] = 0
        roi[:, :, 1] = 0
        red_channel_image = cv2.bitwise_and(roi, red_mask)
        red_channel_image[red_channel_image >= 170] = 255
        red_channel_image[red_channel_image < 170] = 0

        # plt.imshow(red_channel_image)
        # # 显示图像窗口
        # plt.show()

        total_pixels = red_channel_image[:, :, 2].size
        white_pixels = np.sum(red_channel_image == 255)
        score = white_pixels / total_pixels
        # 0.58 -> 100%
        # 0.01 -> dead
        print('血量比:',score)

        health_score = 0
        if score is not None:
            current_health = score
            if current_health is None: return health_score
            # 目前生命值与上一步生命中的差值
            gained_score = current_health - self.previous_health
            self.previous_health = current_health
            print('血量奖励:', int(gained_score * 100))
            return gained_score
        return health_score
    def modify_observation(self,obs):
        # 如果观测值只有三个通道，添加一个全为0的透明度通道
        if obs.shape[-1] == 3:
            obs = np.concatenate(
                (obs, np.zeros((obs.shape[0], obs.shape[1], 1), dtype=obs.dtype)),
                axis=-1)
        return obs
    def display_game_window_stream_with_info(self):
        """
        显示游戏窗口的视频流并在中心画矩形。

        Args:
        title (str): 游戏窗口的标题。
        """
        with mss.mss() as sct:
            coordinates = self.get_game_window_coordinates()
            self.window_location = coordinates
            print("当前 step 窗口坐标:",coordinates)
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
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            self.bgr_frame = frame
            # 在游戏窗口中心画矩形
            self.draw_border_detect_roi(frame, monitor['width'], monitor['height'])
            self.draw_hp_detect_roi(frame, monitor['width'], monitor['height'])
            self.draw_score_detect_roi(frame, monitor['width'], monitor['height'])
            cv2.imshow('HUD Window', frame)
            cv2.waitKey(1)
    def get_game_window_coordinates(self):
        """
        获取游戏窗口的坐标。

        Args:
        title (str): 游戏窗口的标题。

        Returns:
        dict: 包含窗口的left, top, width, height的字典。
        """
        game_windows = gw.getWindowsWithTitle(self.GAME_WINDOW_TITLE)
        if game_windows:
            game_window = game_windows[0]
            return {'left': game_window.left, 'top': game_window.top, 'width': game_window.width,
                    'height': game_window.height}
        else:
            return None
    @staticmethod
    def draw_border_detect_roi(frame, window_width, window_height):
        """
        在游戏窗口中心画一个矩形，作为角色检测碰撞 border 的 roi。

        Args:
        frame (numpy.ndarray): 游戏窗口的图像帧。
        window_width (int): 游戏窗口的宽度。
        window_height (int): 游戏窗口的高度。
        """
        # 计算矩形的宽度和高度（基于游戏窗口宽高比0.25）
        rect_width = int(window_width * 0.13)
        rect_height = int(window_height * 0.18)

        # 计算矩形的中心位置
        center_x = int(window_width / 2)
        center_y = int(window_height / 2)

        # 计算矩形的左上角和右下角坐标
        rect_top_left = (center_x - int(rect_width / 2), center_y - int(rect_height / 2))
        rect_bottom_right = (center_x + int(rect_width / 2), center_y + int(rect_height / 2))

        # 在游戏窗口中心绘制矩形
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 255, 0), 2)
    @staticmethod
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
        center_y = int(window_height / 2.19)

        # 计算矩形的左上角和右下角坐标
        rect_top_left = (center_x - int(rect_width / 2), center_y - int(rect_height / 2))
        rect_bottom_right = (center_x + int(rect_width / 2), center_y + int(rect_height / 2))

        # 在游戏窗口中心绘制矩形
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 255, 0), 2)
    @staticmethod
    def draw_score_detect_roi(frame, window_width, window_height):
        """
        检测角色 分数 的 roi。

        Args:
        frame (numpy.ndarray): 游戏窗口的图像帧。
        window_width (int): 游戏窗口的宽度。
        window_height (int): 游戏窗口的高度。
        """
        rect_width = int(window_width * 0.06)
        rect_height = int(window_height * 0.03)

        center_x = int(window_width / 1.04)
        center_y = int(window_height / 4)

        # 计算矩形的左上角和右下角坐标
        rect_top_left = (center_x - int(rect_width / 2), center_y - int(rect_height / 2))
        rect_bottom_right = (center_x + int(rect_width / 2), center_y + int(rect_height / 2))

        # 在游戏窗口中心绘制矩形
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 255, 0), 2)

    def get_border_detect_roi(self):
        frame = self.bgr_frame
        window_width = self.window_location['width']
        window_height = self.window_location['height']
        # 计算矩形的宽度和高度（基于游戏窗口宽高比0.25）
        rect_width = int(window_width * 0.13)
        rect_height = int(window_height * 0.18)

        # 计算矩形的中心位置
        center_x = int(window_width / 2)
        center_y = int(window_height / 2)

        # 计算矩形的左上角和右下角坐标
        rect_top_left = (center_x - int(rect_width / 2), center_y - int(rect_height / 2))
        rect_bottom_right = (center_x + int(rect_width / 2), center_y + int(rect_height / 2))

        roi = frame[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]
        return roi

    def get_hp_detect_roi(self):
        frame = self.bgr_frame
        window_width = self.window_location['width']
        window_height = self.window_location['height']
        rect_width = int(window_width * 0.09)
        rect_height = int(window_height * 0.03)

        # 计算矩形的中心位置
        center_x = int(window_width / 2)
        center_y = int(window_height / 2.19)

        # 计算矩形的左上角和右下角坐标
        rect_top_left = (center_x - int(rect_width / 2), center_y - int(rect_height / 2))
        rect_bottom_right = (center_x + int(rect_width / 2), center_y + int(rect_height / 2))
        roi = frame[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]
        return roi

    def get_score_detect_roi(self):
        frame = self.bgr_frame
        window_width = self.window_location['width']
        window_height = self.window_location['height']
        rect_width = int(window_width * 0.06)
        rect_height = int(window_height * 0.03)

        center_x = int(window_width / 1.04)
        center_y = int(window_height / 4)

        # 计算矩形的左上角和右下角坐标
        rect_top_left = (center_x - int(rect_width / 2), center_y - int(rect_height / 2))
        rect_bottom_right = (center_x + int(rect_width / 2), center_y + int(rect_height / 2))
        roi = frame[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]
        return roi



