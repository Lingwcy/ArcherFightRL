import time
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
import matplotlib.pyplot as plt
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
        self.border_type = None
        self.score_threshold = None
        self.max_steps = None
        self.previous_health = None
        self.previous_exp = None
        self.previous_score = None
        self.previous_move_time = None
        self.current_step = None
        self.current_total_score = None
        self.current_hp_rate = None
        self.is_dead = False
        # info 初始化
        self.current_score_reward = None
        self.current_border_reward = None
        self.current_hp_reward = None
        self.current_move_reward =None
        # 窗口ID
        self.window_id = win32gui.FindWindow(None, self.GAME_WINDOW_TITLE)
        # 行为
        self.action_space = spaces.MultiDiscrete(
            [8, 5 ,5]  # 移动  # 单次移动时间(0-10)->(0-1s)
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super(ArcherFightEnv, self).reset(seed=seed)
        self.display_game_window_stream_with_info()
        self.done = False
        self.previous_health = 0
        self.previous_exp = 0
        self.previous_score = 0
        self.previous_move_time = 0
        self.current_step = 0
        self.current_total_score = 0
        self.current_score_reward = 0
        self.current_border_reward = 0
        self.current_hp_reward = 0
        self.current_move_reward = 0
        self.current_hp_rate = 0
        # 终止条件
        self.max_steps = 32
        self.score_threshold = 5000
        # 如果是角色死亡引起的重置，则发起等待...，然后启动重启逻辑
        if self.is_dead:
            self.is_dead = False
            reset_game('offline',self.window_location)

        time.sleep(3)
        # 观测图像，原始图像
        obs = self.get_game_frame()
        # 判断 border 颜色类型
        self.border_type = detect_border_type(obs)
        # 使用观测图像进行ocr检测 拿取 reward
        self.get_current_reward(obs)
        return self.modify_observation(obs), {}

    def step(self, actions: np.ndarray):
        self.detect_window_focused()
        next_state = self.get_next_state(actions)
        reward = self.get_current_reward(next_state)

        self.current_total_score += reward
        self.current_step += 1

        # 复合的终止条件，使得训练的每一轮在满足以下任一条件时 done 为 True
        self.check_epoch_end()
        self.check_dead()
        info = {
            '步次': self.current_step,
            '当前步奖励': reward,
            '当前轮次总分': self.current_total_score,
            '分数奖励': self.current_score_reward,
            '血量奖励': self.current_hp_reward,
            '边界奖励': self.current_border_reward,
            '移动惩罚': self.current_move_reward,
            '移动:': self.MOVEMENT_KEYS[actions[0]],
            '移动时间':int(self.previous_move_time),
            '血量比': self.current_hp_rate
        }
        return (
            self.modify_observation(next_state),
            reward,
            self.done,
            False,
            info
        )
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
    def get_next_state(self,actions):
        # 设置移动时间
        self.previous_move_time = actions[1] * (1 / max(actions[2], 1))
        # 开始执行动作
        key_down(self.MOVEMENT_KEYS[actions[0]])
        # 定义一个列表来存储时间步的观测序列
        obs_sequence = []
        # 使用非阻塞的方式进行移动并记录观测
        start_time = time.time()
        move_duration = self.previous_move_time
        while (time.time() - start_time) < move_duration:
            # 获取并记录每个时间步的观测
            obs = self.get_game_frame()  # 获取实时观测
            # obs_sequence.append(obs)
            self.display_game_window_stream_with_info()
        key_up(self.MOVEMENT_KEYS[actions[0]])  # 停止移动
        final_obs = self.get_game_frame()
        # # # 在循环完成后获取最终观测
        # # final_obs = self.get_game_frame()
        # obs_sequence.append(final_obs)  # 将最终观测添加到序列中
        return final_obs
    def get_current_reward(self , game_frame):
        """
        计算一个 step 的 reward 和 punishment
        :param game_frame:
        :return:
        """
        reward = 0
        # reward += self.get_exp_gained(game_frame) * 100  #等级的奖励是 10 倍率
        self.current_score_reward = int(self.get_score_gained() * 3) # 分数奖励是 1 倍率
        reward += self.current_score_reward

        self.current_hp_reward = int(self.get_health_gained() * 500)
        reward += self.current_hp_reward

        self.current_border_reward = int(self.get_border_gained() * 20)
        reward += self.current_border_reward

        self.current_move_reward = int(self.get_move_gained())# 长时间移动惩罚
        reward += self.current_move_reward

        return reward
    def check_dead(self):
        # 若 self.previous_health > 0.75 则 游戏界面处于 重启游戏的 2.png
        if self.previous_health < 0.06 or self.previous_health > 0.75:
            print("-----------------角色死亡判断------------------")
            # 预先准备的角色死亡截图路径
            death_img_path = 'source/offline_channel/1.png'
            death_img_path2 = 'source/offline_channel/2.png'
            print('死亡判断....')
            with mss.mss() as sct:
                sc_grab = sct.grab(self.window_location)
            screen_image = cv2.cvtColor(np.array(sc_grab), cv2.COLOR_RGB2BGR)
            similarity = compare_images(death_img_path, screen_image)
            similarity2 = compare_images(death_img_path2, screen_image)
            threshold = 0.6
            print('相似度1:', similarity)
            print('相似度2:', similarity2)
            if similarity > threshold or similarity2 > threshold:
                print("角色死亡，游戏结束，准备重启游戏...")
                self.done = True
                self.is_dead = True
            else:
                print("游戏正在进行中...")
    def check_epoch_end(self):
        """
        达到最终步数
        达到分数阈值
        引发的轮次结束
        通过这种方式，有效地控制训练过程中每一轮的执行时间，
        同时确保AI能够在达到一定的性能标准后结束当前回合，提高训练的效率和效果。
        :return:
        """
        self.done = (self.previous_score >= self.score_threshold or
                self.current_step >= self.max_steps)
    def get_move_gained(self)-> int:
        """
        ai经常走3秒以上一步，这是完全无效的长时间移动，如果AI走的越久，惩罚越大
        :return:
        """
        move_time = self.previous_move_time
        if move_time > 2:  # 如果移动时间超过2秒，开始施加惩罚
            punishment = move_time - 2  # 超过的时间越长，惩罚越大
            punishment = -(punishment * 5)
            # print('移动惩罚分:', punishment)
            return punishment # 惩罚系数可以根据需要调整
        else: return 0
    def get_border_gained(self) -> float:
        roi = self.get_border_detect_roi()
        border_image = detect_border(roi, self.border_type)
        border_frame = cv2.cvtColor(border_image, cv2.COLOR_BGR2RGB)

        total_pixels = border_frame[:, :, 2].size
        border_pixels = np.sum(border_frame == 255)
        score = border_pixels / total_pixels

        # 线性分布，接触墙体越多，扣分越多；不接触或少接触墙体，加分
        # 假设墙体接触程度在0到1之间，通过一个线性变换将其映射到-1到1之间
        # 使用简单的线性变换：score * (-2) + 1
        # 当score=0（不接触墙体）时，分数为1；当score=1（完全接触墙体）时，分数为-1
        factor = 0
        if self.border_type == 'blue': factor = -45
        else: factor = -15
        border_score = score * factor + 1

        # 将分数缩放到一个合适的范围，例如乘以30
        return border_score
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
    def get_score_gained(self) -> int:
        roi = self.get_score_detect_roi()
        # 分数 颜色范围
        # low: 100 100 35
        # up: 240 240 60
        low = np.array([35,90,90])
        up = np.array([60,240,240])
        mask = cv2.inRange(roi, low, up)
        result = cv2.bitwise_and(roi, roi, mask=mask)
        # result = cv2.dilate(result, (1, 1), iterations=1)
        # result[result > 0] = 255
        # cv2.imshow('s',result)
        # cv2.waitKey(1)
        score: str = pytesseract.image_to_string(result)
        # print('ocr检测分数：', score)
        gained_score = 0
        if len(score) > 1:
            numbers = re.findall(r'\d+', score)
            score_number = int(numbers[0]) if numbers else None
            if score_number is None:
                return gained_score
            # 确保gained_score非负
            gained_score = max(score_number - self.previous_score, 0)
            # 如果获得的分数不是10的倍数，代表 ocr 检测失败, 放弃此次 奖励 更新
            if gained_score % 10 != 0:
                return 0
            self.previous_score = score_number
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
        # print('血量比:',score)
        self.current_hp_rate = score
        health_score = 0
        if score is not None:
            current_health = score
            if current_health is None: return health_score
            # 目前生命值与上一步生命中的差值
            gained_score = current_health - self.previous_health
            self.previous_health = current_health
            # print('血量奖励:', int(gained_score * 100))
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
            #print("当前 step 窗口坐标:",coordinates)
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
        rect_width = int(window_width * 0.15)
        rect_height = int(window_height * 0.20)

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
        center_y = int(window_height / 2.27)

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
        center_y = int(window_height / 4.5)

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
        rect_width = int(window_width * 0.15)
        rect_height = int(window_height * 0.20)

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
        center_y = int(window_height / 2.27)

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
        center_y = int(window_height / 4.5)

        # 计算矩形的左上角和右下角坐标
        rect_top_left = (center_x - int(rect_width / 2), center_y - int(rect_height / 2))
        rect_bottom_right = (center_x + int(rect_width / 2), center_y + int(rect_height / 2))
        roi = frame[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]
        return roi
    def detect_window_focused(self):
        foreground_id = win32gui.GetForegroundWindow()
        while foreground_id != self.window_id:
            foreground_id = win32gui.GetForegroundWindow()
            time.sleep(1)
            print("等待鼠标瞄准游戏窗口...")

