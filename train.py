from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import os
from game_env import ArcherFightEnv
from stable_baselines3.common.logger import configure
class SaveOnStepCallback(BaseCallback):
    """
    自定义回调类，用于在每个训练迭代后保存模型。
    """
    def __init__(self, save_freq, save_path, verbose=0):
        super(SaveOnStepCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        """
        在每个训练迭代后调用。
        """
        if self.num_timesteps % self.save_freq == 0:
            # 获取当前时间并格式化为字符串
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # 创建带时间戳的文件名
            save_path = os.path.join(self.save_path, f"model_{timestamp}_{self.num_timesteps}_steps.zip")
            self.model.save(save_path)
            print(f"模型已保存至 {save_path}")
        return True

# 假设 ArcherFightEnv 是你的环境类
env = ArcherFightEnv()

# 配置TensorBoard日志记录
log_name = "PPO_ArcherFight_" + datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = os.path.join("logs/", log_name)

# 配置日志记录器，同时输出到文件和控制台
configure(log_path, ["stdout","log"])

# 初始化模型
model = PPO("CnnPolicy", env, verbose=2, batch_size=256, n_steps=256)


# 设置每轮保存一次模型
save_freq = 256  # 假设每256步保存一次模型
save_path = "models2/test4"

# 确保保存路径存在
os.makedirs(save_path, exist_ok=True)
# 创建回调实例
callback = SaveOnStepCallback(save_freq, save_path)
# 加载模型
model = PPO.load('models2/test4/model_20241101-210616_768_steps')


# 继续训练
model.set_env(env)

# 开始训练，并传入回调
model.learn(total_timesteps=10000, reset_num_timesteps=False, callback=callback)

