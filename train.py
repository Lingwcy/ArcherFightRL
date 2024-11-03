from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import os
from game_env import ArcherFightEnv
from stable_baselines3.common.logger import configure
from torch.utils.tensorboard import SummaryWriter

class SaveOnStepCallback(BaseCallback):
    """
    自定义回调类，用于在每个训练迭代后保存模型，并打印step的输出info。
    """

    def __init__(self, save_freq, save_path, log_path, verbose=0):
        super(SaveOnStepCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.log_path = log_path
        # 初始化TensorBoard的SummaryWriter
        os.makedirs(log_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_path , 'tensorboard'))

    def _on_step(self) -> bool:
        """
        在每个训练迭代后调用。
        """
        # 打印当前轮次总分
        print(self.locals['infos'][0]['当前轮次总分'])

        # 记录到TensorBoard
        current_score = self.locals['infos'][0]['当前轮次总分']
        self.writer.add_scalar('Score', current_score, self.num_timesteps)

        # 检查是否需要保存模型
        if self.num_timesteps % self.save_freq == 0:
            # 获取当前时间并格式化为字符串
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # 创建带时间戳的文件名
            save_path = os.path.join(self.save_path, f"model_{timestamp}_{self.num_timesteps}_steps.zip")
            self.model.save(save_path)
            if self.verbose > 0:
                print(f"模型已保存至 {save_path}")
        return True

    def _on_training_end(self) -> bool:
        # 结束训练时关闭SummaryWriter
        self.writer.close()
        return True
# 假设 ArcherFightEnv 是你的环境类
env = ArcherFightEnv()

log_name = "PPO_ArcherFight_" + datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = os.path.join("logs/", log_name)

# 配置日志记录器，同时输出到文件和控制台
configure(log_path, ["stdout"])

# 初始化模型
model = PPO("CnnPolicy", env, verbose=2,
            learning_rate=1.7e-4,
            gamma=0.99,  # 折扣因子
            n_steps=128,  # 每个环境中的步数
            batch_size=128,  # 批量大小
            n_epochs=5,  # 优化器的迭代次数
            clip_range=0.2,  # 裁剪参数
            gae_lambda=0.95,  # GAE参数
            ent_coef=0.0,  # 熵正则化系数，
            vf_coef=0.5,  # 价值函数系数
            max_grad_norm=0.5,  # 最大梯度范数，用于梯度裁剪
            target_kl=0.01,  # 目标KL散度，用于早期停止
            policy_kwargs=None,  # 传递给策略的额外参数
            )


# 设置每轮保存一次模型
save_freq = 256  # 假设每256步保存一次模型
save_path = "models2/5"

# 确保保存路径存在
os.makedirs(save_path, exist_ok=True)
# 创建回调实例
callback = SaveOnStepCallback(save_freq, save_path, log_path)
# # 加载模型
# model = PPO.load('models2/test4/model_20241101-210616_768_steps')
#
#
# # 继续训练
# model.set_env(env)

# 开始训练，并传入回调
model.learn(total_timesteps=10000, reset_num_timesteps=False, callback=callback)

