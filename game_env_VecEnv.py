import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces
from game_env import ArcherFightEnv
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

class ArcherFightVecEnv(VecEnv):
    def __init__(self, env):
        """
        初始化VecEnv环境。
        :param env: ArcherFightEnv环境实例。
        """
        # 调用父类的构造函数，传递必要的参数
        super(ArcherFightVecEnv, self).__init__(num_envs=1,
                                                observation_space=env.observation_space,
                                                action_space=env.action_space)
        self.env = env

    def step_async(self, actions):
        """
        异步发送动作到环境中。
        :param actions: 动作数组。
        """
        self.actions = actions

    def step_wait(self):
        """
        等待异步步骤完成，并返回结果。
        :return: 观测序列、奖励、完成标志和额外信息。
        """
        obs_sequences, rewards, dones, _, infos = self.env.step(self.actions)
        # 将观测序列包装成NumPy数组，以匹配VecEnv的接口
        return obs_sequences, rewards, dones, infos

    def reset(self):
        """
        重置环境，并返回序列观测。
        """
        obs, info = self.env.reset()
        return obs, info

    def close(self):
        """
        关闭环境。
        """
        self.env.close()

    def render(self, mode='human'):
        """
        渲染环境。
        """
        return self.env.render(mode)

    def get_attr(self, attr_name, indices=None):
        """
        获取环境中的属性。
        """
        if indices is not None:
            return [getattr(self.env, attr_name) for _ in range(len(indices))]
        else:
            return getattr(self.env, attr_name)

    def set_attr(self, attr_name, value, indices=None):
        """
        设置环境中的属性。
        """
        if indices is not None:
            for i, idx in enumerate(indices):
                setattr(self.env, attr_name, value[i])
        else:
            setattr(self.env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        """
        调用环境中的方法。
        """
        if indices is not None:
            return [getattr(self.env, method_name)(*args, **kwargs) for _ in range(len(indices))]
        else:
            return getattr(self.env, method_name)(*args, **kwargs)

    def env_is_wrapped(self, wrapper_class):
        """
        检查环境是否被特定的包装器包装。
        :param wrapper_class: 包装器类。
        :return: 布尔值，指示环境是否被该包装器包装。
        """
        current_env = self.env
        while isinstance(current_env, VecEnv):
            current_env = current_env.env
        return isinstance(current_env, wrapper_class)

# 使用 ArcherFightVecEnv
from stable_baselines3 import PPO

# 创建 ArcherFightEnv 实例
env = ArcherFightEnv()

# 包装成 VecEnv
vec_env = ArcherFightVecEnv(env)
print(vec_env.action_space)
print(vec_env.observation_space)
# 初始化模型
model = PPO("CnnPolicy", vec_env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)