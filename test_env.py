from game_env import ArcherFightEnv
env = ArcherFightEnv()
episodes = 10
game_window = "MuMu模拟器12"
total_scores = []  # 用来存储每个episode的总分数

for episode in range(episodes):
    obs = env.reset()
    episode_score = 0  # 初始化当前episode的分数
    while True:
        random_action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(random_action)
        print(info)
        episode_score += reward  # 累加当前步骤的奖励到episode分数
        if terminated:
            print("本轮游戏结束...")
            total_scores.append(episode_score)  # 将当前episode的分数添加到总分数列表中
            print(f"轮次 {episode} 分数: {episode_score}")
            break