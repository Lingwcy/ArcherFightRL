import os
# 设置每轮保存一次模型
save_freq = 256  # 假设每256步保存一次模型
save_path = "models2/test5"

# 确保保存路径存在
os.makedirs(save_path, exist_ok=True)