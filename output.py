import os
import numpy as np
import matplotlib.pyplot as plt
import xlrd


# PPO agent 数据结构
class PPOAgent:
    def __init__(self, episodes, mean_rewards, critic_losses):
        self.episodes = episodes
        self.mean_rewards = mean_rewards
        self.critic_losses = critic_losses


# 平滑函数
def moving_average(data, window_size=10):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# 绘图函数
def plot_ppo(ppo, figurename, parameterlist, variable="reward", smooth=False, window=10, show_raw=False, dpi=300,
             output_svg=False):
    plt.figure(figsize=(12, 8), dpi=dpi)
    has_data = False
    suffix = "smoothed" if smooth else "raw" if show_raw else "normal"
    ext = "svg" if output_svg else "png"
    save_path = f"figures/{figurename}_{suffix}_{variable}.{ext}"
    os.makedirs("figures", exist_ok=True)

    for i in range(len(parameterlist)):
        x = np.array(ppo[i].episodes, dtype=np.float64)
        y = np.array(ppo[i].mean_rewards if variable == "reward" else ppo[i].critic_losses[:len(x)], dtype=np.float64)

        if len(x) == 0 or len(y) == 0:
            continue

        valid_mask = ~np.isnan(y) & ~np.isinf(y)
        x = x[:len(y)]
        x = x[valid_mask]
        y = y[valid_mask]

        if len(x) == 0 or len(y) == 0:
            continue

        if show_raw:
            plt.plot(x, y, label=f"{parameterlist[i]} (raw)", linestyle='--', linewidth=1.5, alpha=0.6)
            has_data = True
        if smooth:
            y_smooth = moving_average(y, window_size=window)
            x_smooth = x[len(x) - len(y_smooth):]
            if len(y_smooth) > 0:
                plt.plot(x_smooth, y_smooth, label=f"{parameterlist[i]} (smoothed)", linewidth=2.5)
                has_data = True

    plt.xlabel("Episodes", fontsize=13)
    plt.ylabel("Average Reward" if variable == "reward" else "Critic Loss", fontsize=13)
    plt.title(f"PPO {'Reward' if variable == 'reward' else 'Critic Loss'} Performance ({suffix})", fontsize=15)
    if has_data:
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"✅ Saved plot: {save_path}")
    else:
        print(f"⚠️ No data to plot for {figurename}, variable={variable}")
    plt.close()


# 从 Excel 文件读取数据
def load_data_from_excel(file_path):
    workbook = xlrd.open_workbook(file_path)

    # 打印所有 sheet 名称以调试
    print("Available sheets:", workbook.sheet_names())

    # 读取奖励数据
    sheet_reward = workbook.sheet_by_name("Change_federation_2_rewar")
    episodes = [int(sheet_reward.cell_value(row, 0)) for row in range(1, sheet_reward.nrows)]
    rewards_true = [float(sheet_reward.cell_value(row, 1)) for row in range(1, sheet_reward.nrows)]
    rewards_false = [float(sheet_reward.cell_value(row, 2)) for row in range(1, sheet_reward.nrows)]

    # 读取损失数据（假设 Change_federation_2_criti 包含损失）
    sheet_loss = workbook.sheet_by_name("Change_federation_2_criti")
    losses_true = [float(sheet_loss.cell_value(row, 1)) for row in range(1, sheet_loss.nrows)]
    losses_false = [float(sheet_loss.cell_value(row, 2)) for row in range(1, sheet_loss.nrows)]

    # 创建 PPOAgent 对象
    agent_true = PPOAgent(episodes, rewards_true, losses_true)
    agent_false = PPOAgent(episodes, rewards_false, losses_false)
    ppo_list = [agent_true, agent_false]
    modes = ["mode=True", "mode=False"]

    return ppo_list, modes


# 主程序
if __name__ == "__main__":
    file_path = r"C:\Users\Tom\Desktop\FRL\FRL-MAPPO\excel\Excel_ppo_20250710_014202.xls"
    try:
        ppo_list, modes = load_data_from_excel(file_path)

        # 生成图像
        plot_ppo(ppo_list, "Change_federation_2", modes, variable="reward", show_raw=True, dpi=300)
        plot_ppo(ppo_list, "Change_federation_2", modes, variable="reward", smooth=True, window=5, dpi=300)
        plot_ppo(ppo_list, "Change_federation_2", modes, variable="critic", show_raw=True, dpi=300)
        plot_ppo(ppo_list, "Change_federation_2", modes, variable="critic", smooth=True, window=5, dpi=300)
    except Exception as e:
        print(f"Error: {e}")