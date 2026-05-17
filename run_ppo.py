import os
import asyncio
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from MAPPO import MAPPO
from Model import NUMBER
import matplotlib.pyplot as plt
from env import S_EPSILON, MecBCEnv
import xlrd
import logging
import time
import numpy as np
from openpyxl.utils.exceptions import IllegalCharacterError
import xlwt
MAX_EPISODES = 100
EPISODES_BEFORE_TRAIN = 0


# def create_ppo(env, critic_lr=0.001, actor_lr=0.001, noise=0, tau=300):
#     ppo = MAPPO(env=env, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size,
#                   action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound,
#                   critic_lr=critic_lr, actor_lr=actor_lr, noise=noise, tau=tau)
#     while ppo.n_episodes < MAX_EPISODES:
#         ppo.interact()
#         if ppo.n_episodes >= EPISODES_BEFORE_TRAIN:
#             ppo.train()
#     return ppo
# 测试有无联邦聚合

# 旧方法
# def create_ppo(env, critic_lr=0.001, actor_lr=0.001, noise=0, tau=300, use_federated_aggregation=True):
#     ppo = MAPPO(env=env, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size,
#                   action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound,
#                   critic_lr=critic_lr, actor_lr=actor_lr, noise=noise, tau=tau,
#                   use_federated_aggregation=use_federated_aggregation)
#     while ppo.n_episodes < MAX_EPISODES:
#         ppo.interact()
#         if ppo.n_episodes >= EPISODES_BEFORE_TRAIN:
#             ppo.train()
#     return ppo
async def create_ppo(env, critic_lr=0.0003, actor_lr=0.0003, noise=0, tau=1, use_federated_aggregation=True):
    ppo = MAPPO(
        env=env, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size,
        action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound,
        critic_lr=critic_lr, actor_lr=actor_lr, noise=noise, tau=tau,
        use_federated_aggregation=use_federated_aggregation, max_episodes=MAX_EPISODES, entropy_reg=0.2
    )
    while ppo.n_episodes < MAX_EPISODES:
        logging.info(f"Starting episode {ppo.n_episodes + 1} for ppo with federated_aggregation={use_federated_aggregation}")
        await ppo.interact()
        if ppo.n_episodes >= EPISODES_BEFORE_TRAIN:
            ppo.train()
    logging.info(f"Completed {MAX_EPISODES} episodes for ppo with federated_aggregation={use_federated_aggregation}")
    return ppo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def writeExcel(agent, workbook, sheetname, parameterlist, variable="reward", smoothed=False):
    try:
        os.makedirs("excel", exist_ok=True)
        base_sheetname = f"{sheetname}_{variable}"[:25]
        if smoothed:
            base_sheetname = f"{sheetname}_smoothed_{variable}"[:25]
        counter = 1
        existing_sheets = [sheet.name for sheet in workbook._Workbook__worksheets]
        sheetname = base_sheetname
        while sheetname in existing_sheets:
            sheetname = f"{base_sheetname}_{counter}"[:31]
            counter += 1

        logging.info(f"Creating sheet: {sheetname}")

        if not agent or not parameterlist:
            logging.warning(f"No agents or parameters provided for {sheetname}")
            sheet = workbook.add_sheet(sheetname)
            sheet.write(0, 0, "Episodes")
            for i, param in enumerate(parameterlist, 1):
                sheet.write(0, i, f"{variable.capitalize()}(mode={param})")
                sheet.write(1, i, "N/A")
            sheet.write(1, 0, "N/A")
            workbook.save("output.xlsx")
            return workbook

        episode_lengths = [len(a.episodes) for a in agent if a.episodes]
        data_lengths = [len(a.mean_rewards if variable == "reward" else a.critic_losses) for a in agent]
        if smoothed and variable == "reward":
            data_lengths = [len(moving_average(a.mean_rewards, window_size=10)) for a in agent]
        elif smoothed and variable == "critic_loss":
            data_lengths = [len(moving_average(a.critic_losses, window_size=10)) for a in agent]
        if not episode_lengths or not data_lengths:
            logging.warning(f"No valid episodes or {variable} data for {sheetname}")
            sheet = workbook.add_sheet(sheetname)
            sheet.write(0, 0, "Episodes")
            for i, param in enumerate(parameterlist, 1):
                sheet.write(0, i, f"{variable.capitalize()}(mode={param})")
                sheet.write(1, i, "N/A")
            sheet.write(1, 0, "N/A")
            workbook.save("output.xlsx")
            return workbook

        min_episodes = min(episode_lengths)
        min_data = min(data_lengths)
        if min_episodes != min_data:
            logging.warning(f"Mismatch in episodes ({min_episodes}) and {variable} ({min_data}) lengths")
            min_episodes = min(min_episodes, min_data)

        logging.info(f"Writing to sheet: {sheetname}, min_episodes={min_episodes}, {variable}_lengths={data_lengths}")

        data = {"Episodes": []}
        for param in parameterlist:
            data[f"{variable.capitalize()}(mode={param})"] = []

        valid_rows = 0
        for j in range(min_episodes):
            if j >= len(agent[0].episodes) or any(j >= len(a.mean_rewards if variable == "reward" else a.critic_losses) for a in agent):
                logging.warning(f"Skipping episode {j + 1} due to missing data")
                continue
            episode = agent[0].episodes[j]
            has_valid_data = False
            row_data = {}
            for i, param in enumerate(parameterlist):
                if variable == "reward" and smoothed:
                    value = moving_average(agent[i].mean_rewards, window_size=10)[j] if j < len(moving_average(agent[i].mean_rewards, window_size=10)) else None
                elif variable == "critic_loss" and smoothed:
                    value = moving_average(agent[i].critic_losses, window_size=10)[j] if j < len(moving_average(agent[i].critic_losses, window_size=10)) else None
                else:
                    value = agent[i].mean_rewards[j] if variable == "reward" else agent[i].critic_losses[j]
                if not np.isnan(value) and not np.isinf(value):
                    has_valid_data = True
                    row_data[f"{variable.capitalize()}(mode={param})"] = value
                else:
                    row_data[f"{variable.capitalize()}(mode={param})"] = None
                    logging.warning(f"Invalid {variable} data for episode {episode}, mode={param}, value={value}")
            if has_valid_data:
                data["Episodes"].append(episode)
                for param in parameterlist:
                    data[f"{variable.capitalize()}(mode={param})"].append(row_data[f"{variable.capitalize()}(mode={param})"])
                valid_rows += 1
            else:
                logging.warning(f"No valid {variable} data for episode {episode}")

        if valid_rows > 0:
            sheet = workbook.add_sheet(sheetname)
            sheet.write(0, 0, "Episodes")
            for i, param in enumerate(parameterlist, 1):
                sheet.write(0, i, f"{variable.capitalize()}(mode={param})")
            for row in range(valid_rows):
                sheet.write(row + 1, 0, data["Episodes"][row])
                for i, param in enumerate(parameterlist, 1):
                    if data[f"{variable.capitalize()}(mode={param})"][row] is not None:
                        sheet.write(row + 1, i, data[f"{variable.capitalize()}(mode={param})"][row])
        else:
            logging.warning(f"No valid data to write for {sheetname}")
            sheet = workbook.add_sheet(sheetname)
            sheet.write(0, 0, "Episodes")
            for i, param in enumerate(parameterlist, 1):
                sheet.write(0, i, f"{variable.capitalize()}(mode={param})")
                sheet.write(1, i, "N/A")
            sheet.write(1, 0, "N/A")

        output_file = os.path.join("excel", f"Excel_ppo_{time.strftime('%Y%m%d_%H%M%S')}.xls")
        workbook.save(output_file)
        logging.info(f"Created sheet: {sheetname}, valid rows={valid_rows}, saved to {output_file}")
        return workbook

    except IllegalCharacterError as e:
        logging.error(f"Illegal character in sheetname: {e}")
        sheetname = f"sheet_{counter}"[:31]
        sheet = workbook.add_sheet(sheetname)
        sheet.write(0, 0, "Episodes")
        sheet.write(1, 0, "Error: Illegal characters")
        workbook.save("output.xlsx")
        return workbook
    except Exception as e:
        logging.error(f"Error in writeExcel: {e}")
        workbook.save("output.xlsx")
        return workbook

def moving_average(data, window_size=20):
    if len(data) == 0:
        return np.array([])
    window_size = min(window_size, len(data))
    if window_size < 1:
        window_size = 1
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='same')


def plot_ppo(ppo, figurename, parameterlist, variable="reward", smooth=False, window=10, show_raw=False):
    plt.figure(figsize=(10, 6))
    has_data = False
    save_path = f"figures/{figurename}_{variable}.png"
    if smooth:
        save_path = f"figures/{figurename}_smoothed_{variable}.png"
    elif show_raw:
        save_path = f"figures/{figurename}_raw_{variable}.png"
    os.makedirs("figures", exist_ok=True)

    for i in range(len(parameterlist)):
        x = np.array(ppo[i].episodes, dtype=np.float64)
        y = np.array(ppo[i].mean_rewards if variable == "reward" else ppo[i].critic_losses[:len(x)], dtype=np.float64)
        print(f"Plotting {variable}, mode={parameterlist[i]}, raw x={x}, raw y={y}")

        if len(x) == 0 or len(y) == 0:
            print(f"Warning: No data for {variable}, mode={parameterlist[i]}, x_len={len(x)}, y_len={len(y)}")
            continue

        valid_mask = ~np.isnan(y) & ~np.isinf(y)
        x = x[:len(y)]  # 截取 x 到与 y 相同长度
        x = x[valid_mask]
        y = y[valid_mask]

        print(f"After filtering: {variable}, mode={parameterlist[i]}, x={x}, y={y}, valid_len={len(y)}")

        if len(x) == 0 or len(y) == 0:
            print(f"Warning: No valid data after filtering for {variable}, mode={parameterlist[i]}")
            continue

        if show_raw:
            plt.plot(x, y, label=f"mode={parameterlist[i]} (raw)", linestyle='--', alpha=0.5)
            has_data = True
        if smooth:
            y_smooth = moving_average(y, window_size=window)
            x_smooth = x[len(x) - len(y_smooth):]  # 同步截取 x
            print(f"After smoothing: {variable}, mode={parameterlist[i]}, x_smooth={x_smooth}, y_smooth={y_smooth}, smooth_len={len(y_smooth)}")
            if len(x_smooth) != len(y_smooth):
                print(f"Warning: Length mismatch for {variable}, mode={parameterlist[i]}, x_smooth={len(x_smooth)}, y_smooth={len(y_smooth)}")
                continue
            if len(y_smooth) > 0:
                plt.plot(x_smooth, y_smooth, label=f"mode={parameterlist[i]} (smoothed)")
                has_data = True

    plt.xlabel("Episodes")
    plt.ylabel("Average Reward" if variable == "reward" else "Critic Loss")
    plt.title(f"PPO {'Reward' if variable == 'reward' else 'Critic Loss'} Performance{' (Smoothed)' if smooth else (' (Raw)' if show_raw else '')}")
    if has_data:
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        print(f"Saved plot with data: {save_path}")
    else:
        print(f"Warning: No data plotted for {variable}, saving empty plot: {save_path}")
    plt.close()
async def run(times, variable="reward"):
    loop = asyncio.get_event_loop()
    try:
        aggregation_settings = [True, False]
        MODE = "normal"
        env_list = [MecBCEnv(n_agents=NUMBER, S_EPSILON=S_EPSILON, mode=MODE) for _ in range(2)]
        ppo_list = [
            await create_ppo(env_list[0], use_federated_aggregation=True),
            await create_ppo(env_list[1], use_federated_aggregation=False)
        ]
        for i, ppo in enumerate(ppo_list):
            logging.info(
                f"PPO {i}: episodes_len={len(ppo.episodes)}, mean_rewards_len={len(ppo.mean_rewards)}, critic_losses_len={len(ppo.critic_losses)}")
            if not ppo.episodes or not ppo.mean_rewards or not ppo.critic_losses:
                logging.warning(
                    f"PPO {i} has incomplete data: episodes={len(ppo.episodes)}, mean_rewards={len(ppo.mean_rewards)}, critic_losses={len(ppo.critic_losses)}")
        wworkbook = xlwt.Workbook()
        wworkbook = writeExcel(ppo_list, wworkbook, f"Change_federation_{times}", aggregation_settings, "reward")
        wworkbook = writeExcel(ppo_list, wworkbook, f"Change_federation_{times}", aggregation_settings, "reward", smoothed=True)
        wworkbook = writeExcel(ppo_list, wworkbook, f"Change_federation_{times}", aggregation_settings, "critic_loss")
        wworkbook = writeExcel(ppo_list, wworkbook, f"Change_federation_{times}", aggregation_settings, "critic_loss", smoothed=True)
        output_file = os.path.join("excel", f"Excel_ppo_{time.strftime('%Y%m%d_%H%M%S')}.xls")
        wworkbook.save(output_file)
        logging.info(f"Saved Excel file: {output_file}")
        plot_ppo(ppo_list, f"federation_{times}", aggregation_settings, "reward", show_raw=True)
        plot_ppo(ppo_list, f"federation_{times}", aggregation_settings, "reward", smooth=True, window=20)
        plot_ppo(ppo_list, f"federation_{times}", aggregation_settings, "critic_loss", show_raw=True)
        plot_ppo(ppo_list, f"federation_{times}", aggregation_settings, "critic_loss", smooth=True, window=10)


    except Exception as e:

        logging.error(f"Run function failed: {str(e)}")

        ppo_list = []  # 确保 ppo_list 初始化

        wworkbook = xlwt.Workbook() if 'wworkbook' not in locals() else wworkbook

        wworkbook = writeExcel(ppo_list, wworkbook, f"Interrupted_federation_{times}", aggregation_settings, "reward")

        wworkbook = writeExcel(ppo_list, wworkbook, f"Interrupted_federation_{times}", aggregation_settings, "reward",
                               smoothed=True)

        wworkbook = writeExcel(ppo_list, wworkbook, f"Interrupted_federation_{times}", aggregation_settings,
                               "critic_loss")

        wworkbook = writeExcel(ppo_list, wworkbook, f"Interrupted_federation_{times}", aggregation_settings,
                               "critic_loss", smoothed=True)

        if ppo_list:  # 仅在 ppo_list 非空时调用 plot_ppo

            plot_ppo(ppo_list, f"interrupted_federation_{times}", aggregation_settings, "reward", show_raw=True)

            plot_ppo(ppo_list, f"interrupted_federation_{times}", aggregation_settings, "reward", smooth=True,
                     window=10)

            plot_ppo(ppo_list, f"interrupted_federation_{times}", aggregation_settings, "critic_loss", show_raw=True)

            plot_ppo(ppo_list, f"interrupted_federation_{times}", aggregation_settings, "critic_loss", smooth=True,
                     window=10)

        output_file = os.path.join("excel", f"Excel_ppo_interrupted_{time.strftime('%Y%m%d_%H%M%S')}.xls")

        wworkbook.save(output_file)

        logging.info(f"训练结果已保存到 {output_file}")

        raise
    # env_mode_list = [MecBCEnv(n_agents=NUMBER, S_EPSILON=S_EPSILON, mode=mode) for mode in All_modes]
    # ppo_mode_list = [create_ppo(env) for env in env_mode_list]
    # wworkbook = writeExcel(ppo_mode_list, wworkbook, f"Change_mode_{times}", All_modes, variable)
    # plot_ppo(ppo_mode_list, f"mode_{times}", All_modes, variable)

    # # change ddl
    # env_ddl_list = [MecBCEnv(n_agents=NUMBER, S_DDL=All_ddl[i]) for i in range(len(All_ddl))]
    # ppo_ddl_list = [create_ppo(env_ddl_list[i], noise=noise[i], tau=tau[i]) for i in range(len(env_ddl_list))]
    # wworkbook = writeExcel(ppo_ddl_list, wworkbook, "Change_ddl_%s"%times, All_ddl, variable)
    # plot_ppo(ppo_ddl_list, "ddl_%s"%times, All_ddl, variable)

    # change epsilon
    # env_epsilon_list = [MecBCEnv(n_agents=NUMBER, S_EPSILON=All_epsilon[i]) for i in range(len(All_epsilon))]
    # ppo_epsilon_list = [create_ppo(env_epsilon_list[i]) for i in range(len(env_epsilon_list))]
    # wworkbook = writeExcel(ppo_epsilon_list, wworkbook, "Change_epsilon_%s"%times, All_epsilon, variable)
    # plot_ppo(ppo_epsilon_list, "epsilon_%s"%times, All_epsilon, variable)

    # # change bandwidth
    # env_bandwidth_list = [MecBCEnv(n_agents=NUMBER, W_BANDWIDTH=All_bandwidth[i]) for i in range(len(All_bandwidth))]
    # ppo_bandwidth_list = [create_ppo(env_bandwidth_list[i]) for i in range(len(env_bandwidth_list))]
    # wworkbook = writeExcel(ppo_bandwidth_list, wworkbook, "Change_bandwidth_%s"%times, All_bandwidth, variable)
    # plot_ppo(ppo_bandwidth_list, "bandwidth_%s"%times, All_bandwidth, variable)

    # # change agents
    # env_agents_list = [MecBCEnv(n_agents=All_agents[i]) for i in range(len(All_agents))]
    # ppo_agents_list = [create_ppo(env_agents_list[i], noise=noise[i], tau=tau[i]) for i in range(len(env_agents_list))]
    # wworkbook = writeExcel(ppo_agents_list, wworkbook, "Change_agents_%s"%times, All_agents, variable)
    # plot_ppo(ppo_agents_list, "agents_%s"%times, All_agents, variable)

if __name__ == "__main__":
    try:
        asyncio.run(run(2, "reward"))
    except KeyboardInterrupt:
        logging.info("主程序被用户中断，退出。")
        raise