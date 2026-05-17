import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from MAPPO import MAPPO
from MADDPG import MADDPG
from Model import NUMBER
import matplotlib.pyplot as plt
import time
from env import MecBCEnv
import xlrd
from xlutils.copy import copy as xl_copy

MAX_EPISODES = 2000
EPISODES_BEFORE_TRAIN = 0

def create_ppo(env, critic_lr=0.001, actor_lr=0.001, noise=0, tau=300, use_federated_aggregation=True,
               roll_out_n_steps=10, batch_size=10, entropy_reg=0.00, reward_gamma=0.99):
    ppo = MAPPO(
        env=env,
        n_agents=env.n_agents,
        state_dim=env.state_size,
        action_dim=env.action_size,
        action_lower_bound=env.action_lower_bound,
        action_higher_bound=env.action_higher_bound,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        noise=noise,
        tau=tau,
        use_federated_aggregation=use_federated_aggregation,
        roll_out_n_steps=roll_out_n_steps,
        batch_size=batch_size,
        entropy_reg=entropy_reg,
        reward_gamma=reward_gamma
    )
    while ppo.n_episodes < MAX_EPISODES:
        ppo.interact()
        if ppo.n_episodes >= EPISODES_BEFORE_TRAIN:
            ppo.train()
    return ppo

def create_maddpg(env, critic_lr=0.001, actor_lr=0.001):
    maddpg = MADDPG(
        env=env,
        n_agents=env.n_agents,
        state_dim=env.state_size,
        action_dim=env.action_size,
        action_lower_bound=env.action_lower_bound,
        action_higher_bound=env.action_higher_bound,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        training_strategy="centralized",
        episodes_before_train=EPISODES_BEFORE_TRAIN
    )
    while maddpg.n_episodes < MAX_EPISODES:
        maddpg.interact()
        if maddpg.n_episodes >= EPISODES_BEFORE_TRAIN:
            maddpg.train()
    return maddpg

def writeExcel(agents, workbook, sheetname, algorithm_list, variable="reward"):
    sheetname = f"{sheetname}_{int(time.time())}"
    sheetname = sheetname[:20]
    sheet = workbook.add_sheet(sheetname)
    sheet.write(0, 0, "Episodes")
    max_episodes = max(len(agent.episodes) for agent in agents)
    for j in range(max_episodes):
        sheet.write(j + 1, 0, j + 1 if j < len(agents[0].episodes) else "")
    for i in range(len(algorithm_list)):
        if variable == "reward":
            sheet.write(0, i + 1, f"Rewards(algorithm={algorithm_list[i]})")
            for j in range(max_episodes):
                if j < len(agents[i].mean_rewards):
                    sheet.write(j + 1, i + 1, agents[i].mean_rewards[j])
                else:
                    sheet.write(j + 1, i + 1, "")
        elif variable == "phi":
            sheet.write(0, i + 1, f"Phi(algorithm={algorithm_list[i]})")
            for j in range(max_episodes):
                if j < len(agents[i].mean_phi):
                    sheet.write(j + 1, i + 1, agents[i].mean_phi[j])
                else:
                    sheet.write(j + 1, i + 1, "")
    return workbook

def plot_ppo(agents, parameter, algorithm_list, variable="reward"):
    plt.figure(figsize=(12, 5))

    # Subplot 1: Fed=True
    plt.subplot(1, 2, 1)
    fed_true_agents = [agents[i] for i in range(0, 3)]  # First 3 agents are Fed=True
    fed_true_labels = [algorithm_list[i] for i in range(0, 3)]
    for i, agent in enumerate(fed_true_agents):
        episodes = agent.episodes[:len(agent.mean_rewards)]
        plt.plot(episodes, agent.mean_rewards, label=fed_true_labels[i])
    # Calculate average line for Fed=True (last 400 episodes)
    last_400_start = max(0, MAX_EPISODES - 400)
    avg_rewards = [sum(agent.mean_rewards[last_400_start:]) / len(agent.mean_rewards[last_400_start:])
                   if len(agent.mean_rewards) > last_400_start else 0
                   for agent in fed_true_agents]
    avg_reward = sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0
    plt.axhline(y=avg_reward, color='r', linestyle='--', label=f'Avg Reward (Last 400)')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Performance with Fed=True")
    plt.legend()
    plt.grid(True)

    # Subplot 2: Fed=False
    plt.subplot(1, 2, 2)
    fed_false_agents = [agents[i] for i in range(3, 6)]  # Last 3 agents are Fed=False
    fed_false_labels = [algorithm_list[i] for i in range(3, 6)]
    for i, agent in enumerate(fed_false_agents):
        episodes = agent.episodes[:len(agent.mean_rewards)]
        plt.plot(episodes, agent.mean_rewards, label=fed_false_labels[i])
    # Calculate average line for Fed=False (last 400 episodes)
    avg_rewards = [sum(agent.mean_rewards[last_400_start:]) / len(agent.mean_rewards[last_400_start:])
                   if len(agent.mean_rewards) > last_400_start else 0
                   for agent in fed_false_agents]
    avg_reward = sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0
    plt.axhline(y=avg_reward, color='r', linestyle='--', label=f'Avg Reward (Last 400)')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Performance with Fed=False")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"./output/ppo_roll_out_comparison_{parameter}.png")
    plt.close()

def run(times, variable):
    algorithm_list = [
        "roll_out=1, Fed=True", "roll_out=5, Fed=True", "roll_out=10, Fed=True",
        "roll_out=1, Fed=False", "roll_out=5, Fed=False", "roll_out=10, Fed=False"
    ]
    MODE = "normal"
    S_EPSILON = 0.86
    roll_out_values = [1, 5, 10]

    rworkbook = xlrd.open_workbook('excel/Excel_ppo.xls', formatting_info=True)
    wworkbook = xl_copy(rworkbook)

    env_list = [MecBCEnv(n_agents=NUMBER, S_EPSILON=S_EPSILON, mode=MODE) for _ in algorithm_list]
    agent_list = []
    for roll_out in roll_out_values:
        for fed in [True, False]:
            agent = create_ppo(
                env_list.pop(0),
                use_federated_aggregation=fed,
                noise=0.15,
                roll_out_n_steps=roll_out,
                batch_size=128,
                tau=400,
                actor_lr=0.0005,
                critic_lr=0.001,
                entropy_reg=0.01,
                reward_gamma=0.95
            )
            agent_list.append(agent)

    wworkbook = writeExcel(agent_list, wworkbook, f"Compare_RollOut_{times}", algorithm_list, variable)
    plot_ppo(agent_list, f"roll_out_{times}", algorithm_list, variable)
    wworkbook.save(f"excel/Excel_ppo_roll_out_comparison_{times}.xls")

if __name__ == "__main__":
    run(2, "reward")