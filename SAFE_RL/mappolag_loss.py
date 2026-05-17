import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import math
from torch.utils.tensorboard import SummaryWriter


# ================================================================
# 1. 基础组件与网络定义
# ================================================================
class SimpleLogger:
    def __init__(self, log_dir, seed, output_fname, use_tensorboard):
        self.log_dir = log_dir
        self.output_file = open(os.path.join(log_dir, output_fname), 'w')

    def log(self, msg): print(msg)

    def close(self): self.output_file.close()


def get_current_lagrangian_lr(epoch, total_epochs):
    PHASE_1_END = 200
    LR_HIGH = 0.02
    LR_LOW = 0.0002
    if epoch < PHASE_1_END:
        return LR_HIGH
    else:
        denom = max(1, total_epochs - PHASE_1_END)
        progress = (epoch - PHASE_1_END) / denom
        progress = min(1.0, max(0.0, progress))
        current_lr = LR_HIGH - progress * (LR_HIGH - LR_LOW)
        return current_lr


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        act = activation if j < len(sizes) - 2 else output_activation
        layers.append(act())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden) + [act_dim])
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def forward(self, obs):
        mu = self.net(obs)
        std = torch.exp(self.log_std)
        return mu, std

    def get_action(self, obs_np, deterministic=False, device="cpu"):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mu, std = self.forward(obs)
        act = mu if deterministic else mu + std * torch.randn_like(mu)
        return act.detach().cpu().numpy()

    def log_prob(self, obs, act):
        mu, std = self.forward(obs)
        var = std.pow(2) + 1e-8
        logp = -0.5 * (((act - mu) ** 2) / var + 2 * torch.log(std + 1e-8) + math.log(2 * math.pi))
        return logp.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=(128, 128)):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden) + [1])

    def forward(self, obs):
        return torch.squeeze(self.v(obs), -1)


class OnPolicyBuffer:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.obs, self.acts, self.rews, self.costs, self.dones = [], [], [], [], []

    def add(self, o, a, r, done, cost):
        self.obs.append(np.array(o, copy=True))
        self.acts.append(np.array(a, copy=True))
        self.rews.append(np.array(r, copy=True))
        self.costs.append(np.array(cost, copy=True))
        self.dones.append(done)

    def clear(self):
        self.obs, self.acts, self.rews, self.costs, self.dones = [], [], [], [], []

    def get(self):
        if len(self.obs) == 0: return None
        return (
        np.stack(self.obs), np.stack(self.acts), np.stack(self.rews), np.stack(self.costs), np.array(self.dones))


# ================================================================
# 2. 算法核心 (MAPPO_Lagrangian)
# ================================================================
class MAPPO_Lagrangian:
    def __init__(self, state_dim, action_dim, n_agents, logger=None, device="cpu", episode_length=100, **kwargs):
        self.device = device
        self.logger = logger
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_length = episode_length

        self.hidden = kwargs.get("hidden_size", 128)
        self.actor_lr = kwargs.get("actor_lr", 1e-4)
        self.critic_lr = kwargs.get("critic_lr", 5e-4)
        self.clip = kwargs.get("clip_param", 0.15)
        self.ppo_epochs = kwargs.get("ppo_epochs", 4)
        self.minibatch = kwargs.get("minibatch_size", 512)
        self.gamma = kwargs.get("gamma", 0.98)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)

        self.cost_limit = float(kwargs.get("cost_limit", 19.0))
        self.lagrangian_lr = float(kwargs.get("lagrangian_lr", 0.0005))
        self.lambda_global = 0.0

        self.actors, self.actor_opts = [], []
        self.critics, self.critic_opts = [], []
        self.cost_critics, self.cost_critic_opts = [], []

        for _ in range(self.n_agents):
            a = GaussianActor(state_dim, action_dim, (self.hidden, self.hidden)).to(self.device)
            v = Critic(state_dim, (self.hidden, self.hidden)).to(self.device)
            c = Critic(state_dim, (self.hidden, self.hidden)).to(self.device)
            self.actors.append(a);
            self.critics.append(v);
            self.cost_critics.append(c)
            self.actor_opts.append(optim.Adam(a.parameters(), lr=self.actor_lr))
            self.critic_opts.append(optim.Adam(v.parameters(), lr=self.critic_lr))
            self.cost_critic_opts.append(optim.Adam(c.parameters(), lr=self.critic_lr))
        self.buffer = OnPolicyBuffer(self.n_agents)

    def select_action(self, obs, deterministic=False):
        actions = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
        for i in range(self.n_agents):
            actions[i] = self.actors[i].get_action(obs[i:i + 1], deterministic, self.device)[0]
        return actions

    def store(self, obs, actions, rewards, dones, info):
        cost = info.get("cost", np.zeros(self.n_agents, dtype=np.float32))
        self.buffer.add(obs, actions, rewards, dones, cost)

    def _gae(self, obs_b, rews_b, dones_b, critics, normalize=True):
        T = obs_b.shape[0]
        advs = np.zeros((T, self.n_agents), dtype=np.float32)
        rets = np.zeros((T, self.n_agents), dtype=np.float32)
        for i in range(self.n_agents):
            obs_i = torch.tensor(obs_b[:, i], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                vals = critics[i](obs_i).cpu().numpy()
            last = 0.0
            for t in reversed(range(T)):
                mask = 0.0 if dones_b[t] else 1.0
                next_v = 0.0 if t == T - 1 else vals[t + 1]
                delta = rews_b[t, i] + self.gamma * next_v * mask - vals[t]
                last = delta + self.gamma * self.gae_lambda * mask * last
                advs[t, i] = last
            rets[:, i] = advs[:, i] + vals
            if normalize: advs[:, i] = (advs[:, i] - advs[:, i].mean()) / (advs[:, i].std() + 1e-8)
        return rets, advs

    def update(self):
        batch = self.buffer.get()
        if batch is None: return 0.0, 0.0

        obs_b, acts_b, rews_b, costs_b, dones_b = batch
        ret_r, adv_r = self._gae(obs_b, rews_b, dones_b, self.critics)
        ret_c, adv_c = self._gae(obs_b, costs_b, dones_b, self.cost_critics)

        epoch_critic_losses = []
        epoch_cost_critic_losses = []

        for i in range(self.n_agents):
            obs = torch.tensor(obs_b[:, i], dtype=torch.float32, device=self.device)
            act = torch.tensor(acts_b[:, i], dtype=torch.float32, device=self.device)
            adv = torch.tensor(adv_r[:, i], dtype=torch.float32, device=self.device)
            cadv = torch.tensor(adv_c[:, i], dtype=torch.float32, device=self.device)
            ret = torch.tensor(ret_r[:, i], dtype=torch.float32, device=self.device)
            cret = torch.tensor(ret_c[:, i], dtype=torch.float32, device=self.device)

            with torch.no_grad():
                old_logp = self.actors[i].log_prob(obs, act)
            ds = TensorDataset(obs, act, adv, cadv, ret, cret, old_logp)
            dl = DataLoader(ds, batch_size=self.minibatch, shuffle=True)

            for _ in range(self.ppo_epochs):
                for o, a, ad, cad, r, cr, olp in dl:
                    logp = self.actors[i].log_prob(o, a)
                    ratio = torch.exp(logp - olp)
                    adv_hybrid = ad - self.lambda_global * cad
                    surr1 = ratio * adv_hybrid.unsqueeze(-1)
                    surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_hybrid.unsqueeze(-1)
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = ((self.critics[i](o) - r) ** 2).mean()
                    cost_value_loss = ((self.cost_critics[i](o) - cr) ** 2).mean()

                    self.actor_opts[i].zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                    self.actor_opts[i].step()

                    self.critic_opts[i].zero_grad()
                    value_loss.backward()
                    self.critic_opts[i].step()

                    self.cost_critic_opts[i].zero_grad()
                    cost_value_loss.backward()
                    self.cost_critic_opts[i].step()

                    epoch_critic_losses.append(value_loss.item())
                    epoch_cost_critic_losses.append(cost_value_loss.item())

        step_cost_mean = costs_b.mean()
        estimated_ep_cost = step_cost_mean * self.episode_length
        cost_violation = estimated_ep_cost - self.cost_limit
        self.lambda_global = max(0.0, self.lambda_global + self.lagrangian_lr * cost_violation)
        self.buffer.clear()

        return np.mean(epoch_critic_losses), np.mean(epoch_cost_critic_losses)


# ================================================================
# 3. 专属绘图函数: 绘制以 Episode 为横坐标的 Cost Critic Loss (仅原始值)
# ================================================================
def plot_cost_critic_loss_per_episode(loss_data, save_dir):
    """
    loss_data: list of tuples -> [(total_episodes, cost_loss_value), ...]
    """
    if not loss_data: return

    # 拆包数据
    episodes = [item[0] for item in loss_data]
    losses = [item[1] for item in loss_data]

    # 转为 DataFrame
    df = pd.DataFrame({"Episode": episodes, "Loss": losses})

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 6))

    # 仅使用醒目的橘色绘制原始数据 (Raw)
    plt.plot(df["Episode"], df["Loss"], color='tab:orange', linewidth=2.0, linestyle='-', label='Raw Cost Critic Loss')

    plt.xlabel("Episode")
    plt.ylabel("Loss (MSE)")
    plt.title("Cost Critic Loss: Adaptive Method")
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "adaptive_cost_critic_loss_episode_raw.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n✅ 成功！Cost Critic Loss (仅原始值) 曲线已保存至: {save_path}")


# ================================================================
# 4. 单次实验运行
# ================================================================
def run_single_experiment(cost_limit, num_agents=4, epochs=200, steps_per_epoch=30000, seed=0,
                          log_dir_root="./logs/mec_compare_instant", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = "Adaptive"
    log_dir = os.path.join(log_dir_root, f"Limit{cost_limit}_{exp_name}_seed{seed}")
    os.makedirs(log_dir, exist_ok=True)

    try:
        from safepo.utils import EpochLogger
        logger = EpochLogger(log_dir=log_dir, seed=str(seed), output_fname="progress.csv", use_tensorboard=False)
    except ImportError:
        logger = SimpleLogger(log_dir=log_dir, seed=str(seed), output_fname="progress.csv", use_tensorboard=False)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # 导入环境
    from safepo.common.env import make_ma_mec_env
    env = make_ma_mec_env(n_agents=num_agents)

    obs_example = env.reset()
    if isinstance(obs_example, (list, tuple)): obs_example = np.array(obs_example)
    obs_dim = int(obs_example.shape[1])
    act_dim = int(env.action_size)

    algo = MAPPO_Lagrangian(
        state_dim=obs_dim, action_dim=act_dim, n_agents=num_agents, logger=logger,
        device=device, episode_length=100, hidden_size=128, actor_lr=1e-4, critic_lr=5e-4,
        clip_param=0.15, ppo_epochs=4, minibatch_size=512, gamma=0.98, gae_lambda=0.95,
        max_grad_norm=0.5, cost_limit=float(cost_limit), lagrangian_lr=0.0005,
    )

    cost_loss_data = []  # 专门记录 (Episode, Loss) 的列表
    total_episodes = 0

    for epoch in range(epochs):
        algo.lagrangian_lr = get_current_lagrangian_lr(epoch, epochs)

        o = env.reset()
        ep_ret, ep_cost, ep_len, steps = 0.0, 0.0, 0, 0
        episode_rets, episode_costs = [], []

        while steps < steps_per_epoch:
            a = algo.select_action(o)
            o2, r, done, truncated, info = env.step(a)
            r_arr = np.array(r, dtype=np.float32)
            c_val = np.mean(info.get("cost", 0.0))

            ep_ret += r_arr.sum()
            ep_cost += c_val
            ep_len += 1
            steps += 1
            algo.store(o, a, r_arr, done, info)
            o = o2

            if done or ep_len >= 100:
                total_episodes += 1
                episode_rets.append(ep_ret)
                episode_costs.append(ep_cost)
                ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
                o = env.reset()

        # ==============================================================
        # 获取 Loss，并将当前的 total_episodes 作为 X 轴坐标记录下来
        # ==============================================================
        current_reward_loss, current_cost_loss = algo.update()

        # 记录二元组：(当前总Episode数, 当前Loss)
        cost_loss_data.append((total_episodes, current_cost_loss))

        mean_ret = np.mean(episode_rets) if episode_rets else 0.0
        mean_cost = np.mean(episode_costs) if episode_costs else 0.0

        print(
            f"Adaptive | Epoch {epoch:3d} | Episodes {total_episodes:5d} | Ret {mean_ret:5.1f} | Cost {mean_cost:5.1f} | Cost-Loss {current_cost_loss:.3f}")

    if hasattr(logger, 'close'): logger.close()

    return cost_loss_data


# ================================================================
# 5. 主控函数
# ================================================================
def run_comparison_experiment(args):
    TARGET_LIMIT = args.limit
    save_dir = "./logs/mec_compare_instant/combined_plots"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n>>> 启动实验: 提取 Cost Critic Loss (Epochs={args.epochs}, 预计 60000 Episodes)...")

    # 运行一次实验，拿到 Loss 数据
    loss_data = run_single_experiment(
        cost_limit=TARGET_LIMIT,
        num_agents=args.agents,
        epochs=args.epochs,  # 这里强制传入 200 以达到 60000 episode
        steps_per_epoch=args.steps,
        seed=args.seed
    )

    # 画出专门的以 Episode 为横坐标的图
    plot_cost_critic_loss_per_episode(loss_data, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=4)
    # 默认 200 个 epoch，用来跑够 ~60000 episodes
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=float, default=19.0)
    args = parser.parse_args()

    run_comparison_experiment(args)