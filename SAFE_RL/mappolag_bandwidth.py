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


# ================================================================
# 1. 基础组件 (Logger, Network, Buffer) - 保持不变
# ================================================================
class SimpleLogger:
    def __init__(self, log_dir, seed, output_fname):
        self.log_dir = log_dir
        self.output_file = open(os.path.join(log_dir, output_fname), 'w')

    def log(self, msg): print(msg)

    def close(self): self.output_file.close()


def get_current_lagrangian_lr(epoch, total_epochs):
    PHASE_1_END = int(total_epochs * 0.4)
    LR_HIGH = 0.02
    LR_LOW = 0.0002
    if epoch < PHASE_1_END:
        return LR_HIGH
    else:
        denom = max(1, total_epochs - PHASE_1_END)
        progress = (epoch - PHASE_1_END) / denom
        progress = min(1.0, max(0.0, progress))
        return LR_HIGH - progress * (LR_HIGH - LR_LOW)


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
# 2. 算法核心 (MAPPO_Lagrangian) - 保持不变
# ================================================================
class MAPPO_Lagrangian:
    def __init__(self, state_dim, action_dim, n_agents, logger=None, device="cpu", episode_length=100, **kwargs):
        self.device = device
        self.logger = logger
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_length = episode_length

        self.hidden = kwargs.get("hidden_size", 256)
        self.actor_lr = kwargs.get("actor_lr", 1e-4)
        self.critic_lr = kwargs.get("critic_lr", 5e-4)
        self.clip = kwargs.get("clip_param", 0.2)
        self.ppo_epochs = kwargs.get("ppo_epochs", 5)
        self.minibatch = kwargs.get("minibatch_size", 1000)
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)

        self.cost_limit = float(kwargs.get("cost_limit", 19.0))
        # 建议使用较小的 lagrangian_lr 以获得平滑表现
        self.lagrangian_lr = float(kwargs.get("lagrangian_lr", 0.001))
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

                    self.actor_opts[i].zero_grad();
                    policy_loss.backward();
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm);
                    self.actor_opts[i].step()
                    self.critic_opts[i].zero_grad();
                    value_loss.backward();
                    self.critic_opts[i].step()
                    self.cost_critic_opts[i].zero_grad();
                    cost_value_loss.backward();
                    self.cost_critic_opts[i].step()

        step_cost_mean = costs_b.mean()
        estimated_ep_cost = step_cost_mean * self.episode_length
        cost_violation = estimated_ep_cost - self.cost_limit
        self.lambda_global = max(0.0, self.lambda_global + self.lagrangian_lr * cost_violation)
        self.buffer.clear()
        return 0, 0


# ================================================================
# 3. 运行单个 Bandwidth 实验 (修正版)
# ================================================================
def run_bandwidth_episode(cost_limit, fixed_lambda, bandwidth, num_agents, epochs, steps_per_epoch, seed, device):
    try:
        from safepo.common.env import make_ma_mec_env
        # 传入带宽参数
        env = make_ma_mec_env(n_agents=num_agents, W_BANDWIDTH=float(bandwidth))
    except ImportError:
        print("Env import error");
        return 0.0, 0.0

    np.random.seed(seed);
    torch.manual_seed(seed)

    class NullLogger:
        def log(self, msg): pass

    obs_example = env.reset()
    if isinstance(obs_example, (list, tuple)): obs_example = np.array(obs_example)
    obs_dim = int(obs_example.shape[1]);
    act_dim = int(env.action_size)

    algo = MAPPO_Lagrangian(
        state_dim=obs_dim, action_dim=act_dim, n_agents=num_agents,
        logger=NullLogger(), device=device, episode_length=100,
        cost_limit=float(cost_limit), lagrangian_lr=0.001
    )

    if fixed_lambda is not None:
        algo.lambda_global = float(fixed_lambda)
        algo.lagrangian_lr = 0.0

    final_rewards = []
    final_success_rates = []

    for epoch in range(epochs):
        if fixed_lambda is None:
            algo.lagrangian_lr = get_current_lagrangian_lr(epoch, epochs)

        o = env.reset()
        ep_len = 0;
        steps = 0

        # --- [统计变量] ---
        epoch_ret_sum = 0.0
        episodes_in_epoch = 0
        ep_success_sum = 0.0  # 累计整个 epoch 所有 step 的成功率

        ep_ret = 0.0  # 当前 Episode 的累计奖励

        while steps < steps_per_epoch:
            a = algo.select_action(o)
            o2, r, done, truncated, info = env.step(a)
            algo.store(o, a, np.array(r, dtype=np.float32), done, info)

            # 1. 累加当前 Episode 的奖励 (关键修正)
            ep_ret += np.mean(r)

            # 2. 累加 Step 成功率 (用于计算平均成功率)
            step_success = np.mean(info.get("is_success", [0.0]))
            ep_success_sum += step_success

            steps += 1;
            ep_len += 1
            o = o2

            if done or ep_len >= 100:
                # 3. Episode 结束，记录总分
                epoch_ret_sum += ep_ret
                episodes_in_epoch += 1

                # 重置环境和计数器
                o = env.reset();
                ep_len = 0;
                ep_ret = 0.0

        algo.update()

        # 计算本 Epoch 平均指标
        avg_ep_reward = epoch_ret_sum / max(1, episodes_in_epoch)
        avg_ep_success = ep_success_sum / max(1, steps)  # 平均每步的成功率

        # 只取最后 20% 的 Epoch 数据
        if epoch >= int(epochs * 0.8):
            final_rewards.append(avg_ep_reward)
            final_success_rates.append(avg_ep_success)

        if epoch % 10 == 0:
            # 打印修正后的 Reward，应该能看到几十甚至上百的数值了
            print(f"   > Ep {epoch}/{epochs} | AvgRet: {avg_ep_reward:.1f} | AvgSucc: {avg_ep_success:.2f}")

    return np.mean(final_rewards), np.mean(final_success_rates)

# ================================================================
# 4. 绘图：双Y轴图表 (Reward 和 Success Rate)
# ================================================================
def plot_bandwidth_analysis(bandwidths, results, scenarios, save_dir):
    plt.rcParams.update({'font.size': 14})

    # ------------------ 图1: Reward vs Bandwidth ------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.2  # 柱子宽度
    x = np.arange(len(bandwidths))

    for i, sc in enumerate(scenarios):
        # 提取数据
        rewards = [results[bw][sc['label']]['reward'] for bw in bandwidths]
        # 偏移位置
        offset = (i - len(scenarios) / 2 + 0.5) * width
        ax.bar(x + offset, rewards, width, label=sc['label'], color=sc['color'], edgecolor='black', alpha=0.8)

    ax.set_xlabel('Bandwidth (MHz)')
    ax.set_ylabel('Average Episode Reward')
    ax.set_title('Impact of Bandwidth on Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bandwidths])
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bandwidth_reward.png"), dpi=300)
    plt.close()

    # ------------------ 图2: Success Rate vs Bandwidth ------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    for sc in scenarios:
        rates = [results[bw][sc['label']]['success'] for bw in bandwidths]
        # 用折线图表现趋势
        ax.plot(bandwidths, rates, marker='o', linewidth=3, label=sc['label'], color=sc['color'])

    ax.set_xlabel('Bandwidth (MHz)')
    ax.set_ylabel('Task Success Rate')
    ax.set_title('Impact of Bandwidth on Reliability')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bandwidth_success.png"), dpi=300)
    plt.close()

    print(f"图表已保存至: {save_dir}")


# ================================================================
# 5. 主控程序
# ================================================================
def run_bandwidth_experiment(args):
    # 【配置】带宽列表
    BANDWIDTHS = [5, 10, 20, 30]

    scenarios = [
        {"label": "Adaptive", "fixed_val": None, "color": "tab:red"},
        {"label": "Fix $\lambda=100$", "fixed_val": 10.0, "color": "tab:green"},
        {"label": "Fixed $\lambda=50$", "fixed_val": 8.0, "color": "tab:purple"},
        {"label": "Fix $\lambda=1$", "fixed_val": 0.1, "color": "tab:blue"},
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./logs/mec_bandwidth"
    os.makedirs(save_dir, exist_ok=True)

    # 存储结构: results[bw][label] = {'reward': x, 'success': y}
    results = {}

    print(f">>> 开始带宽敏感性实验 (Bandwidths: {BANDWIDTHS})")

    for bw in BANDWIDTHS:
        print(f"\n========== Testing with Bandwidth = {bw} MHz ==========")
        results[bw] = {}

        for sc in scenarios:
            print(f"Running: {sc['label']}...")

            # 运行实验
            final_reward, final_success = run_bandwidth_episode(
                cost_limit=args.limit,
                fixed_lambda=sc["fixed_val"],
                bandwidth=bw,
                num_agents=args.agents,  # 默认使用 4 个智能体
                epochs=args.epochs,
                steps_per_epoch=args.steps,
                seed=args.seed,
                device=device
            )

            results[bw][sc['label']] = {'reward': final_reward, 'success': final_success}
            print(f"--> {sc['label']} Done. R: {final_reward:.1f}, Succ: {final_success:.2f}")

    print("\n>>> 绘图...")
    plot_bandwidth_analysis(BANDWIDTHS, results, scenarios, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=4)
    # 同样建议 epochs 设为 50-100 用于快速验证趋势
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=float, default=25.0)
    args = parser.parse_args()

    run_bandwidth_experiment(args)