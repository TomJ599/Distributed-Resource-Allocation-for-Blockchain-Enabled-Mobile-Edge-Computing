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
# 1. 基础组件
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
# 2. 算法核心
# ================================================================
class MAPPO_Lagrangian:
    def __init__(self, state_dim, action_dim, n_agents, logger=None, device="cpu", episode_length=100, **kwargs):
        self.device = device
        self.logger = logger
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_length = episode_length

        # Params
        self.hidden = kwargs.get("hidden_size", 256)
        self.actor_lr = kwargs.get("actor_lr", 1e-4)
        self.critic_lr = kwargs.get("critic_lr", 5e-4)
        self.clip = kwargs.get("clip_param", 0.2)
        self.ppo_epochs = kwargs.get("ppo_epochs", 5)
        self.minibatch = kwargs.get("minibatch_size", 1000)
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)

        # Lagrangian
        self.cost_limit = float(kwargs.get("cost_limit", 19.0))
        self.lagrangian_lr = float(kwargs.get("lagrangian_lr", 0.005))
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
        if batch is None: return

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

        step_cost_mean = costs_b.mean()
        estimated_ep_cost = step_cost_mean * self.episode_length
        cost_violation = estimated_ep_cost - self.cost_limit
        self.lambda_global = max(0.0, self.lambda_global + self.lagrangian_lr * cost_violation)
        self.buffer.clear()


# ================================================================
# 3. 绘图函数 (绘制 SuccessRate 和 OffloadRate)
# ================================================================
def plot_combined_metrics_curves(results_list, save_dir):
    plt.rcParams.update({'font.size': 14})

    # --- 1. Success Rate Plot ---
    plt.figure(figsize=(12, 7))
    for item in results_list:
        df = item['df'];
        label = item['label'];
        color = item['color']
        if df is not None:
            plt.plot(df["Epoch"], df["SuccessRate"], color=color, linewidth=2.5, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Success Rate")
    plt.title("Task Success Rate Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_success_rate.png"), dpi=300)
    plt.close()

    # --- 2. Offload Rate Plot ---
    plt.figure(figsize=(12, 7))
    for item in results_list:
        df = item['df'];
        label = item['label'];
        color = item['color']
        if df is not None:
            plt.plot(df["Epoch"], df["OffloadRate"], color=color, linewidth=2.5, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Offloading Rate")
    plt.title("Offloading Rate Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_offload_rate.png"), dpi=300)
    plt.close()

    print(f"指标对比图已保存至: {save_dir}")


# ================================================================
# 4. 单次实验运行 (捕获 SuccessRate/OffloadRate)
# ================================================================
def run_single_experiment(cost_limit, fixed_lambda=None, custom_label=None, num_agents=4, epochs=500,
                          steps_per_epoch=30000, seed=0,
                          log_dir_root="./logs/mec_metrics_compare", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if fixed_lambda is None:
        exp_name = "Adaptive"
    else:
        exp_name = f"FixedLambda_{fixed_lambda}"

    if custom_label is None: custom_label = exp_name

    log_dir = os.path.join(log_dir_root, f"Limit{cost_limit}_{exp_name}_seed{seed}")
    os.makedirs(log_dir, exist_ok=True)

    try:
        from safepo.utils import EpochLogger
        logger = EpochLogger(log_dir=log_dir, seed=str(seed), output_fname="progress.csv", use_tensorboard=False)
    except ImportError:
        logger = SimpleLogger(log_dir=log_dir, seed=str(seed), output_fname="progress.csv", use_tensorboard=False)

    np.random.seed(seed);
    torch.manual_seed(seed)
    from safepo.common.env import make_ma_mec_env
    env = make_ma_mec_env(n_agents=num_agents)

    obs_example = env.reset()
    if isinstance(obs_example, (list, tuple)): obs_example = np.array(obs_example)
    obs_dim = int(obs_example.shape[1]);
    act_dim = int(env.action_size)

    algo = MAPPO_Lagrangian(
        state_dim=obs_dim,
        action_dim=act_dim,
        n_agents=num_agents,
        logger=logger,
        device=device,
        episode_length=100,
        cost_limit=float(cost_limit),
    )

    if fixed_lambda is not None:
        algo.lambda_global = float(fixed_lambda)
        algo.lagrangian_lr = 0.0

    # 数据容器
    epoch_data = {
        "Epoch": [],
        "SuccessRate": [],
        "OffloadRate": []
    }

    for epoch in range(epochs):
        if fixed_lambda is None:
            algo.lagrangian_lr = get_current_lagrangian_lr(epoch, epochs)

        o = env.reset()
        ep_len = 0;
        steps = 0

        # 临时列表，用于计算本 Epoch 的平均值
        ep_success_list = []
        ep_offload_list = []

        while steps < steps_per_epoch:
            a = algo.select_action(o)
            o2, r, done, truncated, info = env.step(a)
            algo.store(o, a, np.array(r, dtype=np.float32), done, info)

            # === 【关键】从 Info 中获取指标 ===
            # 注意：info['is_success'] 是一个列表（每个 agent 一个值）
            # 我们取所有 agent 的平均值作为这一步的系统性能

            # 安全获取：如果没有修改 env，为了不报错，给默认值 0
            s_rate = np.mean(info.get("is_success", [0.0]))
            o_rate = np.mean(info.get("is_offloaded", [0.0]))

            ep_success_list.append(s_rate)
            ep_offload_list.append(o_rate)

            ep_len += 1;
            steps += 1
            o = o2

            if done or ep_len >= 100:
                o = env.reset();
                ep_len = 0

        # 更新策略
        algo.update()

        # 计算本 Epoch 平均值
        avg_success = np.mean(ep_success_list) if ep_success_list else 0.0
        avg_offload = np.mean(ep_offload_list) if ep_offload_list else 0.0

        # 记录
        epoch_data["Epoch"].append(epoch)
        epoch_data["SuccessRate"].append(avg_success)
        epoch_data["OffloadRate"].append(avg_offload)

        print(f"{custom_label} | Ep {epoch} | Success {avg_success:.2f} | Offload {avg_offload:.2f}")

    if hasattr(logger, 'close'): logger.close()

    df = pd.DataFrame(epoch_data)
    return df


# ================================================================
# 5. 主控入口
# ================================================================
def run_comparison_experiment(args):
    scenarios = [
        {"label": "Adaptive Method", "fixed_val": None, "color": "tab:red"},
        {"label": "Fixed $\lambda=100$", "fixed_val": 5.0, "color": "tab:green"},
        {"label": "Fixed $\lambda=50$", "fixed_val": 4.0, "color": "tab:purple"},
        {"label": "Fixed $\lambda=1$", "fixed_val": 0.0, "color": "tab:blue"},
    ]

    TARGET_LIMIT = args.limit
    results_list = []

    print(f"\n>>> 启动指标对比实验 (Limit={TARGET_LIMIT}, Steps={args.steps})")
    print("注意：请确保 mec_env.py 已修改以返回 'is_success' 和 'is_offloaded'！")

    for sc in scenarios:
        print(f"\n=== Running Scenario: {sc['label']} ===")
        df = run_single_experiment(
            cost_limit=TARGET_LIMIT,
            fixed_lambda=sc["fixed_val"],
            custom_label=sc["label"],
            num_agents=args.agents,
            epochs=args.epochs,
            steps_per_epoch=args.steps,
            seed=args.seed
        )
        results_list.append({"df": df, "label": sc["label"], "color": sc["color"]})

    print("\n>>> 生成对比图...")
    save_dir = "./logs/mec_metrics_compare/combined_plots"
    os.makedirs(save_dir, exist_ok=True)

    plot_combined_metrics_curves(results_list, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=float, default=30.0)
    args = parser.parse_args()

    run_comparison_experiment(args)