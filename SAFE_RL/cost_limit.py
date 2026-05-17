import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from torch.utils.tensorboard import SummaryWriter


# ================================================================
# 简单的 Logger (如果你没有 safepo.utils)
# ================================================================
class SimpleLogger:
    def __init__(self, log_dir, seed, output_fname, use_tensorboard):
        self.log_dir = log_dir
        self.output_file = open(os.path.join(log_dir, output_fname), 'w')

    def log(self, msg): print(msg)

    def log_tabular(self, k, v): pass

    def dump_tabular(self): pass

    def close(self): self.output_file.close()


# ================================================================
# 1. LR 调度计算函数
# ================================================================
def get_current_lagrangian_lr(epoch, total_epochs):
    """
    策略:
    - 0 ~ 200 epoch: 保持高 LR (0.05)
    - 200 ~ 500 epoch: 线性衰减到低 LR (0.02)
    """
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


# ================================================================
# 2. 单次实验运行函数
# ================================================================
def run_single_experiment(cost_limit, num_agents=4, epochs=500, steps_per_epoch=30000, seed=0,
                          log_dir_root="./logs/mec_mappolag", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = os.path.join(log_dir_root, f"agents{num_agents}_limit{cost_limit}_seed{seed}")
    os.makedirs(log_dir, exist_ok=True)

    try:
        from safepo.utils import EpochLogger
        logger = EpochLogger(log_dir=log_dir, seed=str(seed), output_fname="progress.csv", use_tensorboard=False)
    except ImportError:
        logger = SimpleLogger(log_dir=log_dir, seed=str(seed), output_fname="progress.csv", use_tensorboard=False)

    tb = SummaryWriter(log_dir)

    np.random.seed(seed);
    torch.manual_seed(seed)

    # 假设 make_ma_mec_env 已经导入
    # from safepo.mec_env import make_ma_mec_env
    env = make_ma_mec_env(n_agents=num_agents)

    obs_example = env.reset()
    if isinstance(obs_example, (list, tuple)): obs_example = np.array(obs_example)
    obs_dim = int(obs_example.shape[1]);
    act_dim = int(env.action_size)

    logger.log(f"--- 开始实验: Limit={cost_limit} | Epochs={epochs} (LR Scheduled) ---")

    algo = MAPPO_Lagrangian(
        state_dim=obs_dim,
        action_dim=act_dim,
        n_agents=num_agents,
        logger=logger,
        device=device,
        episode_length=100,

        # PPO 参数
        hidden_size=128,
        actor_lr=0.001,
        critic_lr=0.0005,
        clip_param=0.15,
        ppo_epochs=4,
        minibatch_size=512,
        gamma=0.98,
        gae_lambda=0.95,
        entropy_coef=0.03,
        max_grad_norm=0.5,

        # Lagrangian 参数
        cost_limit=float(cost_limit),
        lagrangian_lr=0.0002,  # 初始值，会被调度器覆盖
    )

    epochs_logged = []
    epret_list = []
    epcost_list = []
    lambda_list = []
    lr_list = []

    for epoch in range(epochs):
        # 应用 LR 调度
        current_lr = get_current_lagrangian_lr(epoch, epochs)
        algo.lagrangian_lr = current_lr

        o = env.reset()
        ep_ret = 0.0;
        ep_cost = 0.0;
        ep_len = 0
        episode_rets = [];
        episode_costs = []
        steps = 0

        while steps < steps_per_epoch:
            a = algo.select_action(o)
            o2, r, done, truncated, info = env.step(a)

            r_arr = np.array(r, dtype=np.float32)
            c_val = np.mean(info.get("cost", 0.0))

            ep_ret += r_arr.sum()
            ep_cost += c_val
            ep_len += 1
            steps += 1

            force_done = (ep_len == steps_per_epoch)
            algo.store(o, a, r_arr, done, info)

            o = o2

            if done or ep_len >= 100:
                episode_rets.append(ep_ret)
                episode_costs.append(ep_cost)
                ep_ret = 0.0;
                ep_cost = 0.0;
                ep_len = 0
                o = env.reset()

        algo.update()

        mean_ret = float(np.mean(episode_rets)) if episode_rets else 0.0
        mean_cost = float(np.mean(episode_costs)) if episode_costs else 0.0

        epochs_logged.append(epoch)
        epret_list.append(mean_ret)
        epcost_list.append(mean_cost)
        lambda_list.append(algo.lambda_global)
        lr_list.append(current_lr)

        print(
            f"Limit {cost_limit} | Ep {epoch} | Ret: {mean_ret:.1f} | Cost: {mean_cost:.1f} | Lam: {algo.lambda_global:.3f} | LR: {current_lr:.4f}")

        if tb is not None:
            tb.add_scalar("Metrics/EpRet", mean_ret, epoch)
            tb.add_scalar("Metrics/EpCost", mean_cost, epoch)
            tb.add_scalar("Misc/lambda", algo.lambda_global, epoch)
            tb.add_scalar("Misc/LagrangianLR", current_lr, epoch)

    if tb is not None: tb.close()
    if hasattr(logger, 'close'): logger.close()

    df = pd.DataFrame({
        "Epoch": epochs_logged,
        "EpRet": epret_list,
        "EpCost": epcost_list,
        "Lambda": lambda_list,
        "LagrangianLR": lr_list
    })

    SMOOTH_WIN = 10
    df["EpCost_smooth"] = df["EpCost"].rolling(SMOOTH_WIN, min_periods=1).mean()
    df["EpRet_smooth"] = df["EpRet"].rolling(SMOOTH_WIN, min_periods=1).mean()

    df.to_csv(os.path.join(log_dir, "training_curves.csv"), index=False)

    return df


# ================================================================
# 3. 批处理与绘图主逻辑 (新增 Reward 绘图)
# ================================================================
def run_batch_and_plot(args):
    target_limits = [16, 18, 22, 26, 30]
    all_results = {}

    print(f"\n>>> 启动批处理 (LR Scheduler Enabled), Limits: {target_limits}")

    for limit in target_limits:
        print(f"\n==============================================")
        print(f"Running Experiment: Limit = {limit}")
        print(f"==============================================")

        df = run_single_experiment(
            cost_limit=limit,
            num_agents=args.agents,
            epochs=args.epochs,
            steps_per_epoch=args.steps,
            seed=args.seed
        )
        all_results[limit] = df

    print("\n>>> 实验结束，正在生成三张对比图...")
    cmap = plt.get_cmap('tab10')
    save_dir = "./logs/mec_mappolag/comparison_plots_scheduled"
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 14})

    # === 图 1: Cost 对比 ===
    plt.figure(figsize=(10, 6))
    for i, limit in enumerate(target_limits):
        df = all_results[limit]
        color = cmap(i)
        plt.plot(df["Epoch"], df["EpCost_smooth"], label=f'Limit = {limit}', color=color, linewidth=2.5)
        plt.axhline(y=limit, color=color, linestyle='--', alpha=0.6, linewidth=1.5)

    plt.xlabel("Epoch")
    plt.ylabel("Episode Cost")
    plt.title("Cost Evolution (With LR Decay)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_cost_comparison.png"), dpi=300)
    print(f"Cost 图已保存至: {os.path.join(save_dir, 'combined_cost_comparison.png')}")
    plt.close()

    # === 图 2: Lambda 对比 ===
    plt.figure(figsize=(10, 6))
    for i, limit in enumerate(target_limits):
        df = all_results[limit]
        color = cmap(i)
        plt.plot(df["Epoch"], df["Lambda"], label=f'Limit = {limit}', color=color, linewidth=2.5)

    plt.xlabel("Epoch")
    plt.ylabel("Lagrangian Multiplier ($\lambda$)")
    plt.title("Lambda Evolution (With LR Decay)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_lambda_comparison.png"), dpi=300)
    print(f"Lambda 图已保存至: {os.path.join(save_dir, 'combined_lambda_comparison.png')}")
    plt.close()

    # === 图 3: Reward 对比 (新增) ===
    plt.figure(figsize=(10, 6))
    for i, limit in enumerate(target_limits):
        df = all_results[limit]
        color = cmap(i)
        # 使用平滑后的 Reward 曲线，看起来更清晰
        plt.plot(df["Epoch"], df["EpRet_smooth"], label=f'Limit = {limit}', color=color, linewidth=2.5)

    plt.xlabel("Epoch")
    plt.ylabel("Episode Reward")
    plt.title("Reward Evolution under Different Limits")
    plt.grid(True, alpha=0.3)
    # Reward 通常图例放在右下或最佳位置
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_reward_comparison.png"), dpi=300)
    print(f"Reward 图已保存至: {os.path.join(save_dir, 'combined_reward_comparison.png')}")
    plt.close()


# ================================================================
# 主程序入口
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=4)
    # 默认 500 以匹配 LR 调度
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", action="store_true", help="是否运行批处理对比实验")
    parser.add_argument("--limit", type=float, default=18.0, help="单次运行时使用的 limit")

    args = parser.parse_args()

    if args.batch:
        run_batch_and_plot(args)
    else:
        print(f">>> 运行单次实验 Limit = {args.limit}")
        run_single_experiment(
            cost_limit=args.limit,
            num_agents=args.agents,
            epochs=args.epochs,
            steps_per_epoch=args.steps,
            seed=args.seed
        )
