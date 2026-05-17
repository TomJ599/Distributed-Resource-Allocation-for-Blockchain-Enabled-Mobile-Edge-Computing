import torch
import torch as th
import torch.nn.functional as F
from torch.optim import Adam, RMSprop
import numpy as np
from copy import deepcopy
import logging
from Model import ActorNetwork, CriticNetwork
from utils import to_tensor_var
from Memory import ReplayMemory

import matplotlib.pyplot as plt

EVAL_EPISODES = 1

class MAPPO(object):
    """
    An agent learned with PPO using Advantage Actor-Critic framework
    - Actor takes state as input
    - Critic takes both state and action as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy
    - adam seems better than rmsprop for ppo
    """

    def __init__(self, env, state_dim, action_dim, n_agents, action_lower_bound, action_higher_bound,
                 noise=0, tau=300,
                 memory_capacity=1000, max_steps=100,
                 roll_out_n_steps=20, target_tau=0.01,
                 target_update_steps=10, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=1.,
                 actor_output_act=F.softmax, critic_loss="mse",
                 actor_lr=0.01, critic_lr=0.01,
                 optimizer_type="adam", entropy_reg=0.2,
                 max_grad_norm=1.0, batch_size=32, episodes_before_train=0,
                 use_cuda=False, use_federated_aggregation=True, max_episodes=2000):
        

        self.env = env
        self.n_agents = env.n_agents  # 使用 env.n_agents
        self.action_space_sizes = {f"agent_{i}": 26 for i in range(self.n_agents)}  # 每个智能体动作空间大小为 4
        self.action_dim = 4  # 动作索引维度为 1
        self.state_dim = env.state_size  # 状态维度为 17
        self.actors = [ActorNetwork(state_dim=self.state_dim) for _ in range(self.n_agents)]  # 默认 action_dim=27
        self.actors_target = [ActorNetwork(state_dim=self.state_dim) for _ in range(self.n_agents)]
        self.env_state = self.env.reset()
        self.n_episodes = 0
        self.episode_done = False  # 初始化 episode_done
        self.n_steps = 0
        self.max_steps = max_steps
        self.n_agents = n_agents

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale

        self.action_lower_bound = action_lower_bound
        self.action_higher_bound = action_higher_bound

        self.memory = ReplayMemory(memory_capacity)

        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.noise = noise
        self.tau = tau
        self.max_episodes = max_episodes
        self.use_federated_aggregation = use_federated_aggregation
        self.use_cuda = use_cuda and th.cuda.is_available()

        self.roll_out_n_steps = roll_out_n_steps
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.clip_param = clip_param

        critic_state_dim = self.n_agents * self.state_dim
        critic_action_dim = self.n_agents * self.action_dim
        print(
            f"CriticNetwork: state_dim={critic_state_dim}, action_dim={critic_action_dim}, total_dim={critic_state_dim + critic_action_dim}"
        )
        self.critics = []
        for i in range(self.n_agents):
            critic = CriticNetwork(critic_state_dim, critic_action_dim, 1)
            assert critic.fc1.in_features == critic_state_dim + critic_action_dim, \
                f"Critic {i} fc1 in_features={critic.fc1.in_features}, expected={critic_state_dim + critic_action_dim}"
            self.critics.append(critic)
            # print(f"Created Critic {i} with fc1 in_features={critic.fc1.in_features}")
        # self.critics = [CriticNetwork(critic_state_dim, critic_action_dim, 1)] * self.n_agents
        # to ensure target network and learning network has the same weights
        # self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        if optimizer_type == "adam":
            self.actors_optimizer = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [Adam(c.parameters(), lr=self.critic_lr) for c in self.critics]
        elif optimizer_type == "rmsprop":
            self.actors_optimizer = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.critics]

        if self.use_cuda:
            for a in self.actors:
                a.cuda()
            for c in self.critics:
                c.cuda()
        self.eval_rewards = []
        self.mean_rewards = []
        self.episodes = []
        self.eval_phi = []
        self.mean_phi = []
        self.eval_losses = []  # 每步的平均 loss
        self.mean_losses = []  # 每 EVAL_EPISODES 的平均 loss
        self.eval_pbft_rewards = []  # 初始化
        self.eval_reward_vt = []
        self.mean_reward_vt = []
        self.eval_pbft_reward = []
        self.mean_pbft_reward = []
        self.eval_energy_pbft = []
        self.mean_energy_pbft = []
        self.eval_time_penalty = []
        self.mean_time_penalty = []
        self.critic_losses = []

        # 初始化 Actor 网络权重
        for actor in self.actors:
            for param in actor.parameters():
                if param.dim() >= 2:  # 仅对权重（2D 张量）应用 Xavier 初始化
                    torch.nn.init.xavier_uniform_(param)
                else:  # 对偏置（1D 张量）应用零初始化
                    torch.nn.init.zeros_(param)

        # 初始化 Critic 网络权重
        for critic in self.critics:
            for param in critic.parameters():
                if param.dim() >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)

    # agent interact with the environment to collect experience 每十步收集一次
    # MAPPO.py: interact() 约 104-165 行
    async def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
            print(f"Reset environment due to max_steps={self.max_steps} reached")

        states, actions, rewards = [], [], []
        total_reward = 0.0
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = self.choose_action(self.env_state)
            print(f"Interact: step={i}, action=\n{action}")
            assert np.all(action[:, 1] >= 0) and np.all(action[:, 1] <= 3), f"Invalid A_channel: {action[:, 1]}"
            next_state, reward, done, _, *_ = await self.env.step(action)
            reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
            actions.append(action.copy())
            rewards.append(reward)
            total_reward += np.mean(reward)
            self.env_state = next_state
            self.n_steps += 1  # 移动到循环内，确保每步增加
            logging.info(f"Interact: step={i + 1}, reward={reward}, total_reward={total_reward}")
            # if done:
            #     print(f"Episode {self.n_episodes + 1} done, resetting environment")
            #     self.env_state = self.env.reset()
            #     break

        final_r = [0.0] * self.n_agents
        if len(rewards) >= self.roll_out_n_steps:  # 每 roll_out_n_steps 增加 episode
            self.n_episodes += 1
            logging.info(f"Episode {self.n_episodes} completed, total steps={self.n_steps}")

        rewards = np.array(rewards)
        for agent_id in range(self.n_agents):
            rewards[:, agent_id] = self._discount_reward(rewards[:, agent_id], final_r[agent_id])
        rewards = rewards.tolist()

        if self.n_episodes <= self.max_episodes:
            mean_reward = total_reward / len(rewards) if rewards else 0.0
            self.eval_rewards.append(mean_reward)
            logging.info(
                f"Episode {self.n_episodes}: mean_reward={mean_reward}, eval_rewards_len={len(self.eval_rewards)}")
            if self.n_episodes % EVAL_EPISODES == 0:
                mean_reward = np.mean(np.array(self.eval_rewards)) if self.eval_rewards else 0.0
                self.mean_rewards.append(mean_reward)
                self.episodes.append(self.n_episodes)
                logging.info(f"Episode {self.n_episodes}: Average Reward={mean_reward}")
                self.eval_rewards = []
        self.memory.push(states, actions, rewards)

        if self.n_episodes >= self.max_episodes:
            logging.info(f"Reached max_episodes={self.max_episodes}, stopping interaction")

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train or len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, 4)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)
        whole_states_var = states_var.view(-1, self.n_agents * self.state_dim)
        whole_actions_var = actions_var.view(-1, self.n_agents * 4)
        agent_losses = []
        for agent_id in range(self.n_agents):
            # critic值计算
            with torch.no_grad():    # 避免梯度跟踪
                values_detached = self.critics[agent_id](whole_states_var, whole_actions_var)
            values = self.critics[agent_id](whole_states_var, whole_actions_var)
            advantages = rewards_var[:, agent_id, :] - values_detached   # 优势值
            # actor损失
            self.actors_optimizer[agent_id].zero_grad()    # 清空优化器梯度
            # 计算当前策略的概率分布
            current_log_probs = self.actors[agent_id](states_var[:, agent_id, :])
            actions = actions_var[:, agent_id, :]
            decision_probs = current_log_probs[:, :2]
            channel_probs = current_log_probs[:, 2:6]
            res_probs = current_log_probs[:, 6:16]
            power_probs = current_log_probs[:, 16:26]
            # 将动作分量转换为长整型索引
            decision_indices = actions_var[:, agent_id, 0].long()
            channel_indices = torch.clamp(actions_var[:, agent_id, 1].long(), 0, 3)
            res_indices = actions_var[:, agent_id, 2].long()
            power_indices = actions_var[:, agent_id, 3].long()
            # 限制索引范围
            channel_indices = torch.clamp(channel_indices, min=0, max=3)
            res_indices = torch.clamp(res_indices, min=0, max=9)
            power_indices = torch.clamp(power_indices, min=0, max=9)
            # 选择动作概率
            selected_decision_probs = decision_probs.gather(1, decision_indices.unsqueeze(1))
            selected_channel_probs = channel_probs.gather(1, channel_indices.unsqueeze(1))
            selected_res_probs = res_probs.gather(1, res_indices.unsqueeze(1))
            selected_power_probs = power_probs.gather(1, power_indices.unsqueeze(1))
            # 计算当前策略的对数概率
            current_log_probs = (torch.log(selected_decision_probs + 1e-10) +
                                 torch.log(selected_channel_probs + 1e-10) +
                                 torch.log(selected_res_probs + 1e-10) +
                                 torch.log(selected_power_probs + 1e-10)).sum(1)
            # 计算目标策略的对数概率
            old_log_probs = self.actors_target[agent_id](states_var[:, agent_id, :]).detach()
            old_decision_probs = old_log_probs[:, :2]
            old_channel_probs = old_log_probs[:, 2:6]
            old_res_probs = old_log_probs[:, 6:16]
            old_power_probs = old_log_probs[:, 16:26]
            old_selected_decision_probs = old_decision_probs.gather(1, decision_indices.unsqueeze(1))
            old_selected_channel_probs = old_channel_probs.gather(1, channel_indices.unsqueeze(1))
            old_selected_res_probs = old_res_probs.gather(1, res_indices.unsqueeze(1))
            old_selected_power_probs = old_power_probs.gather(1, power_indices.unsqueeze(1))
            old_log_probs = (torch.log(old_selected_decision_probs + 1e-10) +
                             torch.log(old_selected_channel_probs + 1e-10) +
                             torch.log(old_selected_res_probs + 1e-10) +
                             torch.log(old_selected_power_probs + 1e-10)).sum(1)
            # 计算当前策略与旧策略的概率比率
            ratio = torch.exp(current_log_probs - old_log_probs)
            # PPO损失计算
            surr1 = ratio * advantages   # 原始概率比率乘以优势
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages  # 剪切比率乘以优势
            actor_loss = -torch.mean(torch.min(surr1, surr2))   # 取两者最小值，计算平均负损失作为 Actor 损失
            # 熵正则化
            entropy = -torch.sum(current_log_probs * torch.exp(current_log_probs), dim=-1).mean()   # 计算当前策略的熵
            actor_loss -= self.entropy_reg * entropy   # 将熵项加到actor损失
            # Actor 梯度更新
            actor_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actors_optimizer[agent_id].step()
            # Critic 损失计算
            self.critics_optimizer[agent_id].zero_grad()
            critic_loss = 0.5 * (values - rewards_var[:, agent_id, :]).pow(2).mean()
            # Critic 梯度更新
            critic_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), self.max_grad_norm)
            self.critics_optimizer[agent_id].step()

            agent_losses.append(critic_loss.item())     # 存储critic损失
        if self.n_episodes % EVAL_EPISODES == 0:
            mean_critic_loss = np.mean(agent_losses) if agent_losses else 0.0
            self.critic_losses.append(mean_critic_loss)
            logging.info(
                f"Episode {self.n_episodes}: Recorded critic_loss={mean_critic_loss}, critic_losses_len={len(self.critic_losses)}")
        if self.use_federated_aggregation and self.n_episodes % 5 == 0 and self.n_episodes > 0:
            losses = th.tensor(agent_losses)
            rewards = rewards_var.mean(dim=0).squeeze(-1)
            S_res_list = states_var[:, :, 7].mean(dim=0)
            S_com_list = states_var[:, :, 8].mean(dim=0)
            S_size_list = states_var[:, :, 3].mean(dim=0)
            self._federated_aggregate(losses, rewards, S_res_list, S_com_list, S_size_list)
        if self.n_steps % 1 == 0 and self.n_steps > 0:
            for agent_id in range(self.n_agents):
                self._soft_update_target(self.actors_target[agent_id], self.actors[agent_id])
                self._soft_update_target(self.critics_target[agent_id], self.critics[agent_id])
        if self.episode_done and ((self.n_episodes + 1) % EVAL_EPISODES == 0):
            mean_reward = np.mean(np.array(self.eval_rewards)) if self.eval_rewards else 0.0
            self.mean_rewards.append(mean_reward)
            self.episodes.append(self.n_episodes + 1)
            print(f"Episode: {self.n_episodes + 1}, Average Reward: {mean_reward}")
            self.eval_rewards = []

    def _federated_aggregate(self, losses, rewards, S_res, S_com, S_size):
        """使用注意力机制加权的联邦参数聚合"""
        # 构造 Q 和 K
        L = losses  # [n_agents]
        R = rewards  # [n_agents]
        com = S_res * S_com  # [n_agents]
        B = S_size  # [n_agents]

        # 标准化 L, R, com, B 用于 Q
        L_norm = (L - th.mean(L)) / (th.std(L) + 1e-6)
        R_norm = (R - th.mean(R)) / (th.std(R) + 1e-6)
        com_norm = (com - th.mean(com)) / (th.std(com) + 1e-6)
        B_norm = (B - th.mean(B)) / (th.std(B) + 1e-6)
        Q = th.stack([L_norm, R_norm, com_norm, B_norm], dim=1)  # [n_agents, 4]

        # 标准化 Lavg, Ravg, comavg, Bavg 用于 K
        Lavg = th.mean(L).unsqueeze(0)
        Ravg = th.mean(R).unsqueeze(0)
        comavg = th.mean(com).unsqueeze(0)
        Bavg = th.mean(B).unsqueeze(0)
        K_values = th.cat([Lavg, Ravg, comavg, Bavg], dim=0)  # [4]
        K_norm = (K_values - th.mean(K_values)) / (th.std(K_values) + 1e-6)  # 标准化 K
        K = K_norm.unsqueeze(0)  # [1, 4]

        # Attention 权重计算: softmax(QK^T / sqrt(dk))
        dk = Q.size(1)
        scores = (Q @ K.T) / (dk ** 0.5) + 1e-6  # [n_agents, 1]
        weights = th.softmax(scores, dim=0)  # 最小权重 0.1
        logging.info(f"Q={Q}, K={K}, scores={scores}, weights={weights}")
        # 聚合 Actor 参数
        actor_params = []
        for agent_id in range(self.n_agents):
            params = [param.data.clone() for param in self.actors[agent_id].parameters()]
            actor_params.append(params)

        avg_actor_params = []
        for i in range(len(actor_params[0])):  # 对每一层
            stacked = th.stack([agent_params[i] for agent_params in actor_params], dim=0)  # [n_agents, ...]
            weighted = th.sum(weights.view(-1, 1) * stacked.view(self.n_agents, -1), dim=0)
            avg_param = weighted.view_as(stacked[0])
            avg_actor_params.append(avg_param)

        for agent_id in range(self.n_agents):
            for param, avg_param in zip(self.actors[agent_id].parameters(), avg_actor_params):
                param.data.copy_(avg_param)

        # 聚合 Critic 参数（同样方式）
        critic_params = []
        for agent_id in range(self.n_agents):
            params = [param.data.clone() for param in self.critics[agent_id].parameters()]
            critic_params.append(params)

        avg_critic_params = []
        for i in range(len(critic_params[0])):
            stacked = th.stack([agent_params[i] for agent_params in critic_params], dim=0)  # [n_agents, ...]
            weighted = th.sum(weights.view(-1, 1) * stacked.view(self.n_agents, -1), dim=0)
            avg_param = weighted.view_as(stacked[0])
            avg_critic_params.append(avg_param)

        for agent_id in range(self.n_agents):
            for param, avg_param in zip(self.critics[agent_id].parameters(), avg_critic_params):
                param.data.copy_(avg_param)

    def _soft_update_target(self, target, source):
        """目标网络软更新"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1.0 - self.target_tau) * target_param.data + self.target_tau * source_param.data
            )

    def choose_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        action = np.zeros((self.n_agents, 4), dtype=np.float64)
        for agent_id in range(self.n_agents):
            with torch.no_grad():
                action_probs = self.actors[agent_id](state_var[:, agent_id, :]).cpu().numpy().squeeze()
            print(f"Agent {agent_id}: action_probs_shape={action_probs.shape}, action_probs={action_probs}")
            assert len(action_probs) == 26, f"Expected action_probs length=26, got {len(action_probs)}"

            # 初始化默认概率分布
            decision_probs = np.ones(2) / 2
            channel_probs = np.ones(4) / 4
            res_probs = np.ones(10) / 10
            power_probs = np.ones(10) / 10

            try:
                decision_logits = action_probs[:2]
                channel_logits = action_probs[2:6]
                res_logits = action_probs[6:16]
                power_logits = action_probs[16:26]

                # 添加探索噪声
                exploration_noise = np.random.normal(0, 0.01, size=res_logits.shape)
                res_logits = res_logits + exploration_noise
                exploration_noise = np.random.normal(0, 0.01, size=power_logits.shape)
                power_logits = power_logits + exploration_noise

                # 计算概率分布
                decision_probs = F.softmax(torch.tensor(decision_logits, dtype=torch.float32), dim=0).numpy()
                channel_probs = F.softmax(torch.tensor(channel_logits, dtype=torch.float32), dim=0).numpy()
                res_probs = F.softmax(torch.tensor(res_logits, dtype=torch.float32), dim=0).numpy()
                power_probs = F.softmax(torch.tensor(power_logits, dtype=torch.float32), dim=0).numpy()
            except Exception as e:
                print(f"Agent {agent_id}: Error computing probabilities: {e}")
                # 使用默认均匀分布
                decision_probs = np.ones(2) / 2
                channel_probs = np.ones(4) / 4
                res_probs = np.ones(10) / 10
                power_probs = np.ones(10) / 10

            # 记录概率以进行调试
            print(f"Agent {agent_id}: res_probs={res_probs}, power_probs={power_probs}")

            # 归一化处理
            decision_probs = decision_probs / (np.sum(decision_probs) + 1e-10)
            channel_probs = channel_probs / (np.sum(channel_probs) + 1e-10)
            res_probs = res_probs / (np.sum(res_probs) + 1e-10)
            power_probs = power_probs / (np.sum(power_probs) + 1e-10)

            decision_probs = np.clip(decision_probs * [0.2, 0.8], 0, 1)
            decision_probs /= np.sum(decision_probs) + 1e-10

            # 检查无效值
            if np.any(np.isnan(channel_probs)) or np.any(channel_probs < 0):
                print(f"Warning: Invalid channel_probs={channel_probs}, using uniform distribution")
                channel_probs = np.ones(4) / 4
            if np.any(np.isnan(power_probs)) or np.any(power_probs < 0):
                print(f"Warning: Invalid power_probs={power_probs}, using uniform distribution")
                power_probs = np.ones(10) / 10
            if np.any(np.isnan(res_probs)) or np.any(res_probs < 0):
                print(f"Warning: Invalid res_probs={res_probs}, using uniform distribution")
                res_probs = np.ones(10) / 10
            if np.any(np.isnan(decision_probs)) or np.any(decision_probs < 0):
                print(f"Warning: Invalid decision_probs={decision_probs}, using uniform distribution")
                decision_probs = np.ones(2) / 2

            action[agent_id, 0] = np.random.choice([0, 1], p=decision_probs)  # A_decision
            action[agent_id, 1] = np.random.choice(range(4), p=channel_probs)  # A_channel
            action[agent_id, 2] = np.random.choice(range(10), p=res_probs)  # A_res
            action[agent_id, 3] = np.random.choice(range(10), p=power_probs)  # A_power

            assert 0 <= action[agent_id, 1] <= 3, f"Invalid A_channel[{agent_id}]={action[agent_id, 1]}"
            assert 0 <= action[agent_id, 2] <= 9, f"Invalid A_res[{agent_id}]={action[agent_id, 2]}"
            assert 0 <= action[agent_id, 3] <= 9, f"Invalid A_power[{agent_id}]={action[agent_id, 3]}"

        print(f"Final action before return:\n{action}")
        return action

    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action_var = to_tensor_var([action], self.use_cuda)
        whole_state_var = state_var.view(-1, self.n_agents * self.state_dim)
        whole_action_var = action_var.view(-1, self.n_agents * self.action_dim)
        print(f"whole_state_var shape={whole_state_var.shape}, whole_action_var shape={whole_action_var.shape}")
        values = []
        for agent_id in range(self.n_agents):
            print(f"Critic {agent_id} fc1 weight shape={self.critics[agent_id].fc1.weight.shape}")
            value_var = self.critics[agent_id](whole_state_var, whole_action_var)
            if self.use_cuda:
                value = value_var.data.cpu().numpy()[0]
            else:
                value = value_var.data.numpy()[0]
            values.append(value)
        return values
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

