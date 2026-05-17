import logging
import time
import numpy as np
import asyncio
import queue
import hashlib
import json
from nacl.signing import SigningKey, VerifyKey
import pandas as pd

K_CHANNEL = 4   # 可用信道数量

MIN_SIZE = 4  # 任务大小范围（MB）   0.1-1
MAX_SIZE = 10

MIN_CYCLE = 0.4   # Gcycles   0.4-4
MAX_CYCLE = 4

MIN_DDL = 0.5  # 任务截止时间(单位)1000ms
MAX_DDL = 1     # 秒 s

MIN_RES = 0.1  # 计算资源的范围  GHz（cycles/s）  ?
MAX_RES = 1

MIN_POWER = 0.01
MAX_POWER = 0.1  # 最大发射功率 瓦特（W）  ？

MAX_GAIN = 2 * 10**(-5)  # 信道增益范围  ？
MIN_GAIN = 1 * 10**(-5)

V_L = 0.125  # 本地和MEC计算延时的权重
V_E = 0.13

THETA_L = 1/1600   # 计算延时的系数
THETA_E = 1/1700

K_ENERGY_LOCAL = 5e-27   #k = 0.8 * 10 ^(-27) * M * G^2#  ？
K_ENERGY_MEC = 4e-27     # 计算能耗的系数（5*10**（-27））

NOISE_VARIANCE = 10**(-10)  # 通信噪声方差 10**-10瓦特（W）？

CAPABILITY_E = 5   # 服务器的计算能力 GHz（cycles/s）

MIN_EPSILON = 0.56
MAX_EPSILON = 0.93

KSI = 0.5
LAMBDA = 0.5
ALPHA = 0.5   # 共识奖励系数
BETA = 0.1   # 奖励函数中的惩罚系数

# 状态变量的初始值
S_POWER = 20
S_GAIN = 8
S_SIZE = 10
S_CYCLE = 3
S_RESOLU = 1
S_RES = 0.5
S_COM = 0.6
S_DDL = 0.1
S_EPSILON = 0.86

# PBFT 常量
MESSAGE_ENERGY = 0.00005  # 每条消息的能耗（焦耳）广播  每个阶段都有
CONSENSUS_REWARD = 1  # 共识参与奖励
TIMEOUT = 5.0  # 共识轮超时（秒）
MESSAGE_DELAY = 0.0005  # 消息延迟（秒）
BLOCK_GEN_ENERGY = 0.05  # 每块生成能耗（焦耳） 公式  J/MB
BLOCK_COMP_ENERGY = 0.01  # 每条消息验证能耗（焦耳）公式
MU_PBFT = 0.4  # PBFT 奖励权重
DELAY_PENALTY = 0.05  # 未完成共识任务的延迟惩罚
PBFT_RES_PER_MSG = 0.0001  # 每条 PBFT 消息的资源需求（单位：GHz资源/消息）收到消息后节点处理消息 每个阶段都有

# ECC 函数（从 ecc.py）
# 生成消息的数字签名，用于确保消息的真实性和完整性。
# def generate_sign(chk_msg):   # chk_msg：需要签名的消息
#     signing_key = SigningKey.generate()  # 生成一个新的签名密钥
#     signed_msg = signing_key.sign(str(chk_msg).encode())   # 将消息转换为字符串并编码为字节，使用签名密钥生成签名
#     verify_key = signing_key.verify_key  # 获取对应的公钥，用于验证签名
#     public_key = verify_key.encode()  # 将公钥编码为字节
#     return signed_msg + b'split' + public_key  # 签名后的消息和公钥的字节串（中间用 b'split' 分隔）
#
# # 验证签名消息的真实性
# def generate_verify(self, signed_msg):
#     try:
#         verify_key = VerifyKey(signed_msg['public_key'])
#         message_bytes = verify_key.verify(signed_msg['message'], signed_msg['signature'])
#         return json.loads(message_bytes.decode('utf-8'))
#     except Exception as e:
#         print(f"Verification failed: {e}")
#         raise

# 对输入实体生成 SHA-256 哈希值，用于生成消息摘要
def hashing_function(entity):   # entity：需要哈希的实体
    h = hashlib.sha256()    # 创建 SHA-256 哈希对象
    h.update(str(entity).encode())  # 将实体转换为字符串并编码为字节，更新哈希对象
    return h.hexdigest()  # 返回哈希值的十六进制表示
class MecBCEnv(object):
    def __init__(self, n_agents, S_DDL=1, S_EPSILON=0.86, W_BANDWIDTH=20,    # MHz（兆赫兹）
        S_one_power=20, S_one_gamma=0.6, mode="normal"):   # S_one_power：默认功率 S_one_gamma：默认通信资源

        self.state_size = 17
        self.action_size = 4
        self.n_agents = n_agents

        # PBFT 初始化
        self.N_NODES = n_agents  # 节点数
        self.F_TOLERANCE = (self.N_NODES - 1) // 3  # 最大容错节点数
        self.primary_node = 0   # 初始主节点ID
        self.view_number = 0    # 初始视图编号
        self.sequence_number = 1   # 初始消息序列号
        self.faulty_nodes = np.zeros(self.n_agents, dtype=bool)        # 布尔数组，标记故障节点
        self.consensus_reached = np.zeros(self.n_agents, dtype=bool)   # 布尔数组，标记每个节点是否达成共识
        self.pre_prepare_msgs = np.zeros(self.n_agents, dtype=bool)    # 布尔数组，标记是否收到预准备消息
        self.prepare_msgs = np.zeros(self.n_agents, dtype=int)         # 整数数组，记录收到的准备消息数量
        self.commit_msgs = np.zeros(self.n_agents, dtype=int)          # 整数数组，记录收到的提交消息数量
        self.confirm_msgs = np.zeros(self.n_agents, dtype=int)         # 记录 CONFIRM 消息数量
        self.message_energy = np.zeros(self.n_agents)                  # 记录每个节点的消息处理能耗
        self.message_count = np.zeros(self.n_agents, dtype=int)        # 记录每个节点发送/接收的消息数量
        self.message_queues = [queue.Queue() for _ in range(self.n_agents)]   # 为每个节点初始化一个队列，用于存储待处理消息
        self.signing_keys = [SigningKey.generate() for _ in range(self.n_agents)]   # 为每个节点生成签名密钥
        self.public_keys = [key.verify_key.encode() for key in self.signing_keys]   # 保存每个节点的公钥
        for i, (sk, pk) in enumerate(zip(self.signing_keys, self.public_keys)):
            pk_hex = pk.encode().hex() if isinstance(pk, VerifyKey) else pk.hex()
            # print(f"节点 {i}: 签名密钥类型={type(sk)}, 公钥类型={type(pk)}, 公钥={pk_hex}")
        self.pbft_progress = np.zeros(self.n_agents)  # PBFT 阶段进度（0-1）
        self.pending_messages = [[] for _ in range(self.n_agents)]  # 未处理消息列表
        self.pending_message_count = np.zeros(self.n_agents, dtype=int)  # 未处理消息数
        self.pbft_reward = np.zeros(self.n_agents)   # 记录每个节点的 PBFT 共识奖励
        self.Energy_pbft = np.zeros(self.n_agents)   # 记录 PBFT 共识的能耗

        self.Time_penalty = np.zeros(self.n_agents)  # 记录时间惩罚
        self.Energy_local = np.zeros(self.n_agents)  # 记录本地计算的能耗
        self.Energy_n = np.zeros(self.n_agents)      # 记录实际总能耗

        # MEC 初始化 保存传入默认参数
        self.S_DDL = S_DDL
        self.S_EPSILON = S_EPSILON
        self.W_BANDWIDTH = W_BANDWIDTH
        self.S_one_power = S_one_power
        self.S_one_gamma = S_one_gamma

        # state
        self.S_channel = np.zeros(self.n_agents)  # 信道分配
        self.S_power = np.zeros(self.n_agents)    # 发射功率
        self.S_gain = np.zeros(self.n_agents)     # 信道增益
        self.S_size = np.zeros(self.n_agents)     # 任务数据大小
        self.S_cycle = np.zeros(self.n_agents)    # 任务计算周期
        self.S_resolu = np.zeros(self.n_agents)   # 任务分辨率
        self.S_ddl = np.zeros(self.n_agents)      # 任务截止时间
        self.S_res = np.zeros(self.n_agents)      # 计算资源
        self.S_com = np.zeros(self.n_agents)      # 通信资源
        self.S_epsilon = np.zeros(self.n_agents)  # 任务性能阈值
        self.mode = mode
        # 连续动作边界
        # self.action_lower_bound = [0, 0, 0.01, MIN_RES, MIN_COM, 1]
        # self.action_higher_bound = [1, K_CHANNEL, 0.99, MAX_RES, MAX_COM, MAX_POWER]
        # 定义离散动作空间
        self.num_res_levels = 10    # 定义计算资源和发射功率的离散级别数
        self.num_power_levels = 10
        self.power_levels = np.linspace(MIN_POWER, MAX_POWER, self.num_power_levels, dtype=np.float64)
        self.res_levels = np.linspace(MIN_RES, MAX_RES, self.num_res_levels, dtype=np.float64)
        self.action_lower_bound = [0, 0, MIN_RES, MIN_POWER]
        self.action_higher_bound = [1, K_CHANNEL, MAX_RES, MAX_POWER]
          

        self.epoch = 0    # 时间步

    # 重置
    def reset(self):
        self.S_channel = np.zeros(self.n_agents)
        self.S_power = np.zeros(self.n_agents)
        self.S_gain = np.full(self.n_agents, (MIN_GAIN + MAX_GAIN) / 2)
        self.S_size = np.clip(np.random.normal(S_SIZE, 0.02, self.n_agents), MIN_SIZE, MAX_SIZE)
        self.S_cycle = np.clip(np.random.normal(S_CYCLE, 0.02, self.n_agents), MIN_CYCLE, MAX_CYCLE)
        self.S_resolu = np.full(self.n_agents, S_RESOLU)
        self.S_ddl = np.clip(np.random.normal(self.S_DDL, 0.02, self.n_agents), MIN_DDL, MAX_DDL * 1.5)
        self.S_res = np.random.choice(self.res_levels, self.n_agents)
        self.S_com = np.full(self.n_agents, S_COM)
        self.S_epsilon = np.random.uniform(0.5, 1.0, self.n_agents)
        self.faulty_nodes = np.zeros(self.n_agents, dtype=bool)
        self.message_queues = [queue.Queue() for _ in range(self.n_agents)]
        self.pending_messages = [[] for _ in range(self.n_agents)]
        self.pending_message_count = np.zeros(self.n_agents, dtype=int)
        self.message_count = np.zeros(self.n_agents, dtype=int)
        self.message_energy = np.zeros(self.n_agents)
        self.pre_prepare_msgs = np.zeros(self.n_agents, dtype=bool)
        self.prepare_msgs = np.zeros(self.n_agents, dtype=int)
        self.commit_msgs = np.zeros(self.n_agents, dtype=int)
        self.confirm_msgs = np.zeros(self.n_agents, dtype=int)
        self.consensus_reached = np.zeros(self.n_agents, dtype=bool)
        self.pbft_progress = np.zeros(self.n_agents)
        self.view_number = 0
        self.sequence_number = 0
        self.primary_node = 0
        self.epoch = 0

        State_ = [[self.S_channel[n], self.S_power[n], self.S_gain[n], self.S_size[n], self.S_cycle[n],
                   self.S_resolu[n], self.S_ddl[n], self.S_res[n], self.S_com[n], self.S_epsilon[n],
                   self.view_number, self.consensus_reached[n], self.sequence_number, self.primary_node,
                   self.message_count[n], self.pbft_progress[n], self.pending_message_count[n]]
                  for n in range(self.n_agents)]
        return np.array(State_)


    def generate_sign(self, message):
        message_bytes = json.dumps(message, sort_keys=True).encode('utf-8')
        client_id = message['client_id']
        signed = self.signing_keys[client_id].sign(message_bytes)
        public_key = self.public_keys[client_id].encode() if isinstance(self.public_keys[client_id], VerifyKey) else \
        self.public_keys[client_id]
        # print(
        #     f"节点 {client_id}: 签名消息={message_bytes.hex()}, 签名={signed.signature.hex()}, 公钥={public_key.hex()}")
        return {
            'message': message_bytes,
            'signature': signed.signature,
            'public_key': public_key
        }

    def generate_verify(self, signed_msg):
        try:
            verify_key = VerifyKey(signed_msg['public_key'])
            message_bytes = verify_key.verify(signed_msg['message'], signed_msg['signature'])
            # print(f"验证成功，消息内容: {message_bytes.hex()}")
            return json.loads(message_bytes.decode('utf-8'))
        except Exception as e:
            # print(
            #     f"验证失败: {e}, 消息={signed_msg['message'].hex()}, 签名={signed_msg['signature'].hex()}, 公钥={signed_msg['public_key'].hex()}")
            return None

    # 异步广播消息到指定节点，模拟 PBFT 共识中的消息传递
    async def broadcast_message(self, sender_id, msg, nodes):
        start = time.time()
        signed_msg = self.generate_sign(msg)
        # print(f"节点 {sender_id}: 广播 {msg['message_type']} 到节点 {nodes}, view_number={msg['view_number']}")
        for node_id in nodes:
            if node_id != sender_id and not self.faulty_nodes[node_id]:
                await asyncio.sleep(MESSAGE_DELAY)
                self.message_queues[node_id].put(signed_msg)
                self.pending_message_count[node_id] += 1
                self.message_count[sender_id] += 1
                self.message_count[node_id] += 1
                self.message_energy[sender_id] += MESSAGE_ENERGY
                self.message_energy[node_id] += MESSAGE_ENERGY
        #         print(
        #             f"节点 {sender_id} -> 节点 {node_id}: 发送 {msg['message_type']}, 消息计数={self.message_count}")
        # print(f"节点 {sender_id}: 广播耗时 {time.time() - start:.3f} 秒")
    # 异步处理节点的未处理消息和队列中的新消息，更新 PBFT 共识进度
    async def process_messages(self, node_id, available_res):
        # print(f"节点 {node_id}: 开始处理消息, available_res={available_res}")
        messages_processed = 0
        max_messages = int(available_res / PBFT_RES_PER_MSG * 1.5)
        # print(
        #     f"节点 {node_id}: 最大消息数={max_messages}, 待处理消息数={len(self.pending_messages[node_id])}, 消息队列大小={self.message_queues[node_id].qsize()}")

        # 优先处理新消息队列
        while not self.message_queues[node_id].empty() and messages_processed < max_messages:
            signed_msg = self.message_queues[node_id].get()
            msg = self.generate_verify(signed_msg)
            if msg is None:
                # print(f"节点 {node_id}: 跳过无效消息")
                continue
            if msg["message_type"] != "REQUEST" and msg.get("view_number") != self.view_number:
                # print(
                #     f"节点 {node_id}: 跳过消息，视图不匹配，消息视图={msg.get('view_number', 'N/A')}, 当前视图={self.view_number}")
                continue
            self.message_energy[node_id] += BLOCK_COMP_ENERGY
            await self.process_pbft_message(node_id, signed_msg)
            messages_processed += 1
            # print(f"节点 {node_id}: 已处理 {messages_processed}/{max_messages} 条消息")

        # 处理待处理消息
        while self.pending_messages[node_id] and messages_processed < max_messages:
            signed_msg = self.pending_messages[node_id].pop(0)
            self.pending_message_count[node_id] -= 1
            msg = self.generate_verify(signed_msg)
            if msg is None:
                # print(f"节点 {node_id}: 跳过无效消息")
                continue
            if msg["message_type"] != "REQUEST" and msg.get("view_number") != self.view_number:
                # print(
                #     f"节点 {node_id}: 跳过消息，视图不匹配，消息视图={msg.get('view_number', 'N/A')}, 当前视图={self.view_number}")
                continue
            self.message_energy[node_id] += BLOCK_COMP_ENERGY
            await self.process_pbft_message(node_id, signed_msg)
            messages_processed += 1
            # print(f"节点 {node_id}: 已处理 {messages_processed}/{max_messages} 条消息")

        total_messages = max_messages + self.pending_message_count[node_id]
        self.pbft_progress[node_id] = min(1.0, messages_processed / (total_messages + 1e-6))
        # print(f"Node {node_id}: Finished process_messages, pbft_progress={self.pbft_progress[node_id]}")

    # 处理 PBFT 共识的单个消息，模拟 PBFT 的四个阶段（REQUEST, PREPREPARE, PREPARE, COMMIT）
    async def process_pbft_message(self, node_id, signed_msg):
        try:
            msg = self.generate_verify(signed_msg)
            if msg is None:
                # print(f"节点 {node_id}: 跳过无效消息")
                return
            msg_type = msg["message_type"]
            # print(f"节点 {node_id}: 处理 {msg_type}, 视图={msg.get('view_number', 'N/A')}, 序列号={msg.get('sequence_number', 'N/A')}")
            if msg_type != "REQUEST" and msg.get("view_number") != self.view_number:
                # print(
                #     f"节点 {node_id}: 跳过消息，视图不匹配，消息视图={msg.get('view_number', 'N/A')}, 当前视图={self.view_number}")
                return
            if msg_type == "REQUEST" and node_id == self.primary_node and not self.faulty_nodes[node_id]:
                preprepare_msg = {
                    "message_type": "PREPREPARE",
                    "view_number": self.view_number,
                    "sequence_number": self.sequence_number,
                    "request": msg["request"],
                    "client_id": msg["client_id"],
                    "node_id": node_id
                }
                # print(f"节点 {node_id}: 为REQUEST {msg['request']} 生成PREPREPARE")
                await self.broadcast_message(node_id, preprepare_msg, range(self.n_agents))
            elif msg_type == "PREPREPARE" and msg["view_number"] == self.view_number:
                confirm_msg = {
                    "message_type": "CONFIRM",
                    "view_number": self.view_number,
                    "sequence_number": msg["sequence_number"],
                    "request": msg["request"],
                    "node_id": node_id,
                    "client_id": msg["client_id"]
                }
                # print(f"节点 {node_id}: 为PREPREPARE {msg['request']} 发送CONFIRM")
                await self.broadcast_message(node_id, confirm_msg, range(self.n_agents))
            elif msg_type == "CONFIRM" and msg["view_number"] == self.view_number:
                self.confirm_msgs[node_id] += 1
                # print(f"节点 {node_id}: 收到CONFIRM, confirm_msgs={self.confirm_msgs[node_id]}")
                if self.confirm_msgs[node_id] >= 2 * self.F_TOLERANCE + 1:
                    self.consensus_reached[node_id] = True
                    # print(f"节点 {node_id}: 达成共识")
        except Exception as e:
            print(f"节点 {node_id}: 消息处理失败, 错误={e}")
    async def run_pbft_consensus(self, S_res_remain):
        tasks = [self.process_messages(n, S_res_remain[n]) for n in range(self.n_agents) if not self.faulty_nodes[n]]
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            print("PBFT consensus timed out")
            self.pbft_progress = np.zeros(self.n_agents)
            for q in self.message_queues:
                while not q.empty():
                    q.get()  # 清空队列
            self.pending_messages = [[] for _ in range(self.n_agents)]
            self.pending_message_count = np.zeros(self.n_agents, dtype=int)
        print("Finished PBFT consensus")


    # 执行一步环境更新，处理智能体的动作，更新状态，计算奖励，模拟 PBFT 共识
    async def step(self, action):
        start_time = time.time()    # 记录开始时间
        print(f"Step started at {start_time}, action shape={action.shape}, action=\n{action}")
        # print(f"power_levels={self.power_levels}, res_levels={self.res_levels}")
        assert action.shape == (self.n_agents, 4)   # 验证动作形状是否正确
        # action
        A_decision = np.zeros(self.n_agents, dtype=np.float64)   # 卸载决定
        A_channel = np.zeros(self.n_agents, dtype=np.float64)    # 信道选择
        A_res = np.zeros(self.n_agents, dtype=np.float64)        # 本地计算资源
        A_power = np.zeros(self.n_agents, dtype=np.float64)      # 发射功率
        A_resolu = np.full(self.n_agents, S_RESOLU, dtype=np.float64)  # 固定为0.6
        A_com = np.full(self.n_agents, S_COM, dtype=np.float64)  # 固定为0.6
        # for n in range(self.n_agents):
        #     A_decision[n] = action[n, 0]
        #     A_channel[n] = action[n, 1]
        #     A_res[n] = np.clip(action[n, 2], MIN_RES, MAX_RES)  # 直接使用 action[:, 2] 并裁剪
        #     # A_power[n] = self.power_levels[int(np.clip(action[n, 3], 0, self.num_power_levels - 1))]
        #     assert 0 <= A_channel[n] <= 3, f"Invalid A_channel[{n}]={A_channel[n]}"
        #     print(
        #         f"Agent {n}: A_decision={A_decision[n]}, A_channel={A_channel[n]}, A_res={A_res[n]}, A_power={A_power[n]}")

        # 默认为normal
        if self.mode == "normal":
            for n in range(self.n_agents):
                A_decision[n] = 1 if action[n][0] > 0.3 else 0
                A_channel[n] = int(np.clip(action[n][1], 0, K_CHANNEL - 1))  # 直接使用信道索引
                A_res[n] = self.res_levels[int(np.clip(action[n, 2], 0, self.num_res_levels - 1))]  # 直接使用索引
                A_power[n] = self.power_levels[int(np.clip(action[n, 3], 0, self.num_power_levels - 1))]  # 直接使用索引
                print(
                    f"Agent {n}: action[2]={action[n, 2]}, A_res={A_res[n]}, action[3]={action[n, 3]}, A_power={A_power[n]}")
        elif self.mode == "NAC":
            for n in range(self.n_agents):
                A_decision[n] = action[n][0]
                A_channel[n] = action[n][1]
                A_resolu[n] = S_RESOLU  # 固定为0.6（与要求一致）
                A_res[n] = self.res_levels[int(action[n][2])]
                A_com[n] = S_COM  # 固定为0.6
                A_power[n] = self.power_levels[int(action[n][3])]
        elif self.mode == "ALLES":
            for n in range(self.n_agents):
                A_decision[n] = 1  # 强制卸载
                A_channel[n] = action[n][1]
                A_resolu[n] = S_RESOLU  # 固定为0.6
                A_res[n] = np.clip(action[n, 2], MIN_RES, MAX_RES)  # 直接使用 action[:, 2] 并裁剪
                A_power[n] = np.clip(action[n, 3], MIN_POWER, MAX_POWER)
                A_com[n] = S_COM  # 固定为0.6
        else:
            print("Wrong!")

        # 更新状态
        S_channel = self.S_channel.copy()
        S_power = A_power.copy()
        S_gain = np.clip(self.S_gain, MIN_GAIN, MAX_GAIN)
        S_size = np.clip(np.random.normal(S_SIZE, 0.5, size=self.n_agents), MIN_SIZE, MAX_SIZE)  # 任务大小
        S_cycle = np.clip(np.random.normal(S_CYCLE, 0.5, size=self.n_agents), MIN_CYCLE, MAX_CYCLE)
        S_resolu = self.S_resolu
        S_ddl = np.clip(np.random.normal(self.S_DDL, 1, size=self.n_agents), MIN_DDL, MAX_DDL * 1.5)
        S_res = A_res.copy()  # 使用动作中的计算资源
        S_com = self.S_com
        S_epsilon = self.S_epsilon

        # 根据 S_task, S_channel 调整 A_decision
        for n in range(self.n_agents):
            conflict = False
            for m in range(self.n_agents):
                if m != n and A_decision[m] == 1 and A_channel[m] == A_channel[n]:
                    conflict = True
                    # print(f"Conflict detected: Agent {n} and Agent {m} on channel {A_channel[n]}")
                    break
            if conflict and A_decision[n] == 1:
                new_channel = np.random.choice([i for i in range(4) if i != A_channel[n]])
                A_channel[n] = new_channel
                S_channel[n] = new_channel
                # print(f"Agent {n}: Reassigned channel to {new_channel} due to conflict")
            else:
                S_channel[n] = A_channel[n]

        x_n = A_decision.copy()  # 直接使用 A_decision 作为卸载决定
        # 根据卸载决定更新 S_channel
        # for n in range(self.n_agents):
        #     if A_decision[n] == 1:
        #         self.S_channel[n] = A_channel[n]
        #         S_channel[n] = A_channel[n]
        #     else:
        #         self.S_channel[n] = 0
        #         S_channel[n] = 0

        total_power = np.minimum(np.sum(x_n * S_power * S_gain), 1000)
        # print(f"total_power={total_power}, x_n={x_n}, S_power={S_power}, S_gain={S_gain}")
        # Phi_local = V_L * np.log(1 + S_resolu / THETA_L)    # 本地性能
        # Phi_off = V_E * np.log(1 + S_resolu / THETA_E)      # MEC性能
        # Phi_n = (1 - x_n) * Phi_local + x_n * Phi_off       # 实际性能
        # Phi_penalty = np.maximum((S_epsilon - Phi_n) / S_epsilon, 0)    # 性能惩罚
        #
        # total_com = np.sum(S_com)
        DataRate = np.zeros(self.n_agents)
        for n in range(self.n_agents):
            if x_n[n] == 1 and S_power[n] > 0:
                DataRate[n] = self.W_BANDWIDTH * np.log(1 + S_power[n] * S_gain[n] / NOISE_VARIANCE) / np.log(2)
            else:
                DataRate[n] = self.W_BANDWIDTH * np.log(1 + S_power[n] * S_gain[n] / NOISE_VARIANCE) / np.log(2)  # 默认数据率 0.1 Mbps
            # print(f"Agent {n}: DataRate={DataRate[n]} Mbps")
        # 计算延时
        Time_proc = S_resolu * S_cycle / CAPABILITY_E    # MEC处理时间
        Time_local = S_resolu * S_cycle / S_res          # 本地计算时间
        Time_off = np.minimum(S_size / np.maximum(DataRate, 0.1), 100)       # 卸载传输时间
        Time_n = (1 - x_n) * Time_local + x_n * (Time_off + Time_proc)    # 实际完成时间

        T_mean = np.mean(Time_n)       # 平均完成时间
        # R_mine = KSI * S_com / total_com * np.exp(-LAMBDA * T_mean / S_ddl)    # 任务完成奖励

        Time_penalty = np.maximum((Time_n - S_ddl) / (S_ddl + 1e-6), 0) * 0.1              # 时间惩罚

        Energy_local = K_ENERGY_LOCAL * S_size * S_resolu * (S_res**2) * 1e27    # 本地能量
        Energy_off = S_power * Time_off           # 卸载能量
        # Energy_mine = OMEGA * S_com                         # 通信能量
        Energy_n = (1 - x_n) * Energy_local + x_n * Energy_off   # 实际能量
        # print(f"S_size={S_size}, S_res={S_res}, DataRate={DataRate}, Time_off={Time_off}")
        print(f"Energy_local={Energy_local}, Energy_n={Energy_n}")

        self.Energy_local = Energy_local
        self.Energy_n = Energy_n
        # 剩余资源计算
        S_res_remain = np.zeros(self.n_agents)
        for n in range(self.n_agents):
            mec_res_needed = S_resolu[n] * S_cycle[n]   # 当前所需资源
            if not x_n[n]:
                S_res_remain[n] = max(0, S_res[n] - mec_res_needed)   # 扣除所需资源
            else:
                S_res_remain[n] = S_res[n]
            # print(f"Agent {n}: mec_res_needed={mec_res_needed}, S_res_remain={S_res_remain[n]}")
        # 重置 PBFT 状态
        # self.pre_prepare_msgs = np.zeros(self.n_agents, dtype=bool)
        # self.prepare_msgs = np.zeros(self.n_agents, dtype=int)
        # self.commit_msgs = np.zeros(self.n_agents, dtype=int)
        # self.consensus_reached = np.zeros(self.n_agents, dtype=bool)
        # self.message_energy = np.zeros(self.n_agents)
        # self.message_count = np.zeros(self.n_agents, dtype=int)
        Energy_pbft = np.zeros(self.n_agents)

        # 生成 PBFT 请求
        for n in range(self.n_agents):
            if not self.faulty_nodes[n] and S_res_remain[n] >= PBFT_RES_PER_MSG:  # 对非故障节点且资源足够的节点
                request = {
                    "message_type": "REQUEST",
                    "request": f"MEC_task_{n}_{self.S_size[n]}_{self.S_cycle[n]}",
                    "timestamp": self.epoch,
                    "client_id": n,
                    "view_number": self.view_number,
                    "sequence_number": self.sequence_number
                }
                # 创建消息，签名并放入主节点，增加生成能耗，扣除资源
                self.message_count[self.primary_node] += 1
                self.message_energy[n] += BLOCK_GEN_ENERGY * self.S_size[n]
                S_res_remain[n] -= PBFT_RES_PER_MSG
                # print(f"Agent {n}: Generated REQUEST: {request}")
                await self.process_pbft_message(self.primary_node, self.generate_sign(request))        # 运行共识
        print("Starting PBFT consensus")
        await self.run_pbft_consensus(S_res_remain)
        # print(
        #     f"consensus_reached={self.consensus_reached}, message_count={self.message_count}, confirm_msgs={self.confirm_msgs}")
        # 共识结果处理
        consensus_success = np.sum(self.consensus_reached) >= 2 * self.F_TOLERANCE + 1
        if not consensus_success:
            # print(
            #     f"Consensus failed, changing view to {self.view_number + 1}, new primary={(self.primary_node + 1) % self.n_agents}")
            self.view_number += 1
            self.primary_node = (self.primary_node + 1) % self.n_agents
            self.sequence_number = 0  # 重置序列号以匹配新视图
            self.Time_penalty += 0.05
            self.message_queues = [queue.Queue() for _ in range(self.n_agents)]
            self.pending_messages = [[] for _ in range(self.n_agents)]
            self.pending_message_count = np.zeros(self.n_agents, dtype=int)
        else:
            self.sequence_number += 1
        # 达成共识，分配奖励，保存奖励、能耗、时间惩罚
        pbft_reward = np.zeros(self.n_agents)
        if consensus_success:
            mask = self.consensus_reached & ~self.faulty_nodes
            pbft_reward[mask] = CONSENSUS_REWARD * S_size[mask]  # 只对掩码为 True 的节点赋值
        self.pbft_reward = pbft_reward
        self.Energy_pbft = self.message_energy + BLOCK_GEN_ENERGY * S_size
        # print(f"pbft_reward={pbft_reward}, Energy_pbft={self.Energy_pbft}")
        self.Time_penalty = Time_penalty

        # Reward_vt = np.clip((Energy_local - Energy_n) / (Energy_local + 1e-6), -1.0, 1.0)
        # Reward_vt = pd.Series(np.clip((Energy_local - Energy_n) / (Energy_local + 1e-6), -1.0, 1.0)).rolling(window=10, min_periods=1).mean().value
        # Time_penalty = np.maximum((Time_n - S_ddl) / (S_ddl + 1e-6), 0) * 0.1
        # Time_penalty = pd.Series(np.maximum((Time_n - S_ddl) / (S_ddl + 1e-6), 0) * 0.1).rolling(window=10, min_periods=1).mean().values
        Reward_vt = pd.Series(np.clip((Energy_local - Energy_n) / (Energy_local + 1e-6), -1.0, 1.0)).rolling(window=10,
                                                                                                             min_periods=1).mean().values
        Time_penalty = pd.Series(np.maximum((Time_n - S_ddl) / (S_ddl + 1e-6), 0) * 0.1).rolling(window=10,
                                                                                                 min_periods=1).mean().values
        Reward = Reward_vt + ALPHA * pbft_reward - self.Energy_pbft - BETA * Time_penalty
        print(
            f"Reward components: Reward_vt={Reward_vt}, pbft_reward={pbft_reward}, Energy_pbft={Energy_pbft}, Time_penalty={Time_penalty}")

        for n in range(self.n_agents):
            if int(A_decision[n]):
                self.S_channel[n] = A_channel[n]
        self.S_resolu = A_resolu
        self.S_res = A_res
        self.S_com = np.array(A_com) if not isinstance(A_com, np.ndarray) else A_com
        self.S_power = np.array(A_power) if not isinstance(A_power, np.ndarray) else A_power
        self.S_channel = np.array(self.S_channel).flatten()
        self.S_power = np.array(self.S_power).flatten()
        self.S_gain = np.array(self.S_gain).flatten()
        self.S_size = np.array(self.S_size).flatten()
        self.S_cycle = np.array(self.S_cycle).flatten()
        self.S_resolu = np.array(self.S_resolu).flatten()
        self.S_ddl = np.array(self.S_ddl).flatten()
        self.S_res = np.array(self.S_res).flatten()
        self.S_com = np.array(self.S_com).flatten()
        self.S_epsilon = np.array(self.S_epsilon).flatten()
        State_ = [[self.S_channel[n], self.S_power[n], self.S_gain[n], self.S_size[n], self.S_cycle[n],
                   self.S_resolu[n], self.S_ddl[n], self.S_res[n], self.S_com[n], self.S_epsilon[n],
                   self.view_number, self.consensus_reached[n], self.sequence_number, self.primary_node,
                   self.message_count[n], self.pbft_progress[n], self.pending_message_count[n]]
                  for n in range(self.n_agents)]
        State_ = np.array(State_)
        # 若 epoch > 10 或任意 Time_penalty > 1.0，调用 reset 并设置 done = True
        self.epoch += 1
        done = False  # 移除 epoch 限制，依靠 MAPPO 的 max_episodes 控制
        if done:
            self.reset()
        end_time = time.time()
        logging.info(f"Step completed in {end_time - start_time:.2f} seconds, consensus_success={consensus_success}")
        return State_, Reward, done, True, Energy_n,

    # 重定义的方法
    # async def process_messages(self, node_id, available_res):
    #     print(f"Node {node_id}: Starting process_messages, available_res={available_res}")
    #     messages_processed = 0
    #     max_messages = min(int(available_res / PBFT_RES_PER_MSG), 10)
    #     print(f"Node {node_id}: max_messages={max_messages}, pending_messages={len(self.pending_messages[node_id])}")
    #     while self.pending_messages[node_id] and messages_processed < max_messages:
    #         signed_msg = self.pending_messages[node_id].pop(0)
    #         self.pending_message_count[node_id] -= 1
    #         try:
    #             msg, public_key = signed_msg.split(b'split')
    #             msg = generate_verify(public_key, msg)
    #             msg = json.loads(msg.replace("'", "\""))
    #         except Exception as e:
    #             print(f"Node {node_id}: Failed to verify message, error={e}")
    #             continue
    #         self.message_energy[node_id] += BLOCK_COMP_ENERGY
    #         await self.process_pbft_message(node_id, msg)
    #         messages_processed += 1
    #         print(f"Node {node_id}: Processed {messages_processed}/{max_messages} messages")
    #     while not self.message_queues[node_id].empty() and messages_processed < max_messages:
    #         signed_msg = self.message_queues[node_id].get()
    #         self.pending_messages[node_id].append(signed_msg)
    #         self.pending_message_count[node_id] += 1
    #         messages_processed += 1
    #         print(f"Node {node_id}: Queued new message, total_processed={messages_processed}")
    #     total_messages = max_messages + self.pending_message_count[node_id]
    #     self.pbft_progress[node_id] = min(1.0, messages_processed / total_messages if total_messages > 0 else 0.0)
    #     print(f"Node {node_id}: Finished process_messages, pbft_progress={self.pbft_progress[node_id]}")
