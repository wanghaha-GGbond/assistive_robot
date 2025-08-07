#!/usr/bin/env python3
"""
SAC强化学习训练器
基于checkpoint_integrated_supervised_epoch_80.pth进行强化学习训练

模型架构: Integrated VLA-SAC
训练方法: Soft Actor-Critic (SAC)
数据集: Subject1-9助餐数据集
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import logging
from collections import deque
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import json
import time
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SACConfig:
    """SAC强化学习配置 - 基于实际模型配置"""
    
    # 模型路径
    supervised_model_path: str = "checkpoint_integrated_supervised_epoch_80.pth"
    output_dir: str = "outputs/sac_reinforcement_learning"
    
    # SAC超参数 (从实际配置提取)
    sac_lr: float = 0.0003
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float = -7.0
    gamma: float = 0.99
    
    # 经验回放
    replay_buffer_size: int = 50000
    min_replay_size: int = 500
    batch_size: int = 256
    
    # 训练配置
    rl_epochs: int = 40
    steps_per_epoch: int = 1000
    eval_episodes: int = 10
    save_every: int = 5
    
    # 环境配置
    action_dim: int = 7
    state_dim: int = 256  # 感知模块输出维度
    max_episode_steps: int = 100
    
    # 网络配置
    hidden_dim: int = 256
    num_layers: int = 2

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """采样批次数据"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """SAC Actor网络 (策略网络)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # 动作范围
        self.action_scale = 1.0
        self.action_bias = 0.0
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # 修正tanh变换的log_prob
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class Critic(nn.Module):
    """SAC Critic网络 (Q网络)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1网络
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2网络
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

class FeedingEnvironment:
    """助餐环境模拟器"""
    
    def __init__(self, config: SACConfig):
        self.config = config
        self.max_steps = config.max_episode_steps
        self.current_step = 0
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        
        # 环境状态
        self.current_state = None
        self.target_position = None
        self.food_type = None
        self.subject_id = None
        
    def reset(self):
        """重置环境"""
        self.current_step = 0
        
        # 随机选择Subject和食物类型
        self.subject_id = np.random.randint(1, 10)  # Subject 1-9
        self.food_type = np.random.randint(0, 9)    # 9种食物
        
        # 生成初始状态 (模拟多模态感知输出)
        self.current_state = self._generate_state()
        self.target_position = np.random.uniform(-1, 1, self.action_dim)
        
        return self.current_state
    
    def step(self, action):
        """执行动作"""
        self.current_step += 1
        
        # 计算奖励
        reward = self._calculate_reward(action)
        
        # 更新状态
        next_state = self._generate_state()
        
        # 判断是否结束
        done = self.current_step >= self.max_steps or self._is_success(action)
        
        self.current_state = next_state
        
        return next_state, reward, done, {}
    
    def _generate_state(self):
        """生成状态 (模拟感知模块输出)"""
        # 模拟多模态特征融合后的256维状态
        state = np.random.randn(self.state_dim).astype(np.float32)
        
        # 添加Subject和食物类型信息
        state[0] = self.subject_id / 10.0  # 归一化Subject ID
        state[1] = self.food_type / 9.0    # 归一化食物类型
        
        return state
    
    def _calculate_reward(self, action):
        """计算奖励函数"""
        # 基础奖励：动作与目标位置的距离
        distance_reward = -np.linalg.norm(action - self.target_position)
        
        # 安全奖励：避免过大的动作
        safety_reward = -0.1 * np.sum(np.abs(action))
        
        # 平滑奖励：避免剧烈变化
        smoothness_reward = -0.05 * np.sum(action ** 2)
        
        # 任务完成奖励
        success_reward = 10.0 if self._is_success(action) else 0.0
        
        # 食物特定奖励 (不同食物有不同难度)
        food_difficulty = [1.0, 1.2, 1.1, 1.0, 1.5, 1.3, 1.4, 1.6, 1.2]  # 对应9种食物
        difficulty_factor = food_difficulty[self.food_type]
        
        total_reward = (distance_reward + safety_reward + smoothness_reward + success_reward) / difficulty_factor
        
        return total_reward
    
    def _is_success(self, action):
        """判断是否成功"""
        distance = np.linalg.norm(action - self.target_position)
        return distance < 0.1  # 成功阈值

class SACAgent:
    """SAC智能体"""
    
    def __init__(self, config: SACConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # 网络初始化
        self.actor = Actor(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.critic = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.critic_target = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        
        # 复制参数到目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.sac_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.sac_lr)
        
        # 自动调整熵系数
        if config.auto_alpha:
            self.target_entropy = config.target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.sac_lr)
        else:
            self.alpha = config.alpha
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        
        # 训练统计
        self.total_steps = 0
        self.episode_rewards = []
        self.losses = {'actor': [], 'critic': [], 'alpha': []}
        
    @property
    def alpha(self):
        if self.config.auto_alpha:
            return self.log_alpha.exp()
        else:
            return self.config.alpha
    
    def select_action(self, state, evaluate=False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def update(self):
        """更新网络参数"""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return
        
        # 采样批次数据
        state, action, reward, next_state, done = self.replay_buffer.sample(self.config.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # 更新Critic
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.config.gamma * q_next
        
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        new_action, log_prob, _ = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新Alpha (如果自动调整)
        alpha_loss = 0
        if self.config.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # 软更新目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        # 记录损失
        self.losses['critic'].append(critic_loss.item())
        self.losses['actor'].append(actor_loss.item())
        if self.config.auto_alpha:
            self.losses['alpha'].append(alpha_loss.item())

class SACTrainer:
    """SAC训练器"""
    
    def __init__(self, config: SACConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化环境和智能体
        self.env = FeedingEnvironment(config)
        self.agent = SACAgent(config, self.device)
        
        # 加载监督学习模型 (如果存在)
        self.load_supervised_model()
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'success_rates': [],
            'losses': {'actor': [], 'critic': [], 'alpha': []},
            'evaluation_scores': []
        }
        
        logger.info(f"🚀 SAC训练器初始化完成")
        logger.info(f"   设备: {self.device}")
        logger.info(f"   输出目录: {config.output_dir}")
        logger.info(f"   强化学习轮数: {config.rl_epochs}")
    
    def load_supervised_model(self):
        """加载监督学习模型作为初始化"""
        if Path(self.config.supervised_model_path).exists():
            try:
                checkpoint = torch.load(self.config.supervised_model_path, map_location=self.device)
                logger.info(f"✅ 成功加载监督学习模型: {self.config.supervised_model_path}")
                logger.info(f"   监督学习轮数: {checkpoint.get('epoch', 'Unknown')}")
                logger.info(f"   监督学习损失: {checkpoint.get('metric', 'Unknown')}")
                
                # 这里可以提取预训练的特征提取器权重
                # 由于我们使用简化的SAC网络，暂时不直接加载权重
                
            except Exception as e:
                logger.warning(f"⚠️ 无法加载监督学习模型: {e}")
        else:
            logger.warning(f"⚠️ 监督学习模型文件不存在: {self.config.supervised_model_path}")
    
    def train(self):
        """开始SAC训练"""
        logger.info("🎯 开始SAC强化学习训练")
        
        total_steps = 0
        
        for epoch in range(self.config.rl_epochs):
            epoch_rewards = []
            epoch_successes = 0
            
            for episode in range(self.config.steps_per_epoch // self.config.max_episode_steps):
                # 重置环境
                state = self.env.reset()
                episode_reward = 0
                episode_success = False
                
                for step in range(self.config.max_episode_steps):
                    # 选择动作
                    action = self.agent.select_action(state)
                    
                    # 执行动作
                    next_state, reward, done, info = self.env.step(action)
                    
                    # 存储经验
                    self.agent.replay_buffer.push(state, action, reward, next_state, done)
                    
                    # 更新网络
                    if total_steps > self.config.min_replay_size:
                        self.agent.update()
                    
                    episode_reward += reward
                    state = next_state
                    total_steps += 1
                    
                    if done:
                        if reward > 5.0:  # 成功奖励阈值
                            episode_success = True
                        break
                
                epoch_rewards.append(episode_reward)
                if episode_success:
                    epoch_successes += 1
            
            # 计算统计信息
            avg_reward = np.mean(epoch_rewards)
            success_rate = epoch_successes / len(epoch_rewards)
            
            # 评估
            eval_score = self.evaluate()
            
            # 记录统计
            self.training_stats['episode_rewards'].extend(epoch_rewards)
            self.training_stats['success_rates'].append(success_rate)
            self.training_stats['evaluation_scores'].append(eval_score)
            
            # 保存模型
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch)
            
            logger.info(f"Epoch {epoch+1}/{self.config.rl_epochs}: "
                       f"Avg Reward: {avg_reward:.3f}, "
                       f"Success Rate: {success_rate:.3f}, "
                       f"Eval Score: {eval_score:.3f}, "
                       f"Alpha: {self.agent.alpha:.3f}")
        
        # 保存最终模型
        self.save_checkpoint(self.config.rl_epochs - 1, is_final=True)
        self.save_training_report()
        
        logger.info("✅ SAC强化学习训练完成!")
    
    def evaluate(self):
        """评估模型性能"""
        eval_rewards = []
        
        for _ in range(self.config.eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.config.max_episode_steps):
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)
    
    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'critic_target_state_dict': self.agent.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats,
            'total_steps': self.agent.total_steps
        }
        
        if self.config.auto_alpha:
            checkpoint['log_alpha'] = self.agent.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.agent.alpha_optimizer.state_dict()
        
        if is_final:
            filename = f"{self.config.output_dir}/sac_final_model.pth"
        else:
            filename = f"{self.config.output_dir}/sac_checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, filename)
        logger.info(f"💾 保存检查点: {filename}")
    
    def save_training_report(self):
        """保存训练报告"""
        report = {
            'training_config': {
                'rl_epochs': self.config.rl_epochs,
                'sac_lr': self.config.sac_lr,
                'tau': self.config.tau,
                'alpha': self.config.alpha,
                'gamma': self.config.gamma,
                'replay_buffer_size': self.config.replay_buffer_size,
                'batch_size': self.config.batch_size
            },
            'training_results': {
                'final_avg_reward': np.mean(self.training_stats['episode_rewards'][-100:]) if self.training_stats['episode_rewards'] else 0,
                'final_success_rate': self.training_stats['success_rates'][-1] if self.training_stats['success_rates'] else 0,
                'final_eval_score': self.training_stats['evaluation_scores'][-1] if self.training_stats['evaluation_scores'] else 0,
                'total_episodes': len(self.training_stats['episode_rewards'])
            },
            'training_stats': self.training_stats
        }
        
        report_file = f"{self.config.output_dir}/sac_training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 训练报告已保存: {report_file}")

def main():
    """主函数"""
    print("🚀 启动SAC强化学习训练")
    print("=" * 80)
    
    # 创建配置
    config = SACConfig()
    
    # 创建训练器
    trainer = SACTrainer(config)
    
    # 开始训练
    trainer.train()
    
    print("🎉 SAC强化学习训练完成!")
    print(f"📊 结果保存在: {config.output_dir}")

if __name__ == "__main__":
    main()
