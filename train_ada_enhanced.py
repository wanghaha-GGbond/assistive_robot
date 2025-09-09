#!/usr/bin/env python3
"""
Ada LeRobot Enhanced Perception Model 专用训练脚本
直接训练 AdaLeRobotEnhancedPerceptionModel 
"""

import os
import sys
import yaml
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import wandb
import logging

# 导入模型
from ada_lerobot_enhanced_perception import AdaLeRobotEnhancedPerceptionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaExpandedDataset(Dataset):
    """专门为 ada_expanded_dataset 设计的数据集加载器 - 增强版支持数据增强
    现在支持我们生成的10000样本数据集格式
    """

    def __init__(self, h5_file_path: str, window_size: int = 128, split: str = 'train',
                 augment: bool = False, augment_prob: float = 0.5):
        self.h5_file_path = h5_file_path
        self.window_size = window_size
        self.split = split
        self.augment = augment and (split == 'train')  # 只在训练时进行数据增强
        self.augment_prob = augment_prob

        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """从HDF5文件加载数据 - 支持LeRobot多模态格式和我们的10K数据集格式"""
        with h5py.File(self.h5_file_path, 'r') as f:
            logger.info(f"正在加载数据集: {self.h5_file_path}")

            # 检查数据集格式
            available_keys = list(f.keys())
            logger.info(f"数据集顶级键: {available_keys}")

            # 检查是数据集格式 (trajectory_0, trajectory_1, ...)
            if 'trajectory_0' in f and f.attrs.get('total_trajectories', 0) > 0:
                logger.info("检测到我们生成的10K数据集格式")
                self._load_trajectory_format(f)

            # 支持LeRobot多模态格式
            elif 'observation' in f and 'slip_binary' in f:
                logger.info("检测到LeRobot多模态格式")

                # 加载观测数据
                obs_group = f['observation']
                if 'state' in obs_group:
                    state_data = obs_group['state'][:]  # shape: (N, 128, 13)
                    # 分离力和关节数据
                    self.sensor_data = state_data[:, :, :6]  # 前6维是力/扭矩
                    self.joint_data = state_data[:, :, 6:]   # 后7维是关节
                    logger.info(f"状态数据形状: {state_data.shape}")
                else:
                    raise KeyError("LeRobot格式中未找到observation/state数据")

                # 加载图像数据（如果存在）
                if 'image_rgb' in obs_group:
                    self.image_data = obs_group['image_rgb'][:]
                    logger.info(f"图像数据形状: {self.image_data.shape}")
                else:
                    self.image_data = None
                    logger.warning("未找到图像数据，将使用随机特征")

                # 加载标签
                self.slip_labels = f['slip_binary'][:]
                self.texture_labels = f['texture_labels'][:]

                # 获取纹理类别名称
                if 'texture_classes' in f.attrs:
                    self.texture_classes = [s.decode('utf-8') for s in f.attrs['texture_classes']]
                else:
                    unique_textures = np.unique(self.texture_labels)
                    self.texture_classes = [f"texture_{i}" for i in unique_textures]

                logger.info(f"纹理类别: {self.texture_classes}")

            # 兼容旧格式
            elif 'data' in f and 'joint_data' in f:
                logger.info("检测到传统格式")
                self.sensor_data = f['data'][:]  # 传感器数据
                self.joint_data = f['joint_data'][:]  # 关节数据
                self.image_data = None

                # 动态加载标签，兼容多种格式
                if 'labels' in f and 'slip_binary' in f['labels']:
                    # 格式1: 嵌套标签格式 (e.g., ada_fully_labeled_dataset)
                    logger.info("检测到 'labels/slip_binary' 嵌套格式")
                    self.slip_labels = f['labels/slip_binary'][:]
                    self.texture_labels = f['labels/texture_labels'][:]
                elif 'slip_label' in f:
                    # 格式2: 根级标签格式 (改进后的数据集)
                    logger.info("检测到 'slip_label' 根级格式（改进数据集）")
                    self.slip_labels = f['slip_label'][:]
                    self.texture_labels = f['texture_label'][:]
                else:
                    logger.error(f"无法识别标签格式。可用键: {available_keys}")
                    raise KeyError("无法在HDF5文件中找到可识别的滑移或纹理标签。")
            else:
                logger.error(f"无法识别数据集格式。可用键: {available_keys}")
                raise KeyError("无法识别数据集格式")
            
            # 数据质量分析
            slip_counts = np.bincount(self.slip_labels.astype(int))
            texture_counts = np.bincount(self.texture_labels.astype(int))
            
            # 计算平衡度指标
            slip_balance = min(slip_counts) / max(slip_counts) if max(slip_counts) > 0 else 0
            texture_balance = min(texture_counts) / max(texture_counts) if max(texture_counts) > 0 else 0
            
            logger.info(f"滑移标签分布: {slip_counts}")
            logger.info(f"纹理标签分布: {texture_counts}")
            logger.info(f"滑移平衡度: {slip_balance:.3f}")
            logger.info(f"纹理平衡度: {texture_balance:.3f}")
            
            # 数据集质量评估
            quality_score = 0
            if slip_balance >= 0.8:
                quality_score += 3
            elif slip_balance >= 0.6:
                quality_score += 2
            elif slip_balance >= 0.4:
                quality_score += 1
                
            if texture_balance >= 0.9:
                quality_score += 3
            elif texture_balance >= 0.7:
                quality_score += 2
            elif texture_balance >= 0.5:
                quality_score += 1
            
            if quality_score >= 5:
                logger.info("✅ A级数据集质量 - 高平衡度，适合高精度训练")
                self.augment_intensity = 0.3  # 降低增强强度
            elif quality_score >= 3:
                logger.info("⚠️ B级数据集质量 - 中等平衡度")
                self.augment_intensity = 0.5  # 中等增强强度
            else:
                logger.warning("❌ C级数据集质量 - 低平衡度，需要额外增强")
                self.augment_intensity = 0.7  # 增加增强强度
            
            # 获取数据统计
            total_windows = len(self.sensor_data)
            logger.info(f"总窗口数: {total_windows}")
            logger.info(f"传感器数据形状: {self.sensor_data.shape}")
            logger.info(f"关节数据形状: {self.joint_data.shape}")
            logger.info(f"滑移标签数量: {len(self.slip_labels)}")
            logger.info(f"纹理标签数量: {len(self.texture_labels)}")
            
            # 分割数据集
            val_size = int(0.15 * total_windows)
            test_size = int(0.1 * total_windows)
            train_size = total_windows - val_size - test_size
            
            if self.split == 'train':
                self.start_idx = 0
                self.end_idx = train_size
            elif self.split == 'val':
                self.start_idx = train_size
                self.end_idx = train_size + val_size
            else:  # test
                self.start_idx = train_size + val_size
                self.end_idx = total_windows
                
            logger.info(f"{self.split} split: {self.start_idx} - {self.end_idx} ({self.end_idx - self.start_idx} samples)")

    def _load_trajectory_format(self, f):
        """加载我们生成的轨迹格式数据集"""
        total_trajectories = f.attrs.get('total_trajectories', 0)
        logger.info(f"加载轨迹格式数据集，总轨迹数: {total_trajectories}")

        # 预分配数组
        all_sensor_data = []
        all_joint_data = []
        all_slip_labels = []
        all_texture_labels = []

        # 食物类型到纹理标签的映射
        food_to_texture = {
            'rice': 0, 'noodles': 1, 'soup': 2, 'chicken_cube': 3,
            'tofu': 4, 'broccoli': 5
        }

        for i in range(total_trajectories):
            traj_key = f'trajectory_{i}'
            if traj_key not in f:
                continue

            traj = f[traj_key]

            # 获取力数据 (作为传感器数据)
            forces = traj['forces']
            normal_forces = forces['total_normal_forces'][:]
            tangential_forces = forces['total_tangential_forces'][:]
            contact_count = forces['contact_count'][:]

            # 组合力数据为6维传感器数据 (3力+3扭矩，这里简化为normal, tangential, contact重复)
            sensor_data = np.column_stack([
                normal_forces, tangential_forces, contact_count,
                normal_forces * 0.1, tangential_forces * 0.1, contact_count * 0.1  # 模拟扭矩
            ])

            # 获取关节数据
            obs = traj['observations']
            joint_positions = obs['joint_positions'][:]
            joint_velocities = obs['joint_velocities'][:]

            # 获取滑移标签
            slip_labels = traj['slip_labels']
            slip_detected = slip_labels['detected'][:]

            # 获取食物类型并转换为纹理标签
            food_type = traj.attrs.get('food_type', 'rice')
            texture_label = food_to_texture.get(food_type, 0)

            # 确保数据长度一致，填充或截断到window_size
            seq_len = len(sensor_data)
            if seq_len < self.window_size:
                # 填充
                pad_len = self.window_size - seq_len
                sensor_data = np.pad(sensor_data, ((0, pad_len), (0, 0)), mode='edge')
                joint_positions = np.pad(joint_positions, ((0, pad_len), (0, 0)), mode='edge')
                joint_velocities = np.pad(joint_velocities, ((0, pad_len), (0, 0)), mode='edge')
                slip_detected = np.pad(slip_detected, (0, pad_len), mode='edge')
            elif seq_len > self.window_size:
                # 截断
                sensor_data = sensor_data[:self.window_size]
                joint_positions = joint_positions[:self.window_size]
                joint_velocities = joint_velocities[:self.window_size]
                slip_detected = slip_detected[:self.window_size]

            # 组合关节数据 (位置+速度，但保持7维)
            joint_data = joint_positions  # 只使用位置，保持7维

            # 添加到列表
            all_sensor_data.append(sensor_data)
            all_joint_data.append(joint_data)
            # 使用轨迹级别的滑移标签 (取平均值)
            all_slip_labels.append(float(np.mean(slip_detected)))
            all_texture_labels.append(texture_label)

        # 转换为numpy数组
        self.sensor_data = np.array(all_sensor_data)  # (N, window_size, 6)
        self.joint_data = np.array(all_joint_data)    # (N, window_size, 7)
        self.slip_labels = np.array(all_slip_labels)  # (N,)
        self.texture_labels = np.array(all_texture_labels)  # (N,)

        # 设置纹理类别
        self.texture_classes = list(food_to_texture.keys())

        # 图像数据设为None（我们的数据集没有图像）
        self.image_data = None

        logger.info(f"传感器数据形状: {self.sensor_data.shape}")
        logger.info(f"关节数据形状: {self.joint_data.shape}")
        logger.info(f"滑移标签数量: {len(self.slip_labels)}")
        logger.info(f"纹理标签数量: {len(self.texture_labels)}")
        logger.info(f"纹理类别: {self.texture_classes}")

        # 数据质量分析
        slip_counts = np.bincount(self.slip_labels.astype(int))
        texture_counts = np.bincount(self.texture_labels.astype(int))

        # 计算平衡度指标
        slip_balance = min(slip_counts) / max(slip_counts) if len(slip_counts) >= 2 and max(slip_counts) > 0 else 0
        texture_balance = min(texture_counts) / max(texture_counts) if len(texture_counts) >= 2 and max(texture_counts) > 0 else 0

        logger.info(f"滑移标签分布: {slip_counts}")
        logger.info(f"纹理标签分布: {texture_counts}")
        logger.info(f"滑移平衡度: {slip_balance:.3f}")
        logger.info(f"纹理平衡度: {texture_balance:.3f}")

        # 数据集质量评估
        quality_score = 0
        if slip_balance >= 0.8:
            quality_score += 3
        elif slip_balance >= 0.6:
            quality_score += 2
        elif slip_balance >= 0.4:
            quality_score += 1

        if texture_balance >= 0.9:
            quality_score += 3
        elif texture_balance >= 0.7:
            quality_score += 2
        elif texture_balance >= 0.5:
            quality_score += 1

        if quality_score >= 5:
            logger.info("✅ A级数据集质量 - 高平衡度，适合高精度训练")
            self.augment_intensity = 0.3  # 降低增强强度
        elif quality_score >= 3:
            logger.info("⚠️ B级数据集质量 - 中等平衡度")
            self.augment_intensity = 0.5  # 中等增强强度
        else:
            logger.warning("❌ C级数据集质量 - 低平衡度，需要额外增强")
            self.augment_intensity = 0.7  # 增加增强强度

        # 获取数据统计
        total_windows = len(self.sensor_data)
        logger.info(f"总窗口数: {total_windows}")

        # 分割数据集
        val_size = int(0.15 * total_windows)
        test_size = int(0.1 * total_windows)
        train_size = total_windows - val_size - test_size

        if self.split == 'train':
            self.start_idx = 0
            self.end_idx = train_size
        elif self.split == 'val':
            self.start_idx = train_size
            self.end_idx = train_size + val_size
        else:  # test
            self.start_idx = train_size + val_size
            self.end_idx = total_windows

    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        
        # 传感器数据 (304, 128, 6) - 包含力数据等
        sensor_window = self.sensor_data[actual_idx]  # (128, 6)
        
        # 关节数据 (304, 128, 7) 
        joint_window = self.joint_data[actual_idx]  # (128, 7)
        
        # 先转换为torch tensor
        sensor_tensor = torch.tensor(sensor_window, dtype=torch.float32)  # (128, 6)
        joint_tensor = torch.tensor(joint_window, dtype=torch.float32)    # (128, 7)
        
        # 数据增强 (仅在训练时)
        if self.augment and torch.rand(1).item() < self.augment_prob:
            sensor_tensor, joint_tensor = self._apply_augmentation(sensor_tensor, joint_tensor)
        
        # 保持时序结构的特征提取 - 适配新的模型架构
        # 视觉特征 (使用真实图像数据或占位符)
        if hasattr(self, 'image_data') and self.image_data is not None:
            # 使用真实图像数据，但需要转换为特征向量
            # 这里我们使用简单的全局平均池化来生成512维特征
            image = torch.from_numpy(self.image_data[actual_idx]).float() / 255.0  # 归一化到[0,1]
            image = image.permute(2, 0, 1)  # 转换为(C, H, W)格式

            # 简单的特征提取：全局平均池化 + 线性变换
            # 将(3, 224, 224)转换为512维特征向量
            pooled = torch.mean(image, dim=(1, 2))  # (3,) -> 全局平均池化
            # 扩展到512维
            visual_features = torch.cat([
                pooled.repeat(170),  # 3*170 = 510
                pooled[:2]           # 再加2维，总共512维
            ])
        else:
            # 占位符，实际项目中可用摄像头数据
            visual_features = torch.zeros(512)
        
        # 触觉特征 = 传感器时序数据 (batch, seq_len=128, features=6)
        tactile_features = sensor_tensor  # 保持形状 (128, 6)
        
        # 力数据 = 传感器时序数据 (batch, seq_len=128, features=6) 
        force_features = sensor_tensor  # 保持形状 (128, 6)
        
        # 本体感觉特征 = 关节时序数据 (batch, seq_len=128, features=7)
        proprioceptive_features = joint_tensor  # 保持形状 (128, 7)
        
        # 标签
        slip_label = torch.tensor(float(self.slip_labels[actual_idx]), dtype=torch.float32)
        texture_label = torch.tensor(int(self.texture_labels[actual_idx]), dtype=torch.long)
        
        return {
            'visual_features': visual_features,
            'tactile_features': tactile_features,
            'force_features': force_features,
            'proprioceptive_features': proprioceptive_features,
            'slip_label': slip_label,
            'texture_label': texture_label
        }
    
    def _apply_augmentation(self, sensor_tensor, joint_tensor):
        """
        对传感器和关节数据应用数据增强
        
        Args:
            sensor_tensor: (128, 6) 传感器数据
            joint_tensor: (128, 7) 关节数据
            
        Returns:
            增强后的传感器和关节数据
        """
        # 1. 高斯噪声增强 (适用于传感器读数的微小随机变化)
        if torch.rand(1).item() < 0.7:  # 70%概率添加噪声
            noise_std = 0.02  # 2%的标准差
            sensor_noise = torch.randn_like(sensor_tensor) * noise_std
            joint_noise = torch.randn_like(joint_tensor) * noise_std * 0.5  # 关节数据噪声更小
            
            sensor_tensor = sensor_tensor + sensor_noise
            joint_tensor = joint_tensor + joint_noise
        
        # 2. 随机缩放 (模拟不同的力度和速度)
        if torch.rand(1).item() < 0.5:  # 50%概率进行缩放
            # 传感器数据缩放 (力/扭矩的强度变化)
            sensor_scale = 0.85 + torch.rand(1, 6) * 0.3  # 85%-115%缩放
            sensor_tensor = sensor_tensor * sensor_scale
            
            # 关节数据缩放 (运动速度的变化)
            joint_scale = 0.9 + torch.rand(1, 7) * 0.2  # 90%-110%缩放
            joint_tensor = joint_tensor * joint_scale
        
        # 3. 时间偏移 (模拟采样时间的微小变化)
        if torch.rand(1).item() < 0.3:  # 30%概率进行时间偏移
            max_shift = 3  # 最大偏移3个时间步
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            
            if shift > 0:  # 向前移动, 用第一个值填充开头
                sensor_tensor = torch.roll(sensor_tensor, shifts=shift, dims=0)
                sensor_tensor[:shift] = sensor_tensor[shift]
                joint_tensor = torch.roll(joint_tensor, shifts=shift, dims=0)
                joint_tensor[:shift] = joint_tensor[shift]
            elif shift < 0:  # 向后移动, 用最后一个值填充结尾
                sensor_tensor = torch.roll(sensor_tensor, shifts=shift, dims=0)
                sensor_tensor[shift:] = sensor_tensor[shift-1]
                joint_tensor = torch.roll(joint_tensor, shifts=shift, dims=0)
                joint_tensor[shift:] = joint_tensor[shift-1]
        
        # 4. 随机通道遮盖 (模拟传感器故障)
        if torch.rand(1).item() < 0.2:  # 20%概率进行通道遮盖
            # 随机选择1-2个传感器通道进行遮盖
            num_mask_channels = torch.randint(1, 3, (1,)).item()
            if sensor_tensor.size(1) > 1:
                mask_channels = torch.randperm(sensor_tensor.size(1))[:num_mask_channels]
                for channel in mask_channels:
                    channel_mean = torch.mean(sensor_tensor[:, channel])
                    sensor_tensor[:, channel] = channel_mean
        
        return sensor_tensor, joint_tensor

    def get_window_strategy(self, is_training=True):
        """
        根据训练/验证阶段返回窗口策略参数

        训练阶段使用无重叠窗口以提高训练效率和数据多样性
        验证阶段使用50%重叠窗口以获得更全面的评估覆盖

        Args:
            is_training (bool): True表示训练阶段，False表示验证阶段

        Returns:
            dict: 包含'stride'和'overlap'参数的字典
                - stride: 窗口滑动步长
                - overlap: 重叠比例 (0.0-1.0)
                - description: 策略描述
        """
        if is_training:
            # 训练阶段：无重叠，提高训练速度和数据多样性
            return {
                'stride': self.window_size,
                'overlap': 0.0,
                'description': '训练模式-无重叠窗口'
            }
        else:
            # 验证阶段：50%重叠，提供更全面的评估覆盖
            return {
                'stride': self.window_size // 2,
                'overlap': 0.5,
                'description': '验证模式-重叠窗口'
            }

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdaPerceptionLoss(nn.Module):
    """Ada感知模型的多任务损失函数 - 增强版支持Focal Loss"""

    def __init__(self, task_weights: Dict[str, float] = None, class_weights: Dict[str, torch.Tensor] = None, use_focal_loss: bool = True):
        super().__init__()

        self.task_weights = task_weights or {
            'slip_risk_assessment': 2.0,
            'object_position_3d': 1.5,
            'grasp_quality_score': 1.5,
            'object_material_type': 1.0,
            'food_recognition': 1.0
        }

        # 类别权重 - 解决数据不平衡问题
        self.class_weights = class_weights or {}
        self.use_focal_loss = use_focal_loss

        self.mse_loss = nn.MSELoss()

        # 为滑移检测使用加权BCE损失
        slip_weights = self.class_weights.get('slip', None)
        if slip_weights is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=slip_weights)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

        # 为材质分类使用Focal Loss或加权CE损失
        material_weights = self.class_weights.get('material', None)
        if self.use_focal_loss:
            self.material_loss = FocalLoss(alpha=1, gamma=2, weight=material_weights)
        else:
            if material_weights is not None:
                self.material_loss = nn.CrossEntropyLoss(weight=material_weights)
            else:
                self.material_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        # 确保total_loss是一个在正确设备上的tensor
        if not outputs:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True), {}
            
        # 从outputs中获取第一个实际的tensor来确定设备
        sample_tensor = None
        for key, value in outputs.items():
            if torch.is_tensor(value):
                sample_tensor = value
                break
            elif isinstance(value, dict):
                # 如果value是字典，从中找第一个tensor
                for sub_key, sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        sample_tensor = sub_value
                        break
                if sample_tensor is not None:
                    break
        
        if sample_tensor is None:
            # 如果找不到tensor，使用默认设备
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = sample_tensor.device
            
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_dict = {}
        
        # 处理模型输出结构 - 模型返回的是 {'task_outputs': {...}, ...}
        task_outputs = outputs.get('task_outputs', {})
        
        # 滑移风险评估
        if 'slip_risk_assessment' in task_outputs and 'slip_label' in targets:
            slip_loss = self.bce_loss(
                task_outputs['slip_risk_assessment'].squeeze(),
                targets['slip_label']
            )
            total_loss = total_loss + self.task_weights.get('slip_risk_assessment', 1.0) * slip_loss
            loss_dict['slip_loss'] = slip_loss.item()
        
        # 物体材质分类 (纹理) - 使用Focal Loss
        if 'object_material_type' in task_outputs and 'texture_label' in targets:
            material_loss = self.material_loss(
                task_outputs['object_material_type'],
                targets['texture_label']
            )
            total_loss = total_loss + self.task_weights.get('object_material_type', 1.0) * material_loss
            loss_dict['material_loss'] = material_loss.item()
        
        # 其他任务的损失可以在这里添加
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict

def freeze_early_layers(model, freeze_ratio=0.7):
    """
    冻结模型的早期层以减少过拟合
    
    Args:
        model: 要冻结的模型
        freeze_ratio: 冻结层的比例 (0.7 = 冻结前70%的层)
    """
    logger.info(f"正在冻结前 {freeze_ratio*100:.0f}% 的模型层...")
    
    # 获取所有可训练参数的层
    trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    
    # 计算要冻结的层数
    freeze_count = int(len(trainable_params) * freeze_ratio)
    
    # 冻结前 freeze_count 个参数
    frozen_count = 0
    for name, param in trainable_params[:freeze_count]:
        param.requires_grad = False
        frozen_count += 1
    
    # 统计冻结后的参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"冻结了 {frozen_count} 个参数组")
    logger.info(f"冻结前总参数: {total_params:,}")
    logger.info(f"冻结后可训练参数: {trainable_params_after:,}")
    logger.info(f"参数减少: {((total_params - trainable_params_after) / total_params * 100):.1f}%")

def compute_class_weights(dataset_path: str, device: torch.device):
    """计算类别权重以处理数据不平衡"""
    with h5py.File(dataset_path, 'r') as f:
        # 动态加载标签，兼容多种格式
        if 'trajectory_0' in f and f.attrs.get('total_trajectories', 0) > 0:
            # 格式0: 我们生成的轨迹格式
            total_trajectories = f.attrs.get('total_trajectories', 0)
            slip_labels = []
            texture_labels = []

            # 食物类型到纹理标签的映射
            food_to_texture = {
                'rice': 0, 'noodles': 1, 'soup': 2, 'chicken_cube': 3,
                'tofu': 4, 'broccoli': 5
            }

            # 采样轨迹进行分析，确保多样性
            sample_indices = np.linspace(0, total_trajectories-1, min(2000, total_trajectories), dtype=int)
            for i in sample_indices:
                traj_key = f'trajectory_{i}'
                if traj_key in f:
                    traj = f[traj_key]

                    # 获取滑移标签
                    if 'slip_labels' in traj:
                        slip_detected = traj['slip_labels']['detected'][:]
                        slip_labels.append(float(np.mean(slip_detected)))

                    # 获取食物类型并转换为纹理标签
                    food_type = traj.attrs.get('food_type', 'rice')
                    texture_label = food_to_texture.get(food_type, 0)
                    texture_labels.append(texture_label)

            slip_labels = np.array(slip_labels)
            texture_labels = np.array(texture_labels)

        elif 'observation' in f and 'slip_binary' in f:
            # 格式1: LeRobot多模态格式
            slip_labels = f['slip_binary'][:]
            texture_labels = f['texture_labels'][:]
        elif 'labels' in f and 'slip_binary' in f['labels']:
            # 格式2: 嵌套标签格式
            slip_labels = f['labels/slip_binary'][:]
            texture_labels = f['labels/texture_labels'][:]
        elif 'slip_label' in f:
            # 格式3: 根级标签格式（改进后的数据集）
            slip_labels = f['slip_label'][:]
            texture_labels = f['texture_label'][:]
        else:
            available_keys = list(f.keys())
            raise KeyError(f"无法找到可识别的滑移或纹理标签。可用键: {available_keys}")
    
    # 计算滑移标签的权重
    slip_counts = np.bincount(slip_labels.astype(int))
    slip_total = len(slip_labels)
    
    # 改进的权重计算 - 更平衡的权重分配
    if len(slip_counts) >= 2 and slip_counts[1] > 0:
        # 使用改进的权重计算方法
        slip_ratio = slip_counts[1] / slip_total
        # 为少数类设置权重，但不要过于极端
        slip_weights = slip_total / (2.0 * slip_counts[1]) if slip_counts[1] > 0 else 1.0
        slip_weights = min(slip_weights, 3.0)  # 限制最大权重为3.0
        slip_weight_tensor = torch.tensor(slip_weights, dtype=torch.float32, device=device)
    else:
        slip_weight_tensor = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    # 计算纹理标签的权重
    texture_counts = np.bincount(texture_labels.astype(int))
    texture_total = len(texture_labels)
    num_classes = len(texture_counts)
    
    # 改进的纹理权重计算
    if num_classes > 1:
        # 使用平滑的逆频率权重
        texture_weights = []
        for count in texture_counts:
            if count > 0:
                weight = texture_total / (num_classes * count)
                weight = min(weight, 2.0)  # 限制最大权重
                texture_weights.append(weight)
            else:
                texture_weights.append(1.0)
        texture_weight_tensor = torch.tensor(texture_weights, dtype=torch.float32, device=device)
    else:
        texture_weight_tensor = None
    
    logger.info(f"滑移检测权重 (正类): {slip_weight_tensor.item():.3f}")
    logger.info(f"纹理分类权重: {texture_weights if texture_weight_tensor is not None else 'None'}")
    
    return {
        'slip': slip_weight_tensor,
        'material': texture_weight_tensor
    }

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # 将数据移到设备
        visual_features = batch['visual_features'].to(device)
        tactile_features = batch['tactile_features'].to(device)
        force_features = batch['force_features'].to(device)
        proprioceptive_features = batch['proprioceptive_features'].to(device)
        
        targets = {
            'slip_label': batch['slip_label'].to(device),
            'texture_label': batch['texture_label'].to(device)
        }
        
        # 前向传播
        optimizer.zero_grad()
        
        outputs = model(
            visual_features=visual_features,
            tactile_features=tactile_features,
            force_features=force_features,
            proprioceptive_features=proprioceptive_features,
            return_attention=False,
            return_uncertainty=True
        )
        
        # 计算损失
        loss, loss_dict = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
        })
        
        # 记录到WandB (如果已初始化)
        if batch_idx % 100 == 0 and hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/slip_loss': loss_dict.get('slip_loss', 0),
                'train/material_loss': loss_dict.get('material_loss', 0),
                'epoch': epoch,
                'batch': batch_idx
            })
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device, epoch):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    correct_slip = 0
    correct_material = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # 将数据移到设备
            visual_features = batch['visual_features'].to(device)
            tactile_features = batch['tactile_features'].to(device)
            force_features = batch['force_features'].to(device)
            proprioceptive_features = batch['proprioceptive_features'].to(device)
            
            targets = {
                'slip_label': batch['slip_label'].to(device),
                'texture_label': batch['texture_label'].to(device)
            }
            
            # 前向传播
            outputs = model(
                visual_features=visual_features,
                tactile_features=tactile_features,
                force_features=force_features,
                proprioceptive_features=proprioceptive_features,
                return_attention=False,
                return_uncertainty=False
            )
            
            # 计算损失
            loss, loss_dict = criterion(outputs, targets)
            total_loss += loss.item()
            
            # 计算准确率
            if 'task_outputs' in outputs:
                task_outputs = outputs['task_outputs']
                
                # 滑移风险评估
                if 'slip_risk_assessment' in task_outputs:
                    slip_pred = torch.sigmoid(task_outputs['slip_risk_assessment']) > 0.5
                    correct_slip += (slip_pred.squeeze() == targets['slip_label']).sum().item()
                
                # 材质识别
                if 'object_material_type' in task_outputs:
                    material_pred = task_outputs['object_material_type'].argmax(dim=1)
                    correct_material += (material_pred == targets['texture_label']).sum().item()
            
            total_samples += visual_features.size(0)
    
    avg_loss = total_loss / len(dataloader)
    slip_acc = (correct_slip / total_samples * 100) if total_samples > 0 else 0  # 转换为百分比
    material_acc = (correct_material / total_samples * 100) if total_samples > 0 else 0  # 转换为百分比
    
    # 记录验证结果 (如果wandb已初始化)
    if hasattr(wandb, 'run') and wandb.run is not None:
        wandb.log({
            'val/loss': avg_loss,
            'val/slip_accuracy': slip_acc,  # 已经是百分比格式
            'val/material_accuracy': material_acc,  # 已经是百分比格式
            'epoch': epoch
        })
    
    logger.info(f"Validation - Loss: {avg_loss:.4f}, Slip Acc: {slip_acc:.2f}%, Material Acc: {material_acc:.2f}%")

    return avg_loss, slip_acc, material_acc

def main():
    parser = argparse.ArgumentParser(description="Ada LeRobot Enhanced Perception Training")
    parser.add_argument("--config", type=str, default="configs/ada_enhanced_config.yaml", help="配置文件路径")
    parser.add_argument("--dataset", type=str, help="指定数据集路径（可选）")
    parser.add_argument("--gpus", type=str, default="0", help="GPU设备")
    parser.add_argument("--wandb", action="store_true", help="启用WandB日志")
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"配置文件 {args.config} 不存在，使用默认配置")
        config = {
            'data': {
                'source_file': '/root/workspace/assistive_mvp/9_datasets/improved/improved_ada_dataset_20250727_210943.h5',
                'window_size': 128,
                'batch_size': 32,
                'num_workers': 4,
                'use_augmentation': True,
                'augment_prob': 0.3
            },
            'model': {
                'visual_dim': 512,
                'tactile_dim': 6,         # 修正：触觉数据实际是6维特征(来自传感器数据)
                'force_dim': 6,           # 力数据是6维(3力+3扭矩)
                'proprioceptive_dim': 7,  # 修正：关节数据是7维
                'd_model': 256,
                'num_layers': 6,
                'num_heads': 8,
                'dropout': 0.1
            },
            'training': {
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'epochs': 200,  # 增加到200轮，约1小时训练
                'patience': 25,  # 增加耐心值
                'loss_weights': {'slip': 1.0, 'texture': 1.0}
            },
            'experiment': {
                'name': 'ada_enhanced_improved_dataset',
                'output_dir': '/root/workspace/assistive_mvp/models'
            },
            'logging': {
                'use_wandb': False,
                'wandb_project': 'ada-perception'
            }
        }
    
    # 优先使用命令行指定的数据集路径
    if args.dataset:
        config['data']['source_file'] = args.dataset
        logger.info(f"使用命令行指定的数据集: {args.dataset}")
    
    # 数据集路径优先级检查 - 支持改进数据集
    improved_dataset_candidates = [
        config['data']['source_file'],  # 配置文件指定的路径
        "/root/workspace/assistive_mvp/9_datasets/improved/improved_ada_dataset_20250727_210943.h5",
        "/root/workspace/assistive_mvp/9_datasets/ada_data/ada_fully_labeled_dataset.h5"
    ]
    
    dataset_path = None
    for candidate in improved_dataset_candidates:
        if candidate and Path(candidate).exists():
            dataset_path = candidate
            break
    
    if not dataset_path:
        # 尝试查找最新的改进数据集
        improved_dir = Path("/root/workspace/assistive_mvp/9_datasets/improved")
        if improved_dir.exists():
            improved_files = list(improved_dir.glob("improved_ada_dataset_*.h5"))
            if improved_files:
                dataset_path = str(sorted(improved_files)[-1])  # 选择最新的
    
    if not dataset_path:
        logger.error("❌ 未找到可用的数据集文件")
        logger.info("请确保以下路径之一存在：")
        for path in improved_dataset_candidates:
            if path:
                logger.info(f"  - {path}")
        return
    
    # 更新配置中的数据集路径
    config['data']['source_file'] = dataset_path
    logger.info(f"✅ 使用数据集: {dataset_path}")
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 分析数据集质量
    logger.info("📊 分析数据集质量...")
    try:
        with h5py.File(dataset_path, 'r') as f:
            # 检测数据集格式
            if 'trajectory_0' in f and f.attrs.get('total_trajectories', 0) > 0:
                logger.info("✅ 检测到我们生成的10K轨迹格式")
                # 从轨迹中提取标签进行分析
                total_trajectories = f.attrs.get('total_trajectories', 0)
                slip_labels = []
                texture_labels = []

                # 食物类型到纹理标签的映射
                food_to_texture = {
                    'rice': 0, 'noodles': 1, 'soup': 2, 'chicken_cube': 3,
                    'tofu': 4, 'broccoli': 5
                }

                # 采样轨迹进行分析，确保多样性
                sample_indices = np.linspace(0, total_trajectories-1, min(1000, total_trajectories), dtype=int)
                for i in sample_indices:
                    traj_key = f'trajectory_{i}'
                    if traj_key in f:
                        traj = f[traj_key]

                        # 获取滑移标签
                        if 'slip_labels' in traj:
                            slip_detected = traj['slip_labels']['detected'][:]
                            slip_labels.append(float(np.mean(slip_detected)))

                        # 获取食物类型并转换为纹理标签
                        food_type = traj.attrs.get('food_type', 'rice')
                        texture_label = food_to_texture.get(food_type, 0)
                        texture_labels.append(texture_label)

                slip_labels = np.array(slip_labels)
                texture_labels = np.array(texture_labels)
                dataset_format = 'trajectory'

            elif 'observation' in f and 'slip_binary' in f:
                logger.info("✅ 检测到LeRobot多模态格式")
                slip_labels = f['slip_binary'][:]
                texture_labels = f['texture_labels'][:]
                dataset_format = 'lerobot'
            elif 'slip_label' in f and 'texture_label' in f:
                logger.info("✅ 检测到改进数据集格式（根级标签）")
                slip_labels = f['slip_label'][:]
                texture_labels = f['texture_label'][:]
                dataset_format = 'improved'
            elif 'labels' in f and 'slip_binary' in f['labels']:
                logger.info("✅ 检测到标准数据集格式（嵌套标签）")
                slip_labels = f['labels/slip_binary'][:]
                texture_labels = f['labels/texture_labels'][:]
                dataset_format = 'standard'
            else:
                available_keys = list(f.keys())
                logger.error(f"❌ 无法识别数据集格式。可用键: {available_keys}")
                return
            
            # 质量评估
            total_samples = len(slip_labels)
            slip_counts = np.bincount(slip_labels.astype(int))
            texture_counts = np.bincount(texture_labels.astype(int))
            
            slip_balance = min(slip_counts) / max(slip_counts) if len(slip_counts) >= 2 and max(slip_counts) > 0 else 0
            texture_balance = min(texture_counts) / max(texture_counts) if len(texture_counts) >= 2 and max(texture_counts) > 0 else 0
            
            logger.info(f"📈 数据集统计:")
            logger.info(f"  总样本数: {total_samples}")
            logger.info(f"  滑移分布: {dict(zip(range(len(slip_counts)), slip_counts))}")
            logger.info(f"  纹理分布: {dict(zip(range(len(texture_counts)), texture_counts))}")
            logger.info(f"  滑移平衡度: {slip_balance:.3f}")
            logger.info(f"  纹理平衡度: {texture_balance:.3f}")
            
            # 质量等级评估
            quality_score = 0
            if slip_balance >= 0.6: quality_score += 3
            elif slip_balance >= 0.4: quality_score += 2
            elif slip_balance >= 0.3: quality_score += 1
            
            if texture_balance >= 0.8: quality_score += 3
            elif texture_balance >= 0.6: quality_score += 2
            elif texture_balance >= 0.4: quality_score += 1
            
            if quality_score >= 5:
                logger.info("🏆 A级数据集质量 - 预期训练精度>90%")
                # A级数据集优化配置
                config['training'].update({
                    'learning_rate': 0.0005,  # 降低学习率
                    'batch_size': min(config['data'].get('batch_size', 32), 24),  # 减小批次
                    'epochs': max(config['training'].get('epochs', 50), 60),  # 增加训练轮数
                    'patience': 15
                })
                config['data']['augment_prob'] = 0.2  # 降低增强概率
            elif quality_score >= 3:
                logger.info("⚡ B级数据集质量 - 预期训练精度>85%")
                # B级数据集标准配置
            else:
                logger.info("⚠️ C级数据集质量 - 建议进一步改进数据")
                # C级数据集鲁棒配置
                config['training'].update({
                    'learning_rate': 0.002,   # 提高学习率
                    'weight_decay': 1e-3      # 增强正则化
                })
                config['data']['augment_prob'] = 0.5  # 增加增强概率
            
    except Exception as e:
        logger.error(f"数据集分析失败: {e}")
        return
    
    # 初始化WandB
    if args.wandb and config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['wandb_project'],
            name=config['experiment']['name'],
            config=config
        )
    
    # 创建输出目录
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据集
    train_dataset = AdaExpandedDataset(
        config['data']['source_file'], 
        window_size=config['data']['window_size'],
        split='train',
        augment=config['data'].get('use_augmentation', False),
        augment_prob=config['data'].get('augment_prob', 0.5)
    )
    val_dataset = AdaExpandedDataset(
        config['data']['source_file'],
        window_size=config['data']['window_size'], 
        split='val',
        augment=False  # 验证集不进行数据增强
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # 获取数据集的纹理类别数
    with h5py.File(config['data']['source_file'], 'r') as f:
        if 'texture_classes' in f.attrs:
            texture_classes = [s.decode('utf-8') for s in f.attrs['texture_classes']]
            num_texture_classes = len(texture_classes)
        else:
            # 从数据中推断类别数
            if 'trajectory_0' in f and f.attrs.get('total_trajectories', 0) > 0:
                # 轨迹格式：使用食物类型映射
                food_to_texture = {
                    'rice': 0, 'noodles': 1, 'soup': 2, 'chicken_cube': 3,
                    'tofu': 4, 'broccoli': 5
                }
                num_texture_classes = len(food_to_texture)
            elif 'observation' in f and 'slip_binary' in f:
                texture_labels = f['texture_labels'][:]
                num_texture_classes = len(np.unique(texture_labels))
            elif 'texture_label' in f:
                texture_labels = f['texture_label'][:]
                num_texture_classes = len(np.unique(texture_labels))
            elif 'labels' in f and 'texture_labels' in f['labels']:
                texture_labels = f['labels/texture_labels'][:]
                num_texture_classes = len(np.unique(texture_labels))
            else:
                # 默认值
                num_texture_classes = 6

    logger.info(f"检测到 {num_texture_classes} 个纹理类别")

    # 创建自定义任务配置，使用实际的纹理类别数
    custom_task_configs = {
        # 基础感知任务
        'object_position_3d': 3,
        'object_orientation': 4,
        'grasp_point_detection': 3,
        'grasp_quality_score': 1,
        'object_material_type': num_texture_classes,  # 使用实际的纹理类别数
        'object_temperature': 1,
        'object_weight': 1,
        'object_fragility': 1,
        'slip_risk_assessment': 1,
        'collision_detection': 1,

        # 动作控制任务
        'trajectory_planning': 7,
        'velocity_control': 7,
        'force_control': 3,
        'gripper_control': 2,
        'motion_smoothness': 1,
        'precision_requirement': 1,
        'safety_margin': 1,
        'emergency_stop_trigger': 1,
        'approach_strategy': 5,
        'retreat_strategy': 5,

        # 高级餐饮任务
        'food_recognition': 20,
        'portion_estimation': 1,
        'feeding_sequence': 10,
        'utensil_selection': 5,
        'serving_technique': 8,
        'spillage_prevention': 1,
        'user_preference': 10,
        'dietary_restriction': 15,
        'meal_timing': 1,
        'satisfaction_prediction': 1
    }

    # 创建模型，传入自定义任务配置
    model = AdaLeRobotEnhancedPerceptionModel(
        visual_dim=config['model']['visual_dim'],
        tactile_dim=config['model']['tactile_dim'],
        force_dim=config['model']['force_dim'],
        proprioceptive_dim=config['model']['proprioceptive_dim'],
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        task_configs=custom_task_configs  # 传入自定义任务配置
    ).to(device)
    
    # 应用层冻结以减少过拟合 (根据配置设置冻结比例)
    freeze_ratio = config.get('training', {}).get('progressive_unfreeze', {}).get('freeze_ratio', 0.7)
    freeze_early_layers(model, freeze_ratio=freeze_ratio)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    
    # 计算类别权重以处理数据不平衡
    logger.info("正在计算类别权重...")
    class_weights = compute_class_weights(config['data']['source_file'], device)
    
    # 创建优化器和损失函数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = AdaPerceptionLoss(
        task_weights=config['training']['loss_weights'],
        class_weights=class_weights,
        use_focal_loss=True  # 启用Focal Loss处理类别不平衡
    )
    
    # 从checkpoint恢复（如果指定）
    start_epoch = 0
    best_val_loss = float('inf')

    # 初始化材质准确率跟踪
    model.best_material_acc = 0.0

    if config['experiment'].get('resume_from_checkpoint'):
        checkpoint_path = config['experiment']['resume_from_checkpoint']
        if Path(checkpoint_path).exists():
            logger.info(f"从checkpoint恢复: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✅ 模型权重加载成功")
            
            # 加载优化器状态（可选，用于精调时可能不需要）
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("✅ 优化器状态加载成功")
            except:
                logger.info("⚠️ 优化器状态加载失败，使用新的优化器状态")
            
            # 记录之前的最佳损失和材质准确率
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            model.best_material_acc = checkpoint.get('val_texture_acc', 0.0)
            start_epoch = checkpoint.get('epoch', 0) + 1

            logger.info(f"从epoch {start_epoch}开始，之前最佳验证损失: {best_val_loss:.4f}, 材质准确率: {model.best_material_acc:.2f}%")
        else:
            logger.warning(f"Checkpoint文件不存在: {checkpoint_path}")
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['training']['epochs'],
        last_epoch=start_epoch-1 if start_epoch > 0 else -1
    )
    
    # 训练循环
    patience_counter = 0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"开始训练 Epoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # 验证
        val_loss, val_slip_acc, val_material_acc = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 记录到WandB (如果已初始化)
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({
                'train/epoch_loss': train_loss,
                'val/epoch_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch
            })
        
        # 保存最佳模型 - 改进策略：同时考虑损失和材质准确率
        save_model = False
        save_reason = ""

        # 策略1: 验证损失改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model = True
            save_reason = f"验证损失改善: {val_loss:.4f}"

        # 策略2: 材质准确率显著提升 (即使损失未改善)
        elif hasattr(model, 'best_material_acc'):
            if val_material_acc > model.best_material_acc + 1.0:  # 阈值1%（现在是百分比格式）
                model.best_material_acc = val_material_acc
                patience_counter = max(0, patience_counter - 50)  # 减少耐心计数
                save_model = True
                save_reason = f"材质准确率显著提升: {val_material_acc:.2f}%"
        else:
            model.best_material_acc = val_material_acc
            patience_counter += 1

        if save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'val_slip_acc': val_slip_acc,
                'val_texture_acc': val_material_acc,
                'config': config
            }, output_dir / f"best_model_epoch_{epoch}.pth")

            logger.info(f"保存最佳模型 (epoch {epoch}, {save_reason})")
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= config['training']['patience']:
            logger.info(f"早停触发 (patience: {config['training']['patience']})")
            break
    
    logger.info("训练完成!")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
