#!/usr/bin/env python3
"""
Ada LeRobot Enhanced Perception Model v2.0
增强版Ada机器人感知模型 - 10M参数版本
专为餐饮助手机器人设计的高精度多模态感知系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

class TemporalAggregator(nn.Module):
    """时序聚合模块，用于处理序列数据"""
    def __init__(self, method='mean'):
        super().__init__()
        self.method = method
    
    def forward(self, x):
        # 输入形状: (batch, seq_len, features)
        # 输出形状: (batch, features)
        if self.method == 'mean':
            return x.mean(dim=1)
        elif self.method == 'max':
            return x.max(dim=1)[0]  # max返回值和索引，我们只要值
        elif self.method == 'last':
            return x[:, -1]  # 取最后一个时间步
        else:
            return x.mean(dim=1)

class EnhancedMultiHeadAttention(nn.Module):
    """增强版多头注意力机制"""
    def __init__(self, d_model=256, num_heads=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 增加头数以获得更细粒度的注意力
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # 相对位置编码 (调整大小以适应更小的序列)
        max_relative_position = 32  # 减少相对位置范围
        self.relative_position_encoding = nn.Parameter(
            torch.randn(2 * max_relative_position - 1, self.head_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # 多头投影
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 添加相对位置编码 (只有在序列长度合适时)
        if seq_len <= 64:  # 限制相对位置编码的使用范围
            relative_scores = self._get_relative_position_scores(seq_len)
            # 调整维度以匹配scores
            relative_scores = relative_scores.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len, head_dim]
            # 取平均或求和来减少维度
            relative_scores = relative_scores.mean(dim=-1)  # [1, 1, seq_len, seq_len]
            scores = scores + relative_scores
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.output_projection(out), attention_weights
    
    def _get_relative_position_scores(self, seq_len):
        """计算相对位置得分"""
        positions = torch.arange(seq_len, device=self.relative_position_encoding.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # 调整到正数范围
        max_relative_position = (self.relative_position_encoding.size(0) + 1) // 2
        relative_positions = relative_positions + max_relative_position - 1
        
        # 确保索引在有效范围内
        max_pos = self.relative_position_encoding.size(0) - 1
        relative_positions = torch.clamp(relative_positions, 0, max_pos)
        
        relative_scores = self.relative_position_encoding[relative_positions]
        return relative_scores

class AdvancedFeedForward(nn.Module):
    """增强版前馈网络"""
    def __init__(self, d_model=256, d_ff=1024, dropout=0.1, activation='gelu'):
        super().__init__()
        
        # 增加网络深度和宽度
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_ff // 2)
        self.linear3 = nn.Linear(d_ff // 2, d_model)
        
        # 多种激活函数选择
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 门控机制
        self.gate = nn.Linear(d_model, d_ff)
        
    def forward(self, x):
        residual = x
        
        # 门控前馈网络
        gate_values = torch.sigmoid(self.gate(x))
        
        # 三层前馈
        out = self.linear1(x)
        out = out * gate_values  # 门控
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear3(out)
        out = self.dropout(out)
        
        # 残差连接和层归一化
        return self.layer_norm(residual + out)

class MultiModalFusion(nn.Module):
    """高级多模态融合模块"""
    def __init__(self, 
                 visual_dim=512, 
                 tactile_dim=128, 
                 force_dim=64, 
                 proprioceptive_dim=32,
                 output_dim=256):
        super().__init__()
        
        # 各模态特征提取器
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )
        
        self.tactile_encoder = nn.Sequential(
            # 处理时序触觉数据：(batch, seq_len, features) -> (batch, features)
            TemporalAggregator(method='mean'),  # 对时间维度求平均
            nn.Linear(tactile_dim, 128),  # tactile_dim现在是特征数量
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        
        self.force_encoder = nn.Sequential(
            # 处理时序力数据：(batch, seq_len, force_features) -> (batch, force_features)
            TemporalAggregator(method='mean'),  # 对时间维度求平均
            nn.Linear(force_dim, 64),  # force_dim是力的特征数量(6)
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
        
        self.proprioceptive_encoder = nn.Sequential(
            # 处理时序关节数据：(batch, seq_len, joint_features) -> (batch, joint_features)
            TemporalAggregator(method='mean'),  # 对时间维度求平均
            nn.Linear(proprioceptive_dim, 64),  # proprioceptive_dim是关节特征数量(7)
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
        
        # 跨模态注意力
        self.cross_modal_attention = EnhancedMultiHeadAttention(
            d_model=output_dim, 
            num_heads=8
        )
        
        # 模态权重学习
        self.modality_weights = nn.Parameter(torch.ones(4))
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(output_dim * 4, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, 
                visual_features, 
                tactile_features, 
                force_features, 
                proprioceptive_features):
        
        # 各模态编码
        visual_encoded = self.visual_encoder(visual_features)
        tactile_encoded = self.tactile_encoder(tactile_features)
        force_encoded = self.force_encoder(force_features)
        proprioceptive_encoded = self.proprioceptive_encoder(proprioceptive_features)
        
        # 堆叠所有模态特征
        all_modalities = torch.stack([
            visual_encoded,
            tactile_encoded,
            force_encoded,
            proprioceptive_encoded
        ], dim=1)  # [batch, 4, output_dim]
        
        # 跨模态注意力
        attended_features, attention_weights = self.cross_modal_attention(all_modalities)
        
        # 学习的模态权重
        modality_weights = F.softmax(self.modality_weights, dim=0)
        weighted_features = attended_features * modality_weights.view(1, 4, 1)
        
        # 融合所有模态
        concatenated = weighted_features.flatten(1)  # [batch, 4*output_dim]
        fused_features = self.fusion_network(concatenated)
        
        return fused_features, attention_weights, modality_weights

class EnhancedTaskHead(nn.Module):
    """增强版任务头 - 支持30个输出任务"""
    def __init__(self, input_dim=256, task_configs=None):
        super().__init__()
        
        if task_configs is None:
            # 30个餐饮相关任务配置
            task_configs = {
                # 基础感知任务 (1-10)
                'object_position_3d': 3,           # 物体3D位置
                'object_orientation': 4,           # 物体姿态(四元数)
                'grasp_point_detection': 3,        # 抓取点检测
                'grasp_quality_score': 1,          # 抓取质量评分
                'object_material_type': 8,         # 材质分类(8种喂食纹理) - 扩展到8个喂食场景类别
                'object_temperature': 1,           # 温度估计
                'object_weight': 1,                # 重量预测
                'object_fragility': 1,             # 脆性程度
                'slip_risk_assessment': 1,         # 滑移风险
                'collision_detection': 1,          # 碰撞检测
                
                # 动作控制任务 (11-20)
                'trajectory_planning': 7,          # 轨迹规划(7DOF)
                'velocity_control': 7,             # 速度控制
                'force_control': 3,                # 力控制(3轴)
                'gripper_control': 2,              # 夹爪控制(开合+力度)
                'motion_smoothness': 1,            # 运动平滑度
                'precision_requirement': 1,        # 精度要求
                'safety_margin': 1,                # 安全边距
                'emergency_stop_trigger': 1,       # 紧急停止
                'approach_strategy': 5,            # 接近策略(5种)
                'retreat_strategy': 5,             # 撤退策略(5种)
                
                # 高级餐饮任务 (21-30)
                'food_recognition': 20,            # 食物识别(20类)
                'portion_estimation': 1,           # 分量估计
                'feeding_sequence': 10,            # 进食序列(10步)
                'utensil_selection': 5,            # 餐具选择(5种)
                'serving_technique': 8,            # 服务技巧(8种)
                'spillage_prevention': 1,          # 防溅策略
                'user_preference': 10,             # 用户偏好(10维)
                'dietary_restriction': 15,         # 饮食限制(15种)
                'meal_timing': 1,                  # 用餐时机
                'satisfaction_prediction': 1       # 满意度预测
            }
        
        self.task_configs = task_configs
        self.num_tasks = len(task_configs)
        
        # 8个喂食场景纹理类别映射
        self.feeding_texture_classes = {
            0: 'liquid_food',      # 液体食物 (如汤、粥、果汁)
            1: 'solid_food',       # 固体食物 (如饼干、面包、苹果)
            2: 'soft_food',        # 软质食物 (如香蕉、蒸蛋、豆腐)
            3: 'hard_food',        # 硬质食物 (如坚果、胡萝卜、饼干)
            4: 'sticky_food',      # 粘性食物 (如蜂蜜、酸奶、果酱)
            5: 'slippery_food',    # 光滑易滑食物 (如果冻、布丁、香蕉)
            6: 'crumbly_food',     # 易碎食物 (如饼干屑、蛋糕、酥脆零食)
            7: 'wet_food'          # 湿润食物 (如水果、汤品、湿润蛋糕)
        }
        
        # 共享特征提取器
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 为每个任务创建专用头
        self.task_heads = nn.ModuleDict()
        for task_name, output_dim in task_configs.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(384, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, output_dim)
            )
        
        # 任务间依赖关系建模
        self.task_interaction = nn.MultiheadAttention(
            embed_dim=384, 
            num_heads=12, 
            batch_first=True
        )
        
    def forward(self, x):
        # 共享特征提取
        shared_features = self.shared_encoder(x)  # [batch, 384]
        
        # 为任务间交互准备特征
        batch_size = shared_features.size(0)
        task_features = shared_features.unsqueeze(1).repeat(1, self.num_tasks, 1)
        
        # 任务间注意力
        attended_features, _ = self.task_interaction(
            task_features, task_features, task_features
        )
        
        # 各任务预测
        task_outputs = {}
        for i, (task_name, head) in enumerate(self.task_heads.items()):
            task_specific_features = attended_features[:, i, :]
            task_outputs[task_name] = head(task_specific_features)
        
        return task_outputs

class UncertaintyQuantification(nn.Module):
    """增强版不确定性量化模块"""
    def __init__(self, input_dim=256, num_monte_carlo=10):
        super().__init__()
        self.num_monte_carlo = num_monte_carlo
        
        # 贝叶斯神经网络层
        self.bayesian_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Dropout(0.2),  # 用于Monte Carlo Dropout
                nn.GELU(),
                nn.Linear(128, 64),
                nn.Dropout(0.2),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(5)  # 5个不同的不确定性估计器
        ])
        
        # 不确定性融合
        self.uncertainty_fusion = nn.Sequential(
            nn.Linear(5, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, training=True):
        if training:
            # 训练时使用Monte Carlo Dropout
            uncertainties = []
            for layer in self.bayesian_layers:
                mc_samples = []
                for _ in range(self.num_monte_carlo):
                    mc_samples.append(layer(x))
                
                # 计算方差作为不确定性
                mc_tensor = torch.stack(mc_samples, dim=0)
                uncertainty = torch.var(mc_tensor, dim=0)
                uncertainties.append(uncertainty)
            
            uncertainties = torch.cat(uncertainties, dim=-1)
            
        else:
            # 推理时直接计算
            uncertainties = []
            for layer in self.bayesian_layers:
                uncertainty = layer(x)
                uncertainties.append(uncertainty)
            uncertainties = torch.cat(uncertainties, dim=-1)
        
        # 融合不确定性
        final_uncertainty = self.uncertainty_fusion(uncertainties)
        
        return final_uncertainty

class AdaLeRobotEnhancedPerceptionModel(nn.Module):
    """
    Ada LeRobot增强版感知模型 - 10M参数版本
    
    增强特性:
    - 30个输出任务 (从18个扩展)
    - 更深的Transformer架构 (8层 -> 12层)
    - 增强的多模态融合
    - 更精细的注意力机制
    - 高级不确定性量化
    - 任务间依赖关系建模
    """
    
    def __init__(self,
                 visual_dim=512,
                 tactile_dim=128,
                 force_dim=64,
                 proprioceptive_dim=32,
                 d_model=256,
                 num_layers=12,  # 增加到12层
                 num_heads=16,   # 增加头数
                 dropout=0.1,
                 task_configs=None):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 多模态融合模块
        self.multimodal_fusion = MultiModalFusion(
            visual_dim=visual_dim,
            tactile_dim=tactile_dim,
            force_dim=force_dim,
            proprioceptive_dim=proprioceptive_dim,
            output_dim=d_model
        )
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, 512, d_model))
        
        # 增强的Transformer层
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': EnhancedMultiHeadAttention(
                    d_model=d_model, 
                    num_heads=num_heads, 
                    dropout=dropout
                ),
                'feedforward': AdvancedFeedForward(
                    d_model=d_model, 
                    d_ff=d_model*4, 
                    dropout=dropout,
                    activation='gelu'
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        # 增强任务头 (30个任务)
        self.task_head = EnhancedTaskHead(input_dim=d_model, task_configs=task_configs)
        
        # 不确定性量化
        self.uncertainty_quantification = UncertaintyQuantification(
            input_dim=d_model, 
            num_monte_carlo=10
        )
        
        # 全局特征聚合
        self.global_aggregation = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                visual_features,
                tactile_features, 
                force_features,
                proprioceptive_features,
                return_attention=False,
                return_uncertainty=True):
        
        batch_size = visual_features.size(0)
        
        # 多模态融合
        fused_features, modal_attention, modal_weights = self.multimodal_fusion(
            visual_features, tactile_features, force_features, proprioceptive_features
        )
        
        # 添加维度以适应Transformer
        x = fused_features.unsqueeze(1)  # [batch, 1, d_model]
        
        # 添加位置编码
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Transformer处理
        attention_weights = []
        for layer in self.transformer_layers:
            # 多头注意力
            attn_out, attn_weights = layer['attention'](x)
            x = layer['norm1'](x + attn_out)
            
            # 前馈网络
            ff_out = layer['feedforward'](x)
            x = layer['norm2'](x + ff_out)
            
            if return_attention:
                attention_weights.append(attn_weights)
        
        # 全局特征聚合
        global_features = self.global_aggregation(x.transpose(1, 2))
        
        # 任务预测
        task_outputs = self.task_head(global_features)
        
        # 不确定性量化
        uncertainty = None
        if return_uncertainty:
            uncertainty = self.uncertainty_quantification(
                global_features, 
                training=self.training
            )
        
        outputs = {
            'task_outputs': task_outputs,
            'global_features': global_features,
            'modal_attention': modal_attention,
            'modal_weights': modal_weights
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            
        if return_uncertainty:
            outputs['uncertainty'] = uncertainty
        
        return outputs
    
    def get_model_stats(self):
        """获取模型统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 各组件参数量
        component_params = {
            'multimodal_fusion': sum(p.numel() for p in self.multimodal_fusion.parameters()),
            'transformer_layers': sum(p.numel() for p in self.transformer_layers.parameters()),
            'task_head': sum(p.numel() for p in self.task_head.parameters()),
            'uncertainty_quantification': sum(p.numel() for p in self.uncertainty_quantification.parameters()),
            'global_aggregation': sum(p.numel() for p in self.global_aggregation.parameters())
        }
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'component_parameters': component_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'num_tasks': len(self.task_head.task_configs),
            'transformer_layers': self.num_layers,
            'model_dimension': self.d_model
        }

def create_enhanced_ada_lerobot_model():
    """创建增强版Ada LeRobot感知模型"""
    model = AdaLeRobotEnhancedPerceptionModel(
        visual_dim=512,      # 视觉特征维度
        tactile_dim=128,     # 触觉特征维度  
        force_dim=64,        # 力觉特征维度
        proprioceptive_dim=32, # 本体感觉维度
        d_model=256,         # 模型维度
        num_layers=12,       # Transformer层数
        num_heads=16,        # 注意力头数
        dropout=0.1
    )
    
    return model

def demo_enhanced_model():
    """演示增强版模型"""
    print("🚀 Ada LeRobot增强版感知模型 v2.0")
    print("=" * 60)
    
    # 创建模型
    model = create_enhanced_ada_lerobot_model()
    model.eval()
    
    # 获取模型统计信息
    stats = model.get_model_stats()
    
    print(f"📊 模型规模统计:")
    print(f"  总参数量: {stats['total_parameters']:,} ({stats['total_parameters']/1000000:.2f}M)")
    print(f"  可训练参数: {stats['trainable_parameters']:,}")
    print(f"  模型大小: {stats['model_size_mb']:.2f} MB")
    print(f"  输出任务数: {stats['num_tasks']}")
    print(f"  Transformer层数: {stats['transformer_layers']}")
    print(f"  模型维度: {stats['model_dimension']}")
    
    print(f"\n🔧 各组件参数分布:")
    for component, params in stats['component_parameters'].items():
        percentage = (params / stats['total_parameters']) * 100
        print(f"  {component}: {params:,} ({percentage:.1f}%)")
    
    # 模拟输入
    batch_size = 4
    visual_features = torch.randn(batch_size, 512)
    tactile_features = torch.randn(batch_size, 128)
    force_features = torch.randn(batch_size, 64)
    proprioceptive_features = torch.randn(batch_size, 32)
    
    print(f"\n🧪 模型推理测试:")
    print(f"  批次大小: {batch_size}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(
            visual_features=visual_features,
            tactile_features=tactile_features,
            force_features=force_features,
            proprioceptive_features=proprioceptive_features,
            return_attention=True,
            return_uncertainty=True
        )
    
    print(f"✅ 推理成功!")
    print(f"  任务输出: {len(outputs['task_outputs'])} 个任务")
    print(f"  全局特征维度: {outputs['global_features'].shape}")
    print(f"  模态注意力权重: {outputs['modal_weights']}")
    print(f"  不确定性得分: {outputs['uncertainty'].mean().item():.4f}")
    
    # 显示30个任务的输出
    print(f"\n📋 30个餐饮任务输出:")
    for i, (task_name, task_output) in enumerate(outputs['task_outputs'].items(), 1):
        print(f"  {i:2d}. {task_name}: {task_output.shape[1]} 维输出")
    
    print(f"\n🎯 模型优势:")
    print(f"  ✅ 参数量提升到 {stats['total_parameters']/1000000:.1f}M")
    print(f"  ✅ 30个餐饮专业任务")
    print(f"  ✅ 4种模态深度融合")
    print(f"  ✅ 12层Transformer架构")
    print(f"  ✅ 高级不确定性量化")
    print(f"  ✅ 任务间依赖关系建模")
    
    return model, stats

if __name__ == "__main__":
    model, stats = demo_enhanced_model()
