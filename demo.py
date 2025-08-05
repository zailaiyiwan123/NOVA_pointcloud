#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云生成可视化工具 - 真实点云 vs 生成点云对比
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse
import logging
import random
from diffnext.pipelines.nova.pipeline_nova_pointcloud_gen import NOVAPointCloudGenerationPipeline
from diffnext.schedulers.scheduling_ddpm import DDPMScheduler
# 从train3.py导入AdvancedShapeNetDataset
import sys
sys.path.append('.')
from train_newloss import AdvancedShapeNetDataset
# 添加CD和EMD计算所需的库
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 类别映射
CLASS_NAME_TO_ID = {
    'airplane': '02691156',
    'car': '02958343', 
    'chair': '03001627',
}

# ====== CD和EMD计算函数 ======
def chamfer_distance(points1, points2):
    """
    计算Chamfer Distance (CD)
    points1, points2: numpy arrays of shape (N, 3) and (M, 3)
    """
    # 计算所有点对之间的距离矩阵
    dist_matrix = cdist(points1, points2)
    
    # 从points1到points2的最小距离
    min_dist_1_to_2 = np.min(dist_matrix, axis=1)
    
    # 从points2到points1的最小距离
    min_dist_2_to_1 = np.min(dist_matrix, axis=0)
    
    # CD = 平均(从1到2的最小距离) + 平均(从2到1的最小距离)
    cd = np.mean(min_dist_1_to_2) + np.mean(min_dist_2_to_1)
    
    return cd

def earth_mover_distance(points1, points2):
    """
    计算Earth Mover's Distance (EMD)
    points1, points2: numpy arrays of shape (N, 3) and (M, 3)
    """
    # 计算距离矩阵
    dist_matrix = cdist(points1, points2)
    
    # 使用匈牙利算法求解最优匹配
    row_indices, col_indices = linear_sum_assignment(dist_matrix)
    
    # 计算EMD
    emd = dist_matrix[row_indices, col_indices].sum()
    
    # 归一化到平均距离
    emd = emd / len(row_indices)
    
    return emd

def compute_metrics(real_points, gen_points):
    """
    计算CD和EMD指标
    """
    # 确保点云点数一致
    if real_points.shape[0] != gen_points.shape[0]:
        if gen_points.shape[0] > real_points.shape[0]:
            # 随机采样
            indices = np.random.choice(gen_points.shape[0], real_points.shape[0], replace=False)
            gen_points = gen_points[indices]
        else:
            # 重复填充
            repeat_times = real_points.shape[0] // gen_points.shape[0] + 1
            gen_points = np.tile(gen_points, (repeat_times, 1))
            gen_points = gen_points[:real_points.shape[0]]
    
    # 计算CD
    cd = chamfer_distance(real_points, gen_points)
    
    # 计算EMD
    emd = earth_mover_distance(real_points, gen_points)
    
    return cd, emd

# ====== 模型组件定义 ======
class ImprovedNOVAPointCloudTransformer(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.patch_size = 16
        self.embed_dim = 768
        self.num_heads = 12
        self.num_layers = 8  # 修复：改为8层，与训练时一致
        self.dropout = 0.1
        self.dtype = torch.float32
        self.device = torch.device('cpu')
        
        # 点云嵌入层
        self.point_embed = torch.nn.Linear(3, self.embed_dim)
        torch.nn.init.xavier_uniform_(self.point_embed.weight)
        torch.nn.init.zeros_(self.point_embed.bias)
        
        # 位置编码 - 修复：改为2048，与训练时一致
        self.pos_embed = torch.nn.Parameter(torch.randn(1, 2048, self.embed_dim) * 0.02)
        
        # Transformer层
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,  # 修复：改为3072，与训练时一致
            dropout=self.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, self.num_layers)
        
        # 输出层
        self.output_proj = torch.nn.Linear(self.embed_dim, 3)
        torch.nn.init.xavier_uniform_(self.output_proj.weight)
        torch.nn.init.zeros_(self.output_proj.bias)
        
        # 时间嵌入
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, self.embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim)
        )
        for layer in self.time_embed:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        
        # 文本嵌入
        self.text_embed = torch.nn.Sequential(
            torch.nn.Linear(768, self.embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim)
        )
        for layer in self.text_embed:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x, timestep, encoder_hidden_states=None, return_dict=True):
        batch_size = x.shape[0]
        x = x.contiguous()
        
        # 点云标准化处理 - 修复：改为2048，与训练时一致
        if x.shape[1] > 2048:
            indices = torch.randperm(x.shape[1])[:2048]
            x = x[:, indices, :]
        elif x.shape[1] < 2048:
            repeat_times = 2048 // x.shape[1] + 1
            x = x.repeat(1, repeat_times, 1)
            x = x[:, :2048, :]
        
        # 点云嵌入
        x = self.point_embed(x)
        
        # 位置编码
        if x.shape[1] <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :x.shape[1], :]
        else:
            x = x + self.pos_embed[:, :, :]
        
        # 时间嵌入
        timestep_emb = self.time_embed(timestep.unsqueeze(-1).float())
        x = x + timestep_emb.unsqueeze(1)
        
        # 文本嵌入（如果提供）
        if encoder_hidden_states is not None:
            text_emb = self.text_embed(encoder_hidden_states)
            # 处理文本嵌入的维度匹配问题
            if text_emb.shape[1] != x.shape[1]:
                # 如果文本序列长度与点云序列长度不匹配，进行插值或重复
                if text_emb.shape[1] < x.shape[1]:
                    # 文本序列较短，进行插值扩展
                    text_emb = text_emb.transpose(1, 2)  # [B, H, L]
                    text_emb = torch.nn.functional.interpolate(
                        text_emb, size=x.shape[1], mode='linear', align_corners=False
                    )
                    text_emb = text_emb.transpose(1, 2)  # [B, L, H]
                else:
                    # 文本序列较长，截断到点云序列长度
                    text_emb = text_emb[:, :x.shape[1], :]
            x = x + text_emb
        
        # Transformer处理
        x = self.transformer(x)
        
        # 输出投影
        x = self.output_proj(x)
        x = x.view(batch_size, -1, 3)
        
        return {'sample': x}

class ImprovedTokenizer:
    """改进的Tokenizer，支持真实文本处理"""
    def __init__(self, vocab_size=1000, max_length=256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
    def __call__(self, text, **kwargs):
        class TokenOutput:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        
        # 简单的文本到token的映射
        if isinstance(text, str):
            text = [text]
        
        input_ids_list = []
        for t in text:
            # 简单的基于字符的tokenization
            tokens = t.lower().split()
            ids = [hash(token) % self.vocab_size for token in tokens]
            # 截断或填充到max_length
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
            else:
                ids = ids + [0] * (self.max_length - len(ids))
            input_ids_list.append(ids)
        
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        return TokenOutput(input_ids)

class ImprovedTextEncoder(torch.nn.Module):
    """改进的TextEncoder，生成有意义的文本嵌入"""
    def __init__(self, vocab_size=1000, hidden_size=768, max_length=256):
        super().__init__()
        self.dtype = torch.float32
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # 词嵌入层
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # 位置编码
        self.pos_embedding = torch.nn.Embedding(max_length, hidden_size)
        torch.nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        
        # Transformer编码器
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=12,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, 6)
        
        # 输出投影
        self.output_proj = torch.nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.output_proj.weight)
        torch.nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        embeddings = self.embedding(input_ids)
        
        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_embedding(positions)
        
        # 组合嵌入
        x = embeddings + pos_embeddings
        
        # Transformer编码
        x = self.transformer(x)
        
        # 输出投影
        x = self.output_proj(x)
        
        return x

def load_model(model_path, device):
    """加载训练好的模型"""
    logger.info(f"加载模型: {model_path}")
    
    # 创建模型组件
    transformer = ImprovedNOVAPointCloudTransformer()
    tokenizer = ImprovedTokenizer()
    text_encoder = ImprovedTextEncoder()
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # 加载权重
    transformer.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建完整pipeline
    pipeline = NOVAPointCloudGenerationPipeline(
        transformer=transformer,
        scheduler=DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2"
        ),
        tokenizer=tokenizer,
        text_encoder=text_encoder
    )
    
    pipeline.to(device)
    pipeline.transformer.eval()
    
    logger.info("模型加载完成")
    return pipeline

# ====== 全局归一化器 BEGIN ======
import json
class GlobalNormalizer:
    """全局归一化器 - 使用训练时保存的统计参数"""
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.is_fitted = False
    
    def load_stats(self, filepath='stats.json'):
        """从stats.json文件加载训练时保存的统计参数"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    stats = json.load(f)
                self.global_mean = torch.tensor(stats['mean'])
                self.global_std = torch.tensor(stats['std'])
                self.is_fitted = True
                logger.info(f"从{filepath}加载归一化参数: mean={self.global_mean}, std={self.global_std}")
                return True
            else:
                logger.warning(f"找不到{filepath}文件，使用默认归一化参数")
                self.global_mean = torch.zeros(3)
                self.global_std = torch.ones(3)
                self.is_fitted = True
                return False
        except Exception as e:
            logger.error(f"加载归一化参数失败: {e}")
            self.global_mean = torch.zeros(3)
            self.global_std = torch.ones(3)
            self.is_fitted = True
            return False
    
    def fit(self, dataset):
        """兼容性方法 - 实际使用load_stats"""
        return self.load_stats()
    
    def __call__(self, points, mode='norm'):
        """归一化或反归一化"""
        if not self.is_fitted:
            logger.warning("归一化器未初始化，尝试加载stats.json")
            self.load_stats()
        
        if mode == 'norm':
            # 归一化: (x - mean) / std
            return (points - self.global_mean.to(points.device)) / self.global_std.to(points.device)
        else:
            # 反归一化: x * std + mean
            return points * self.global_std.to(points.device) + self.global_mean.to(points.device)

# 全局归一化器实例
global_normalizer = GlobalNormalizer()
# ====== 全局归一化器 END ======

def stable_generate_pointcloud(pipeline, text, num_points=2048, device="cuda", num_steps=50):
    """
    抗崩溃的生成算法 - 解决误差累积问题
    分阶段生成：低频结构 -> 高频细节 -> 后处理修复
    """
    # 生成文本嵌入
    with torch.no_grad():
        input_ids = pipeline.tokenizer(text).input_ids.to(device)
        text_embeddings = pipeline.text_encoder(input_ids)
    
    # 阶段1：低频结构生成
    low_freq_steps = num_steps // 2
    noise = torch.randn(1, num_points, 3, device=device)
    
    pipeline.scheduler.set_timesteps(low_freq_steps)
    
    for t in pipeline.scheduler.timesteps:
        with torch.no_grad():
            model_output = pipeline.transformer(
                noise, 
                torch.tensor([t]*1, device=device),
                encoder_hidden_states=text_embeddings
            )['sample']
            noise = pipeline.scheduler.step(model_output, t, noise).prev_sample
    
    # 阶段2：高频细节增强
    high_freq_steps = num_steps - low_freq_steps
    pipeline.scheduler.set_timesteps(high_freq_steps)
    
    for t in pipeline.scheduler.timesteps:
        with torch.no_grad():
            model_output = pipeline.transformer(
                noise, 
                torch.tensor([t]*1, device=device),
                encoder_hidden_states=text_embeddings
            )['sample']
            noise = pipeline.scheduler.step(model_output, t, noise).prev_sample
    
    # 阶段3：后处理修复
    generated_points = noise
    
    # 应用反归一化
    generated_points = global_normalizer(generated_points, 'denorm')
    
    # 点云质量修复
    generated_points = fix_pointcloud_topology(generated_points)
    
    return generated_points.cpu().numpy()[0]

def fix_pointcloud_topology(points):
    """修复点云拓扑结构"""
    # 1. 移除异常点
    mean = points.mean(dim=1, keepdim=True)
    std = points.std(dim=1, keepdim=True)
    mask = torch.abs(points - mean) < 3 * std
    points = points * mask.float()
    
    # 2. 中心化
    centroid = points.mean(dim=1, keepdim=True)
    points = points - centroid
    
    # 3. 归一化到合理范围
    max_dist = torch.norm(points, dim=-1).max(dim=1, keepdim=True)[0]
    points = points / (max_dist + 1e-6)
    
    return points

def visualize_comparison(real_points, gen_points, title, save_path=None):
    """
    完全优化版点云可视化对比 - 解决看不清、点太小问题
    """
    # 1. 创建大尺寸图表
    fig = plt.figure(figsize=(20, 10), dpi=120)
    
    # 2. 真实点云 - 使用深度着色增强3D效果
    ax1 = fig.add_subplot(121, projection='3d')
    # 使用深度值(z轴)着色
    colors_real = real_points[:, 2]
    sc1 = ax1.scatter(
        real_points[:, 0], real_points[:, 1], real_points[:, 2],
        s=100,  # 大幅增加点大小
        alpha=1.0,  # 完全去除透明度
        c=colors_real, 
        cmap='jet',  # 高对比度colormap
        depthshade=True,  # 启用3D深度阴影
        edgecolor='k',  # 黑色边缘增强轮廓
        linewidth=0.5  # 边缘线加粗
    )
    ax1.set_title('真实点云', fontsize=16, fontweight='bold', pad=20)
    
    # 3. 生成点云 - 使用热力图配色增强可读性
    ax2 = fig.add_subplot(122, projection='3d')
    colors_gen = np.linalg.norm(gen_points, axis=1)  # 计算距离原点的距离
    sc2 = ax2.scatter(
        gen_points[:, 0], gen_points[:, 1], gen_points[:, 2],
        s=100,  # 大幅增加点大小
        alpha=1.0, 
        c=colors_gen,
        cmap='hot',  # 高对比度热力图
        depthshade=True,
        edgecolor='k',
        linewidth=0.5
    )
    ax2.set_title('生成点云', fontsize=16, fontweight='bold', pad=20)
    
    # 4. 智能坐标范围设置
    def set_smart_limits(ax, points):
        """动态计算最佳坐标范围"""
        # 计算中点和范围
        center = np.mean(points, axis=0)
        max_range = max(
            np.ptp(points[:, 0]),
            np.ptp(points[:, 1]),
            np.ptp(points[:, 2]),
            1.0  # 确保最小范围
        ) * 1.2  # 增加20%余量
        
        # 设置坐标范围
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
        
        # 设置美观视角
        ax.view_init(elev=30, azim=45)
        
        # 添加网格增强3D感
        ax.grid(True, alpha=0.3)
        
        # 添加坐标轴标签
        ax.set_xlabel('X', fontweight='bold', labelpad=15)
        ax.set_ylabel('Y', fontweight='bold', labelpad=15)
        ax.set_zlabel('Z', fontweight='bold', labelpad=15)
    
    # 应用智能坐标限制
    set_smart_limits(ax1, real_points)
    set_smart_limits(ax2, gen_points)
    
    # 5. 添加色标提供深度参考
    fig.colorbar(sc1, ax=ax1, shrink=0.6, pad=0.1, label='深度(Z)', location='left')
    fig.colorbar(sc2, ax=ax2, shrink=0.6, pad=0.1, label='距离中心', location='right')
    
    # 6. 添加标题并保存
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.96)
    
    if save_path:
        # 高质量保存
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"优化可视化结果已保存: {save_path}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# 在generate_pointcloud函数中添加后处理确保点云范围合理
def generate_pointcloud(pipeline, text, num_points=2048, device="cuda", num_steps=50):
    """生成点云并确保合理范围"""
    # 生成文本嵌入
    with torch.no_grad():
        input_ids = pipeline.tokenizer(text).input_ids.to(device)
        text_embeddings = pipeline.text_encoder(input_ids)
    
    # 创建初始噪声
    noise = torch.randn(1, num_points, 3, device=device)
    
    # 设置扩散步数
    pipeline.scheduler.set_timesteps(num_steps)
    
    # 扩散过程
    for t in pipeline.scheduler.timesteps:
        with torch.no_grad():
            model_output = pipeline.transformer(
                noise, 
                torch.tensor([t]*1, device=device),
                encoder_hidden_states=text_embeddings
            )['sample']
            noise = pipeline.scheduler.step(model_output, t, noise).prev_sample
    
    # 获取点云并确保合理范围
    generated_points = noise.cpu().numpy()[0]
    
    # 确保点云不会太散（缩放点云）
    max_value = np.max(np.abs(generated_points))
    if max_value > 10:
        scale_factor = 10.0 / max_value
        generated_points *= scale_factor
    
    return generated_points

def set_axis_limits(ax, points):
    """智能坐标范围设置"""
    max_val = max(points.max(), abs(points.min()))
    view_range = max(1.0, max_val * 1.2)  # 防止过小
    ax.set_xlim([-view_range, view_range])
    ax.set_ylim([-view_range, view_range])
    ax.set_zlim([-view_range, view_range])
    ax.view_init(elev=25, azim=45)
    ax.grid(True, alpha=0.3)



def main():
    # 固定参数设置 - 只测试airplane类别
    model_path = '/root/NOVA-main/checkpoints2/best_model.pth'
    data_root = '/root/autodl-tmp/.autodl/data'
    class_name = 'airplane'  # 固定为airplane
    num_samples = -1  # 评估所有样本
    output_dir = './evaluations'
    images_dir = './evaluations/images'  # 专门存放图片的文件夹
    device = 'cuda'
    visualize = True  # 生成可视化
    max_visualize = 5  # 最多可视化5个样本
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)  # 创建图片文件夹
    
    # 设备设置
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    pipeline = load_model(model_path, device)
    
    # 只评估airplane类别
    logger.info(f"将评估类别: {class_name}")
    
    # 加载数据集
    class_id = CLASS_NAME_TO_ID[class_name]
    dataset = AdvancedShapeNetDataset(
        data_root=data_root,
        split='test',
        max_points=2048,  # 修复：改为2048，与训练时一致
        normalize=True,
        stage='domain_adapt',
        class_filter=[class_name],
        enable_cache=True,
        cache_size=500,
        precompute_texts=True
    )
    
    # 初始化全局归一化器 - 直接加载训练时保存的统计参数
    global_normalizer.load_stats('stats.json')
    
    # 筛选指定类别的样本
    indices = [i for i, cid in enumerate(dataset.class_ids) if cid == class_id]
    if not indices:
        logger.error(f"类别 {class_name} 没有测试样本！")
        return
    
    # 确定要评估的样本数量
    if num_samples == -1:
        # 评估所有样本
        selected_indices = indices
        logger.info(f"将评估所有 {len(selected_indices)} 个样本")
    else:
        # 随机选择指定数量的样本
        selected_indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False)
        logger.info(f"将评估 {len(selected_indices)} 个样本")
    
    # 存储所有指标
    all_cd_scores = []
    all_emd_scores = []
    all_texts = []
    
    # 评估每个样本
    for i, idx in enumerate(tqdm(selected_indices, desc=f"评估{class_name}样本")):
        sample = dataset[idx]
        real_points = sample['points'].numpy()
        text = sample['text']
        
        # 稳定生成点云
        gen_points = stable_generate_pointcloud(pipeline, [text], num_points=real_points.shape[0], 
                                              device=device, num_steps=100)
        
        # 计算CD和EMD指标
        cd_score, emd_score = compute_metrics(real_points, gen_points)
        
        # 存储结果
        all_cd_scores.append(cd_score)
        all_emd_scores.append(emd_score)
        all_texts.append(text)
        
        # 打印当前样本的指标
        logger.info(f"{class_name} 样本 {i+1}/{len(selected_indices)} - CD: {cd_score:.6f}, EMD: {emd_score:.6f}")
        logger.info(f"文本: {text[:50]}...")
        
        # 如果启用可视化且未超过最大可视化数量
        if visualize and i < max_visualize:
            # 确保生成点云和真实点云点数一致
            if gen_points.shape[0] != real_points.shape[0]:
                if gen_points.shape[0] > real_points.shape[0]:
                    # 随机采样
                    indices = np.random.choice(gen_points.shape[0], real_points.shape[0], replace=False)
                    gen_points = gen_points[indices]
                else:
                    # 重复填充
                    repeat_times = real_points.shape[0] // gen_points.shape[0] + 1
                    gen_points = np.tile(gen_points, (repeat_times, 1))
                    gen_points = gen_points[:real_points.shape[0]]
            
            # 可视化对比
            title = f"{class_name} - CD:{cd_score:.4f} EMD:{emd_score:.4f} - {text[:30]}..."
            save_path = os.path.join(images_dir, f"{class_name}_{i+1}_CD{cd_score:.4f}_EMD{emd_score:.4f}.png")
            visualize_comparison(real_points, gen_points, title, save_path)
    
    # 计算平均指标
    avg_cd = np.mean(all_cd_scores)
    avg_emd = np.mean(all_emd_scores)
    std_cd = np.std(all_cd_scores)
    std_emd = np.std(all_emd_scores)
    
    # 打印最终结果
    logger.info("=" * 60)
    logger.info(f"评估结果 - 类别: {class_name}")
    logger.info(f"总样本数: {len(selected_indices)}")
    logger.info(f"平均CD: {avg_cd:.6f} ± {std_cd:.6f}")
    logger.info(f"平均EMD: {avg_emd:.6f} ± {std_emd:.6f}")
    logger.info("=" * 60)
    
    # 保存详细结果到文件
    results_file = os.path.join(output_dir, f"{class_name}_evaluation_results.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"类别: {class_name}\n")
        f.write(f"总样本数: {len(selected_indices)}\n")
        f.write(f"平均CD: {avg_cd:.6f} ± {std_cd:.6f}\n")
        f.write(f"平均EMD: {avg_emd:.6f} ± {std_emd:.6f}\n")
        f.write("\n详细结果:\n")
        f.write("样本ID\tCD\t\tEMD\t\t文本\n")
        f.write("-" * 80 + "\n")
        for i, (cd, emd, text) in enumerate(zip(all_cd_scores, all_emd_scores, all_texts)):
            f.write(f"{i+1}\t{cd:.6f}\t{emd:.6f}\t{text[:50]}...\n")
    
    logger.info(f"详细结果已保存到: {results_file}")
    
    # 保存指标数据为numpy数组
    metrics_file = os.path.join(output_dir, f"{class_name}_metrics.npz")
    np.savez(metrics_file, 
             cd_scores=np.array(all_cd_scores),
             emd_scores=np.array(all_emd_scores),
             texts=np.array(all_texts))
    logger.info(f"指标数据已保存到: {metrics_file}")

if __name__ == "__main__":
    main() 