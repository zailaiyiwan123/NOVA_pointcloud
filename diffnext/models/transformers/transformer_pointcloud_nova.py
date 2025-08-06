# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""3D Point Cloud transformer model based on NOVA architecture with dynamic partitioning and autoregressive diffusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

from diffnext.models.diffusion_mlp import DiffusionMLP
from diffnext.models.embeddings import PosEmbed, TextEmbed, LabelEmbed
from diffnext.models.normalization import AdaLayerNorm
from diffnext.models.vision_transformer import VisionTransformer
from diffnext.utils.registry import Registry

POINT_CLOUD_ENCODERS = Registry("point_cloud_encoders")
POINT_CLOUD_DECODERS = Registry("point_cloud_decoders")


@POINT_CLOUD_ENCODERS.register("vit_d32w768")
def point_cloud_encoder_vit_d32w768(patch_size, point_cloud_size):
    return PointCloudTransformer(depth=32, embed_dim=768, num_heads=12, patch_size=patch_size, point_cloud_size=point_cloud_size)

@POINT_CLOUD_ENCODERS.register("vit_d32w1024")
def point_cloud_encoder_vit_d32w1024(patch_size, point_cloud_size):
    return PointCloudTransformer(depth=32, embed_dim=1024, num_heads=16, patch_size=patch_size, point_cloud_size=point_cloud_size)

@POINT_CLOUD_ENCODERS.register("vit_d32w1536")
def point_cloud_encoder_vit_d32w1536(patch_size, point_cloud_size):
    return PointCloudTransformer(depth=32, embed_dim=1536, num_heads=16, patch_size=patch_size, point_cloud_size=point_cloud_size)


@POINT_CLOUD_DECODERS.register("mlp_d6w768")
def point_cloud_decoder_mlp_d6w768(patch_size, cond_dim):
    return DiffusionMLP(depth=6, embed_dim=768, cond_dim=cond_dim)

@POINT_CLOUD_DECODERS.register("mlp_d6w1024")
def point_cloud_decoder_mlp_d6w1024(patch_size, cond_dim):
    return DiffusionMLP(depth=6, embed_dim=1024, cond_dim=cond_dim)

@POINT_CLOUD_DECODERS.register("mlp_d6w1536")
def point_cloud_decoder_mlp_d6w1536(patch_size, cond_dim):
    return DiffusionMLP(depth=6, embed_dim=1536, cond_dim=cond_dim)


def dynamic_partition(points, k=20):
    """将点云动态划分为k个子集并生成随机序列"""
    batch_size, num_points, dim = points.shape
    indices = torch.randperm(num_points, device=points.device)
    subset_size = num_points // k
    subsets = []
    
    for i in range(k):
        start_idx = i * subset_size
        end_idx = start_idx + subset_size if i < k - 1 else num_points
        subset_indices = indices[start_idx:end_idx]
        subsets.append(points[:, subset_indices, :])
    
    # 生成随机预测顺序序列
    order = torch.randperm(k, device=points.device)
    return order, subsets


def compute_local_density(points, k_neighbors=8):
    """计算局部密度用于动态资源分配"""
    # 使用KNN计算局部密度
    dist_matrix = torch.cdist(points, points)
    # 获取k个最近邻的距离
    k_distances, _ = torch.topk(dist_matrix, k=k_neighbors + 1, dim=-1, largest=False)
    # 平均距离作为密度指标
    density = torch.mean(k_distances[:, :, 1:], dim=-1)  # 排除自身
    return density


def adaptive_sampling(subset, target_size, density_factor=0.5):
    """动态调整子集点数"""
    if subset.size(1) < target_size:  # 稀疏子集
        return farthest_point_sampling(subset, target_size)
    else:  # 稠密子集
        return feature_aware_interpolation(subset, target_size)


def farthest_point_sampling(points, num_samples):
    """最远点采样"""
    batch_size, num_points, dim = points.shape
    device = points.device
    
    # 随机选择起始点
    start_indices = torch.randint(0, num_points, (batch_size,), device=device)
    selected_indices = torch.zeros(batch_size, num_samples, dtype=torch.long, device=device)
    selected_indices[:, 0] = start_indices
    
    # 计算距离矩阵
    dist_matrix = torch.cdist(points, points)
    
    for i in range(1, num_samples):
        # 找到距离已选点最远的点
        min_distances = torch.min(dist_matrix, dim=1)[0]
        farthest_indices = torch.argmax(min_distances, dim=1)
        selected_indices[:, i] = farthest_indices
        
        # 更新距离矩阵
        new_distances = dist_matrix[torch.arange(batch_size), farthest_indices]
        dist_matrix = torch.minimum(dist_matrix, new_distances.unsqueeze(1))
    
    # 返回采样的点
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_samples)
    return points[batch_indices, selected_indices]


def feature_aware_interpolation(points, target_size):
    """特征感知插值"""
    batch_size, num_points, dim = points.shape
    
    if num_points <= target_size:
        # 如果点数不够，重复采样
        repeat_times = target_size // num_points + 1
        points = points.repeat(1, repeat_times, 1)
        return points[:, :target_size, :]
    
    # 使用KNN插值
    # 随机选择目标点
    indices = torch.randperm(num_points, device=points.device)[:target_size]
    target_points = points[:, indices, :]
    
    # 对每个目标点，找到最近的k个源点
    k = min(8, num_points)
    dist_matrix = torch.cdist(target_points, points)
    _, nearest_indices = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
    
    # 加权平均插值
    weights = torch.softmax(-dist_matrix, dim=-1)
    interpolated = torch.sum(weights.unsqueeze(-1) * points.unsqueeze(1), dim=2)
    
    return interpolated


class EdgeAligner(nn.Module):
    """边缘对齐优化 - 双向注意力融合边缘信息"""
    
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 双向注意力机制
        self.biattn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 边缘特征提取
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # 空间坐标编码
        self.spatial_embed = nn.Linear(3, embed_dim)
        
    def extract_edge_features(self, points, features):
        """提取边缘特征"""
        # 计算每个点到其他点的距离
        dist_matrix = torch.cdist(points, points)
        
        # 找到每个点的最近邻
        _, nearest_indices = torch.topk(dist_matrix, k=min(8, points.size(1)), dim=-1, largest=False)
        
        # 聚合邻居特征
        neighbor_features = torch.gather(features.unsqueeze(1).expand(-1, nearest_indices.size(-1), -1), 
                                       dim=2, index=nearest_indices.unsqueeze(-1).expand(-1, -1, features.size(-1)))
        
        # 边缘特征 = 中心特征 - 邻居特征均值
        edge_features = features.unsqueeze(1) - torch.mean(neighbor_features, dim=2)
        return edge_features.squeeze(1)
    
    def forward(self, current_points, current_features, neighbor_points, neighbor_features):
        """双向注意力融合边缘信息"""
        batch_size = current_points.size(0)
        
        # 提取边缘特征
        current_edge_features = self.extract_edge_features(current_points, current_features)
        neighbor_edge_features = []
        
        for i in range(len(neighbor_points)):
            edge_feats = self.extract_edge_features(neighbor_points[i], neighbor_features[i])
            neighbor_edge_features.append(edge_feats)
        
        # 合并邻居边缘特征
        if neighbor_edge_features:
            all_neighbor_features = torch.cat(neighbor_edge_features, dim=1)
        else:
            all_neighbor_features = current_edge_features
        
        # 双向注意力机制
        aligned_features, _ = self.biattn(
            query=current_edge_features,
            key=all_neighbor_features,
            value=all_neighbor_features
        )
        
        # 空间坐标编码
        spatial_features = self.spatial_embed(current_points)
        
        # 融合边缘对齐特征和空间特征
        final_features = aligned_features + spatial_features
        
        return final_features


class AutoregressiveDiffusion(nn.Module):
    """自回归扩散生成框架"""
    
    def __init__(self, base_transformer, embed_dim=768, num_heads=12):
        super().__init__()
        self.transformer = base_transformer
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 双向注意力用于边缘对齐
        self.biattn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 边缘对齐器
        self.edge_aligner = EdgeAligner(embed_dim, num_heads)
        
        # 特征聚合层
        self.feature_aggregator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 时间步编码
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def _aggregate_features(self, generated_subsets):
        """特征聚合 - 空间注意力机制"""
        if not generated_subsets:
            return None
        
        # 将所有已生成的子集特征连接
        all_features = torch.cat(generated_subsets, dim=1)
        
        # 使用注意力机制聚合特征
        aggregated, _ = self.biattn(
            query=all_features,
            key=all_features,
            value=all_features
        )
        
        # 全局特征
        global_feature = torch.mean(aggregated, dim=1, keepdim=True)
        
        return global_feature
    
    def forward(self, current_subset, generated_subsets, t, current_points=None, neighbor_points=None):
        """自回归扩散生成"""
        batch_size = current_subset.size(0)
        
        # 特征聚合
        context = self._aggregate_features(generated_subsets)
        
        # 时间步编码
        timestep_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # 边缘对齐（如果有邻居信息）
        if current_points is not None and neighbor_points:
            neighbor_features = [subset for subset in generated_subsets if subset is not None]
            edge_aligned = self.edge_aligner(current_points, current_subset, neighbor_points, neighbor_features)
            current_subset = current_subset + edge_aligned
        
        # 添加上下文特征
        if context is not None:
            current_subset = current_subset + context.expand(-1, current_subset.size(1), -1)
        
        # 添加时间步编码
        current_subset = current_subset + timestep_emb.unsqueeze(1)
        
        # 扩散生成
        noisy_points = self.transformer(current_subset, t)
        
        return noisy_points


class PointCloudPatchEmbed(nn.Module):
    """Point Cloud Patch Embedding based on NOVA's patch embedding."""
    
    def __init__(self, point_cloud_dim, embed_dim, patch_size):
        super().__init__()
        self.point_cloud_dim = point_cloud_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Linear projection for point cloud patches (similar to NOVA's image patch embedding)
        self.proj = nn.Linear(patch_size * point_cloud_dim, embed_dim)
        
    def forward(self, x):
        # x: (B, point_cloud_dim, N) -> (B, N//patch_size, embed_dim)
        B, C, N = x.shape
        assert N % self.patch_size == 0, f"Point cloud size {N} must be divisible by patch size {self.patch_size}"
        
        # Reshape to patches (similar to NOVA's image patching)
        x = x.transpose(1, 2)  # (B, N, C)
        x = x.view(B, N // self.patch_size, self.patch_size * C)
        
        # Project to embedding dimension
        x = self.proj(x)
        return x


class PointCloudPosEmbed(nn.Module):
    """3D Positional Embedding for Point Cloud patches."""
    
    def __init__(self, embed_dim, max_patches):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_patches = max_patches
        
        # Learnable positional embedding (similar to NOVA's pos_embed)
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)
        
    def forward(self, x):
        # x: (B, N, embed_dim)
        B, N, D = x.shape
        pos_embed = self.pos_embed[:, :N, :]
        return x + pos_embed


class DepthAwarePositionalEncoding(nn.Module):
    """Depth-aware 3D positional encoding - based on sine/cosine 3D coordinate encoding"""
    
    def __init__(self, embed_dim, max_points=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_points = max_points
        
        # 3D positional encoding parameters
        self.position_scale = 10000.0
        self.dim_div = torch.arange(0, embed_dim, 2) / embed_dim
        self.div_term = self.position_scale ** self.dim_div
        
        # Learnable scaling factors
        self.scale_x = nn.Parameter(torch.ones(1))
        self.scale_y = nn.Parameter(torch.ones(1))
        self.scale_z = nn.Parameter(torch.ones(1))
        
    def forward(self, points):
        # points: [B, N, 3]
        batch_size, num_points, _ = points.shape
        
        # Apply learnable scaling
        scaled_points = points * torch.stack([self.scale_x, self.scale_y, self.scale_z], dim=0)
        
        # Generate 3D positional encoding
        pe = torch.zeros(batch_size, num_points, self.embed_dim, device=points.device)
        
        # X coordinate encoding
        pe[:, :, 0::6] = torch.sin(scaled_points[:, :, 0:1] / self.div_term[:self.embed_dim//6])
        pe[:, :, 1::6] = torch.cos(scaled_points[:, :, 0:1] / self.div_term[:self.embed_dim//6])
        
        # Y coordinate encoding
        pe[:, :, 2::6] = torch.sin(scaled_points[:, :, 1:2] / self.div_term[:self.embed_dim//6])
        pe[:, :, 3::6] = torch.cos(scaled_points[:, :, 1:2] / self.div_term[:self.embed_dim//6])
        
        # Z coordinate encoding
        pe[:, :, 4::6] = torch.sin(scaled_points[:, :, 2:3] / self.div_term[:self.embed_dim//6])
        pe[:, :, 5::6] = torch.cos(scaled_points[:, :, 2:3] / self.div_term[:self.embed_dim//6])
        
        return pe


class PointCloudTransformer(nn.Module):
    """Improved point cloud Transformer - integrates depth awareness and spatial partitioning"""
    
    def __init__(self, depth, embed_dim, num_heads, patch_size, point_cloud_size):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.point_cloud_size = point_cloud_size
        
        # Basic components
        self.patch_embed = PointCloudPatchEmbed(3, embed_dim, patch_size)
        self.pos_embed = PointCloudPosEmbed(embed_dim, point_cloud_size // patch_size)
        
        # Depth-aware positional encoding
        self.depth_pe = DepthAwarePositionalEncoding(embed_dim, point_cloud_size)
        
        # Learnable clustering centers for spatial partitioning
        self.num_clusters = 8
        self.cluster_centers = nn.Parameter(torch.randn(self.num_clusters, 3) * 0.1)  # Initialize with std=0.1
        
        # Cluster feature extraction
        self.cluster_feature_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Cross-cluster attention
        self.cluster_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.1
        )
        
        # Cluster output projection
        self.cluster_output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, 3)
        
    def forward(self, x, encoder_hidden_states=None):
        # x: [B, 3, point_cloud_size]
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, N//patch_size, embed_dim]
        
        # Position embedding
        x = x + self.pos_embed(x)
        
        # Depth-aware positional encoding
        depth_pe = self.depth_pe(x.transpose(1, 2))  # [B, N//patch_size, embed_dim]
        x = x + depth_pe
        
        # Spatial clustering and feature extraction
        # Convert patch embeddings back to point cloud format for clustering
        patch_points = x.transpose(1, 2)  # [B, embed_dim, N//patch_size]
        patch_points = patch_points[:, :3, :]  # Take first 3 dimensions as spatial coordinates
        
        # Calculate distances to cluster centers
        # patch_points: [B, 3, N//patch_size] -> [B, N//patch_size, 3]
        patch_points = patch_points.transpose(1, 2)
        # cluster_centers: [8, 3]
        # distances: [B, N//patch_size, 8]
        distances = torch.cdist(patch_points, self.cluster_centers)
        
        # Soft assignment using softmax
        cluster_weights = torch.softmax(-distances, dim=-1)  # [B, N//patch_size, 8]
        
        # Calculate weighted cluster features
        cluster_features = []
        for i in range(self.num_clusters):
            # Calculate weighted center for this cluster
            weights = cluster_weights[:, :, i:i+1]  # [B, N//patch_size, 1]
            weighted_center = (patch_points * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-8)  # [B, 3]
            
            # Extract cluster features
            cluster_feature = self.cluster_feature_mlp(weighted_center)  # [B, embed_dim]
            cluster_features.append(cluster_feature)
        
        # Stack all cluster features
        cluster_features = torch.stack(cluster_features, dim=1)  # [B, 8, embed_dim]
        
        # Cross-cluster attention
        cluster_features, _ = self.cluster_attention(
            cluster_features, cluster_features, cluster_features
        )
        
        cluster_features = self.cluster_output_proj(cluster_features)
        
        # Expand cluster features to match patch features
        # cluster_features: [B, 8, embed_dim] -> [B, N//patch_size, embed_dim]
        num_patches = x.shape[1]
        expanded_cluster_features = cluster_features.mean(dim=1, keepdim=True).expand(-1, num_patches, -1)
        
        # Add cluster features to patch features
        x = x + expanded_cluster_features
        
        # Add text conditioning if available
        if encoder_hidden_states is not None:
            text_emb = encoder_hidden_states.mean(dim=1, keepdim=True)  # [B, 1, embed_dim]
            x = x + text_emb
        

        x = self.transformer(x)
        
        x = self.output_proj(x)
        
        # Reshape back to point cloud format
        x = x.view(batch_size, -1, 3)
        
        return x
    
    def get_cluster_centers(self):
        """Get current cluster center positions for debugging"""
        return self.cluster_centers.detach().cpu().numpy()
    
    def visualize_cluster_centers(self):
        """Visualize cluster centers in 3D space"""
        centers = self.get_cluster_centers()
        print(f"Cluster centers shape: {centers.shape}")
        print("Cluster center positions:")
        for i, center in enumerate(centers):
            print(f"  Cluster {i}: {center}")
        return centers


class NOVAPointCloudTransformer(nn.Module):
    """NOVA Point Cloud Transformer - main model for point cloud generation with dynamic partitioning"""
    
    def __init__(
        self,
        point_cloud_dim=3,
        point_cloud_size=1024,
        patch_size=16,
        text_token_dim=None,
        text_token_len=None,
        rotary_pos_embed=False,
        arch=("vit_d32w768", "mlp_d6w768"),
        num_subsets=20,  # 动态划分的子集数量
    ):
        super().__init__()
        self.point_cloud_dim = point_cloud_dim
        self.point_cloud_size = point_cloud_size
        self.patch_size = patch_size
        self.text_token_dim = text_token_dim
        self.text_token_len = text_token_len
        self.rotary_pos_embed = rotary_pos_embed
        self.arch = arch
        self.num_subsets = num_subsets
        
        # Add diffusers expected attributes
        self.dtype = torch.float32
        self.device = torch.device('cpu')
        
        # Point cloud embedding layer
        self.point_embed = nn.Linear(point_cloud_dim, 768)
        
        # Position encoding
        self.pos_embed = nn.Parameter(torch.randn(1, point_cloud_size, 768) * 0.02)
        
        # Learnable clustering centers for spatial partitioning
        self.num_clusters = 8
        self.cluster_centers = nn.Parameter(torch.randn(self.num_clusters, 3) * 0.1)  # Initialize with std=0.1
        
        # Cluster feature extraction
        self.cluster_feature_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 768),
            nn.LayerNorm(768)
        )
        
        # Cross-cluster attention
        self.cluster_attention = nn.MultiheadAttention(
            768, 12, batch_first=True, dropout=0.1
        )
        
        # Cluster output projection
        self.cluster_output_proj = nn.Linear(768, 768)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=768 * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, 8)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, 768),
            nn.SiLU(),
            nn.Linear(768, 768)
        )
        
        if text_token_dim is not None:
            self.text_embed = nn.Linear(text_token_dim, 768)
        else:
            self.text_embed = None
        
        self.output_proj = nn.Linear(768, point_cloud_dim)
        
        # 自回归扩散组件
        self.autoregressive_diffusion = AutoregressiveDiffusion(self, 768, 12)
        
        # 边缘对齐器
        self.edge_aligner = EdgeAligner(768, 12)
        
        # 线程池用于并行处理
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        
        self._init_weights()
        
    def _init_weights(self):
        """Improved weight initialization"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'output_proj' in name:
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                elif 'text_embed' in name or 'time_embed' in name:
                    nn.init.xavier_uniform_(module.weight, gain=0.3)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.2)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.001)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def generate_pointcloud_autoregressive(self, global_feature, total_points=15000, text_embeddings=None):
        """完整生成流程 - 动态区域划分与自回归扩散生成"""
        batch_size = global_feature.size(0)
        device = global_feature.device
        
        # 1. 动态划分（20子集）
        order, subsets = dynamic_partition(global_feature, k=self.num_subsets)
        
        # 存储生成的子集
        generated_subsets = {}
        generated_points = {}
        
        # 并行处理子集
        def process_subset(idx, subset_idx):
            with self.lock:
                # 2. 点数动态调整
                base_size = total_points // self.num_subsets
                current_subset = subsets[subset_idx]
                
                # 计算局部密度
                density = compute_local_density(current_subset)
                density_factor = 0.5
                target_size = int(base_size * (1 + density_factor * (density.mean() - 0.5)))
                target_size = max(100, min(target_size, base_size * 2))  # 限制范围
                
                # 自适应采样
                current_points = adaptive_sampling(current_subset, target_size)
                
                # 3. 自回归扩散生成
                t = torch.tensor(idx / float(self.num_subsets), device=device)
                
                # 获取已生成的邻居子集
                neighbor_subsets = []
                neighbor_points = []
                for prev_idx in order[:idx]:
                    if prev_idx in generated_subsets:
                        neighbor_subsets.append(generated_subsets[prev_idx])
                        neighbor_points.append(generated_points[prev_idx])
                
                # 生成当前子集
                generated_subset = self.autoregressive_diffusion(
                    current_subset, 
                    neighbor_subsets, 
                    t,
                    current_points,
                    neighbor_points
                )
                
                return subset_idx, generated_subset, current_points
        
        # 串行生成（保持自回归特性）
        for i, subset_idx in enumerate(order):
            subset_idx, generated_subset, current_points = process_subset(i, subset_idx)
            generated_subsets[subset_idx] = generated_subset
            generated_points[subset_idx] = current_points
        
        # 4. 空间重组
        final_point_cloud = torch.cat([generated_subsets[i] for i in range(self.num_subsets)], dim=1)
        
        return final_point_cloud
    
    def forward(self, x, timestep, encoder_hidden_states=None, return_dict=True):
        # x: [B, 3, point_cloud_size]
        batch_size = x.shape[0]
        
        # 检查是否使用自回归生成
        if hasattr(self, 'use_autoregressive') and self.use_autoregressive:
            return self.generate_pointcloud_autoregressive(x, encoder_hidden_states)
        
        # 标准前向传播
        # Point cloud embedding
        x = x.transpose(1, 2)  # [B, point_cloud_size, 3]
        x = self.point_embed(x)  # [B, point_cloud_size, 768]
        
        # Add position encoding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Spatial clustering and feature extraction
        # Calculate distances to cluster centers
        # x: [B, point_cloud_size, 768] -> [B, point_cloud_size, 3] (take first 3 dims as spatial coords)
        spatial_coords = x[:, :, :3]  # [B, point_cloud_size, 3]
        # cluster_centers: [8, 3]
        # distances: [B, point_cloud_size, 8]
        distances = torch.cdist(spatial_coords, self.cluster_centers)
        
        # Soft assignment using softmax
        cluster_weights = torch.softmax(-distances, dim=-1)  # [B, point_cloud_size, 8]
        
        # Calculate weighted cluster features
        cluster_features = []
        for i in range(self.num_clusters):
            # Calculate weighted center for this cluster
            weights = cluster_weights[:, :, i:i+1]  # [B, point_cloud_size, 1]
            weighted_center = (spatial_coords * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-8)  # [B, 3]
            
            # Extract cluster features
            cluster_feature = self.cluster_feature_mlp(weighted_center)  # [B, 768]
            cluster_features.append(cluster_feature)
        
        # Stack all cluster features
        cluster_features = torch.stack(cluster_features, dim=1)  # [B, 8, 768]
        
        # Cross-cluster attention
        cluster_features, _ = self.cluster_attention(
            cluster_features, cluster_features, cluster_features
        )
        
        # Cluster output projection
        cluster_features = self.cluster_output_proj(cluster_features)
        
        # Expand cluster features to match point cloud features
        # cluster_features: [B, 8, 768] -> [B, point_cloud_size, 768]
        expanded_cluster_features = cluster_features.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
        
        # Add cluster features to point cloud features
        x = x + expanded_cluster_features
        
        # Time embedding
        timestep_emb = self.time_embed(timestep.unsqueeze(-1).float())
        x = x + timestep_emb.unsqueeze(1)
        
        # Text embedding
        if encoder_hidden_states is not None and self.text_embed is not None:
            text_emb = self.text_embed(encoder_hidden_states)
            if self.text_pos_embed is not None:
                text_emb = text_emb + self.text_pos_embed[:, :text_emb.shape[1], :]
        else:
            text_emb = None
        
        # Add text conditioning if available
        if text_emb is not None:
            x = x + text_emb.unsqueeze(1)
        
        # Transformer processing
        x = self.transformer(x)
        
        # Output projection
        x = self.output_proj(x)
        
        # Reshape back to point cloud format
        x = x.view(batch_size, -1, 3)
        
        if return_dict:
            return {'sample': x}
        else:
            return x
    
    def get_cluster_centers(self):
        """Get current cluster center positions for debugging"""
        return self.cluster_centers.detach().cpu().numpy()
    
    def visualize_cluster_centers(self):
        """Visualize cluster centers in 3D space"""
        centers = self.get_cluster_centers()
        print(f"Cluster centers shape: {centers.shape}")
        print("Cluster center positions:")
        for i, center in enumerate(centers):
            print(f"  Cluster {i}: {center}")
        return centers
    
    def enable_autoregressive_generation(self):
        """启用自回归生成模式"""
        self.use_autoregressive = True
        print("✅ Enabled autoregressive point cloud generation with dynamic partitioning")
    
    def disable_autoregressive_generation(self):
        """禁用自回归生成模式"""
        self.use_autoregressive = False
        print("✅ Disabled autoregressive generation, using standard forward pass")
    
 
