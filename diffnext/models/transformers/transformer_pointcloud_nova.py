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
"""3D Point Cloud transformer model based on NOVA architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

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
    """NOVA Point Cloud Transformer - main model for point cloud generation"""
    
    def __init__(
        self,
        point_cloud_dim=3,
        point_cloud_size=1024,
        patch_size=16,
        text_token_dim=None,
        text_token_len=None,
        rotary_pos_embed=False,
        arch=("vit_d32w768", "mlp_d6w768"),
    ):
        super().__init__()
        self.point_cloud_dim = point_cloud_dim
        self.point_cloud_size = point_cloud_size
        self.patch_size = patch_size
        self.text_token_dim = text_token_dim
        self.text_token_len = text_token_len
        self.rotary_pos_embed = rotary_pos_embed
        self.arch = arch
        
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
    
    def forward(self, x, timestep, encoder_hidden_states=None, return_dict=True):
        # x: [B, 3, point_cloud_size]
        batch_size = x.shape[0]
        
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
    
 