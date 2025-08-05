#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import json
import logging
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import swanlab as wandb
from typing import Dict, List, Optional, Tuple
import random
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from diffnext.pipelines.nova.pipeline_nova_pointcloud_gen import NOVAPointCloudGenerationPipeline
from diffnext.models.transformers.transformer_pointcloud_nova import NOVAPointCloudTransformer, PointCloudTransformer
from diffnext.models.diffusion_mlp import DiffusionMLP
from diffnext.models.embeddings import TextEmbed
from diffnext.schedulers.scheduling_ddpm import DDPMScheduler

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gradient protection layer
class GradientGuard(nn.Module):
    """Gradient protection layer - prevents gradient explosion"""
    def __init__(self, max_grad_norm=0.1):
        super().__init__()
        self.max_grad_norm = max_grad_norm
        
    def forward(self, x):
        if self.training:
            x.register_hook(lambda grad: torch.clamp(grad, -self.max_grad_norm, self.max_grad_norm))
        return x


class AdvancedShapeNetDataset(Dataset):
    """Advanced ShapeNet dataset - optimized version"""

    def __init__(self, data_root, split='train', max_points=15000,
                 normalize=True, class_filter=None,
                 enable_cache=True, cache_size=1000, precompute_texts=True):

        self.data_root = Path(data_root)
        self.split = split
        self.max_points = max_points
        self.normalize = normalize
        self.class_filter = class_filter
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.precompute_texts = precompute_texts

        # Cache mechanism
        self.cache = {}
        self.cache_lock = threading.Lock()

        self.file_paths = []
        self.class_ids = []
        self.class_names = []
        self._collect_files()

        # Compute statistics
        if normalize:
            # Enable normalization, compute and save statistics
            self.mean, self.std = self._compute_statistics()
            logger.info(f"Dataset statistics - normalization enabled, mean: {self.mean}, std: {self.std}")
        else:
            # Disable normalization, use default parameters
            self.mean = torch.zeros(3)
            self.std = torch.ones(3)
            logger.info(f"Dataset statistics - normalization disabled, using default parameters")

        # Filter data by class filter
        self._filter_by_class()

        # Precompute text descriptions (Step 3: async text generation)
        if self.precompute_texts:
            logger.info("Precomputing text descriptions...")
            self.texts = self._precompute_texts()
            logger.info(f"Precomputation completed, {len(self.texts)} text descriptions")
        else:
            self.texts = None

        logger.info(f"Found {len(self.file_paths)} {split} samples")

    def _collect_files(self):
        """Collect all .npy files"""
        for class_dir in self.data_root.iterdir():
            if class_dir.is_dir() and len(class_dir.name) == 8:
                split_dir = class_dir / self.split
                if split_dir.exists():
                    for npy_file in split_dir.glob("*.npy"):
                        self.file_paths.append(str(npy_file))
                        self.class_ids.append(class_dir.name)
                        self.class_names.append(self._get_class_name(class_dir.name))

    def _get_class_name(self, class_id):
        synsetid_to_cate = {
            '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
            '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
            '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
            '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
            '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
            '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
            '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
            '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
            '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
            '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
            '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
            '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
            '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
            '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
            '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
            '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
            '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
            '04554684': 'washer', '02843684': 'birdhouse', '02871439': 'bookshelf',
            '02874714': 'bowl', '02879718': 'bowl', '02920083': 'bicycle',
            '02933112': 'cabinet', '02942699': 'camera', '02946921': 'can',
            '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
            '03046257': 'clock', '03085013': 'keyboard', '03207941': 'dishwasher',
            '03211117': 'monitor', '03261776': 'earphone', '03325088': 'faucet',
            '03337140': 'file', '03467517': 'guitar', '03513137': 'helmet',
            '03593526': 'jar', '03624134': 'knife', '03636649': 'lamp',
            '03642806': 'laptop', '03691459': 'speaker', '03710193': 'mailbox',
            '03759954': 'microphone', '03761084': 'microwave', '03790512': 'motorcycle',
            '03797390': 'mug', '03928116': 'piano', '03938244': 'pillow',
            '03948459': 'pistol', '03991062': 'pot', '04004475': 'printer',
            '04074963': 'remote_control', '04090263': 'rifle', '04099429': 'rocket',
            '04225987': 'skateboard', '04256520': 'sofa', '04330267': 'stove',
            '04530566': 'vessel', '04554684': 'washer'
        }
        return synsetid_to_cate.get(class_id, 'unknown')

    def _filter_by_class(self):
        if self.class_filter:
            filtered_paths = []
            filtered_ids = []
            filtered_names = []
            for i, class_id in enumerate(self.class_ids):
                if self._get_class_name(class_id) in self.class_filter:
                    filtered_paths.append(self.file_paths[i])
                    filtered_ids.append(self.class_ids[i])
                    filtered_names.append(self.class_names[i])
            self.file_paths = filtered_paths
            self.class_ids = filtered_ids
            self.class_names = filtered_names

    def _compute_statistics(self, sample_size=1000):
        logger.info("Computing dataset statistics...")
        sample_indices = random.sample(range(len(self.file_paths)), min(sample_size, len(self.file_paths)))

        all_points = []
        for idx in tqdm(sample_indices, desc="Computing statistics"):
            points = np.load(self.file_paths[idx])
            if len(points) > self.max_points:
                indices = np.random.choice(len(points), self.max_points, replace=False)
                points = points[indices]
            all_points.append(points)

        all_points = np.concatenate(all_points, axis=0)
        mean = np.mean(all_points, axis=0)
        std = np.std(all_points, axis=0)

        # Update global normalizer
        global global_normalizer
        global_normalizer.update_stats(torch.tensor(mean), torch.tensor(std))

        # Save statistics
        global_normalizer.save_stats('stats.json')

        return mean, std

    def _precompute_texts(self):
        """pre-generate all text descriptions"""
        texts = []
        for i in tqdm(range(len(self)), desc="Precomputing text descriptions"):
            class_name = self.class_names[i]
            texts.append(self._generate_advanced_text_prompt(class_name))
        return texts

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.enable_cache:
            with self.cache_lock:
                if idx in self.cache:
                    return self.cache[idx]

        try:
            points = np.load(self.file_paths[idx], mmap_mode='r')
            points = np.array(points) 
        except:
            points = np.load(self.file_paths[idx])

        # Optimize point cloud sampling algorithm
        if len(points) > self.max_points:
            # torch.randperm is 10x faster than np.random.choice
            indices = torch.randperm(len(points))[:self.max_points].numpy()
            points = points[indices]

        points = torch.from_numpy(points).float()
        
        if self.normalize:
            points = normalize(points)
        else:
            pass 

        if self.texts is not None:
            text_prompt = self.texts[idx]
        else:
            text_prompt = self._generate_advanced_text_prompt(self.class_names[idx])

        result = {
            'points': points,
            'text': text_prompt,
            'class_name': self.class_names[idx],
            'class_id': self.class_ids[idx]
        }

        if self.enable_cache:
            with self.cache_lock:
                # Limit cache size
                if len(self.cache) >= self.cache_size:
                    # Randomly remove one cache item
                    remove_key = random.choice(list(self.cache.keys()))
                    del self.cache[remove_key]
                self.cache[idx] = result

        return result

    def _generate_advanced_text_prompt(self, class_name):
        """Generate simple text description"""
        # Just use the class name directly - 3D point clouds don't need complex prompts
        return f"a {class_name}"


# ====== Load training set normalization parameters BEGIN ======
class GlobalNormalizer:
    """Global normalizer - unified normalization strategy for training and evaluation"""

    def __init__(self, mean=None, std=None):
        self.mean = mean if mean is not None else torch.zeros(3)
        self.std = std if std is not None else torch.ones(3)

    def __call__(self, points, mode='norm'):
        """Normalize or denormalize point cloud"""
        if mode == 'norm':
            return (points - self.mean.to(points.device)) / self.std.to(points.device)
        else:  # denorm
            return points * self.std.to(points.device) + self.mean.to(points.device)

    def update_stats(self, mean, std):
        """Update statistics"""
        self.mean = mean
        self.std = std

    def save_stats(self, filepath='stats.json'):
        """Save statistics"""
        stats = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(stats, f)

    @classmethod
    def load_stats(cls, filepath='stats.json'):
        """Load statistics from file"""
        if os.path.exists(filepath):
            with open(filepath) as f:
                stats = json.load(f)
            return cls(
                mean=torch.tensor(stats['mean']),
                std=torch.tensor(stats['std'])
            )
        return cls()


# Global normalizer instance
global_normalizer = GlobalNormalizer()


def normalize(points):
    """Normalize point cloud"""
    return global_normalizer(points, 'norm')


def denormalize(points):
    """Denormalize point cloud"""
    return global_normalizer(points, 'denorm')


# ====== Load training set normalization parameters END ======

# ====== Official CD/EMD metrics implementation BEGIN ======
import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    
    # Numerical stability: clip input
    x = torch.clamp(x, -1.0, 1.0)
    y = torch.clamp(y, -1.0, 1.0)
    
    # Use log space calculation to avoid numerical overflow
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    y_norm = torch.norm(y, dim=-1, keepdim=True)
    
    # Avoid zero norm
    x_norm = torch.clamp(x_norm, min=1e-8)
    y_norm = torch.clamp(y_norm, min=1e-8)
    
    # Normalize
    x_normalized = x / x_norm
    y_normalized = y / y_norm
    
    # Calculate distance matrix
    dist_matrix = torch.cdist(x_normalized, y_normalized)
    
    # Numerical stability: ensure distance is non-negative
    dist_matrix = torch.clamp(dist_matrix, min=1e-8)
    
    # Use log space calculation
    log_dist_matrix = torch.log(dist_matrix + 1e-8)
    log_dist_matrix = torch.clamp(log_dist_matrix, min=-10, max=10)
    
    dl = log_dist_matrix.min(2)[0].exp().mean()
    dr = log_dist_matrix.min(1)[0].exp().mean()
    
    return dl, dr


def emd_approx(x, y):
    bs, npts, mpts, dim = x.size(0), x.size(1), y.size(1), x.size(2)
    assert npts == mpts, "EMD only works if two point clouds are equal size"
    dim = x.shape[-1]
    
    # Numerical stability: clip input
    x = torch.clamp(x, -2.0, 2.0)
    y = torch.clamp(y, -2.0, 2.0)
    
    x = x.reshape(bs, npts, 1, dim)
    y = y.reshape(bs, 1, mpts, dim)
    dist = (x - y).norm(dim=-1, keepdim=False)  # (bs, npts, mpts)
    
    # Numerical stability: ensure distance is non-negative
    dist = torch.clamp(dist, min=1e-8)

    emd_lst = []
    dist_np = dist.cpu().detach().numpy()
    for i in range(bs):
        d_i = dist_np[i]
        r_idx, c_idx = linear_sum_assignment(d_i)
        emd_i = d_i[r_idx, c_idx].mean()
        emd_lst.append(emd_i)
    emd = np.stack(emd_lst).reshape(-1)
    emd_torch = torch.from_numpy(emd).to(x)
    return emd_torch


# Compatibility functions
def robust_chamfer_distance(pred, gt):
    """Use official CD implementation"""
    dl, dr = distChamfer(pred, gt)
    return (dl.mean() + dr.mean()) / 2


def robust_emd(pred, gt):
    """Use official EMD implementation"""
    return emd_approx(pred, gt).mean()


# ====== Official CD/EMD metrics implementation END ======


class PointCloudLoss(nn.Module):
    """Improved point cloud diffusion loss function - combines diffusion loss, CD and EMD"""

    def __init__(self, scheduler, cd_weight=0.1, emd_weight=0.05, diffusion_weight=1.0):
        super().__init__()
        self.scheduler = scheduler
        self.cd_weight = cd_weight
        self.emd_weight = emd_weight
        self.diffusion_weight = diffusion_weight

    def robust_chamfer_distance(self, pred, target):
        """Calculate Chamfer Distance"""
        # pred, target: [B, N, 3]
        # Ensure device consistency
        pred = pred.contiguous()
        target = target.contiguous()
        # Use official CD implementation
        dl, dr = distChamfer(pred, target)
        return (dl.mean() + dr.mean()) / 2

    def robust_emd(self, pred, target):
        """Calculate Earth Mover's Distance"""
        # Ensure device consistency
        pred = pred.contiguous()
        target = target.contiguous()
        # Use official EMD implementation
        return emd_approx(pred, target).mean()



    def forward(self, noise_pred, noise_target, pred_points=None, target_points=None, use_only_diffusion=False):
        """Calculate combined loss - diffusion loss + CD + EMD"""
        # Numerical stability: check input
        if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
            noise_pred = torch.clamp(noise_pred, -2.0, 2.0)
        if torch.isnan(noise_target).any() or torch.isinf(noise_target).any():
            noise_target = torch.clamp(noise_target, -2.0, 2.0)
            
        # Base diffusion loss
        diffusion_loss = nn.functional.mse_loss(noise_pred, noise_target, reduction='mean')
        
        # Numerical stability: check diffusion loss
        if torch.isnan(diffusion_loss) or torch.isinf(diffusion_loss):
            diffusion_loss = torch.tensor(0.1, device=noise_pred.device, dtype=noise_pred.dtype)
        
        total_loss = self.diffusion_weight * diffusion_loss
        
        # If only using diffusion loss or no point cloud data provided, skip geometric loss
        if use_only_diffusion or pred_points is None or target_points is None:
            # Only record diffusion loss
            try:
                wandb.log({
                    'loss_components/diffusion_loss': diffusion_loss.item(),
                    'loss_components/total_loss': total_loss.item()
                })
            except Exception as e:
                pass
            return total_loss
        
        # Calculate geometric loss
        # Ensure point cloud shape consistency
        if pred_points.shape[1] != target_points.shape[1]:
            # Sample to same number of points - ensure device consistency
            min_points = min(pred_points.shape[1], target_points.shape[1])
            if pred_points.shape[1] > min_points:
                indices = torch.randperm(pred_points.shape[1], device=pred_points.device)[:min_points]
                pred_points = pred_points[:, indices, :]
            if target_points.shape[1] > min_points:
                indices = torch.randperm(target_points.shape[1], device=target_points.device)[:min_points]
                target_points = target_points[:, indices, :]
        
        # Chamfer Distance loss
        try:
            cd_loss = self.robust_chamfer_distance(pred_points, target_points)
            # Numerical stability: check CD loss
            if torch.isnan(cd_loss) or torch.isinf(cd_loss):
                cd_loss = torch.tensor(0.0, device=pred_points.device, dtype=pred_points.dtype)
            else:
                total_loss += self.cd_weight * cd_loss
        except Exception as e:
            logger.warning(f"CD loss calculation failed: {e}")
            cd_loss = torch.tensor(0.0, device=pred_points.device, dtype=pred_points.dtype)
        
        # Earth Mover's Distance loss
        try:
            emd_loss = self.robust_emd(pred_points, target_points)
            # Numerical stability: check EMD loss
            if torch.isnan(emd_loss) or torch.isinf(emd_loss):
                emd_loss = torch.tensor(0.0, device=pred_points.device, dtype=pred_points.dtype)
            else:
                total_loss += self.emd_weight * emd_loss
        except Exception as e:
            logger.warning(f"EMD loss calculation failed: {e}")
            emd_loss = torch.tensor(0.0, device=pred_points.device, dtype=pred_points.dtype)
        
        # Numerical stability: final check of total loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = diffusion_loss  # Only use diffusion loss
        
        # Record various losses
        try:
            wandb.log({
                'loss_components/diffusion_loss': diffusion_loss.item(),
                'loss_components/cd_loss': cd_loss.item(),
                'loss_components/emd_loss': emd_loss.item(),
                'loss_components/total_loss': total_loss.item(),
                'loss_weights/diffusion_weight': self.diffusion_weight,
                'loss_weights/cd_weight': self.cd_weight,
                'loss_weights/emd_weight': self.emd_weight
            })
        except Exception as e:
            pass

        return total_loss




class AdvancedNOVATrainer:
    """Advanced NOVA trainer - optimized version"""

    def __init__(self, args):
        self.args = args


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.pipeline = self._create_pipeline()
        self.model = self.pipeline.transformer
        
        # Enable gradient checkpointing (if specified)
        if self.args.enable_gradient_checkpointing:
            self.model.use_checkpoint = True
            logger.info("✅ Enabled gradient checkpointing, saving memory")

        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.train_loader, self.val_loader = self._create_data_loaders()

        # Create loss function
        self.criterion = PointCloudLoss(
            self.pipeline.scheduler,
            cd_weight=args.cd_weight,
            emd_weight=args.emd_weight,
            diffusion_weight=args.diffusion_weight
        )

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        self.current_epoch = 0
        self.best_loss = float('inf')

        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb - SwanLab compatible configuration
        try:
            # Set environment variables to ensure SwanLab compatibility
            os.environ["WANDB_MODE"] = "online"
            os.environ["WANDB_SILENT"] = "false"

            wandb.init(
                project="nova-pointcloud-training",
                name=f"nova-pointcloud-{time.strftime('%Y%m%d-%H%M%S')}",
                config=vars(args),
                tags=["pointcloud", "nova"],
                notes="NOVA point cloud generation training",
                reinit=True
            )
            logger.info(f"✅ wandb initialization successful, project: nova-pointcloud-training")
            logger.info(f"   Experiment name: nova-pointcloud")
            logger.info(f"   View URL: {wandb.run.get_url()}")
        except Exception as e:
            logger.warning(f"❌ wandb initialization failed: {e}")
            logger.warning("Continue training but not logging to wandb")

    def _create_pipeline(self):
        logger.info("Creating NOVA point cloud generation pipeline...")

        class DummyTokenizer:
            class TokenOutput:
                def __init__(self, input_ids):
                    self.input_ids = input_ids

            def __call__(self, text, **kwargs):
                # Create virtual token IDs
                input_ids = torch.randint(0, 1000, (1, len(text.split())))
                return self.TokenOutput(input_ids)

        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dtype = torch.float32
                self.hidden_size = 768 

            def forward(self, input_ids):
                batch_size = input_ids.shape[0]
                return torch.randn(batch_size, input_ids.shape[1], self.hidden_size)

        class SimpleNOVAPointCloudTransformer(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.patch_size = getattr(args, 'patch_size', 16)
                self.embed_dim = getattr(args, 'embed_dim', 768)  
                self.num_heads = getattr(args, 'num_heads', 12)  
                self.num_layers = getattr(args, 'num_layers', 8) 
                self.dropout = getattr(args, 'dropout', 0.1)

                # Add diffusers expected attributes
                self.dtype = torch.float32
                self.device = torch.device('cpu')

                # Point cloud embedding layer - use better initialization
                self.point_embed = nn.Linear(3, self.embed_dim)
                nn.init.xavier_uniform_(self.point_embed.weight)
                nn.init.zeros_(self.point_embed.bias)

                # Position encoding - support 2048 points, add gradient protection
                self.pos_embed = nn.Parameter(torch.randn(1, 2048, self.embed_dim) * 0.02)
                self.pos_embed_guard = GradientGuard(max_grad_norm=20.0)  # Position encoding specific protection

                # Transformer layers - use pre-normalization for stability
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.embed_dim * 4,  
                    dropout=self.dropout,
                    batch_first=True,
                    norm_first=True,  # Use Pre-LN structure for stability
                    layer_norm_eps=1e-6  # Increase numerical stability
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)

                # Output layer
                self.output_proj = nn.Linear(self.embed_dim, 3)
                nn.init.xavier_uniform_(self.output_proj.weight)
                nn.init.zeros_(self.output_proj.bias)
                self.output_guard = GradientGuard(max_grad_norm=15.0)  # Output layer specific protection

                # Time embedding 
                self.time_embed = nn.Sequential(
                    nn.Linear(1, self.embed_dim),
                    nn.SiLU(),
                    nn.Linear(self.embed_dim, self.embed_dim)
                )
                for layer in self.time_embed:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)

                # Text embedding
                self.text_embed = nn.Sequential(
                    nn.Linear(768, self.embed_dim),  
                    nn.SiLU(),
                    nn.Linear(self.embed_dim, self.embed_dim)
                )
                for layer in self.text_embed:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)

                self._init_weights()
                logger.info(f"SimpleNOVAPointCloudTransformer initialization completed, parameter count: {sum(p.numel() for p in self.parameters())}")

            def _init_weights(self):
                """More stable weight initialization strategy"""
                for name, param in self.named_parameters():
                    if 'weight' in name and param.dim() > 1:
                        # Use smaller initialization range
                        if 'point_embed' in name:
                            nn.init.xavier_uniform_(param, gain=0.1)  # Point cloud embedding layer uses small gain
                        elif 'output_proj' in name:
                            nn.init.xavier_uniform_(param, gain=0.5)  # Output layer uses medium gain
                        elif 'time_embed' in name or 'text_embed' in name:
                            nn.init.xavier_uniform_(param, gain=0.3)  # Embedding layers use small gain
                        else:
                            nn.init.xavier_uniform_(param, gain=0.2)  # Other layers use small gain
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.001)  # Smaller bias initialization
                    elif 'pos_embed' in name:
                        nn.init.normal_(param, std=0.001)  # Position encoding uses smaller variance

            def forward(self, x, timestep, encoder_hidden_states=None, return_dict=True):
                batch_size = x.shape[0]
                x = x.contiguous()

                try:
                    # Directly embed point cloud without patch operation
                    # Ensure correct input dimensions
                    if x.shape[1] > 2048:
                        # If too many points, random sample
                        indices = torch.randperm(x.shape[1])[:2048]
                        x = x[:, indices, :]
                    elif x.shape[1] < 2048:
                        # If too few points, repeat to fill
                        repeat_times = 2048 // x.shape[1] + 1
                        x = x.repeat(1, repeat_times, 1)
                        x = x[:, :2048, :]

                    x = self.point_embed(x)

                    # Add position encoding - use gradient protection
                    if x.shape[1] <= self.pos_embed.shape[1]:
                        pos_embed = self.pos_embed_guard(self.pos_embed[:, :x.shape[1], :])
                        x = x + pos_embed
                    else:
                        # If sequence too long, truncate position encoding
                        pos_embed = self.pos_embed_guard(self.pos_embed)
                        x = x + pos_embed[:, :, :]

                    # Time embedding
                    timestep_emb = self.time_embed(timestep.unsqueeze(-1).float())
                    x = x + timestep_emb.unsqueeze(1)

                    # Text embedding
                    if encoder_hidden_states is not None:
                        text_emb = self.text_embed(encoder_hidden_states)
                        x = x + text_emb.unsqueeze(1)

                    # Transformer processing - use post-normalization to save memory
                    if hasattr(self, 'use_checkpoint') and self.use_checkpoint:
                        x = torch.utils.checkpoint.checkpoint(self.transformer, x)
                    else:
                        x = self.transformer(x)

                    # Output projection - use gradient protection
                    x = self.output_guard(self.output_proj(x))

                    # Reshape back to point cloud format
                    x = x.view(batch_size, -1, 3)

                    # Ensure output requires gradients
                    x = x.requires_grad_(True)

                    return {'sample': x}

                except Exception as e:
                    logger.error(f"Transformer forward pass error: {e}")
                    # Return zero tensor as fallback
                    return {'sample': torch.zeros_like(x).requires_grad_(True)}

        # Create pipeline
        pipeline = NOVAPointCloudGenerationPipeline(
            transformer=SimpleNOVAPointCloudTransformer(self.args),
            scheduler=DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2"
            ),
            tokenizer=DummyTokenizer(),
            text_encoder=DummyTextEncoder()
        )

        pipeline.to(self.device)
        logger.info("NOVA pipeline creation completed")

        return pipeline

    def _create_optimizer(self):
        """Create optimizer"""
        # Ensure all parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999)
        )

        logger.info(f"Optimizer created, learning rate: {self.args.learning_rate}")
        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.args.scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.num_epochs,
                eta_min=self.args.learning_rate * 0.01
            )
        else:
            scheduler = MultiStepLR(
                self.optimizer,
                milestones=[self.args.num_epochs // 2, self.args.num_epochs * 3 // 4],
                gamma=0.1
            )

        logger.info(f"Learning rate scheduler created, type: {self.args.scheduler_type}")
        return scheduler

    def _create_data_loaders(self):
        """Create data loaders - Step 2: parallel data preprocessing"""
        logger.info("Creating data loaders...")

        # Training dataset
        train_dataset = AdvancedShapeNetDataset(
            data_root=self.args.data_root,
            split='train',
            max_points=self.args.max_points,
            normalize=True,  # Enable normalization, unified coordinate space
            class_filter=self.args.class_filter,
            enable_cache=True,  
            cache_size=10000,
            precompute_texts=True 
        )

        # Validation dataset 
        val_dataset = AdvancedShapeNetDataset(
            data_root=self.args.data_root,
            split='val',
            max_points=self.args.max_points,
            normalize=True,  # Enable normalization, unified coordinate space
            class_filter=self.args.class_filter,
            enable_cache=True,
            cache_size=2000, 
            precompute_texts=True
        )

        # parallel data preprocessing 
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8, 
            pin_memory=True,  
            drop_last=True,
            persistent_workers=True, 
            prefetch_factor=4  
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size * 2,  # Use larger batch size for validation
            shuffle=False,
            num_workers=4,  # Increase number of worker processes
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=4
        )

        logger.info(f"Data loaders created - Training: {len(train_loader)} batches, Validation: {len(val_loader)} batches")
        return train_loader, val_loader

    def train_epoch(self):
        """Train one epoch: mixed precision training"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            try:

                points = batch['points'].to(self.device, non_blocking=True)
                texts = batch['text']
                # Ensure data requires gradients
                points = points.requires_grad_(True)

                # optimize mixed precision strategy
                with torch.cuda.amp.autocast():
                    # Model forward using mixed precision
                    loss = self._compute_diffusion_loss(points, texts)

                # Backward propagation
                if loss.requires_grad:
                    self.scaler.scale(loss).backward()

                    self._monitor_gradients(batch_idx, "before_clip")

                    # Smart gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    self._smart_grad_clip(max_norm=50.0)  # Use safer threshold

                    self._monitor_gradients(batch_idx, "after_clip")

                    # If gradients are abnormal, perform detailed diagnosis
                    total_norm = 0.0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            total_norm += param.grad.norm().item() ** 2
                    total_norm = total_norm ** 0.5
                    
                    if total_norm < 1e-6 or total_norm > 1000:
                        self._detailed_gradient_diagnosis(batch_idx)

                    # Adaptive learning rate adjustment - only lower when gradient explodes, don't raise when gradient vanishes
                    self._adaptive_lr_adjustment()

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.optimizer.zero_grad()

                # Update statistics
                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / num_batches:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })

                # Log to wandb - minimal logging to improve speed
                if batch_idx % 100 == 0:  # Log every 100 batches, significantly reduce wandb overhead
                    try:
                        wandb.log({
                            'train_loss': loss.item(),
                            'train_avg_loss': total_loss / num_batches,
                            'learning_rate': self.optimizer.param_groups[0]['lr'],
                            'epoch': self.current_epoch + batch_idx / len(self.train_loader),
                            'batch': batch_idx,
                            'gpu_memory_used': torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0,
                            'batch_size': points.shape[0]
                        })
                    except Exception as e:
                        logger.debug(f"wandb logging failed: {e}")  # Use debug level, don't affect training

            except Exception as e:
                logger.error(f"Training batch {batch_idx} error: {e}")
                continue

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Epoch {self.current_epoch + 1} training completed, average loss: {avg_loss:.4f}")

        return avg_loss

    def _smart_grad_clip(self, max_norm=50.0):  # Increase threshold
        """Safe gradient clipping - avoid over-clipping causing gradient vanishing"""
        total_norm = 0.0
        param_count = 0
        clipped_params = 0
        
        # First pass: calculate total norm and preprocess abnormal gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Preprocess abnormal values
                if torch.isnan(param.grad).any():
                    print(f"⚠️  NaN detected in parameter: {name}")
                    param.grad.data = torch.randn_like(param.grad.data) * 0.001
                    continue
                    
                if torch.isinf(param.grad).any():
                    print(f"⚠️  Inf detected in parameter: {name}")
                    param.grad.data = torch.clamp(param.grad.data, -10, 10)
                    continue
                
                # Pre-clip extremely large gradients
                if param.grad.data.abs().max() > 1000:
                    print(f"⚠️  Extremely large gradient detected in parameter: {name}, value: {param.grad.data.abs().max():.2f}")
                    param.grad.data = torch.clamp(param.grad.data, -10, 10)
                
                param_norm = param.grad.data.norm(2)
                if torch.isnan(param_norm) or torch.isinf(param_norm):
                    continue
                    
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            
            # safe clipping strategy - relax clipping conditions
            if total_norm > max_norm:
                # Calculate clipping coefficient, but ensure sufficient signal retention
                clip_coef = max_norm / (total_norm + 1e-8)
                # Retain at least 50% signal, avoid complete clipping
                clip_coef = max(0.5, min(0.9, clip_coef))  # Relax to 50%-90%
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None and not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        # Layer-specific clipping thresholds - set more relaxed thresholds for key layers
                        layer_scale = 1.0
                        if 'output_proj.weight' in name:
                            layer_scale = 1.5  # Output layer more relaxed
                        elif 'time_embed' in name:
                            layer_scale = 1.3  # Time embedding layer
                        elif 'pos_embed' in name:
                            layer_scale = 1.2  # Position encoding slightly relaxed
                        elif 'point_embed' in name:
                            layer_scale = 1.1  # Point cloud embedding layer
                        
                        layer_max_norm = max_norm * layer_scale
                        param_norm = param.grad.data.norm(2)
                        
                        if param_norm > layer_max_norm:
                            param_clip_coef = layer_max_norm / (param_norm + 1e-8)
                            param_clip_coef = max(0.5, min(0.9, param_clip_coef))  # Safe range
                            param.grad.data.mul_(param_clip_coef)
                            clipped_params += 1
                        else:
                            param.grad.data.mul_(clip_coef)
                            clipped_params += 1
                
                print(f"🔧  Safe clipping: total norm {total_norm:.2f} → {total_norm * clip_coef:.2f}, clipped parameters: {clipped_params}/{param_count}")
            else:
                # If gradient norm is normal, don't perform any clipping
                print(f"✅  Gradient normal, no clipping needed: {total_norm:.2f}")
        
        return total_norm

    def _monitor_gradients(self, batch_idx, stage):
        """Monitor gradient information - check for gradient explosion"""
        try:
            total_norm = 0.0
            param_count = 0
            max_grad = 0.0
            min_grad = float('inf')
            grad_norms = []
            has_nan = False
            has_inf = False
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Check if gradient contains nan or inf
                    if torch.isnan(param.grad).any():
                        has_nan = True
                        print(f"⚠️  NaN detected in parameter: {name}")
                        continue
                        
                    if torch.isinf(param.grad).any():
                        has_inf = True
                        print(f"⚠️  Inf detected in parameter: {name}")
                        continue
                    
                    param_norm = param.grad.data.norm(2)
                    if torch.isnan(param_norm) or torch.isinf(param_norm):
                        continue
                        
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    max_grad = max(max_grad, param.grad.data.abs().max().item())
                    min_grad = min(min_grad, param.grad.data.abs().min().item())
                    grad_norms.append(param_norm.item())
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                avg_grad_norm = total_norm / param_count
                
                # Calculate signal retention ratio (numerical)
                signal_retention_ratio = min(1.0, (total_norm / 50.0))
                signal_retention_status = 1 if 0.3 < signal_retention_ratio < 0.7 else 0  # Use numerical
                
                # Output gradient information
                print(f"\n=== Gradient Monitoring [Batch {batch_idx}, {stage}] ===")
                print(f"Total gradient norm: {total_norm:.6f}")
                print(f"Average gradient norm: {avg_grad_norm:.6f}")
                print(f"Maximum gradient value: {max_grad:.6f}")
                print(f"Minimum gradient value: {min_grad:.6f}")
                print(f"Parameter count: {param_count}")
                print(f"Signal retention status: {'good' if signal_retention_status == 1 else 'warning'}")
                if has_nan:
                    print(f"⚠️  NaN gradients detected, processed")
                if has_inf:
                    print(f"⚠️  Inf gradients detected, processed")
                
                # Check for gradient explosion
                if total_norm > 1000.0:
                    print(f"⚠️  Warning: gradient explosion detected! Total gradient norm: {total_norm:.6f}")
                elif total_norm > 100.0:
                    print(f"⚠️  Note: large gradient! Total gradient norm: {total_norm:.6f}")
                elif total_norm < 1e-6:
                    print(f"⚠️  Warning: gradient vanishing detected! Total gradient norm: {total_norm:.6f}")
                else:
                    print(f"✅  Gradient status normal: {total_norm:.6f}")
                
                # Log to logger
                logger.info(f"Batch {batch_idx} {stage} - Total gradient norm: {total_norm:.6f}, max gradient: {max_grad:.6f}, status: {'good' if signal_retention_status == 1 else 'warning'}")
                
                # Try to log to wandb (fix data type issues)
                try:
                    wandb.log({
                        f'gradients/{stage}_total_norm': total_norm,
                        f'gradients/{stage}_avg_norm': avg_grad_norm,
                        f'gradients/{stage}_max_grad': max_grad,
                        f'gradients/{stage}_min_grad': min_grad,
                        f'gradients/{stage}_param_count': param_count,
                        f'gradients/{stage}_signal_retention_ratio': signal_retention_ratio,  # Use numerical
                        f'gradients/{stage}_status': signal_retention_status,  # Use numerical status code
                        'batch': batch_idx
                    })
                except Exception as e:
                    pass
            else:
                print(f"\n=== Gradient Monitoring [Batch {batch_idx}, {stage}] ===")
                print(f"⚠️  All parameter gradients contain NaN or Inf, cleaned")
                    
        except Exception as e:
            print(f"Gradient monitoring error: {e}")

    def _adaptive_lr_adjustment(self):
        """Improved adaptive learning rate adjustment - avoid raising learning rate when gradient vanishes"""
        try:
            # Calculate current gradient norm
            total_norm = 0.0
            param_count = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                
                base_lr = self.optimizer.param_groups[0]['lr']
                min_lr = 1e-7  # Minimum learning rate
                max_lr = 1e-4  # Maximum learning rate
                
                # More conservative learning rate adjustment strategy - only lower when gradient explodes, don't raise when gradient vanishes
                if total_norm > 1000:
                    new_lr = max(min_lr, base_lr * 0.5) 
                    print(f"⚠️  Gradient too large({total_norm:.2f}), learning rate lowered to: {new_lr:.2e}")
                elif total_norm > 100:
                    new_lr = max(min_lr, base_lr * 0.8) 
                    print(f"⚠️  Large gradient({total_norm:.2f}), learning rate lowered to: {new_lr:.2e}")
                elif total_norm < 1e-4:
                    # When gradient vanishes, don't raise learning rate, keep current learning rate
                    new_lr = base_lr  # Keep unchanged
                    print(f"⚠️  Gradient vanishing({total_norm:.2f}), learning rate kept: {new_lr:.2e}")
                elif total_norm < 1e-2:
                    # When gradient is very small, also keep learning rate unchanged
                    new_lr = base_lr
                    print(f"⚠️  Small gradient({total_norm:.2f}), learning rate kept: {new_lr:.2e}")
                else:
                    new_lr = base_lr  # Keep unchanged
                    print(f"✅  Gradient normal({total_norm:.2f}), learning rate kept: {new_lr:.2e}")
                
                # Apply new learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                    
        except Exception as e:
            print(f"Adaptive learning rate adjustment failed: {e}")

    def _compute_diffusion_loss(self, points, texts):
        """Calculate diffusion loss - more stable version"""
        batch_size = points.shape[0]
        
        # Numerical stability: limit point cloud range (maintain dataset original range)
        points = torch.clamp(points, -1.0, 1.0)  # Maintain -1 to 1 range
        
        # Ensure point cloud size is 2048 (consistent with max_points)
        target_size = 2048
        if points.shape[1] != target_size:
            if points.shape[1] > target_size:
                # Random sampling - ensure device consistency
                indices = torch.randperm(points.shape[1], device=points.device)[:target_size]
                points = points[:, indices, :]
            else:
                # Padding
                pad_size = target_size - points.shape[1]
                points = torch.cat([points, torch.zeros(batch_size, pad_size, 3, device=points.device)], dim=1)
        
        # Generate random noise and maintain reasonable range
        noise = torch.randn_like(points)
        noise = torch.clamp(noise, -1.0, 1.0)  # Maintain -1 to 1 noise range
        
        # Random timesteps - ensure device consistency, use smaller timestep range
        max_timesteps = min(100, self.pipeline.scheduler.config.num_train_timesteps)  # Limit maximum timesteps
        timesteps = torch.randint(0, max_timesteps, (batch_size,), device=points.device)
        
        # Use more stable noise addition method
        try:
            # Simplified noise addition process
            t_ratio = timesteps.float() / max_timesteps
            t_ratio = t_ratio.view(-1, 1, 1)  # Expand dimensions to match point cloud shape
            # Numerical stability: ensure ratio is in reasonable range
            t_ratio = torch.clamp(t_ratio, 0.01, 0.99)
            noisy_points = (1 - t_ratio) * points + t_ratio * noise
        except Exception as e:
            logger.warning(f"Using scheduler to add noise failed: {e}, using default method")
            noisy_points = self.pipeline.scheduler.add_noise(points, noise, timesteps)
        
        # Numerical stability: final clipping
        noisy_points = torch.clamp(noisy_points, -1.0, 1.0)
        noisy_points = noisy_points.requires_grad_(True)
        
        try:
            # Model noise prediction
            model_pred = self.model(noisy_points, timesteps)
            
            # Extract prediction results
            if isinstance(model_pred, dict):
                model_pred = model_pred['sample']
            
            # Numerical stability: clip model predictions
            model_pred = torch.clamp(model_pred, -1.0, 1.0)
            model_pred = model_pred.requires_grad_(True)
            
            # Use simplified loss calculation, avoid complex geometric loss
            if self.args.use_only_diffusion:
                # Only use diffusion loss
                loss = nn.functional.mse_loss(model_pred, noise, reduction='mean')
            else:
                # Calculate combined loss - diffusion loss + CD + EMD
                loss = self.criterion(
                    noise_pred=model_pred, 
                    noise_target=noise,
                    pred_points=model_pred,  # Predicted point cloud
                    target_points=points,    # Real point cloud
                    use_only_diffusion=False
                )
            
            # Numerical stability: check if loss is nan or inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Loss is nan or inf, using fallback loss")
                loss = nn.functional.mse_loss(noisy_points, points)
            
            # Additional loss clipping
            loss = torch.clamp(loss, 0.0, 10.0)  # Limit loss range
            
            return loss
            
        except Exception as e:
            logger.error(f"Diffusion loss calculation error: {e}")
            # Use simple MSE loss as fallback
            loss = nn.functional.mse_loss(noisy_points, points)
            return loss



    def validate(self):
        """Validation - simplified version, don't calculate CD/EMD"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                try:
                    points = batch['points'].to(self.device, non_blocking=True)
                    texts = batch['text']
                    
                    with torch.cuda.amp.autocast(enabled=False):
                        loss = self._compute_diffusion_loss(points, texts)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Also output some gradient information during validation (though no gradients are calculated during validation, can monitor model output)
                    if batch_idx % 10 == 0:  # Output every 10 batches
                        print(f"\n=== Validation Monitoring [Batch {batch_idx}] ===")
                        print(f"Validation loss: {loss.item():.6f}")
                        print(f"Point cloud shape: {points.shape}")
                        print(f"Average loss: {total_loss / num_batches:.6f}")
                        
                except Exception as e:
                    logger.error(f"Validation batch error: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Validation completed, average loss: {avg_loss:.4f}")
        
        # Add wandb logging
        try:
            wandb.log({
                'val_loss': avg_loss,
                'val_best_loss': self.best_loss,
                'epoch': self.current_epoch + 1,
                'validation/avg_loss': avg_loss,
                'validation/num_batches': num_batches,
                'validation/is_best': avg_loss < self.best_loss,
                'validation/improvement': self.best_loss - avg_loss if avg_loss < self.best_loss else 0,
                'training_progress/epoch': self.current_epoch + 1,
                'model_state/best_loss_so_far': self.best_loss,
                'model_state/current_lr': self.optimizer.param_groups[0]['lr'],
                'system/gpu_memory_used': torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0,
                'system/gpu_memory_cached': torch.cuda.memory_reserved() / 1024 ** 3 if torch.cuda.is_available() else 0
            })
        except Exception as e:
            logger.debug(f"wandb validation logging failed: {e}")
        
        return avg_loss



    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'args': self.args
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

    def train(self):
        logger.info(f"Starting training")
        logger.info(f"Training configuration: {self.args}")

        start_time = time.time()
        patience = self.args.patience
        patience_counter = 0

        for epoch in range(self.args.num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch()

            # Reduce validation frequency to improve training speed
            if (epoch + 1) % self.args.val_freq == 0:
                val_loss = self.validate()
            else:
                val_loss = float('inf') 

            # Check if it's the best model
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1

            self.save_checkpoint(is_best=is_best)

            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered! {patience} epochs without improvement, stopping training")
                break

            self.scheduler.step()

            try:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss_epoch': train_loss,
                    'val_loss_epoch': val_loss,
                    'best_loss': self.best_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_summary/train_loss': train_loss,
                    'epoch_summary/val_loss': val_loss,
                    'epoch_summary/is_best': is_best,
                    'epoch_summary/epoch': epoch + 1,
                    'epoch_summary/total_epochs': self.args.num_epochs,
                    'epoch_summary/progress': (epoch + 1) / self.args.num_epochs,
                    'model_performance/best_loss_so_far': self.best_loss,
                    'model_performance/current_lr': self.optimizer.param_groups[0]['lr'],
                    'system_info/gpu_memory': torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
                })
            except Exception as e:
                pass

            logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}: "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Best Val Loss: {self.best_loss:.4f}, "
                        f"Patience: {patience_counter}/{patience}")

        total_time = time.time() - start_time
        logger.info(f"Training completed! Total time: {total_time:.2f} seconds")
        logger.info(f"Best validation loss: {self.best_loss:.4f}")

    def _detailed_gradient_diagnosis(self, batch_idx):
        """Detailed gradient diagnosis - identify specific locations of gradient problems"""
        try:
            print(f"\n=== Detailed Gradient Diagnosis [Batch {batch_idx}] ===")
            
            layer_grads = {}
            total_params = 0
            zero_grad_params = 0
            nan_grad_params = 0
            inf_grad_params = 0
            large_grad_params = 0
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    total_params += 1
                    grad_norm = param.grad.norm().item()
                    
                    # Classify gradient problems
                    if grad_norm == 0:
                        zero_grad_params += 1
                    elif torch.isnan(param.grad).any():
                        nan_grad_params += 1
                    elif torch.isinf(param.grad).any():
                        inf_grad_params += 1
                    elif grad_norm > 10:
                        large_grad_params += 1
                    
                    # Group by layer
                    layer_name = name.split('.')[0] if '.' in name else name
                    if layer_name not in layer_grads:
                        layer_grads[layer_name] = []
                    layer_grads[layer_name].append(grad_norm)
            
            # Output statistics
            print(f"Total parameter count: {total_params}")
            print(f"Zero gradient parameters: {zero_grad_params} ({zero_grad_params/total_params*100:.1f}%)")
            print(f"NaN gradient parameters: {nan_grad_params} ({nan_grad_params/total_params*100:.1f}%)")
            print(f"Inf gradient parameters: {inf_grad_params} ({inf_grad_params/total_params*100:.1f}%)")
            print(f"Large gradient parameters: {large_grad_params} ({large_grad_params/total_params*100:.1f}%)")
            
            # Analyze by layer
            print(f"\nGradient analysis by layer:")
            for layer_name, grads in layer_grads.items():
                avg_grad = np.mean(grads)
                max_grad = np.max(grads)
                min_grad = np.min(grads)
                zero_count = sum(1 for g in grads if g == 0)
                
                print(f"  {layer_name}: avg={avg_grad:.6f}, max={max_grad:.6f}, min={min_grad:.6f}, zero_gradients={zero_count}/{len(grads)}")
            
            # Problem diagnosis
            if zero_grad_params / total_params > 0.5:
                print(f"⚠️  Critical problem: over 50% of parameters have zero gradients!")
                print(f"   Suggestion: check loss function, if learning rate is too small, if model is frozen")
            
            if nan_grad_params > 0:
                print(f"⚠️  Problem: found {nan_grad_params} NaN gradients!")
                print(f"   Suggestion: check loss function numerical stability")
            
            if inf_grad_params > 0:
                print(f"⚠️  Problem: found {inf_grad_params} Inf gradients!")
                print(f"   Suggestion: check loss function, gradient clipping")
            
            if large_grad_params > 0:
                print(f"⚠️  Problem: found {large_grad_params} large gradients!")
                print(f"   Suggestion: lower learning rate or strengthen gradient clipping")
            
            # Log to wandb
            try:
                wandb.log({
                    'gradient_diagnosis/total_params': total_params,
                    'gradient_diagnosis/zero_grad_ratio': zero_grad_params / total_params,
                    'gradient_diagnosis/nan_grad_ratio': nan_grad_params / total_params,
                    'gradient_diagnosis/inf_grad_ratio': inf_grad_params / total_params,
                    'gradient_diagnosis/large_grad_ratio': large_grad_params / total_params,
                    'batch': batch_idx
                })
            except Exception as e:
                pass
                
        except Exception as e:
            print(f"Gradient diagnosis failed: {e}")


def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description="NOVA point cloud generation fine-tuning")

    # ===== Data parameters =====
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/.autodl/data',
                        help='ShapeNet data root directory')
    parser.add_argument('--max_points', type=int, default=1024,
                        help='Maximum number of points per point cloud')
    parser.add_argument('--batch_size', type=int, default=32,  # Significantly increase batch size
                        help='Batch size (fully utilize memory)')

    # ===== Training parameters =====
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-6,  # Significantly lower learning rate
                        help='Base learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100,  # Reduce warmup steps
                        help='Learning rate warmup steps')
    parser.add_argument('--lr_decay_steps', type=int, default=1000,  # Reduce decay steps
                        help='Learning rate decay steps')
    parser.add_argument('--grad_clip', type=float, default=50.0,  # Safer gradient clipping threshold
                        help='Gradient clipping threshold')
    parser.add_argument('--weight_decay', type=float, default=1e-5,  # Reduce weight decay
                        help='Weight decay')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                        choices=['cosine', 'step', 'linear'],
                        help='Learning rate scheduler type')

    # ===== Training parameters =====
    parser.add_argument('--class_filter', nargs='*', default=None,
                        help='Class filter')

    # ===== Validation parameters =====
    parser.add_argument('--val_freq', type=int, default=1,  # Validate every epoch
                        help='Validation frequency (validate every N epochs)')
    parser.add_argument('--patience', type=int, default=8,  # Compromise solution
                        help='Early stopping patience (stop after N epochs without improvement)')
    parser.add_argument('--early_stopping', action='store_true', default=True,
                        help='Enable early stopping mechanism')
    parser.add_argument('--save_checkpoint_freq', type=int, default=2,
                        help='Checkpoint save frequency')

    # ===== Output parameters =====
    parser.add_argument('--output_dir', type=str, default='./checkpoints2',
                        help='Output directory')

    # ===== Optimization parameters =====
    parser.add_argument('--enable_cache', action='store_true', default=True,
                        help='Enable data caching')
    parser.add_argument('--cache_size', type=int, default=2000,
                        help='Cache size')
    parser.add_argument('--precompute_texts', action='store_true', default=True,
                        help='Precompute text descriptions')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Data loader worker processes')
    
    # ===== Acceleration techniques =====
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Enable mixed precision training')
    parser.add_argument('--grad_scale', type=float, default=1024,
                        help='Mixed precision gradient scaling coefficient')

    # ===== Loss function parameters =====
    parser.add_argument('--diffusion_weight', type=float, default=0.85,
                        help='Diffusion loss weight')
    parser.add_argument('--cd_weight', type=float, default=0.12,
                        help='Chamfer Distance loss weight')
    parser.add_argument('--emd_weight', type=float, default=0.08,
                        help='Earth Mover\'s Distance loss weight')
    parser.add_argument('--use_only_diffusion', action='store_true', default=False,  # Default not enabled
                        help='Only use diffusion loss (don\'t calculate CD and EMD)')
    
    # ===== Memory optimization =====
    parser.add_argument('--enable_gradient_checkpointing', action='store_true', default=True,
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--memory_efficient_attention', action='store_true', default=True,
                        help='Enable memory efficient attention mechanism')

    return parser.parse_args()


def main():

    args = get_args()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    trainer = AdvancedNOVATrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()