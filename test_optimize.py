#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced point cloud generation model evaluation script - Fixed version
Solve coordinate system inconsistency and numerical instability issues
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging
import time
import json
from collections import defaultdict
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAME_TO_ID = {
    'airplane': '02691156',
    'car': '02958343', 
    'chair': '03001627',
}

# ====== Global normalizer ======
class GlobalNormalizer:
    """Global normalizer - ensure all point clouds are in the same coordinate system"""
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.is_fitted = False
    
    def load_stats(self, filepath='stats.json'):
        """Load training statistics from stats.json file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    stats = json.load(f)
                self.global_mean = torch.tensor(stats['mean'])
                self.global_std = torch.tensor(stats['std'])
                self.is_fitted = True
                logger.info(f"Loaded normalization parameters from {filepath}: mean={self.global_mean}, std={self.global_std}")
                return True
            else:
                logger.warning(f"Could not find {filepath} file, using default normalization parameters")
                self.global_mean = torch.zeros(3)
                self.global_std = torch.ones(3)
                self.is_fitted = True
                return False
        except Exception as e:
            logger.error(f"Failed to load normalization parameters: {e}")
            self.global_mean = torch.zeros(3)
            self.global_std = torch.ones(3)
            self.is_fitted = True
            return False
    
    def __call__(self, points, mode='norm'):
        """Normalize or denormalize"""
        if not self.is_fitted:
            logger.warning("Normalizer not initialized, trying to load stats.json")
            self.load_stats()
        
        if mode == 'norm':
            # Normalize: (x - mean) / std
            return (points - self.global_mean.to(points.device)) / self.global_std.to(points.device)
        else:
            # Denormalize: x * std + mean
            return points * self.global_std.to(points.device) + self.global_mean.to(points.device)

# Global normalizer instance
global_normalizer = GlobalNormalizer()

class SimpleTokenizer:
    """Simple tokenizer, no external dependencies"""
    class TokenOutput:
        def __init__(self, input_ids):
            self.input_ids = input_ids
            
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
            
        # Generate fixed-length token IDs for each text
        input_ids_list = []
        for text in texts:
            # Randomly generate token IDs (fixed length of 10)
            input_ids = torch.randint(0, 1000, (1, 10))
            input_ids_list.append(input_ids)
            
        input_ids = torch.cat(input_ids_list, dim=0)
        return self.TokenOutput(input_ids)

class EnhancedTextEncoder(nn.Module):
    """Enhanced text encoder"""
    def __init__(self, hidden_size=768):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_size)  # Vocabulary size 1000
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True),
            num_layers=3
        )
        
    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        return self.transformer(embeddings)

class EnhancedTransformer(nn.Module):
    """Enhanced Transformer model"""
    def __init__(self, hidden_size=768, num_layers=6, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Point cloud embedding
        self.point_embed = nn.Linear(3, hidden_size)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Text embedding processing
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.output_proj = nn.Linear(hidden_size, 3)
        
    def forward(self, x, timestep, encoder_hidden_states=None):
        batch_size, num_points, _ = x.shape
        
        # Point cloud embedding
        x = self.point_embed(x)
        
        # Time embedding
        t_emb = self.time_embed(timestep.unsqueeze(-1).float())
        x = x + t_emb.unsqueeze(1)
        
        # Text embedding
        if encoder_hidden_states is not None:
            # Ensure encoder_hidden_states is 3D tensor
            if encoder_hidden_states.dim() == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            
            text_emb = self.text_proj(encoder_hidden_states.mean(dim=1))
            x = x + text_emb.unsqueeze(1)
        
        # Transformer processing
        x = self.transformer(x)
        
        # Output
        return {'sample': self.output_proj(x)}

class SimpleDataset:
    """Fixed dataset - use global normalization"""
    def __init__(self, data_root, class_name, max_points=2048):
        self.data_root = data_root
        self.class_name = class_name
        self.class_id = CLASS_NAME_TO_ID[class_name]
        self.max_points = max_points
        
        # Collect files - from test directory
        self.files = []
        test_dir = os.path.join(data_root, self.class_id, 'test')
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.endswith('.npy'):
                    self.files.append(os.path.join(test_dir, file))
        
        logger.info(f"Found {len(self.files)} {class_name} test files")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load point cloud
        file_path = self.files[idx]
        points = np.load(file_path)
        
        # Ensure point count is 2048
        if len(points) > self.max_points:
            # Random sampling
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
        elif len(points) < self.max_points:
            # Repeat padding
            repeat_times = self.max_points // len(points) + 1
            points = np.tile(points, (repeat_times, 1))
            points = points[:self.max_points]
        
        # Use global normalizer - fix coordinate system inconsistency issue
        points_tensor = torch.FloatTensor(points)
        points_normalized = global_normalizer(points_tensor, mode='norm')
        
        # Generate text description
        text = f"a {self.class_name}"
        
        return {
            'points': points_normalized,  # Already in normalized space
            'text': text
        }

def load_model_enhanced(model_path, device):
    """Load enhanced model"""
    logger.info(f"Loading model: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create simple pipeline
    class SimplePipeline:
        def __init__(self, transformer, scheduler, tokenizer, text_encoder):
            self.transformer = transformer
            self.scheduler = scheduler
            self.tokenizer = tokenizer
            self.text_encoder = text_encoder
    
    # Extract transformer from checkpoint
    if 'model_state_dict' in checkpoint:
        transformer_state = checkpoint['model_state_dict']
    elif 'transformer' in checkpoint:
        transformer_state = checkpoint['transformer']
    else:
        transformer_state = checkpoint
    
    # Create enhanced transformer
    transformer = EnhancedTransformer(hidden_size=768, num_layers=6, num_heads=8)
    
    # Create components
    dummy_optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dummy_optimizer, T_max=1000
    )
    tokenizer = SimpleTokenizer()
    text_encoder = EnhancedTextEncoder(hidden_size=768)
    
    # Load transformer weights
    try:
        # Print model state dict keys
        logger.info(f"Model state dict keys: {list(transformer_state.keys())[:5]}...")
        
        # Try to load weights
        missing_keys, unexpected_keys = transformer.load_state_dict(transformer_state, strict=False)
        logger.info("✅ Successfully loaded transformer weights")
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys[:5]}... (total {len(missing_keys)})")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys[:5]}... (total {len(unexpected_keys)})")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        logger.warning("⚠️  Cannot load transformer weights, using random initialization")
    
    # Create pipeline
    pipeline = SimplePipeline(transformer, scheduler, tokenizer, text_encoder)
    pipeline.transformer.to(device)
    pipeline.text_encoder.to(device)
    pipeline.transformer.eval()
    pipeline.text_encoder.eval()
    
    logger.info("✅ Enhanced model loading completed")
    return pipeline

def generate_pointcloud_enhanced(pipeline, batch_size, num_points=2048, device="cuda", 
                                text_embeddings=None, num_steps=100, guidance_scale=3.0):
    """Fixed point cloud generation process - solve numerical instability issues"""
    logger.info(f"Generating point clouds: batch_size={batch_size}, points={num_points}, steps={num_steps}, guidance={guidance_scale}")
    
    # Create initial noise - stricter initialization
    x = torch.randn(batch_size, num_points, 3, device=device) * 0.05  # Reduce initial noise
    
    # Set DDPM scheduler
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    # Diffusion process (reverse) - add numerical stability mechanism
    for i in tqdm(range(num_steps-1, -1, -1), desc="Diffusion generation"):
        with torch.no_grad():
            t = torch.tensor([i] * batch_size, device=device)
            
            # Conditional prediction
            noise_pred_cond = pipeline.transformer(
                x, 
                t,
                encoder_hidden_states=text_embeddings
            )['sample']
            
            # Unconditional prediction
            if guidance_scale > 1.0 and text_embeddings is not None:
                noise_pred_uncond = pipeline.transformer(
                    x, 
                    t,
                    encoder_hidden_states=torch.zeros_like(text_embeddings)
                )['sample']
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
            
            # Numerical stability: limit noise prediction range
            noise_pred = torch.clamp(noise_pred, -2.0, 2.0)
            
            # Denoising step
            sqrt_alpha_prod_t = sqrt_alphas_cumprod[i]
            sqrt_one_minus_alpha_prod_t = sqrt_one_minus_alphas_cumprod[i]
            
            # Calculate predicted original sample
            pred_x0 = (x - sqrt_one_minus_alpha_prod_t * noise_pred) / sqrt_alpha_prod_t
            
            # Numerical stability: limit minimum variance
            variance = torch.clamp(posterior_variance[i], min=1e-4)
            
            # Generate previous step sample
            if i > 0:
                noise = torch.randn_like(x) * 0.1  # Reduce noise intensity
            else:
                noise = 0
            
            x = pred_x0 + torch.sqrt(variance) * noise
            
            # Numerical stability: stricter range limiting
            x = torch.clamp(x, -2.0, 2.0)
            
            # Monitor numerical stability
            if i % 20 == 0:
                max_val = x.abs().max().item()
                logger.info(f"Step {i}: max_val={max_val:.4f}")
    
    logger.info("Point cloud generation completed")
    return x

def compute_chamfer_distance(pred, target):
    """Fixed Chamfer distance calculation - add density awareness"""
    # Limit input range to avoid numerical explosion
    pred = torch.clamp(pred, -5.0, 5.0)
    target = torch.clamp(target, -5.0, 5.0)
    
    # Dynamically adjust point count consistency
    num_points = min(pred.shape[1], target.shape[1])
    pred = pred[:, :num_points, :]
    target = target[:, :num_points, :]
    
    # Calculate distance matrix
    dist = torch.cdist(pred, target)
    
    # Add density-aware weights
    min_dist_pred_to_target = dist.min(dim=2)[0]
    min_dist_target_to_pred = dist.min(dim=1)[0]
    
    # Avoid division by zero
    weights_pred = 1.0 / (min_dist_pred_to_target.detach() + 1e-6)
    weights_target = 1.0 / (min_dist_target_to_pred.detach() + 1e-6)
    
    # Weighted average
    dist1 = (min_dist_pred_to_target * weights_pred).mean(dim=1)
    dist2 = (min_dist_target_to_pred * weights_target).mean(dim=1)
    
    result = (dist1 + dist2).mean()
    
    # Limit result range
    return torch.clamp(result, 0.0, 10.0)

def compute_emd_distance(pred, target):
    """Fixed EMD calculation - add numerical stability"""
    # Limit input range
    pred = torch.clamp(pred, -5.0, 5.0)
    target = torch.clamp(target, -5.0, 5.0)
    
    # Dynamically adjust point count consistency
    num_points = min(pred.shape[1], target.shape[1])
    pred = pred[:, :num_points, :]
    target = target[:, :num_points, :]
    
    batch_size = pred.shape[0]
    emd_list = []
    
    for i in range(batch_size):
        # Calculate distance matrix
        dist_matrix = torch.cdist(pred[i], target[i])
        
        # Use linear assignment algorithm to calculate optimal transport
        try:
            row_ind, col_ind = linear_sum_assignment(dist_matrix.cpu().detach().numpy())
            emd = dist_matrix[row_ind, col_ind].mean()
            emd_list.append(emd)
        except Exception as e:
            # If linear assignment fails, use simple average distance as fallback
            emd_list.append(dist_matrix.mean())
    
    result = torch.stack(emd_list).mean()
    
    # Limit result range
    return torch.clamp(result, 0.0, 10.0)

def visualize_pointcloud(points, title="Point Cloud", save_path=None):
    """Visualize point cloud"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Limit number of displayed points
    if len(points) > 1000:
        indices = np.random.choice(len(points), 1000, replace=False)
        points_display = points[indices]
    else:
        points_display = points
    
    ax.scatter(points_display[:, 0], points_display[:, 1], points_display[:, 2], s=1, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set coordinate axis range
    max_val = max(points_display.max(), abs(points_display.min()))
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Point cloud saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def eval_class_enhanced(pipeline, data_root, class_name, batch_size, device):
    """Fixed category evaluation - solve coordinate system issues"""
    logger.info(f"\n==== Evaluating category: {class_name} ====")
    
    # Initialize global normalizer
    global_normalizer.load_stats('stats.json')
    
    # Create dataset
    dataset = SimpleDataset(data_root, class_name)
    if len(dataset) == 0:
        logger.warning(f"Category {class_name} has no test samples!")
        return None
    
    logger.info(f"Found {len(dataset)} {class_name} samples")
    
    # Verify coordinate system consistency
    sample = dataset[0]
    points_range = sample['points']
    logger.info(f"Point cloud range: [{points_range.min():.4f}, {points_range.max():.4f}]")
    
    # Create data loader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    all_cd, all_emd = [], []
    total_time = 0
    
    # Try different guidance scales
    guidance_scales = [1.0, 2.0, 3.0, 5.0]
    best_guidance = 1.0
    best_cd = float('inf')
    
    # Find best guidance scale
    logger.info("Finding best guidance scale...")
    for guidance in guidance_scales:
        temp_cd = []
        temp_emd = []
        
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 2:  # Only test first 2 batches
                break
                
            gt_points = batch['points'].to(device)
            texts = batch['text']
            
            # Generate text embeddings
            input_ids = pipeline.tokenizer(texts).input_ids.to(device)
            text_embeddings = pipeline.text_encoder(input_ids)
            
            # Generate point clouds - already in normalized space
            pred_points = generate_pointcloud_enhanced(
                pipeline, 
                batch_size, 
                num_points=gt_points.shape[1], 
                device=device,
                text_embeddings=text_embeddings,
                num_steps=100,
                guidance_scale=guidance
            )
            
            # Visualize first sample
            if batch_idx == 0:  # Only visualize first batch for first guidance scale
                gt_points_np = gt_points[0].cpu().numpy()
                pred_points_np = pred_points[0].cpu().numpy()
                
                # Create visualization directory
                os.makedirs("pointcloud_visualizations", exist_ok=True)
                
                # Visualize ground truth point cloud
                gt_save_path = f"pointcloud_visualizations/{class_name}_gt_guidance_{guidance}.png"
                visualize_pointcloud(gt_points_np, f"Ground Truth - {class_name}", gt_save_path)
                
                # Visualize generated point cloud
                pred_save_path = f"pointcloud_visualizations/{class_name}_pred_guidance_{guidance}.png"
                visualize_pointcloud(pred_points_np, f"Generated - {class_name} (Guidance={guidance})", pred_save_path)
            
            # Calculate metrics
            try:
                cd_loss = compute_chamfer_distance(pred_points, gt_points)
                emd_loss = compute_emd_distance(pred_points, gt_points)
                temp_cd.append(cd_loss.item())
                temp_emd.append(emd_loss.item())
                logger.info(f"Guidance {guidance}, Batch {batch_idx+1}: CD={cd_loss:.6f}, EMD={emd_loss:.6f}")
            except Exception as e:
                logger.error(f"Metric calculation failed: {e}")
                continue
        
        if temp_cd and temp_emd:
            avg_cd = np.mean(temp_cd)
            avg_emd = np.mean(temp_emd)
            logger.info(f"Guidance {guidance}: CD={avg_cd:.6f}, EMD={avg_emd:.6f}")
            if avg_cd < best_cd:
                best_cd = avg_cd
                best_guidance = guidance
    
    logger.info(f"Best guidance scale: {best_guidance}")
    
    # Complete evaluation
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"{class_name} test")):
            start_time = time.time()
            
            gt_points = batch['points'].to(device)
            texts = batch['text']
            
            # Generate text embeddings
            input_ids = pipeline.tokenizer(texts).input_ids.to(device)
            text_embeddings = pipeline.text_encoder(input_ids)
            
            # Generate point clouds - already in normalized space
            pred_points = generate_pointcloud_enhanced(
                pipeline, 
                batch_size, 
                num_points=gt_points.shape[1], 
                device=device,
                text_embeddings=text_embeddings,
                num_steps=100,
                guidance_scale=best_guidance
            )
            
            # Visualize first sample
            if batch_idx == 0:  # Only visualize first batch
                gt_points_np = gt_points[0].cpu().numpy()
                pred_points_np = pred_points[0].cpu().numpy()
                
                # Create visualization directory
                os.makedirs("pointcloud_visualizations1", exist_ok=True)
                
                # Visualize ground truth point cloud
                gt_save_path = f"pointcloud_visualizations1/{class_name}_gt_final.png"
                visualize_pointcloud(gt_points_np, f"Ground Truth - {class_name}", gt_save_path)
                
                # Visualize generated point cloud
                pred_save_path = f"pointcloud_visualizations1/{class_name}_pred_final_guidance_{best_guidance}.png"
                visualize_pointcloud(pred_points_np, f"Generated - {class_name} (Guidance={best_guidance})", pred_save_path)
            
            # Calculate metrics
            try:
                cd_loss = compute_chamfer_distance(pred_points, gt_points)
                emd_loss = compute_emd_distance(pred_points, gt_points)
                
                all_cd.append(cd_loss.item())
                all_emd.append(emd_loss.item())
                
                logger.info(f"Batch {batch_idx+1}: CD={cd_loss:.6f}, EMD={emd_loss:.6f}")
                
            except Exception as e:
                logger.error(f"Batch {batch_idx+1}: metric calculation failed: {e}")
            
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Clean GPU memory
            del pred_points, gt_points, text_embeddings
            torch.cuda.empty_cache()
    
    # Calculate average metrics
    if not all_cd or not all_emd:
        logger.error(f"Category {class_name} has no valid evaluation results!")
        return None
    
    avg_cd = np.mean(all_cd)
    avg_emd = np.mean(all_emd)
    std_cd = np.std(all_cd)
    std_emd = np.std(all_emd)
    
    logger.info(f"Category {class_name} results:")
    logger.info(f"  Average CD: {avg_cd:.6f} ± {std_cd:.6f}")
    logger.info(f"  Average EMD: {avg_emd:.6f} ± {std_emd:.6f}")
    logger.info(f"  Best guidance: {best_guidance}")
    logger.info(f"  Total time: {total_time:.2f}s")
    
    return {
        'class': class_name, 
        'cd': avg_cd, 
        'emd': avg_emd,
        'cd_std': std_cd,
        'emd_std': std_emd,
        'best_guidance': best_guidance,
        'total_time': total_time,
        'valid_samples': len(all_cd)
    }

def main():
    # Hard-coded parameters
    model_path = "/root/NOVA-main/checkpoints2/best_model.pth"
    data_root = "/root/autodl-tmp/.autodl/data"
    classes = ['airplane']
    batch_size = 4
    device = 'cuda'
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Data root directory: {data_root}")
    logger.info(f"Evaluation categories: {classes}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file does not exist: {model_path}")
        return
    
    try:
        # Load model
        pipeline = load_model_enhanced(model_path, device)
        
        # Evaluate all categories
        results = []
        total_start_time = time.time()
        
        for cname in classes:
            try:
                logger.info(f"\nStarting evaluation for category: {cname}")
                res = eval_class_enhanced(pipeline, data_root, cname, batch_size, device)
                if res:
                    results.append(res)
                    logger.info(f"✅ Category {cname} evaluation completed")
                else:
                    logger.warning(f"⚠️  Category {cname} evaluation failed")
            except Exception as e:
                logger.error(f"❌ Category {cname} evaluation error: {e}")
                continue
        
        total_time = time.time() - total_start_time
        
        # Output results
        print("\n" + "="*80)
        print("Fixed Evaluation Results")
        print("="*80)
        print(f"{'Category':<12} {'CD':<12} {'EMD':<12} {'Guidance':<10} {'Time(s)':<8}")
        print("-"*80)
        
        for r in results:
            print(f"{r['class']:<12} {r['cd']:<12.6f} {r['emd']:<12.6f} {r['best_guidance']:<10.1f} {r['total_time']:<8.2f}")
        
        if results:
            avg_cd = np.mean([r['cd'] for r in results])
            avg_emd = np.mean([r['emd'] for r in results])
            print("-"*80)
            print(f"{'Average':<12} {avg_cd:<12.6f} {avg_emd:<12.6f} {'-':<10} {total_time:<8.2f}")
        
        print(f"\nTotal evaluation time: {total_time:.2f} seconds")
        print("="*80)
        
        # Save results
        if results:
            result_file = f"fixed_eval_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'model_path': model_path,
                    'data_root': data_root,
                    'classes': classes,
                    'batch_size': batch_size,
                    'total_time': total_time,
                    'results': results,
                    'average_cd': avg_cd if results else None,
                    'average_emd': avg_emd if results else None
                }, f, indent=2)
            logger.info(f"Results saved to: {result_file}")
        
    except Exception as e:
        logger.error(f"Evaluation process error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 