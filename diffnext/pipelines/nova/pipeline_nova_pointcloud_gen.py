# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""NOVA Point Cloud Generation Pipeline based on NOVA architecture with dynamic partitioning and autoregressive diffusion."""

from typing import List, Optional, Union
import numpy as np
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffnext.pipelines.nova.pipeline_utils import NOVAPipelineOutput, PipelineMixin


class NOVAPointCloudPipelineOutput:
    """Output class for NOVA point cloud generation."""

    def __init__(self, point_clouds: List[np.ndarray], colors: Optional[List[np.ndarray]] = None):
        self.point_clouds = point_clouds
        self.colors = colors


class NOVAPointCloudGenerationPipeline(DiffusionPipeline, PipelineMixin):
    """Improved NOVA pipeline for 3D point cloud generation from text prompts with dynamic partitioning."""

    _optional_components = ["transformer", "scheduler", "text_encoder", "tokenizer"]
    model_cpu_offload_seq = "text_encoder->transformer"

    def __init__(
        self,
        transformer=None,
        scheduler=None,
        text_encoder=None,
        tokenizer=None,
        trust_remote_code=True,
        use_autoregressive=True,  # 默认启用自回归生成
        num_subsets=20,  # 动态划分的子集数量
    ):
        super().__init__()
        
        # Basic components
        self.transformer = transformer
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.use_autoregressive = use_autoregressive
        self.num_subsets = num_subsets
        
        # Register components
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        
        # 如果transformer支持自回归生成，启用它
        if hasattr(self.transformer, 'enable_autoregressive_generation') and self.use_autoregressive:
            self.transformer.enable_autoregressive_generation()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 64,
        num_diffusion_steps: int = 25,
        guidance_scale: float = 1.0,  # Default to no guidance for 3D point clouds
        num_points: int = 15000,
        point_cloud_size: int = 1024,
        negative_prompt: Optional[Union[str, List[str]]] = None,  # Not used for 3D point clouds
        num_point_clouds_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # Not used for 3D point clouds
        disable_progress_bar: bool = False,
        output_type: str = "numpy",
        use_autoregressive: Optional[bool] = None,  # 允许运行时切换
        **kwargs,
    ) -> NOVAPointCloudPipelineOutput:
        """Generate 3D point clouds from text prompts using NOVA architecture with dynamic partitioning.

        Args:
            prompt (str or List[str]): The prompt(s) to guide point cloud generation.
            num_inference_steps (int): The number of autoregressive steps.
            num_diffusion_steps (int): The number of denoising steps.
            guidance_scale (float): The classifier guidance scale.
            num_points (int): The number of points in each generated point cloud.
            point_cloud_size (int): The size of the point cloud representation.
            negative_prompt (str or List[str]): The prompt(s) to guide what to not include.
            num_point_clouds_per_prompt (int): The number of point clouds per prompt.
            generator (torch.Generator): A generator to make generation deterministic.
            latents (torch.Tensor): Pre-generated noisy latents.
            prompt_embeds (torch.Tensor): Pre-generated text embeddings.
            negative_prompt_embeds (torch.Tensor): Pre-generated negative text embeddings.
            disable_progress_bar (bool): Whether to disable the progress bar.
            output_type (str): The output format of the generated point clouds.
            use_autoregressive (bool): Whether to use autoregressive generation with dynamic partitioning.

        Returns:
            NOVAPointCloudPipelineOutput: The pipeline output containing generated point clouds.
        """
        self.guidance_scale = guidance_scale
        
        # 运行时切换自回归模式
        if use_autoregressive is not None:
            if use_autoregressive and hasattr(self.transformer, 'enable_autoregressive_generation'):
                self.transformer.enable_autoregressive_generation()
            elif not use_autoregressive and hasattr(self.transformer, 'disable_autoregressive_generation'):
                self.transformer.disable_autoregressive_generation()

        # Prepare inputs (similar to NOVA's input preparation)
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        device = self.transformer.device
        dtype = self.transformer.dtype

        # Encode prompts (reuse NOVA's text encoding)
        prompt_embeds = self.encode_prompt(
            prompt,
            num_point_clouds_per_prompt,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # Prepare latents (similar to NOVA's latent preparation)
        if latents is None:
            latents = self.prepare_latents(
                batch_size=batch_size * num_point_clouds_per_prompt,
                point_cloud_size=point_cloud_size,
                generator=generator,
                device=device,
                dtype=dtype,
            )

        # Set timesteps (similar to NOVA's timestep setting)
        self.scheduler.set_timesteps(num_diffusion_steps)
        timesteps = self.scheduler.timesteps

        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator)

        # Generate point clouds using NOVA's diffusion loop with dynamic partitioning
        point_clouds = []
        colors = []

        for i, t in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual (similar to NOVA's noise prediction)
            noise_pred = self.transformer(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            # Perform guidance (similar to NOVA's classifier-free guidance)
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute the previous noisy sample x_t -> x_t-1 (similar to NOVA's denoising)
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # Decode point clouds (similar to NOVA's decoding)
        latents = 1 / self.scheduler.init_noise_sigma * latents

        # Convert to point clouds with dynamic partitioning
        for i in range(batch_size * num_point_clouds_per_prompt):
            point_cloud = self.latents_to_point_cloud_with_partitioning(
                latents[i:i+1],
                num_points=num_points
            )
            point_clouds.append(point_cloud)

            # Generate colors based on point positions
            color = self.generate_point_colors(point_cloud)
            colors.append(color)

        # Convert to numpy if requested
        if output_type == "numpy":
            point_clouds = [pc.cpu().numpy() for pc in point_clouds]
            colors = [c.cpu().numpy() for c in colors]

        return NOVAPointCloudPipelineOutput(point_clouds=point_clouds, colors=colors)

    def latents_to_point_cloud_with_partitioning(self, latents: torch.Tensor, num_points: int = 15000) -> torch.Tensor:
        """Convert latents to point cloud coordinates with dynamic partitioning."""
        # Reshape latents to point cloud format
        # latents shape: (1, 3, point_cloud_size) -> (num_points, 3)
        point_cloud = latents.squeeze(0).transpose(0, 1)  # (point_cloud_size, 3)

        # 如果启用了自回归生成，使用动态划分
        if hasattr(self.transformer, 'use_autoregressive') and self.transformer.use_autoregressive:
            return self._generate_with_dynamic_partitioning(point_cloud, num_points)
        else:
            # 使用标准方法
            return self._standard_point_cloud_generation(point_cloud, num_points)

    def _generate_with_dynamic_partitioning(self, initial_points: torch.Tensor, num_points: int) -> torch.Tensor:
        """使用动态区域划分与自回归扩散生成点云"""
        from diffnext.models.transformers.transformer_pointcloud_nova import (
            dynamic_partition, compute_local_density, adaptive_sampling
        )
        
        # 1. 动态划分（20子集）
        order, subsets = dynamic_partition(initial_points.unsqueeze(0), k=self.num_subsets)
        
        # 存储生成的子集
        generated_subsets = {}
        
        # 2. 自回归生成每个子集
        for i, subset_idx in enumerate(order):
            current_subset = subsets[subset_idx]
            
            # 计算局部密度用于动态资源分配
            density = compute_local_density(current_subset)
            base_size = num_points // self.num_subsets
            density_factor = 0.5
            target_size = int(base_size * (1 + density_factor * (density.mean() - 0.5)))
            target_size = max(100, min(target_size, base_size * 2))  # 限制范围
            
            # 自适应采样
            current_points = adaptive_sampling(current_subset, target_size)
            
            # 3. 扩散生成当前子集
            t = torch.tensor(i / float(self.num_subsets), device=initial_points.device)
            
            # 获取已生成的邻居子集作为条件
            neighbor_subsets = []
            for prev_idx in order[:i]:
                if prev_idx in generated_subsets:
                    neighbor_subsets.append(generated_subsets[prev_idx])
            
            # 生成当前子集（使用transformer的自回归扩散功能）
            if hasattr(self.transformer, 'autoregressive_diffusion'):
                generated_subset = self.transformer.autoregressive_diffusion(
                    current_points, 
                    neighbor_subsets, 
                    t
                )
            else:
                # 回退到标准生成
                generated_subset = self._standard_subset_generation(current_points, t)
            
            generated_subsets[subset_idx] = generated_subset
        
        # 4. 空间重组
        final_point_cloud = torch.cat([generated_subsets[i] for i in range(self.num_subsets)], dim=1)
        
        return final_point_cloud.squeeze(0)  # 移除batch维度

    def _standard_subset_generation(self, subset: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """标准子集生成方法"""
        # 简单的噪声添加和去噪
        noise = torch.randn_like(subset)
        noisy_subset = subset + 0.1 * noise
        return noisy_subset

    def _standard_point_cloud_generation(self, point_cloud: torch.Tensor, num_points: int) -> torch.Tensor:
        """标准点云生成方法"""
        # Sample points if needed
        if point_cloud.size(0) > num_points:
            indices = torch.randperm(point_cloud.size(0))[:num_points]
            point_cloud = point_cloud[indices]
        elif point_cloud.size(0) < num_points:
            # Repeat points to reach desired number
            repeat_times = num_points // point_cloud.size(0) + 1
            point_cloud = point_cloud.repeat(repeat_times, 1)
            point_cloud = point_cloud[:num_points]

        # Use tanh for non-linear transformation
        point_cloud = torch.tanh(point_cloud)
        
        # Add structured noise to simulate real objects
        noise_scale = 0.1
        structured_noise = torch.randn_like(point_cloud) * noise_scale
        point_cloud = point_cloud + structured_noise
        
        # Re-normalize to reasonable range
        point_cloud = torch.clamp(point_cloud, -1.0, 1.0)
        
        return point_cloud

    def prepare_latents(
        self,
        batch_size: int,
        point_cloud_size: int,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Prepare initial latents for point cloud generation (similar to NOVA's latent preparation)."""
        device = device or self.transformer.device
        dtype = dtype or self.transformer.dtype

        # Create random latents (similar to NOVA's noise initialization)
        latents = torch.randn(
            (batch_size, 3, point_cloud_size),  # 3D coordinates
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def generate_point_colors(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """Generate colors for point cloud based on positions."""
        # Use position-based coloring
        colors = torch.abs(point_cloud)  # Use absolute values as RGB
        colors = torch.clamp(colors, 0, 1)  # Clamp to [0, 1]

        # Add some variation
        colors = colors + 0.1 * torch.randn_like(colors)
        colors = torch.clamp(colors, 0, 1)

        return colors

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_point_clouds_per_prompt: int,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode the prompt into text embeddings (reuse NOVA's text encoding)."""
        if prompt_embeds is None:
            if isinstance(prompt, str):
                prompt = [prompt]

            # Tokenize prompts (similar to NOVA's tokenization)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            # Encode text (similar to NOVA's text encoding)
            prompt_embeds = self.text_encoder(text_input_ids)[0]

        prompt_embeds = prompt_embeds.to(dtype=self.transformer.dtype, device=self.transformer.device)

        # Duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_point_clouds_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_point_clouds_per_prompt, seq_len, -1)

        # Get unconditional embeddings for classifier free guidance
        if self.guidance_scale > 1.0:
            uncond_tokens = [""] * len(prompt)
            max_length = prompt_embeds.shape[1]

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids)[0]
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.transformer.dtype, device=self.transformer.device
            )

            # Duplicate unconditional embeddings for each generation per prompt
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_point_clouds_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                bs_embed * num_point_clouds_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def prepare_extra_step_kwargs(self, generator: Optional[torch.Generator]) -> dict:
        """Prepare extra kwargs for the scheduler step."""
        import inspect
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def enable_autoregressive_generation(self):
        """启用自回归生成模式"""
        self.use_autoregressive = True
        if hasattr(self.transformer, 'enable_autoregressive_generation'):
            self.transformer.enable_autoregressive_generation()
        print("✅ Enabled autoregressive point cloud generation with dynamic partitioning")

    def disable_autoregressive_generation(self):
        """禁用自回归生成模式"""
        self.use_autoregressive = False
        if hasattr(self.transformer, 'disable_autoregressive_generation'):
            self.transformer.disable_autoregressive_generation()
        print("✅ Disabled autoregressive generation, using standard forward pass")

    def visualize_cluster_centers(self):
        """可视化聚类中心（用于调试）"""
        if hasattr(self.transformer, 'visualize_cluster_centers'):
            return self.transformer.visualize_cluster_centers()
        else:
            print("❌ Transformer does not support cluster center visualization")
            return None


 
