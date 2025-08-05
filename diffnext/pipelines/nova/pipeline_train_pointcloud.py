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
"""NOVA Point Cloud training pipeline."""

from typing import Dict
import torch
import numpy as np
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from diffnext.engine import engine_utils
from diffnext.pipelines.builder import build_diffusion_scheduler
from diffnext.pipelines.nova.pipeline_utils import PipelineMixin


class NOVATrainPointCloudPipeline(DiffusionPipeline, PipelineMixin):
    """Pipeline for training NOVA Point Cloud models."""

    _optional_components = ["transformer", "scheduler", "text_encoder", "tokenizer"]

    def __init__(
        self,
        transformer=None,
        scheduler=None,
        text_encoder=None,
        tokenizer=None,
        trust_remote_code=True,
    ):
        super(NOVATrainPointCloudPipeline, self).__init__()
        self.text_encoder = self.register_module(text_encoder, "text_encoder")
        self.tokenizer = self.register_module(tokenizer, "tokenizer")
        self.transformer = self.register_module(transformer, "transformer")
        self.scheduler = self.register_module(scheduler, "scheduler")
        self.transformer.noise_scheduler = build_diffusion_scheduler(self.scheduler)
        self.transformer.sample_scheduler, self.guidance_scale = self.scheduler, 5.0
        
        # 点云数据统计
        self.dataset_mean = None
        self.dataset_std = None

    @property
    def model(self) -> torch.nn.Module:
        """Return the trainable model."""
        return self.transformer

    def configure_model(self, config) -> torch.nn.Module:
        """Configure the trainable model."""
        ckpt_lvl = config.model.get("gradient_checkpointing", 0)
        self.model.loss_repeat = config.model.get("loss_repeat", 4)
        
        # 配置点云编码器
        if hasattr(self.model, "point_cloud_encoder"):
            [setattr(blk, "mlp_checkpointing", ckpt_lvl) for blk in self.model.point_cloud_encoder.blocks]
        
        # 配置点云解码器
        if hasattr(self.model, "point_cloud_decoder"):
            [setattr(blk, "mlp_checkpointing", ckpt_lvl > 2) for blk in self.model.point_cloud_decoder.blocks]
        
        # 冻结文本编码器
        if hasattr(self.model, "text_embed"):
            engine_utils.freeze_module(self.model.text_embed.norm)
            self.model.text_embed.encoders = [self.tokenizer, self.text_encoder]
        
        self.model.pipeline_preprocess = self.preprocess
        return self.model.train()

    def set_dataset_stats(self, mean, std):
        """设置数据集的均值和标准差"""
        self.dataset_mean = torch.from_numpy(mean).to(self.device)
        self.dataset_std = torch.from_numpy(std).to(self.device)

    def prepare_point_cloud_latents(self, inputs: Dict):
        """准备点云潜在表示"""
        if "point_clouds" in inputs:
            # 直接输入点云数据
            point_clouds = torch.as_tensor(inputs.pop("point_clouds"), device=self.device)
            point_clouds = point_clouds.to(dtype=self.dtype)
            
            # 归一化点云数据
            if self.dataset_mean is not None and self.dataset_std is not None:
                point_clouds = (point_clouds - self.dataset_mean) / self.dataset_std
            
            # 转换为潜在表示格式 (B, 3, N)
            if point_clouds.dim() == 3 and point_clouds.shape[-1] == 3:
                point_clouds = point_clouds.transpose(1, 2)  # (B, 3, N)
            
            inputs["x"] = point_clouds
        elif "latents" in inputs:
            # 输入预计算的潜在表示
            x = torch.as_tensor(inputs.pop("latents"), device=self.device)
            x = x.to(dtype=self.dtype if x.is_floating_point() else torch.int64)
            inputs["x"] = x

    def encode_prompt(self, inputs: Dict):
        """编码文本提示"""
        inputs["c"] = inputs.get("c", [])
        if inputs.get("prompt", None) is not None and hasattr(self.transformer, "text_embed"):
            inputs["c"].append(self.transformer.text_embed(inputs.pop("prompt")))

    def preprocess(self, inputs: Dict) -> Dict:
        """定义pipeline预处理"""
        if not self.model.training:
            raise RuntimeError("Expected a trainable model.")
        
        self.prepare_point_cloud_latents(inputs)
        self.encode_prompt(inputs)
        
        return inputs

    def forward(self, inputs: Dict) -> Dict:
        """前向传播"""
        inputs = self.preprocess(inputs)
        return self.model(inputs)

    def sample(self, prompt, num_samples=1, num_points=15000, guidance_scale=5.0):
        """生成点云样本"""
        self.model.eval()
        
        with torch.no_grad():
            # 准备输入
            inputs = {
                "prompt": prompt,
                "num_samples": num_samples,
                "num_points": num_points,
                "guidance_scale": guidance_scale
            }
            
            # 生成点云
            outputs = self.model.generate_point_clouds(inputs)
            
            # 反归一化
            if self.dataset_mean is not None and self.dataset_std is not None:
                outputs = outputs * self.dataset_std + self.dataset_mean
            
            return outputs.cpu().numpy()

    def save_checkpoint(self, path, **kwargs):
        """保存检查点"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "dataset_mean": self.dataset_mean.cpu().numpy() if self.dataset_mean is not None else None,
            "dataset_std": self.dataset_std.cpu().numpy() if self.dataset_std is not None else None,
            **kwargs
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # 加载调度器状态
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # 加载数据集统计
        if "dataset_mean" in checkpoint and checkpoint["dataset_mean"] is not None:
            self.dataset_mean = torch.from_numpy(checkpoint["dataset_mean"]).to(self.device)
        if "dataset_std" in checkpoint and checkpoint["dataset_std"] is not None:
            self.dataset_std = torch.from_numpy(checkpoint["dataset_std"]).to(self.device)
        
        return checkpoint 