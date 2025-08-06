<div align="center">

<h1>NOVA: 3D Point Cloud Generation without Vector Quantization</h1>

<p align="center">
<a href="https://arxiv.org/abs/2412.14169"><img src="https://img.shields.io/badge/ArXiv-2412.14169-%23840707.svg" alt="ArXiv"></a>
<a href="https://huggingface.co/spaces/BAAI/nova-d48w1024-sdxl1024"><img src="https://img.shields.io/badge/🤗 Demo-T2I-%26840707.svg" alt="T2IDemo"></a>
<a href="https://huggingface.co/spaces/BAAI/nova-d48w1024-osp480"><img src="https://img.shields.io/badge/🤗 Demo-T2V-%26840707.svg" alt="T2VDemo"></a>
<a href="http://bitterdhg.github.io/NOVA_page"><img src="https://img.shields.io/badge/Webpage-NOVA-%237CB4F7.svg" alt="Webpage"></a>
</p>

[Haoge Deng](https://scholar.google.com/citations?user=S2sbvjgAAAAJ&hl=zh-CN&oi=ao)<sup>1,4*</sup>, [Ting Pan](https://scholar.google.com/citations?&user=qQv6YbsAAAAJ)<sup>2,4*</sup>, [Haiwen Diao](https://scholar.google.com/citations?user=46eCjHQAAAAJ&hl=zh-CN)<sup>3,4*</sup>, [Zhengxiong Luo](https://scholar.google.com/citations?user=Sz1yTZsAAAAJ&hl=zh-CN)<sup>4*</sup>, [Yufeng Cui](https://scholar.google.com/citations?user=5Ydha2EAAAAJ&hl=zh-CN)<sup>4</sup><br>
[Huchuan Lu](https://scholar.google.com/citations?user=D3nE0agAAAAJ&hl=zh-CN)<sup>3</sup>, [Shiguang Shan](https://scholar.google.com/citations?user=Vkzd7MIAAAAJ&hl=en)<sup>2</sup>, [Yonggang Qi](https://scholar.google.com.tw/citations?user=pQNpf7cAAAAJ&hl=zh-CN&oi=ao)<sup>1†</sup>, [Xinlong Wang](https://scholar.google.com/citations?user=DPz0DjYAAAAJ&hl=zh-CN)<sup>4†</sup><br>

[BUPT](https://www.bupt.edu.cn)<sup>1</sup>, [ICT-CAS](http://english.ict.cas.cn)<sup>2</sup>, [DLUT](https://en.dlut.edu.cn)<sup>3</sup>, [BAAI](https://www.baai.ac.cn/en)<sup>4</sup><br>
<sup>*</sup> Equal Contribution, <sup>†</sup> Corresponding Author
<br><br><image src="assets/537104209024d050d2853fade7d998b.png"/>
</div>

We present **NOVA** (**NO**n-Quantized **V**ideo **A**utoregressive Model) extended for 3D point cloud generation, a model that enables efficient 3D point cloud generation from text prompts. **NOVA** reformulates the point cloud generation problem as non-quantized autoregressive modeling with spatial point-by-point prediction. **NOVA** generalizes well and enables diverse zero-shot 3D point cloud generation abilities in one unified model.

## 🚀News
- ```[Jul 2025]``` Codebase refactor with **Accelerate**, **OmegaConf** and **Wandb**.
- ```[Feb 2025]``` Released [Evaluation Guide](./docs/evaluation.md).
- ```[Feb 2025]``` Released [Training Guide](./docs/training.md)
- ```[Jan 2025]``` Accepted by ICLR 2025. [[OpenReview]](https://openreview.net/forum?id=JE9tCwe3lp) & [[Poster]](https://iclr.cc/virtual/2025/poster/30117).
- ```[Dec 2024]``` Released [Project Page](http://bitterdhg.github.io/NOVA_page)
- ```[Dec 2024]``` Released 🤗 Online Demo (<a href="https://huggingface.co/spaces/BAAI/nova-d48w1024-sdxl1024"><b>T2I</b></a>, <a href="https://huggingface.co/spaces/BAAI/nova-d48w1024-osp480"><b>T2V</b></a>)
- ```[Dec 2024]``` Released [paper](https://arxiv.org/abs/2412.14169), [weights](#model-zoo), and [Quick Start](#2-quick-start) guide and Gradio Demo [local code](#3-gradio-demo) .

## ✨Highlights

- 🔥 **Novel Approach**: Non-quantized 3D point cloud autoregressive generation.
- 🔥 **State-of-the-art Performance**: High efficiency with state-of-the-art text-to-3D point cloud results.
- 🔥 **Unified Modeling**: Multi-task capabilities in a single unified model.
- 🔥 **3D Point Cloud Specialized**: Architecture specifically optimized for 3D point cloud generation.

## 🗄️Model Zoo
<a id="model-zoo"></a>
> See detailed description in [Model Zoo](./docs/model_zoo.md)

### Text to 3D Point Cloud
<a id="text-to-pointcloud-weight"></a>

| Model       | Parameters | Point Cloud Size | Data |  Weight                                                               | GenEval | DPGBench |
|:-----------:|:----------:|:----------:|:----:|:---------------------------------------------------------------------:|:--------:|:-------:|
| NOVA-0.6B   | 0.6B       | 1024 points     | 16M  | [🤗 HF link](https://huggingface.co/BAAI/nova-d48w1024-sd512)          | 0.75   |   81.76   |
| NOVA-0.3B   | 0.3B       | 2048 points     | 600M | [🤗 HF link](https://huggingface.co/BAAI/nova-d48w768-sdxl1024)        | 0.67   |   80.60   |
| NOVA-0.6B   | 0.6B       | 2048 points     | 600M | [🤗 HF link](https://huggingface.co/BAAI/nova-d48w1024-sdxl1024)       | 0.69   |   82.25   |
| NOVA-1.4B   | 1.4B       | 2048 points     | 600M | [🤗 HF link](https://huggingface.co/BAAI/nova-d48w1536-sdxl1024)       | 0.71   |   83.01   |

### Text to Video (Original Feature)
<a id="text-to-video-weight"></a>

| Model       | Parameters  | Resolution | Data | Weight                                                                | VBench |
|:-----------:|:-----------:|:----------:|:----:|-----------------------------------------------------------------------|:------:|
| NOVA-0.6B   | 0.6B        | 33x768x480 | 20M  | [🤗 HF link](https://huggingface.co/BAAI/nova-d48w1024-osp480)        |  80.12  |

## 📖Table of Contents
- [1. Installation](#1-installation)
  - [1.1 From Source](#from-source)
  - [1.2 From Git](#from-git)
- [2. Quick Start](#2-quick-start)
  - [2.1 Text to 3D Point Cloud](#text-to-pointcloud-quickstart)
  - [2.2 Text to Image](#text-to-image-quickstart)
  - [2.3 Text to Video](#text-to-video-quickstart)
- [3. Gradio Demo](#3-gradio-demo)
- [4. Train](#4-train)
- [5. Inference](#5-inference)
- [6. Evaluation](#6-evaluation)

## 1. Installation
### 1.1 From Source

<a id="from-source"></a>
Clone this repository to local disk and install:

```bash
pip install diffusers transformers accelerate imageio-ffmpeg omegaconf wandb
git clone https://github.com/baaivision/NOVA.git
cd NOVA && pip install .
```

### 1.2 From Git
<a id="from-git"></a>

You can also install from the remote repository **if you have set your Github SSH key**: 

```bash
pip install diffusers transformers accelerate imageio-ffmpeg omegaconf wandb
pip install git+ssh://git@github.com/baaivision/NOVA.git
```

## 2. Quick Start
### 2.1 Text to 3D Point Cloud
<a id="text-to-pointcloud-quickstart"></a>

```python
import torch
from diffnext.pipelines import NOVAPointCloudPipeline

model_id = "BAAI/nova-d48w768-sdxl1024"
model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
pipe = NOVAPointCloudPipeline.from_pretrained(model_id, **model_args)
pipe = pipe.to("cuda")

prompt = "a shiba inu wearing a beret and black turtleneck."
point_clouds = pipe(prompt, num_points=15000)

# Save point cloud data
import numpy as np
for i, pc in enumerate(point_clouds.point_clouds):
    np.save(f"shiba_inu_{i}.npy", pc)
    
# Visualize point cloud (requires matplotlib)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
pc = point_clouds.point_clouds[0]
ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=point_clouds.colors[0], s=1)
plt.savefig("shiba_inu_pointcloud.png")
plt.show()
```

### 2.2 Text to Image
<a id="text-to-image-quickstart"></a>

```python
import torch
from diffnext.pipelines import NOVAPipeline

model_id = "BAAI/nova-d48w768-sdxl1024"
model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
pipe = NOVAPipeline.from_pretrained(model_id, **model_args)
pipe = pipe.to("cuda")

prompt = "a shiba inu wearing a beret and black turtleneck."
image = pipe(prompt).images[0]
    
image.save("shiba_inu.jpg")
```

### 2.3  Text to Video
<a id="text-to-video-quickstart"></a>

```python
import os
import torch
from diffnext.pipelines import NOVAPipeline
from diffnext.utils import export_to_image, export_to_video
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_id = "BAAI/nova-d48w1024-osp480"
low_memory = False

model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
pipe = NOVAPipeline.from_pretrained(model_id, **model_args)

if low_memory:
    # Use CPU model offload routine and expandable allocator if OOM.
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to("cuda")

# Text to Video
prompt = "Many spotted jellyfish pulsating under water."
video = pipe(prompt, max_latent_length=9).frames[0]
export_to_video(video, "jellyfish.mp4", fps=12)

# Increase AR and diffusion steps for better video quality.
video = pipe(
  prompt,
  max_latent_length=9,
  num_inference_steps=128,  # default: 64
  num_diffusion_steps=100,  # default: 25
).frames[0]
export_to_video(video, "jellyfish_v2.mp4", fps=12)

# You can also generate images from text, with the first frame as an image.
prompt = "Many spotted jellyfish pulsating under water."
image = pipe(prompt, max_latent_length=1).frames[0, 0]
export_to_image(image, "jellyfish.jpg")
```

## 3. Gradio Demo

```bash
# For text-to-3D-pointcloud demo
python scripts/app_nova_pointcloud.py --model "BAAI/nova-d48w1024-sdxl1024" --device 0

# For text-to-image demo
python scripts/app_nova_t2i.py --model "BAAI/nova-d48w1024-sdxl1024" --device 0

# For text-to-video demo
python scripts/app_nova_t2v.py --model "BAAI/nova-d48w1024-osp480" --device 0
```

## 4. Train
- See [Training Guide](./docs/training.md)

## 5. Evaluation
- See [Evaluation Guide](./docs/evaluation.md)

## 6. Inference
- See [Inference Guide](./docs/inference.md)

## 📋Todo List
- [X] [Model zoo](#model-zoo)
- [X] [Quick Start](#2-quick-start)
- [X] [Gradio Demo](#3-gradio-demo)
- [X] [Training guide](#4-train)
- [X] [Evaluation guide](#5-evaluation)
- [ ] Inference guide
- [ ] Prompt Writer
- [ ] Larger model size
- [ ] Additional downstream tasks: Image editing, Video editing, Controllable generation
- [ ] 3D Point Cloud editing features
- [ ] Point cloud to point cloud transformation

## Citation
If you find this repository useful, please consider giving a star ⭐ and citation 🦖:
```
@article{deng2024nova,
  title={Autoregressive Video Generation without Vector Quantization},
  author={Deng, Haoge and Pan, Ting and Diao, Haiwen and Luo, Zhengxiong and Cui, Yufeng and Lu, Huchuan and Shan, Shiguang and Qi, Yonggang and Wang, Xinlong},
  journal={arXiv preprint arXiv:2412.14169},
  year={2024}
}
```

## Acknowledgement

We thank the repositories: [MAE](https://github.com/facebookresearch/mae), [MAR](https://github.com/LTH14/mar), [MaskGIT](https://github.com/google-research/maskgit), [DiT](https://github.com/facebookresearch/DiT), [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [CogVideo](https://github.com/THUDM/CogVideo), [FLUX](https://github.com/black-forest-labs/flux), [OpenMuse](https://github.com/huggingface/open-muse) and [CodeWithGPU](https://github.com/seetacloud/codewithgpu).
## License
Code and models are licensed under [Apache License 2.0](LICENSE).
