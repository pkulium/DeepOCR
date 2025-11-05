# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import re

import torch
import torch.nn as nn
from timm.models.layers import Mlp
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from safetensors.torch import load_file as safe_load


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class DownSampleBlock(nn.Module):
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x):
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.contiguous()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class DownSample2x2BlockFix(nn.Module):
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = flat_square_2x2(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds


def flat_square_2x2(x):
    n, w, h, c = x.size()
    if w % 2 == 1:
        x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
        n, w, h, c = x.size()
    x = x.contiguous()
    if h % 2 == 1:
        x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
        n, w, h, c = x.size()
    x = x.view(n, w, int(h / 2), int(c * 2))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class DownSample3x3BlockFix(nn.Module):
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = flat_square_3x3(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds


def flat_square_3x3(x):
    n, w, h, c = x.size()
    if w % 3 != 0:
        x = torch.concat([x, torch.zeros((n, 3 - (w % 3), h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
        n, w, h, c = x.size()
    x = x.contiguous()
    if h % 3 != 0:
        x = torch.concat([x, torch.zeros((n, w, 3 - (h % 3), c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
        n, w, h, c = x.size()
    x = x.view(n, w, int(h / 3), int(c * 3))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h / 3), int(w / 3), int(c * 9))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class MultimodalProjectorConfig(PretrainedConfig):
    model_type = "v2l_projector"

    def __init__(self, mm_projector_type: str = None, **kwargs):
        super().__init__()
        self.mm_projector_type = mm_projector_type


class MultimodalProjector(PreTrainedModel):
    config_class = MultimodalProjectorConfig

    def __init__(self, mm_projector_cfg: MultimodalProjectorConfig, config: PretrainedConfig):
        super().__init__(mm_projector_cfg)
        mm_projector_type = mm_projector_cfg.mm_projector_type
        self.downsample_rate = 1
        if mm_projector_type == "identity":
            self.layers = IdentityMap()
        elif mm_projector_type == "linear":
            n_embed = config.hidden_size
            embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
            self.layers = nn.Linear(2048, config.hidden_size)
            torch.manual_seed(42)  
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
            # ckpt = "/lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/checkpoints/sam_clip_ckpt/model_cache/model-00001-of-000001.safetensors"
            # state_dict = safe_load(ckpt)
            
            # # Actually load the parameters into your model
            # if "model.image_newline" in state_dict:
            #     self.image_newline.data = state_dict["model.image_newline"]
                
            # if "model.view_seperator" in state_dict:
            #     self.view_seperator.data = state_dict["model.view_seperator"]
            # del state_dict

            # self.layers = nn.Linear(config.mm_hidden_size, config.hidden_size)
        elif mm_projector_type == "mlp_downsample":
            self.layers = nn.Sequential(
                DownSampleBlock(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.downsample_rate = 2
        elif mm_projector_type == "mlp_downsample_2x2_fix":
            self.layers = nn.Sequential(
                DownSample2x2BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.downsample_rate = 2
        elif mm_projector_type == "mlp_downsample_3x3_fix":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(config.mm_hidden_size * 9, config.mm_hidden_size * 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 3),
                nn.Linear(config.mm_hidden_size * 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.downsample_rate = 3
        elif mm_projector_type == "mlp_downsample_3x3_s2":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(config.mm_hidden_size * 9, config.mm_hidden_size * 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 3),
                nn.Linear(config.mm_hidden_size * 3, config.mm_hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size),
                nn.Linear(config.mm_hidden_size, config.mm_hidden_size // 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size // 3),
                nn.Linear(config.mm_hidden_size // 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        elif mm_projector_type == "mlp_downsample_3x3_s2_new":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(config.mm_hidden_size * 9, config.mm_hidden_size * 4),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.mm_hidden_size * 2),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 2),
                nn.Linear(config.mm_hidden_size * 2, config.mm_hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size),
                nn.Linear(config.mm_hidden_size, config.mm_hidden_size // 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size // 3),
                nn.Linear(config.mm_hidden_size // 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", mm_projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
                self.layers = nn.Sequential(*modules)
            else:
                raise ValueError(f"Unknown projector type: {mm_projector_type}")

        if getattr(config, "ps3", False):
            if getattr(config, "look_close_mode", None) == "after_prompt":
                if getattr(config, "top_down_prompt_head_type", "linear") == "linear":
                    self.top_down_prompt_head = nn.Linear(config.hidden_size, config.mm_hidden_size)
                elif getattr(config, "top_down_prompt_head_type", "linear") == "mlp":
                    self.top_down_prompt_head = Mlp(
                        in_features=config.hidden_size,
                        hidden_features=config.mm_hidden_size * 2,
                        out_features=config.mm_hidden_size,
                        norm_layer=nn.LayerNorm,
                    )
                else:
                    raise NotImplementedError

                for n, p in self.top_down_prompt_head.named_parameters():
                    if "norm" not in n:
                        p.data.uniform_(-0.02, 0.02)

            if getattr(config, "high_res_pos_embed", False):
                self.high_res_pos_embed = nn.Parameter(torch.zeros(1, config.mm_low_res_token_num, config.hidden_size))
                self.high_res_scale_embed = nn.ParameterList(
                    [nn.Parameter(torch.zeros(1, 1, config.hidden_size)) for _ in range(config.mm_scale_num)]
                )

    def forward(self, x, forward_top_down_prompt_head=False, *args, **kwargs):
        if forward_top_down_prompt_head:
            return self.top_down_prompt_head(x)
        if isinstance(x, torch.Tensor):
            return self.layers(x)
        # print(">>>self.image_newline:", self.image_newline)
        images_in_this_batch = []
        for item in x:
            item = item[0]
            if item["local_features"] is None:
                global_features = item["global_features"]
                global_features = self.layers(global_features)
                _, hw, n_dim = global_features.shape
                h = w = int(hw ** 0.5)

                global_features = global_features.view(h, w, n_dim)

                global_features = torch.cat(
                    [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                )

                global_features = global_features.view(-1, n_dim)

                global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)
            else:
                global_features = item["global_features"]
                local_features = item["local_features"]
                crop_shape = item["crop_shape"]
                
                global_features = self.layers(global_features)
                local_features = self.layers(local_features)

                _, hw, n_dim = global_features.shape
                h = w = int(hw ** 0.5)

                _2, hw2, n_dim2 = local_features.shape
                h2 = w2 = int(hw2 ** 0.5)

                width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                global_features = global_features.view(h, w, n_dim)

                global_features = torch.cat(
                    [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                )

                global_features = global_features.view(-1, n_dim)


                local_features = local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2).permute(0, 2, 1, 3, 4).reshape(height_crop_num*h2, width_crop_num*w2, n_dim2)
                local_features = torch.cat(
                    [local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1
                )
                local_features = local_features.view(-1, n_dim2)

                global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)
            # print(">>>global_local_features:", global_local_features.shape)
            images_in_this_batch.append(global_local_features)
            
        # return torch.cat(images_in_this_batch, dim=0)
        return images_in_this_batch

AutoConfig.register("v2l_projector", MultimodalProjectorConfig)
AutoModel.register(MultimodalProjectorConfig, MultimodalProjector)
