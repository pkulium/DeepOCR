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

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import math
from PIL import Image, ImageOps
from transformers import PretrainedConfig
from torchvision import transforms
import numpy as np
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from .config import IMAGE_SIZE, BASE_SIZE, CROP_MODE, PRINT_NUM_VIS_TOKENS, PROMPT
from .deepencoder import build_sam_vit_b, build_clip_l

class SAMCLIPConfig(PretrainedConfig):
    """Configuration class for SAMCLIP model."""
    model_type = "samclip"
    
    def __init__(
        self,
        hidden_size=2048,
        sam_hidden_size=1024,
        clip_hidden_size=1024,
        image_size=640,
        base_size=448,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        normalize=True,
        cropping=True,
        min_crops=1,
        max_crops=12,
        ckpt="",
        tile_tag="2D",
        global_view_pos="",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.sam_hidden_size = sam_hidden_size
        self.clip_hidden_size = clip_hidden_size
        self.image_size = image_size
        self.base_size = base_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        self.cropping = cropping
        self.min_crops = min_crops
        self.max_crops = max_crops
        self.ckpt = ckpt
        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos

class SAMCLIP_hf(PreTrainedModel):
    config_class = SAMCLIPConfig
    
    def __init__(self, config: SAMCLIPConfig):
        super().__init__(config)
        
        # Initialize SAM and CLIP models
        # Note: You'll need to adjust these based on your actual checkpoint loading
        self.sam_model = build_sam_vit_b(config.ckpt)  # Remove checkpoint parameter for now
        self.vision_model = build_clip_l(config.ckpt)    # Remove checkpoint parameter for now

        n_embed = 1280
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

        self.is_loaded = True

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """Custom from_pretrained method."""
        # Create config
        config = cls.config_class()
        
        # Create model instance
        model = cls(config)
        return model

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ):
        sam_model = getattr(self, 'sam_model', None)
        # sam_model = self.sam_model
        vision_model = getattr(self, 'vision_model', None)

        idx = 0
        # sam_model = torch.jit.script(sam_model)
        images = [(images_crop, pixel_values)]
        images_spatial_crop = images_spatial_crop
        # start_time = time.time()
        for image, crop_shape in zip(images, images_spatial_crop):
            images_in_this_batch = []

            patches = image[0]
            image_ori = image[1]

            with torch.no_grad():
            # with torch.inference_mode(): 
                
                if torch.sum(patches).item() != 0:
                    print(">>>with patches")
                    # P, C, H, W = patches.shape
                    crop_flag = 1
                    local_features_1 = sam_model(patches)

                    local_features_2 = vision_model(patches, local_features_1)  
                    # vit_time = time.time()
                    local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                    local_features = self.projector(local_features)


                    global_features_1 = sam_model(image_ori)
                    global_features_2 = vision_model(image_ori, global_features_1) 
                    global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                    global_features = self.projector(global_features)

                    print('=====================')
                    print('BASE: ', global_features.shape)
                    print('PATCHES: ', local_features.shape)
                    print('=====================')

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

                    # end_time = time.time()

                    # print('sam: ', sam_time - start_time)
                    # print('vit: ', vit_time - sam_time)
                    # print('all: ', end_time - start_time)

                    # exit()
                
                else:
                    print(">>>without patches")
                    global_features_1 = sam_model(image_ori)
                    global_features_2 = vision_model(image_ori, global_features_1) 
                    global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                    global_features = self.projector(global_features)
                    print('=====================')
                    print('BASE: ', global_features.shape)
                    print('NO PATCHES')
                    print('=====================')
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)


                    global_features = global_features.view(h, w, n_dim)

                    global_features = torch.cat(
                        [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                    )

                    global_features = global_features.view(-1, n_dim)

                    global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)

                images_in_this_batch.append(global_local_features)

            idx += 1

        return images_in_this_batch
            
    def _process_image_input(
            self, image_input) -> torch.Tensor:
        # image_input: [pixel_values, images_crop, images_spatial_crop]
    
        # >>>pixel_values: torch.Size([1, 3, 1024, 1024])
        pixel_values = image_input[0].to(torch.bfloat16)
        # print(image_input[1][0].shape)
        # print(type(image_input[1]))
        # exit()

        # images_crop = image_input[1].to(torch.bfloat16)
        images_crop = image_input[1]
        # images_crop = image_input[1]
        images_spatial_crop = image_input[2].to(dtype=torch.long)

        # local_start = time.time()
        vision_features = self._pixel_values_to_embedding(
            pixel_values=pixel_values, images_crop = images_crop,  images_spatial_crop=images_spatial_crop)

        # local_total_time = time.time() - local_start

        # print('encoder_time: ', local_total_time)
        # exit()
        return vision_features

    def _parse_and_validate_image_input(
            self, **kwargs: object):
        
        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        images_crop = kwargs.pop("images_crop", None)


        if pixel_values is None or torch.sum(pixel_values).item() == 0:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(images_spatial_crop, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(images_spatial_crop)}")
            
            if not isinstance(images_crop, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image crop. "
                                 f"Got type: {type(images_crop)}")

            return [pixel_values, images_crop, images_spatial_crop]


        raise AssertionError("This line should be unreachable.")
 
    def forward(
        self,
        pixel_values,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[torch.FloatTensor] = None,
        images_spatial_crop=None,
    ):
        # print(">>>pixel_values:", pixel_values)
        image_input = self._parse_and_validate_image_input(**pixel_values)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)


        # Stack embeddings if multiple images
        if len(vision_embeddings) == 1:
            last_hidden_state = vision_embeddings[0].unsqueeze(0)  # Add batch dimension
        else:
            # Pad sequences to same length and stack
            max_len = max(emb.shape[0] for emb in vision_embeddings)
            padded_embeddings = []
            for emb in vision_embeddings:
                if emb.shape[0] < max_len:
                    padding = torch.zeros(
                        max_len - emb.shape[0], 
                        emb.shape[1], 
                        dtype=emb.dtype, 
                        device=emb.device
                    )
                    emb = torch.cat([emb, padding], dim=0)
                padded_embeddings.append(emb)
            last_hidden_state = torch.stack(padded_embeddings, dim=0)
        
        
        
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state
        )
        