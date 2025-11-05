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
from transformers import PretrainedConfig
from llava.model.multimodal_encoder.sam_clip.modeling_sam_clip import SAMCLIP, SAMCLIPConfig
from llava.model.multimodal_encoder.vision_encoder import VisionTower, VisionTowerS2
from transformers.image_processing_utils import BaseImageProcessor
from typing import List, Optional, Tuple, Union
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch.nn as nn
from abc import ABC


def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform



class BaseTransform(ABC):

    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError

class BasicImageTransform(BaseTransform):
    def __init__(
        self, 
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):
        self.mean = mean
        self.std = std
    
        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)
    
    def __call__(self, x):
        x = self.transform(x)
        return x

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=9, image_size=640, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    # print(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # print(target_aspect_ratio)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio




def process_image_for_model(image, base_size=1024, image_size=640, crop_mode=True):
    """
    Process an image file and return pixel values and spatial crop information.
    
    Args:
        image_file: Path to the image file or PIL Image object
        base_size: Size for the global view (default: 1024)
        image_size: Size for cropped patches (default: 640)
        crop_mode: Whether to use dynamic cropping (default: True)
    
    Returns:
        dict: Dictionary containing:
            - pixel_values: Tuple of (images_crop, images_ori) tensors
            - images_spatial_crop: Tensor of crop ratios [width_crop_num, height_crop_num]
            - images_crop: Tensor of cropped image patches
    """
    
    # Initialize image transform
    image_transform = BasicImageTransform(
        mean=(0.5, 0.5, 0.5), 
        std=(0.5, 0.5, 0.5), 
        normalize=True
    )
    
    images_list = []
    images_crop_list = []
    images_spatial_crop = []
    
    if crop_mode:
        # Check if cropping is needed
        if image.size[0] <= 640 and image.size[1] <= 640:
            crop_ratio = [1, 1]
        else:
            # Dynamic preprocessing to get crops
            images_crop_raw, crop_ratio = dynamic_preprocess(image, image_size=image_size)
        
        # Process global view
        global_view = ImageOps.pad(
            image, 
            (base_size, base_size),
            color=tuple(int(x * 255) for x in image_transform.mean)
        )
        images_list.append(image_transform(global_view).to(torch.bfloat16))
        
        width_crop_num, height_crop_num = crop_ratio
        images_spatial_crop.append([width_crop_num, height_crop_num])
        
        # Process local views (patches) if needed
        if width_crop_num > 1 or height_crop_num > 1:
            for i in range(len(images_crop_raw)):
                images_crop_list.append(
                    image_transform(images_crop_raw[i]).to(torch.bfloat16)
                )
    else:
        # No crop mode - single global view
        global_view = ImageOps.pad(
            image,
            (image_size, image_size),
            color=tuple(int(x * 255) for x in image_transform.mean)
        )
        images_list.append(image_transform(global_view).to(torch.bfloat16))
        
        width_crop_num, height_crop_num = 1, 1
        images_spatial_crop.append([width_crop_num, height_crop_num])
    
    # Convert to tensors
    if len(images_list) == 0:
        images_ori = torch.zeros((1, 3, image_size, image_size))
        images_spatial_crop_tensor = torch.zeros((1, 2), dtype=torch.long)
        images_crop = torch.zeros((1, 3, base_size, base_size))
    else:
        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)
        
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, base_size, base_size))
    
    return {
        "pixel_values": images_ori,
        "images_spatial_crop": images_spatial_crop_tensor,
        "images_crop": images_crop
    }

class DeepOCRImagePreprocessor_hf(BaseImageProcessor):
    """
    Image preprocessor for DeepOCR that handles image transformation
    with dynamic cropping support.
    """
    
    model_input_names = ["pixel_values", "images_crop", "images_spatial_crop"]
    
    def __init__(
        self,
        image_size
    ):
        """
        Args:
            image_size: Size for image crops (default: 640)
            base_size: Size for base/global view (default: 448)
            image_mean: Mean values for normalization
            image_std: Std values for normalization
            normalize: Whether to normalize images
            cropping: Whether to use dynamic cropping
            min_crops: Minimum number of crops
            max_crops: Maximum number of crops
        """
        super().__init__()
        self.name = "sam_clip_processor"
        self.image_size = image_size

    @property
    def size(self):
        """Returns the processor size configuration."""
        return {"height": self.image_size, "width": self.image_size}

    def preprocess(
        self,
        images,
        return_tensors=None
    ):
        if isinstance(images, list):
            return [process_image_for_model(image) for image in images]
        return process_image_for_model(images)
        
 
class SAMCLIPVisionTower_hf(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig = None):
        super().__init__(model_name_or_path, config)
        
        self.image_processor = DeepOCRImagePreprocessor_hf()
        

        dtype = torch.bfloat16  # default dtype        
        # Create SAMCLIP config if none provided
        self.cfg_only = SAMCLIPConfig(ckpt=model_name_or_path)
        self.vision_tower = SAMCLIP(self.cfg_only).to(dtype)
            
        self.is_loaded = True 
    
    # ADD THIS PROPERTY:
    @property
    def num_patches(self):
        """Return number of patches from the vision tower."""
        if hasattr(self.vision_tower, 'num_patches'):
            return self.vision_tower.num_patches
        elif hasattr(self.vision_tower, 'clip_model') and hasattr(self.vision_tower.clip_model, 'embeddings'):
            return self.vision_tower.clip_model.embeddings.num_patches
        else:
            image_size = 640
            patch_size = 16
            return (image_size // patch_size) ** 2

class SAMCLIPVisionTowerS2_hf(VisionTowerS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig = None):
        super().__init__(model_name_or_path, config)
        
        self.image_processor = DeepOCRImagePreprocessor_hf()
        
        # Handle config being None
        if config is not None and hasattr(config, 'model_dtype'):
            dtype = eval(config.model_dtype) if isinstance(config.model_dtype, str) else config.model_dtype
        else:
            dtype = torch.float32  # default dtype
        
        # Create SAMCLIP config if none provided
        self.config = SAMCLIPConfig()
        
        # Try to load from pretrained first, fallback to direct instantiation
        try:
            self.vision_tower = SAMCLIP.from_pretrained(model_name_or_path)
        except Exception as e:
            print(f"Warning: from_pretrained failed ({e}), using direct instantiation")
            self.vision_tower = SAMCLIP(self.config)
            
        # Set dtype if needed
        if dtype != torch.float32:
            self.vision_tower = self.vision_tower.to(dtype=dtype)
            
        self.is_loaded = True