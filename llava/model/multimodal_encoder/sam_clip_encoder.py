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
from .sam_clip.image_process import DeepOCRProcessor

class DeepOCRImagePreprocessor(BaseImageProcessor):
    """
    Image preprocessor for DeepOCR that handles image transformation
    with dynamic cropping support.
    """
    
    model_input_names = ["pixel_values", "images_crop", "images_spatial_crop"]
    
    def __init__(
        self
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
        self.processor = DeepOCRProcessor()
        self.name = "sam_clip_processor"

    @property
    def size(self):
        """Returns the processor size configuration."""
        return {"height": self.processor.image_size, "width": self.processor.image_size}

    def preprocess(
        self,
        images,
        return_tensors=None
    ):
        if not isinstance(images, list):
            images = [images]
        results = self.processor.tokenize_with_images(images)
        pixel_values, images_crop, images_spatial_crop = results[0]
        return {"pixel_values": pixel_values, "images_spatial_crop": images_spatial_crop, "images_crop": images_crop}
 
class SAMCLIPVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig = None):
        super().__init__(model_name_or_path, config)
        
        self.image_processor = DeepOCRImagePreprocessor()
        

        dtype = torch.bfloat16  # default dtype        
        # Create SAMCLIP config if none provided
        self.cfg_only = SAMCLIPConfig(ckpt=model_name_or_path)
        self.vision_tower = SAMCLIP(self.cfg_only).to(dtype)
        self.vision_tower.gradient_checkpointing = False
            
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
            patch_size = 14
            return (image_size // patch_size) ** 2

class SAMCLIPVisionTowerS2(VisionTowerS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig = None):
        super().__init__(model_name_or_path, config)
        
        self.image_processor = DeepOCRImagePreprocessor()
        
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