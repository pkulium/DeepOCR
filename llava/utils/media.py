import glob
import os
import tempfile
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import PIL
import PIL.Image
import requests
from transformers import PretrainedConfig

from llava.constants import MEDIA_TOKENS
from llava.media import Image, Video
from llava.utils import make_list
from llava.utils.logging import logger
__all__ = ["extract_media"]

import fitz
import io

def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    pdf2images
    """
    images = []
    
    pdf_document = fitz.open(pdf_path)
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        PIL.Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = PIL.Image.open(io.BytesIO(img_data)).convert("RGB")  # Use PIL.Image explicitly
        else:
            img_data = pixmap.tobytes("png")
            img = PIL.Image.open(io.BytesIO(img_data))  # Use PIL.Image explicitly
            if img.mode in ('RGBA', 'LA'):
                background = PIL.Image.new('RGB', img.size, (255, 255, 255))  # Use PIL.Image explicitly
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
        
        images.append(img)
    
    pdf_document.close()
    return images

def _extract_image(image: Union[Image, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, Image):
        if image.path.startswith("http://") or image.path.startswith("https://"):
            image = PIL.Image.open(requests.get(image.path, stream=True).raw)
        else:
            # Check if it's a PDF file
            if image.path.lower().endswith('.pdf'):
                # Convert PDF to images and return the first page
                pdf_images = pdf_to_images_high_quality(image.path)
                if pdf_images:
                    image = pdf_images[0]  # Use first page
                else:
                    # Fallback to a blank image if PDF conversion fails
                    logger.warning(f"Failed to convert PDF to images: {image.path}. Using blank image as fallback.")
                    image = PIL.Image.new('RGB', (224, 224), color='white')
            else:
                image = PIL.Image.open(image.path)
    return image

def _load_video_bytesio(video_bytesio: BytesIO, *, num_frames: int) -> List[PIL.Image.Image]:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
        temp_video.write(video_bytesio.read())
        temp_video_name = temp_video.name
        return _load_video(temp_video_name, num_frames=num_frames)


def _load_video(video_path: str, *, num_frames: int, fps: float) -> List[PIL.Image.Image]:
    # Load video frames from a directory
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        indices = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        return [PIL.Image.open(frame_paths[index]) for index in indices]

    # Load video frames from a video file
    vidcap = cv2.VideoCapture(video_path)
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)

    # Find the last frame as frame count might not be accurate
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    else:
        raise ValueError(f"Video '{video_path}' has no frames.")

    duration_sec = frame_count / video_fps if video_fps > 0 else 0

    # Extract frames uniformly
    # If FPS is specified, use that to compute timestamps
    if fps > 0:
        timestamps = np.arange(0, duration_sec, 1.0 / fps)
        timestamps = timestamps[:num_frames]  # Clamp
        indices = [int(t * video_fps) for t in timestamps]
        logger.info(f"timestamps used: {timestamps} with frame numbers {indices}")
    else:
        indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)

    frames = {}
    for index in indices:
        if index in frames:
            continue
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = vidcap.read()
        if not success:
            logger.warning(f"Failed to read frame {index} from video '{video_path}'. Skipped.")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = PIL.Image.fromarray(frame)
    return [frames[index] for index in indices if index in frames]


def _extract_video(video: Video, config: PretrainedConfig) -> List[PIL.Image.Image]:
    num_frames = config.num_video_frames
    fps = getattr(config, "fps", 0.0)
    frames = _load_video(video.path, num_frames=num_frames, fps=fps)
    return frames


def extract_media(
    messages: List[Dict[str, Any]],
    config: Optional[PretrainedConfig] = None,
    draft: bool = False,
) -> Dict[str, List[Any]]:
    media = defaultdict(list)
    for message in messages:
        text = ""
        for part in make_list(message["value"]):
            if isinstance(part, str):
                for token in MEDIA_TOKENS.values():
                    if token in part:
                        logger.warning(f"Media token '{token}' found in text: '{part}'. Removed.")
                        part = part.replace(token, "").strip()
                text += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                if draft:
                    media["image"].append(part)
                else:
                    media["image"].append(_extract_image(part))
                text += MEDIA_TOKENS["image"]
            elif isinstance(part, Video):
                if draft:
                    media["image"].append(part)
                else:
                    media["image"].extend(_extract_video(part, config))
                text += MEDIA_TOKENS["image"] * config.num_video_frames
            else:
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text
    return media
