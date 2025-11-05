import argparse
import importlib.util
import json
import os
import re
from typing import List, Optional, Union

import cv2
from pydantic import BaseModel
from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat
import torch
import glob


def get_schema_from_python_path(path: str) -> str:
    schema_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("schema_module", schema_path)
    schema_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_module)

    # Get the Main class from the loaded module
    Main = schema_module.Main
    assert issubclass(
        Main, BaseModel
    ), f"The provided python file {path} does not contain a class Main that describes a JSON schema"
    return Main.schema_json()


def decode_time_token(text: str, *, duration: float, num_time_tokens: int, time_token_format: str) -> str:
    """Replace time tokens in text with actual timestamps."""
    for t in range(num_time_tokens):
        time_token = time_token_format.format(t=t)
        timestamp = round(t * duration / (num_time_tokens - 1), 2)
        text = text.replace(time_token, f"<{timestamp}>")

    # Handle out-of-range time tokens
    excess_pattern = re.compile(rf"<t(\d+)>")
    matches = excess_pattern.findall(text)
    for match in matches:
        t = int(match)
        if t >= num_time_tokens:
            timestamp = round(duration, 2)  # Map to the end of the video
            text = text.replace(f"<t{t}>", f"<{timestamp}>")

    return text


def configure_ps3_and_context_length(model):
    """Configure PS3 settings and adjust context length based on those settings."""

    # get PS3 configs from environment variables
    num_look_close = os.environ.get("NUM_LOOK_CLOSE", None)
    num_token_look_close = os.environ.get("NUM_TOKEN_LOOK_CLOSE", None)
    select_num_each_scale = os.environ.get("SELECT_NUM_EACH_SCALE", None)
    look_close_mode = os.environ.get("LOOK_CLOSE_MODE", None)
    smooth_selection_prob = os.environ.get("SMOOTH_SELECTION_PROB", None)

    # Set PS3 configs
    if num_look_close is not None:
        print("Num look close:", num_look_close)
        num_look_close = int(num_look_close)
        model.num_look_close = num_look_close
    if num_token_look_close is not None:
        print("Num token look close:", num_token_look_close)
        num_token_look_close = int(num_token_look_close)
        model.num_token_look_close = num_token_look_close
    if select_num_each_scale is not None:
        print("Select num each scale:", select_num_each_scale)
        select_num_each_scale = [int(x) for x in select_num_each_scale.split("+")]
        model.get_vision_tower().vision_tower.vision_model.max_select_num_each_scale = select_num_each_scale
    if look_close_mode is not None:
        print("Look close mode:", look_close_mode)
        model.look_close_mode = look_close_mode
    if smooth_selection_prob is not None:
        print("Smooth selection prob:", smooth_selection_prob)
        if smooth_selection_prob.lower() == "true":
            smooth_selection_prob = True
        elif smooth_selection_prob.lower() == "false":
            smooth_selection_prob = False
        else:
            raise ValueError(f"Invalid smooth selection prob: {smooth_selection_prob}")
        model.smooth_selection_prob = smooth_selection_prob

    # Adjust the max context length based on the PS3 config
    context_length = model.tokenizer.model_max_length
    if num_look_close is not None:
        context_length = max(context_length, num_look_close * 2560 // 4 + 1024)
    if num_token_look_close is not None:
        context_length = max(context_length, num_token_look_close // 4 + 1024)
    context_length = max(getattr(model.tokenizer, "model_max_length", context_length), context_length)
    model.config.model_max_length = context_length
    model.config.tokenizer_model_max_length = context_length
    model.llm.config.model_max_length = context_length
    model.llm.config.tokenizer_model_max_length = context_length
    model.tokenizer.model_max_length = context_length



def generate_response(
    model, 
    media: Optional[Union[str, List[str]]] = None, 
    text: Optional[str] = None,
    response_format: Optional[object] = None
) -> str:
    """
    Generate response from LLaVA model given media and text inputs.
    
    Args:
        model: Loaded LLaVA model instance
        media: Single media path or list of media paths (images/videos)
        text: Text prompt to accompany the media
        response_format: Optional response format for structured output
        
    Returns:
        str: Generated response from the model
    """
    # Prepare multi-modal prompt
    has_video = False
    duration = None
    prompt = []
    
    # Process media inputs
    if media is not None:
        # Convert single media to list for uniform processing
        media_list = [media] if isinstance(media, str) else media
        
        for media_path in media_list:
            if any(media_path.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]):
                # Handle image
                media_obj = Image(media_path)
            elif any(media_path.endswith(ext) for ext in [".mp4", ".mkv", ".webm", ".avi"]):
                # Handle video
                cap = cv2.VideoCapture(media_path)
                duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                media_obj = Video(media_path)
                has_video = True
            else:
                raise ValueError(f"Unsupported media type: {media_path}")
            
            prompt.append(media_obj)
    
    # Add text to prompt if provided
    if text is not None:
        prompt.append(text)
    
    # Generate response
    response = model.generate_content(prompt, response_format=response_format)
    
    # Decode time tokens for video if applicable
    if (has_video and 
        hasattr(model.config, 'num_time_tokens') and model.config.num_time_tokens is not None and
        hasattr(model.config, 'time_token_format') and model.config.time_token_format is not None and
        duration is not None):
        response = decode_time_token(
            response,
            duration=duration,
            num_time_tokens=model.config.num_time_tokens,
            time_token_format=model.config.time_token_format,
        )
    
    return response


def get_schema_from_python_path(path: str) -> str:
    """Load JSON schema from a Python file containing a Pydantic model."""
    schema_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("schema_module", schema_path)
    schema_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_module)

    # Get the Main class from the loaded module
    Main = schema_module.Main
    assert issubclass(
        Main, BaseModel
    ), f"The provided python file {path} does not contain a class Main that describes a JSON schema"
    return Main.schema_json()


def configure_ps3_and_context_length(model):
    """Configure PS3 settings and adjust context length based on those settings."""
    # get PS3 configs from environment variables
    num_look_close = os.environ.get("NUM_LOOK_CLOSE", None)
    num_token_look_close = os.environ.get("NUM_TOKEN_LOOK_CLOSE", None)
    select_num_each_scale = os.environ.get("SELECT_NUM_EACH_SCALE", None)
    look_close_mode = os.environ.get("LOOK_CLOSE_MODE", None)
    smooth_selection_prob = os.environ.get("SMOOTH_SELECTION_PROB", None)

    # Set PS3 configs
    if num_look_close is not None:
        print("Num look close:", num_look_close)
        num_look_close = int(num_look_close)
        model.num_look_close = num_look_close
    if num_token_look_close is not None:
        print("Num token look close:", num_token_look_close)
        num_token_look_close = int(num_token_look_close)
        model.num_token_look_close = num_token_look_close
    if select_num_each_scale is not None:
        print("Select num each scale:", select_num_each_scale)
        select_num_each_scale = [int(x) for x in select_num_each_scale.split("+")]
        model.get_vision_tower().vision_tower.vision_model.max_select_num_each_scale = select_num_each_scale
    if look_close_mode is not None:
        print("Look close mode:", look_close_mode)
        model.look_close_mode = look_close_mode
    if smooth_selection_prob is not None:
        print("Smooth selection prob:", smooth_selection_prob)
        if smooth_selection_prob.lower() == "true":
            smooth_selection_prob = True
        elif smooth_selection_prob.lower() == "false":
            smooth_selection_prob = False
        else:
            raise ValueError(f"Invalid smooth selection prob: {smooth_selection_prob}")
        model.smooth_selection_prob = smooth_selection_prob

    # Adjust the max context length based on the PS3 config
    context_length = model.tokenizer.model_max_length
    if num_look_close is not None:
        context_length = max(context_length, num_look_close * 2560 // 4 + 1024)
    if num_token_look_close is not None:
        context_length = max(context_length, num_token_look_close // 4 + 1024)
    context_length = max(getattr(model.tokenizer, "model_max_length", context_length), context_length)
    model.config.model_max_length = context_length
    model.config.tokenizer_model_max_length = context_length
    model.llm.config.model_max_length = context_length
    model.llm.config.tokenizer_model_max_length = context_length
    model.tokenizer.model_max_length = context_length

def process_folder(model, input_folder: str, output_folder: str, text: str = None, response_format = None):
    """
    Process all images in input folder and save results to output folder.
    
    Args:
        model: Loaded LLaVA model instance
        input_folder: Path to folder containing images
        output_folder: Path to folder where results will be saved
        text: Text prompt to accompany each image
        response_format: Optional response format for structured output
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIFF"]
    
    # Get all image files from input folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not image_files:
        print(colored(f"No image files found in {input_folder}", "red"))
        return
    
    print(colored(f"Found {len(image_files)} images to process", "green"))
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(colored(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}", "yellow"))
        
        try:
            # Generate response for this image
            response = generate_response(
                model=model,
                media=image_path,
                text=text,
                response_format=response_format
            )
            
            # Create output filename - CHANGED TO .md FOR OMNIDOCBENCH
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{image_basename}.md"  # ✅ Changed from .txt to .md
            output_path = os.path.join(output_folder, output_filename)

            print(">>>response:", response)
            
            # Save response to markdown file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response)
            
            print(colored(f"✓ Saved result to {output_filename}", "green"))
            
        except Exception as e:
            print(colored(f"✗ Error processing {os.path.basename(image_path)}: {str(e)}", "red"))
            continue
    
    print(colored(f"Finished processing all images. Results saved to {output_folder}", "cyan", attrs=["bold"]))
 

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--lora-path", "-l", type=str, default=None)
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    parser.add_argument("--text", type=str, default="Free OCR.")
    parser.add_argument("--media", type=str, nargs="+", default=["/lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/data/OmniDocBench/images/jiaocaineedrop_jiaocai_needrop_en_3146.jpg"])
    parser.add_argument("--num_video_frames", "-nf", type=int, default=-1)
    parser.add_argument("--video_max_tiles", "-vm", type=int, default=-1)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--json-schema", type=str, default=None)
    parser.add_argument("--input-folder", type=str, help="Path to folder containing images to process")
    parser.add_argument("--output-folder", type=str, help="Path to folder where results will be saved")

    args = parser.parse_args()

    # Convert json mode to response format
    if not args.json_mode:
        response_format = None
    elif args.json_schema is None:
        response_format = ResponseFormat(type="json_object")
    else:
        schema_str = get_schema_from_python_path(args.json_schema)
        print(schema_str)
        response_format = ResponseFormat(type="json_schema", json_schema=JsonSchemaResponseFormat(schema=schema_str))

    # Load model
    if args.lora_path is None:
        model = llava.load(args.model_path, model_base=None)
        model = model.to(torch.bfloat16)
    else:
        model = llava.load(args.lora_path, model_base=args.model_path)

    # Override num_video_frames and video_max_tiles
    if args.num_video_frames > 0:
        model.config.num_video_frames = args.num_video_frames

    if args.video_max_tiles > 0:
        model.config.video_max_tiles = args.video_max_tiles
        model.llm.config.video_max_tiles = args.video_max_tiles

    # Configure PS3 and adjust context length
    configure_ps3_and_context_length(model)

    # Set conversation mode
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

    # # Prepare multi-modal prompt
    # has_video = False
    # prompt = []
    # if args.media is not None:
    #     for media in args.media or []:
    #         if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
    #             media = Image(media)
    #         elif any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
    #             cap = cv2.VideoCapture(media)
    #             duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    #             media = Video(media)
    #             has_video = True
    #         else:
    #             raise ValueError(f"Unsupported media type: {media}")
    #         prompt.append(media)
    # if args.text is not None:
    #     prompt.append(args.text)

    # # Generate response
    # response = model.generate_content(prompt, response_format=response_format)

    # if has_video and model.config.num_time_tokens is not None and model.config.time_token_format is not None:
    #     # Decode time tokens
    #     response = decode_time_token(
    #         response,
    #         duration=duration,
    #         num_time_tokens=model.config.num_time_tokens,
    #         time_token_format=model.config.time_token_format,
    #     )
    # print(colored(response, "cyan", attrs=["bold"]))

    process_folder(
        model=model,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        text=args.text,
        response_format=response_format
    )


if __name__ == "__main__":
    main()
