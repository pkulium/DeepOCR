import json
import fitz
import os
from PIL import Image
import io
from multiprocessing import Pool, cpu_count
import traceback
import signal
import sys

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
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
        
        images.append(img)
    
    pdf_document.close()
    return images

def process_single_item(args):
    """
    Process a single item: convert PDF to PNG and create transformed data.
    This function is called in parallel.
    
    Args:
        args: tuple of (idx, item, pdf_base_dir)
    
    Returns:
        tuple: (success, (pdf_item, png_item) or error_message, idx)
    """
    idx, item, pdf_base_dir = args
    
    try:
        # Extract the PDF filename from pdf_relpath
        pdf_relpath = item.get('pdf_relpath', '')
        
        # Split by ':' to get the part after the colon (the actual file path)
        if ':' in pdf_relpath:
            # Gets "rg_341/rg_341-9/40989274-311170626-page-1.pdf"
            image_path = pdf_relpath.split(':', 1)[1]
            id_path = f"pdf_tarballs/{image_path}"
        else:
            image_path = pdf_relpath
            id_path = pdf_relpath
        
        natural_text = item.get('natural_text', '')
        if not natural_text:
            return (False, f"missing natural_text", idx)

        # Create PDF item (before conversion)
        pdf_item = {
            "id": id_path,
            "image": image_path,  # PDF path
            "conversations": [
                {
                    "from": "human",
                    "value": "Free OCR.\n<image>"
                },
                {
                    "from": "gpt",
                    "value": natural_text
                }
            ]
        }

        # Convert PDF to PNG
        # Construct full PDF path using only the extracted path (after the colon)
        # This assumes PDFs are in pdf_base_dir/pdf_tarballs/{image_path}
        pdf_full_path = os.path.join(pdf_base_dir, "pdf_tarballs", image_path)
        
        # Check if file exists
        if not os.path.exists(pdf_full_path):
            return (False, f"PDF file not found at: {pdf_full_path}", idx)
        
        # Convert PDF to images
        images = pdf_to_images_high_quality(pdf_full_path, dpi=144, image_format="PNG")
        
        if not images:
            return (False, f"no images generated from PDF", idx)
        
        # Create output directory structure
        png_path = image_path.replace('.pdf', '.png')
        png_full_path = f"png_tarballs/{png_path}"
        png_dir = os.path.dirname(png_full_path)
        
        os.makedirs(png_dir, exist_ok=True)
        
        # Save the first page as PNG (or handle multi-page PDFs as needed)
        images[0].save(png_full_path, format='PNG')
        
        # Create PNG item (after conversion)
        png_image_path = image_path.replace('.pdf', '.png')
        png_item = {
            "id": id_path,
            "image": png_image_path,  # PNG path
            "conversations": [
                {
                    "from": "human",
                    "value": "Free OCR.\n<image>"
                },
                {
                    "from": "gpt",
                    "value": natural_text
                }
            ]
        }
        
        return (True, (pdf_item, png_item), idx)
        
    except Exception as e:
        # Catch all exceptions and return detailed error
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        return (False, error_msg, idx)

def save_json_safely(data, output_file):
    """
    Safely save JSON data with atomic write operation.
    """
    temp_file = output_file + '.tmp'
    try:
        print(f"Saving {len(data)} items to {output_file}...")
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # Atomic move
        os.replace(temp_file, output_file)
        print(f"Successfully saved to: {output_file}")
        return True
    except Exception as e:
        print(f"Error saving {output_file}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False

def transform_json_data(input_file, pdf_output_file, png_output_file, pdf_base_dir="", num_workers=None):
    """
    Transform JSON data from original format to new conversation format (parallelized).
    Creates two separate JSON files: one for PDFs and one for PNGs.
    
    Args:
        input_file (str): Path to input JSON file
        pdf_output_file (str): Path to output JSON file for PDF references
        png_output_file (str): Path to output JSON file for PNG references
        pdf_base_dir (str): Base directory where PDFs are stored (if needed)
        num_workers (int): Number of parallel workers. If None, uses cpu_count()
    """
    
    # Read the original JSON file
    print("Reading input file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"Loaded {len(original_data)} items from input file")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Using {num_workers} worker processes")
    print(f"Processing {len(original_data)} items...")
    
    # Prepare arguments for parallel processing
    args_list = [(idx, item, pdf_base_dir) for idx, item in enumerate(original_data)]
    
    # Process items in parallel
    pdf_data = []
    png_data = []
    skipped_count = 0
    pool = None
    
    def cleanup_and_save():
        """Save current progress before exiting"""
        print("\n\nCleaning up and saving progress...")
        if pool is not None:
            pool.terminate()
            pool.join()
        
        # Save whatever we have
        print(f"\nSaving {len(pdf_data)} PDF items and {len(png_data)} PNG items...")
        save_json_safely(pdf_data, pdf_output_file)
        save_json_safely(png_data, png_output_file)
        
        print(f"\n{'='*50}")
        print(f"Saved {len(pdf_data)} items")
        print(f"Skipped {skipped_count} items")
        print(f"Total processed: {len(pdf_data) + skipped_count}/{len(original_data)}")
        print(f"{'='*50}")
    
    # Register signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}")
        cleanup_and_save()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        pool = Pool(processes=num_workers)
        
        # Use imap for ordered results
        print("Starting parallel processing...")
        results = pool.imap(process_single_item, args_list, chunksize=10)
        
        for i, result in enumerate(results, 1):
            try:
                success, data, idx = result
                
                if success:
                    pdf_item, png_item = data
                    pdf_data.append(pdf_item)
                    png_data.append(png_item)
                    if i % 100 == 0:  # Print progress every 100 items
                        print(f"Progress: {i}/{len(original_data)} items processed ({len(pdf_data)} successful, {skipped_count} skipped)")
                        # Periodic save to prevent data loss
                        if i % 1000 == 0:
                            print("Saving checkpoint...")
                            save_json_safely(pdf_data, pdf_output_file + '.checkpoint')
                            save_json_safely(png_data, png_output_file + '.checkpoint')
                else:
                    skipped_count += 1
                    if skipped_count <= 10:  # Only print first 10 errors to avoid spam
                        print(f"Skipping item {idx}: {data}")
                    elif skipped_count == 11:
                        print(f"... (suppressing further error messages)")
            
            except Exception as e:
                print(f"Error processing result {i}: {e}")
                skipped_count += 1
        
        print("\nClosing pool...")
        pool.close()
        pool.join()
        print("All processes completed!")
        pool = None  # Mark as closed
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        cleanup_and_save()
        return
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        print(traceback.format_exc())
        cleanup_and_save()
        return
    
    # Save final results
    print(f"\nWriting final results...")
    save_json_safely(pdf_data, pdf_output_file)
    save_json_safely(png_data, png_output_file)
    
    # Clean up checkpoint files
    for checkpoint_file in [pdf_output_file + '.checkpoint', png_output_file + '.checkpoint']:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"Removed checkpoint file: {checkpoint_file}")
    
    print(f"\n{'='*50}")
    print(f"Successfully transformed {len(pdf_data)} items")
    print(f"Skipped {skipped_count} items")
    print(f"Total processed: {len(pdf_data) + skipped_count}/{len(original_data)}")
    print(f"{'='*50}")

# Usage example
if __name__ == "__main__":
    # Replace with your actual file paths
    input_filename = "/lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/data/olmOCR-mix-1025/combined_train_data.json"
    pdf_output_filename = "/lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/data/olmOCR-mix-1025/transformed_data_pdf.json"
    png_output_filename = "/lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/data/olmOCR-mix-1025/transformed_data_png.json"
    
    # Set the base directory where PDFs are stored (adjust as needed)
    # The PDFs should be in: pdf_base_directory/pdf_tarballs/{extracted_path}
    pdf_base_directory = "/lustre/hdd/LAS/wzhang-lab/mingl/code/vllm/vlm_ocr/workspace/data/olmOCR-mix-1025/"
    
    # Optionally specify number of workers (default is cpu_count())
    # num_workers = 8  # Uncomment to use specific number
    
    try:
        transform_json_data(input_filename, pdf_output_filename, png_output_filename, pdf_base_directory)
        # Or with specific number of workers:
        # transform_json_data(input_filename, pdf_output_filename, png_output_filename, pdf_base_directory, num_workers=8)
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_filename}'")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in input file")
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())