# Data Preparation Guide

This guide explains how to prepare the training datasets for Easy Deep-OCR. The model requires two datasets across two training stages.

## 📋 Overview

| Stage | Dataset | Size | Purpose | Format |
|-------|---------|------|---------|--------|
| Stage 1: Alignment | LLaVA-CC3M-Pretrain-595K | 595K | Initialize projector | Direct download |
| Stage 2: Pretraining | olmOCR-mix-1025 | 260k | Train full model | Requires conversion |

## 🔧 Prerequisites

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Login to Hugging Face (if needed)
huggingface-cli login
```

## 📦 Stage 1: LLaVA-CC3M-Pretrain-595K

### Download

This dataset can be used directly without conversion:

```bash
# Download the dataset
huggingface-cli download liuhaotian/LLaVA-CC3M-Pretrain-595K \
    --repo-type dataset \
    --local-dir ./data/LLaVA-CC3M-Pretrain-595K
```

### Dataset Structure

```
data/LLaVA-CC3M-Pretrain-595K/
├── images/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
└── chat.json
```

The dataset is ready to use for Stage 1 alignment training.

## 📦 Stage 2: olmOCR-mix-1025

This dataset requires format conversion to be compatible with the VILA training pipeline.

### Step 1: Download Raw Data

```bash
# Download the dataset
huggingface-cli download allenai/olmOCR-mix-1025 \
    --repo-type dataset \
    --local-dir ./data/olmOCR-mix-1025-raw
```

### Step 2: Convert to Intermediate JSON Format

The first conversion script transforms the dataset into a standardized JSON format:

```bash
cd data_prepare

python convert_to_json.py

```

Note: You must update the file paths inside the Python script before running it.

**What this does:**
- Parses the original olmOCR format
- Extracts image paths and annotations
- Creates a unified JSON structure
- Validates data integrity

**Output structure:**
```
data/olmOCR-mix-1025-json/
├── annotations.json
└── images/
    ├── doc_0001.pdf (or .png/.jpg)
    ├── doc_0002.pdf
    └── ...
```

### Step 3: Convert to LLaVA Training Format

The second conversion script transforms the data into the format required for VILA training, with images converted to PNG:

```bash
python convert_to_llava.py 

```

**What this does:**
- Converts all images to PNG format
- Creates conversation-style annotations
- Formats data for multi-turn dialogue training
- Generates the final `chat.json` file

**Output structure:**
```
data/olmOCR-mix-1025-llava/
├── images/
│   ├── doc_0001.png
│   ├── doc_0002.png
│   └── ...
└── chat.json
```

### Final Dataset Format

The `chat.json` file follows this structure:

```json
[
    {
        "id": "doc_0001",
        "image": "images/doc_0001.png",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nFree OCR."
            },
            {
                "from": "gpt",
                "value": "Extracted text content..."
            }
        ]
    },
    ...
]
```

## 🚀 Complete Pipeline

Run all steps in sequence:

```bash
# Step 1: Download LLaVA-CC3M
huggingface-cli download liuhaotian/LLaVA-CC3M-Pretrain-595K \
    --repo-type dataset \
    --local-dir ./data/LLaVA-CC3M-Pretrain-595K

# Step 2: Download olmOCR-mix
huggingface-cli download allenai/olmOCR-mix-1025 \
    --repo-type dataset \
    --local-dir ./data/olmOCR-mix-1025-raw

# Step 3: Convert olmOCR to JSON
cd data_prepare
python convert_to_json.py 

# Step 4: Convert to LLaVA format with PNG images
python convert_to_llava.py 
```

## 📁 Final Directory Structure

After completing all steps, your data directory should look like this:

```
data/
├── LLaVA-CC3M-Pretrain-595K/          # Stage 1 data (ready to use)
│   ├── images/
│   └── chat.json
│
├── olmOCR-mix-1025-raw/               # Downloaded raw data
│   └── [original files]
│
├── olmOCR-mix-1025-json/              # Intermediate format
│   ├── annotations.json
│   └── images/
│
└── olmOCR-mix-1025-llava/             # Stage 2 data (ready to use)
    ├── images/                         # All images in PNG format
    └── chat.json
```

## 🔍 Data Preparation Scripts

All conversion scripts are located in the `data_prepare/` folder:

```
data_prepare/
├── convert_to_json.py      # Step 1: Raw data → JSON format
├── convert_to_llava.py     # Step 2: JSON → LLaVA format + PNG conversion
└── README.md               # This file
```


```


## 🐛 Troubleshooting

### Issue: Download fails or is interrupted

```bash
# Resume download by re-running the command
huggingface-cli download <dataset> --resume-download
```

## 📚 Next Steps

After data preparation is complete, proceed to training:

1. **Stage 1 - Alignment**: Use `LLaVA-CC3M-Pretrain-595K`
   ```bash
   bash scripts/NVILA-Lite/align_ocr.sh
   ```

2. **Stage 2 - Pretraining**: Use `olmOCR-mix-1025-llava`
   ```bash
   bash scripts/NVILA-Lite/pretrain_ocr.sh
   ```

See the main [README.md](../README.md) for detailed t