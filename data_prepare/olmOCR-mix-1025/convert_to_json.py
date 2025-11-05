#!/usr/bin/env python3
"""
Script to combine all train parquet files into a single JSON file.
Preserves all columns from the original parquet files.
"""

import pandas as pd
import glob
import json
from pathlib import Path

def combine_train_parquets_to_json(output_file="combined_train_data.json"):
    """
    Load all *_train.parquet files and combine them into a single JSON file.
    
    Args:
        output_file: Name of the output JSON file
    """
    # Find all train parquet files
    train_files = sorted(glob.glob("*_train.parquet"))
    
    if not train_files:
        print("No train parquet files found!")
        return
    
    print(f"Found {len(train_files)} train parquet files:")
    for file in train_files:
        print(f"  - {file}")
    
    # List to store all dataframes
    all_dfs = []
    
    # Load each parquet file
    for file in train_files:
        print(f"\nLoading {file}...")
        df = pd.read_parquet(file)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        all_dfs.append(df)
    
    # Combine all dataframes
    print("\nCombining all dataframes...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined shape: {combined_df.shape}")
    print(f"Combined columns: {list(combined_df.columns)}")
    
    # Save to JSON
    print(f"\nSaving to {output_file}...")
    # Convert to records format (list of dictionaries)
    records = combined_df.to_dict(orient='records')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print(f"\nSuccessfully saved {len(records)} records to {output_file}")
    print(f"File size: {Path(output_file).stat().st_size / (1024**2):.2f} MB")
    
    # Print sample of first record
    print("\nSample of first record:")
    for key, value in list(records[0].items()):
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    combine_train_parquets_to_json()