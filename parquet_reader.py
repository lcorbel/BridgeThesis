#!/usr/bin/env python3
"""
Parquet File Reader and Summary Tool
Reads all parquet files from the bridge analysis output and displays them with pretty formatting
Can also read a specific parquet file or Python file if specified as an argument
"""

import pandas as pd
import os
import argparse
from pathlib import Path
import numpy as np
import ast
import tokenize
from io import BytesIO

def format_bytes(bytes_val):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"

def get_column_info(df):
    """Get detailed column information"""
    info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
        unique_count = df[col].nunique()
        
        # Sample values (first few non-null)
        sample_vals = df[col].dropna().head(3).tolist()
        if len(sample_vals) > 0:
            sample_str = ', '.join([str(x)[:20] for x in sample_vals])
            if len(sample_str) > 60:
                sample_str = sample_str[:57] + "..."
        else:
            sample_str = "No data"
        
        info.append({
            'Column': col,
            'Type': dtype,
            'Nulls': f"{null_count} ({null_pct:.1f}%)",
            'Unique': unique_count,
            'Sample': sample_str
        })
    
    return pd.DataFrame(info)

def analyze_parquet_file(file_path):
    """Analyze a single parquet file and return summary info"""
    try:
        df = pd.read_parquet(file_path)
        file_size = os.path.getsize(file_path)
        
        print(f"\n{'='*80}")
        print(f"📁 FILE: {os.path.basename(file_path)}")
        print(f"{'='*80}")
        
        print(f"📊 BASIC INFO:")
        print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   File Size: {format_bytes(file_size)}")
        print(f"   Memory Usage: {format_bytes(df.memory_usage(deep=True).sum())}")
        
        # Data quality summary
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        null_percentage = (null_cells / total_cells) * 100 if total_cells > 0 else 0
        
        print(f"   Data Quality: {null_cells:,} nulls out of {total_cells:,} cells ({null_percentage:.2f}%)")
        
        # Column information
        print(f"\n📋 COLUMN DETAILS:")
        col_info = get_column_info(df)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 25)
        print(col_info.to_string(index=False))
        
        # Data preview
        print(f"\n👀 DATA PREVIEW (First 5 rows):")
        pd.set_option('display.max_columns', 10)  # Limit columns for preview
        pd.set_option('display.width', 120)
        print(df.head().to_string())
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n📈 NUMERIC SUMMARY:")
            pd.set_option('display.float_format', '{:.3f}'.format)
            print(df[numeric_cols].describe().round(3).to_string())
        
        # Object columns value counts for key columns
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            print(f"\n🔤 CATEGORICAL SUMMARY (Top 5 values per column):")
            for col in object_cols[:5]:  # Limit to first 5 object columns
                print(f"\n   {col}:")
                value_counts = df[col].value_counts().head(5)
                for val, count in value_counts.items():
                    pct = (count / len(df)) * 100
                    print(f"     {str(val)[:30]}: {count:,} ({pct:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR reading {os.path.basename(file_path)}: {e}")
        return False

def main():
    """Main function to read and display parquet files"""
    parser = argparse.ArgumentParser(description='Read and analyze parquet files from bridge dataset')
    parser.add_argument('file', nargs='?', help='Specific parquet file to read (optional)')
    parser.add_argument('--dir', default='bridge_output_run', help='Directory containing parquet files (default: bridge_output_run)')
    
    args = parser.parse_args()
    
    # If specific file provided
    if args.file:
        # Check if it's a full path or just filename
        if os.path.exists(args.file):
            file_path = args.file
        elif os.path.exists(os.path.join(args.dir, args.file)):
            file_path = os.path.join(args.dir, args.file)
        else:
            # Try adding .parquet extension if not present
            if not args.file.endswith('.parquet'):
                test_file = args.file + '.parquet'
                if os.path.exists(os.path.join(args.dir, test_file)):
                    file_path = os.path.join(args.dir, test_file)
                else:
                    print(f"❌ File not found: {args.file}")
                    print(f"   Tried: {args.file}, {os.path.join(args.dir, args.file)}, {os.path.join(args.dir, test_file)}")
                    return
            else:
                print(f"❌ File not found: {args.file}")
                return
        
        print("🔍 BRIDGE DATASET ANALYSIS - SINGLE PARQUET FILE READER")
        print(f"📂 Reading specific file: {os.path.basename(file_path)}")
        print(f"🕐 Analysis started at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if analyze_parquet_file(file_path):
            print(f"\n✅ Successfully analyzed: {os.path.basename(file_path)}")
        else:
            print(f"\n❌ Failed to analyze: {os.path.basename(file_path)}")
        
        print(f"🕐 Analysis completed at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return
    
    # Original logic for reading all files
    output_dir = args.dir
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory '{output_dir}' not found!")
        return
    
    # Get all parquet files
    parquet_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.parquet')])
    
    if not parquet_files:
        print(f"❌ No parquet files found in '{output_dir}'!")
        return
    
    print("🔍 BRIDGE DATASET ANALYSIS - PARQUET FILES READER")
    print(f"📂 Reading {len(parquet_files)} parquet files from: {output_dir}")
    print(f"🕐 Analysis started at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful_reads = 0
    total_size = 0
    total_rows = 0
    
    # Process each file
    for parquet_file in parquet_files:
        file_path = os.path.join(output_dir, parquet_file)
        
        if analyze_parquet_file(file_path):
            successful_reads += 1
            # Add to totals
            try:
                df = pd.read_parquet(file_path)
                total_rows += len(df)
                total_size += os.path.getsize(file_path)
            except:
                pass
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"📋 OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successfully read: {successful_reads}/{len(parquet_files)} files")
    print(f"📊 Total data: {total_rows:,} rows across all files")
    print(f"💾 Total storage: {format_bytes(total_size)}")
    print(f"🕐 Analysis completed at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_reads < len(parquet_files):
        print(f"⚠️  {len(parquet_files) - successful_reads} files had errors")

if __name__ == "__main__":
    # Set pandas display options for better formatting
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.precision', 3)
    
    main()
