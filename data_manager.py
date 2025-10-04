"""
Data Management Script for NASA Exoplanet Detection Challenge 2025
This script helps manage and explore the various exoplanet datasets
"""

import pandas as pd
import glob
import os
from pathlib import Path
import argparse

def list_data_files():
    """List all data files in the data directory"""
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return {}
    
    patterns = {
        "K2 Exoplanet Data": "k2pandc_*.csv",
        "TOI (TESS Objects of Interest)": "TOI_*.csv", 
        "Cumulative Exoplanet Data": "cumulative_*.csv"
    }
    
    files_found = {}
    print("📁 Available Data Files:")
    print("=" * 50)
    
    for data_type, pattern in patterns.items():
        files = list(data_dir.glob(pattern))
        if files:
            # Sort by modification time, newest first
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            files_found[data_type] = files
            
            print(f"\n🔹 {data_type}:")
            for i, file in enumerate(files):
                size_mb = file.stat().st_size / (1024*1024)
                print(f"   {i+1}. {file.name} ({size_mb:.1f} MB)")
        else:
            print(f"\n❌ {data_type}: No files found")
    
    return files_found

def analyze_dataset(file_path):
    """Analyze a single dataset and show basic information"""
    print(f"\n📊 Analyzing: {file_path.name}")
    print("-" * 50)
    
    try:
        # Try to read the file
        df = pd.read_csv(file_path, comment='#')
        
        print(f"📏 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"💾 File Size: {file_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Show column information
        print(f"\n📋 Column Information:")
        for i, col in enumerate(df.columns[:10]):  # Show first 10 columns
            non_null = df[col].notna().sum()
            print(f"   {i+1:2d}. {col[:40]:<40} ({non_null:,} non-null)")
        
        if len(df.columns) > 10:
            print(f"   ... and {len(df.columns) - 10} more columns")
        
        # Show key statistics if it's exoplanet data
        if 'disposition' in df.columns:
            print(f"\n🌌 Exoplanet Statistics:")
            disposition_counts = df['disposition'].value_counts()
            for status, count in disposition_counts.items():
                print(f"   {status}: {count:,}")
        
        if 'discoverymethod' in df.columns:
            print(f"\n🔬 Discovery Methods:")
            method_counts = df['discoverymethod'].value_counts().head(5)
            for method, count in method_counts.items():
                print(f"   {method}: {count:,}")
        
        # Sample data
        print(f"\n📋 Sample Data (first 3 rows):")
        sample_cols = df.columns[:5]  # Show first 5 columns
        print(df[sample_cols].head(3).to_string(index=False))
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def create_data_summary():
    """Create a summary of all available datasets"""
    files_found = list_data_files()
    
    if not files_found:
        print("\n❌ No data files found!")
        return
    
    print(f"\n📊 Dataset Summary:")
    print("=" * 70)
    
    total_size = 0
    total_files = 0
    
    for data_type, files in files_found.items():
        print(f"\n🔹 {data_type}:")
        type_size = 0
        
        for file in files:
            size_mb = file.stat().st_size / (1024*1024)
            type_size += size_mb
            total_size += size_mb
            total_files += 1
            
            # Quick peek at file structure
            try:
                df = pd.read_csv(file, comment='#', nrows=0)  # Just read headers
                cols = len(df.columns)
                print(f"   📄 {file.name}")
                print(f"      Size: {size_mb:.1f} MB, Columns: {cols}")
            except:
                print(f"   📄 {file.name} (error reading)")
        
        print(f"   📊 Total for {data_type}: {type_size:.1f} MB")
    
    print(f"\n📈 Overall Summary:")
    print(f"   Total Files: {total_files}")
    print(f"   Total Size: {total_size:.1f} MB")

def interactive_explorer():
    """Interactive data explorer"""
    files_found = list_data_files()
    
    if not files_found:
        print("\n❌ No data files found!")
        return
    
    # Create a flat list of all files with indices
    all_files = []
    for data_type, files in files_found.items():
        for file in files:
            all_files.append((data_type, file))
    
    print(f"\n🔍 Interactive Data Explorer")
    print("=" * 40)
    print("Select a file to analyze:")
    
    for i, (data_type, file) in enumerate(all_files):
        print(f"{i+1:2d}. [{data_type}] {file.name}")
    
    try:
        choice = input(f"\nEnter choice (1-{len(all_files)}) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            return
        
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(all_files):
            data_type, selected_file = all_files[choice_idx]
            print(f"\n🎯 Selected: [{data_type}] {selected_file.name}")
            analyze_dataset(selected_file)
        else:
            print("❌ Invalid choice!")
            
    except (ValueError, KeyboardInterrupt):
        print("\n👋 Goodbye!")

def check_data_integrity():
    """Check data integrity and provide recommendations"""
    print("🔍 Data Integrity Check")
    print("=" * 40)
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found!")
        print("💡 Recommendation: Create a 'data' directory and add your CSV files")
        return
    
    files_found = list_data_files()
    issues = []
    recommendations = []
    
    # Check for expected file types
    expected_patterns = ["k2pandc_*.csv", "TOI_*.csv", "cumulative_*.csv"]
    
    for pattern in expected_patterns:
        files = list(data_dir.glob(pattern))
        if not files:
            issues.append(f"No files matching pattern: {pattern}")
            recommendations.append(f"Consider adding files matching: {pattern}")
    
    # Check file sizes
    for data_type, files in files_found.items():
        for file in files:
            size_mb = file.stat().st_size / (1024*1024)
            if size_mb < 0.1:
                issues.append(f"Very small file: {file.name} ({size_mb:.2f} MB)")
            elif size_mb > 50:
                issues.append(f"Very large file: {file.name} ({size_mb:.1f} MB)")
    
    # Report results
    if issues:
        print("\n⚠️  Issues Found:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("\n✅ No issues found!")
    
    if recommendations:
        print("\n💡 Recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    print(f"\n📊 Summary: {len(files_found)} dataset types, {sum(len(files) for files in files_found.values())} total files")

def main():
    parser = argparse.ArgumentParser(description="NASA Exoplanet Data Manager")
    parser.add_argument('--action', choices=['list', 'analyze', 'explore', 'check', 'summary'], 
                       default='summary', help='Action to perform')
    parser.add_argument('--file', type=str, help='Specific file to analyze')
    
    args = parser.parse_args()
    
    print("🌌 NASA Exoplanet Detection Challenge 2025")
    print("📁 Data Management Tool")
    print("=" * 50)
    
    if args.action == 'list':
        list_data_files()
    elif args.action == 'analyze' and args.file:
        file_path = Path("data") / args.file
        if file_path.exists():
            analyze_dataset(file_path)
        else:
            print(f"❌ File not found: {file_path}")
    elif args.action == 'explore':
        interactive_explorer()
    elif args.action == 'check':
        check_data_integrity()
    elif args.action == 'summary':
        create_data_summary()
    else:
        print("Usage examples:")
        print("  python data_manager.py --action list")
        print("  python data_manager.py --action analyze --file k2pandc_2025.09.27_13.36.16.csv")
        print("  python data_manager.py --action explore")
        print("  python data_manager.py --action check")

if __name__ == "__main__":
    main()